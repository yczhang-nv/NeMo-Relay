// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable, bounded fixtures independent of cache TTL and eviction.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nemo_relay::error::FlowError;
use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use tokio::sync::Notify;
use uuid::Uuid;

use super::config::{ReplayConfig, ReplayMode, ReplayToolMode, ResponseCacheKeyStrategy};
use super::store::{BoxCacheFuture, CACHE_SCHEMA_VERSION, CacheEntry, CacheStore, now_unix_ms};
use crate::config::ResponseCacheConfig;
use crate::error::{AdaptiveError, Result};

const FORMAT_VERSION: u32 = 2;

/// Counts for one managed execution surface.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplayCounts {
    /// Results served from the fixture.
    pub hits: u64,
    /// Eligible keys absent from the fixture.
    pub misses: u64,
    /// Callbacks invoked live.
    pub live_calls: u64,
    /// Successful calls captured (including identical duplicates).
    pub captured: u64,
    /// Live calls without an eligible complete recording.
    pub uncaptured: u64,
}

/// Bounded diagnostics for the most recent failure; no prompt or credentials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayFailure {
    /// Concrete replay failure reason.
    pub reason: String,
    /// One-based call position in this session.
    pub call_position: u64,
    /// LLM or tool surface.
    pub surface: String,
    /// Provider or tool name.
    pub identity: String,
    /// Available model name for an LLM call.
    pub model: Option<String>,
    /// Key when key derivation succeeded.
    pub key_hash: Option<String>,
}

/// Replay activity and recording coverage, independent of telemetry draining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayReport {
    /// Fixture produced by this session (or loaded for strict replay).
    pub recording_id: String,
    /// Source fixture identity for derived recordings.
    pub source_recording_id: Option<String>,
    /// Policy used during execution.
    pub mode: ReplayMode,
    /// Tool workload performed.
    pub tool_mode: ReplayToolMode,
    /// Number of calls using each delivery path.
    pub delivery: BTreeMap<String, u64>,
    /// Buffered and streaming LLM activity.
    pub llm: ReplayCounts,
    /// Tool activity (including live-tool mode).
    pub tool: ReplayCounts,
    /// Counts by exclusion/error reason.
    pub reasons: BTreeMap<String, u64>,
    /// Most recent failure metadata.
    pub last_failure: Option<ReplayFailure>,
    /// Captures whose responses disagreed with an existing key.
    pub conflicts: u64,
    /// Captures lost to recording failures or budget checks.
    pub failed_writes: u64,
    /// Failed publication attempts; a successful retry preserves all captures.
    pub persistence_errors: u64,
    /// Finalization completed successfully. Harness completion is caller-owned.
    pub finalized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureEntry {
    surface: String,
    key_hash: String,
    response: Json,
    // Responses clients consume item completion order, which the final
    // aggregate does not preserve. Keep the original native frames as well.
    #[serde(skip_serializing_if = "Option::is_none")]
    response_stream: Option<Vec<Json>>,
    recorded_unix_ms: u64,
    identity: Json,
    #[serde(skip_serializing_if = "Option::is_none")]
    request: Option<Json>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format_version: u32,
    key_version: u32,
    recording_id: String,
    created_unix_ms: u64,
    relay_version: String,
    harness_revision: Option<String>,
    key_policy: Json,
    tool_policy: Json,
    capture_requests: bool,
    middleware: Json,
    entry_count: usize,
    checksum: String,
    coverage: ReplayReport,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    manifest: Manifest,
    entries: Vec<FixtureEntry>,
}

#[derive(Default)]
struct State {
    entries: BTreeMap<String, FixtureEntry>,
    bytes: usize,
    active: usize,
    accepting: bool,
    calls: u64,
    report: Option<ReplayReport>,
}

/// A replay fixture and recorder owned by one response-cache registration.
pub struct ReplaySession {
    config: ReplayConfig,
    key_policy: Json,
    tool_policy: Json,
    middleware: Json,
    max_bytes: usize,
    state: Arc<Mutex<State>>,
    idle: Arc<Notify>,
    finalize_lock: tokio::sync::Mutex<()>,
}

fn fixture_error(reason: &str) -> AdaptiveError {
    AdaptiveError::Storage(format!("replay: {reason}"))
}

fn key_policy(config: &ResponseCacheConfig) -> Json {
    let mut headers: Vec<_> = config
        .header_allowlist
        .iter()
        .map(|h| h.to_ascii_lowercase())
        .collect();
    headers.sort();
    headers.dedup();
    json!({"namespace": config.namespace, "strategy": config.key_strategy,
        "header_allowlist": headers, "cache_nondeterministic": config.cache_nondeterministic})
}

fn tool_policy(config: &ResponseCacheConfig) -> Json {
    // TTL, priority and bypass sampling do not affect frozen recording eligibility.
    let mut policy = serde_json::to_value(&config.tools).expect("tool policy serializes");
    if let Some(object) = policy.as_object_mut() {
        object.remove("priority");
        for name in ["default", "classes", "overrides"] {
            if let Some(value) = object.get_mut(name) {
                if name == "default" {
                    if let Some(class) = value.as_object_mut() {
                        class.remove("ttl_seconds");
                        class.remove("bypass_rate");
                    }
                } else if let Some(classes) = value.as_object_mut() {
                    for class in classes.values_mut().filter_map(Json::as_object_mut) {
                        class.remove("ttl_seconds");
                        class.remove("bypass_rate");
                    }
                }
            }
        }
    }
    policy
}

fn checksum(entries: &[FixtureEntry]) -> Result<String> {
    let canonical = serde_json_canonicalizer::to_vec(&entries)
        .map_err(|error| fixture_error(&format!("persistence_error: {error}")))?;
    Ok(format!(
        "sha256:{}",
        Sha256::digest(canonical)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    ))
}

fn resolved_path(path: &str) -> std::io::Result<PathBuf> {
    let path = Path::new(path);
    if path.exists() {
        return path.canonicalize();
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok(parent
        .canonicalize()?
        .join(path.file_name().unwrap_or_default()))
}

/// Validate replay settings before registering any intercepts.
pub(crate) fn validate(config: &ResponseCacheConfig) -> std::result::Result<(), String> {
    let Some(replay) = &config.replay else {
        return Ok(());
    };
    if config.key_strategy != ResponseCacheKeyStrategy::ExactRequest {
        return Err("replay requires key_strategy = exact_request".into());
    }
    if config.backend.kind != "in_memory" {
        return Err("replay requires the in_memory backend and its max_bytes budget".into());
    }
    let present = |path: &Option<String>| path.as_ref().is_some_and(|p| !p.trim().is_empty());
    if replay.mode != ReplayMode::Record && !present(&replay.input_path) {
        return Err("replay modes require input_path".into());
    }
    if replay.mode != ReplayMode::ReplayOnly && !present(&replay.output_path) {
        return Err("recording modes require output_path".into());
    }
    if replay.mode == ReplayMode::Record && replay.input_path.is_some() {
        return Err("record does not accept input_path".into());
    }
    if replay.mode == ReplayMode::ReplayOnly && replay.output_path.is_some() {
        return Err("replay_only does not accept output_path".into());
    }
    if let (Some(input), Some(output)) = (&replay.input_path, &replay.output_path)
        && (input == output
            || matches!((resolved_path(input), resolved_path(output)), (Ok(a), Ok(b)) if a == b))
    {
        return Err("derived recording cannot overwrite input_path".into());
    }
    if replay.tools.mode == ReplayToolMode::Recorded
        && !config.tools.as_ref().is_some_and(|t| t.enabled)
    {
        return Err(
            "recorded tool mode requires tools.enabled and explicit read-only policies".into(),
        );
    }
    Ok(())
}

impl ReplaySession {
    pub(crate) fn load(config: &ResponseCacheConfig) -> Result<Self> {
        validate(config).map_err(AdaptiveError::InvalidConfig)?;
        let replay = config.replay.clone().expect("replay section present");
        let max_bytes = config.backend.max_bytes();
        let policy = key_policy(config);
        let tools = tool_policy(config);
        let mut state = State {
            accepting: true,
            ..State::default()
        };
        let mut source = None;
        let mut inherited_coverage = None;
        let mut recording_id = Uuid::now_v7().to_string();
        if let Some(path) = &replay.input_path {
            let file =
                File::open(path).map_err(|e| fixture_error(&format!("persistence_error: {e}")))?;
            // Bound allocation before parsing; reject even if metadata understates the file.
            let mut bytes = Vec::new();
            file.take(max_bytes.saturating_add(1) as u64)
                .read_to_end(&mut bytes)
                .map_err(|e| fixture_error(&format!("persistence_error: {e}")))?;
            if bytes.len() > max_bytes {
                return Err(fixture_error(
                    "fixture_incompatible: memory budget exceeded",
                ));
            }
            let fixture: Fixture = serde_json::from_slice(&bytes)
                .map_err(|e| fixture_error(&format!("fixture_incompatible: {e}")))?;
            let m = &fixture.manifest;
            if m.format_version != FORMAT_VERSION
                || m.key_version != CACHE_SCHEMA_VERSION
                || m.key_policy != policy
                || m.tool_policy != tools
                || m.entry_count != fixture.entries.len()
                || m.checksum != checksum(&fixture.entries)?
                || m.coverage.conflicts != 0
            {
                return Err(fixture_error(
                    "fixture_incompatible: version, policy, count, checksum or conflicts",
                ));
            }
            if replay.mode == ReplayMode::ReplayOnly
                && (m.coverage.failed_writes != 0
                    || m.coverage.llm.uncaptured != 0
                    || m.coverage.tool.uncaptured != 0
                    || !m.coverage.finalized)
            {
                return Err(fixture_error(
                    "fixture_incompatible: incomplete recording coverage",
                ));
            }
            source = Some(m.recording_id.clone());
            if replay.mode == ReplayMode::ReplayOrRecord {
                inherited_coverage = Some(m.coverage.clone());
            }
            if replay.mode == ReplayMode::ReplayOnly {
                recording_id = m.recording_id.clone();
            }
            for mut entry in fixture.entries {
                if !matches!(entry.surface.as_str(), "llm" | "tool")
                    || entry
                        .key_hash
                        .strip_prefix("sha256:")
                        .is_none_or(|s| s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()))
                {
                    return Err(fixture_error("fixture_incompatible: entry surface or key"));
                }
                // Preserve opt-out when deriving from a fixture that captured requests.
                if !replay.capture_requests {
                    entry.request = None;
                }
                let key = entry.key_hash.clone();
                if let Some(previous) = state.entries.get(&key) {
                    if previous.surface != entry.surface
                        || previous.response != entry.response
                        || previous.response_stream != entry.response_stream
                    {
                        return Err(fixture_error(
                            "recording_conflict: duplicate key responses disagree",
                        ));
                    }
                    continue;
                }
                state.bytes += entry_bytes(&entry)?;
                if state.bytes > max_bytes {
                    return Err(fixture_error(
                        "fixture_incompatible: resident memory budget exceeded",
                    ));
                }
                state.entries.insert(key, entry);
            }
        }
        state.report = Some(ReplayReport {
            recording_id,
            source_recording_id: source,
            mode: replay.mode,
            tool_mode: replay.tools.mode,
            delivery: BTreeMap::new(),
            llm: ReplayCounts::default(),
            tool: ReplayCounts::default(),
            reasons: BTreeMap::new(),
            last_failure: None,
            conflicts: 0,
            failed_writes: 0,
            persistence_errors: 0,
            finalized: false,
        });
        if let Some(source) = inherited_coverage {
            let report = state.report.as_mut().unwrap();
            report.llm.uncaptured = source.llm.uncaptured;
            report.tool.uncaptured = source.tool.uncaptured;
            report.failed_writes = source.failed_writes;
            report.persistence_errors = source.persistence_errors;
            report.reasons = source.reasons;
        }
        Ok(Self {
            config: replay,
            key_policy: policy,
            tool_policy: tools,
            middleware: Json::Null,
            max_bytes,
            state: Arc::new(Mutex::new(state)),
            idle: Arc::new(Notify::new()),
            finalize_lock: tokio::sync::Mutex::new(()),
        })
    }

    pub(crate) fn set_middleware(&mut self, middleware: Json) {
        self.middleware = middleware;
    }

    /// Snapshot activity without exposing captured request bodies.
    pub fn report(&self) -> ReplayReport {
        self.state
            .lock()
            .expect("replay lock poisoned")
            .report
            .as_ref()
            .expect("initialized report")
            .clone()
    }

    pub(crate) fn begin(
        &self,
        surface: &str,
        delivery: &str,
        identity: &str,
    ) -> nemo_relay::error::Result<ReplayCall> {
        let mut state = self.state.lock().expect("replay lock poisoned");
        if !state.accepting {
            return Err(FlowError::Internal("replay: session_finalized".into()));
        }
        state.active += 1;
        state.calls += 1;
        let position = state.calls;
        *state
            .report
            .as_mut()
            .unwrap()
            .delivery
            .entry(delivery.into())
            .or_default() += 1;
        Ok(ReplayCall {
            state: self.state.clone(),
            idle: self.idle.clone(),
            surface: surface.into(),
            identity: identity.into(),
            position,
            key: None,
            live: false,
            captured: false,
            model: None,
            failure_reported: std::sync::atomic::AtomicBool::new(false),
            uncaptured_reason: "failed_call",
        })
    }

    pub(crate) fn lookup(&self, call: &mut ReplayCall, key: &str) -> Option<Json> {
        self.lookup_entry(call, key, |entry| entry.response.clone())
    }

    pub(crate) fn lookup_stream(
        &self,
        call: &mut ReplayCall,
        key: &str,
    ) -> Option<(Json, Option<Vec<Json>>)> {
        self.lookup_entry(call, key, |entry| {
            (entry.response.clone(), entry.response_stream.clone())
        })
    }

    fn lookup_entry<T>(
        &self,
        call: &mut ReplayCall,
        key: &str,
        read: impl FnOnce(&FixtureEntry) -> T,
    ) -> Option<T> {
        call.key = Some(key.into());
        if self.config.mode == ReplayMode::Record {
            return None;
        }
        let mut state = self.state.lock().expect("replay lock poisoned");
        let response = state
            .entries
            .get(key)
            .filter(|e| e.surface == call.surface)
            .map(read);
        let counts = counts(state.report.as_mut().unwrap(), &call.surface);
        if response.is_none() {
            counts.misses += 1;
        }
        response
    }

    pub(crate) fn strict(&self) -> bool {
        self.config.mode == ReplayMode::ReplayOnly
    }
    pub(crate) fn recorded_tools(&self) -> bool {
        self.config.tools.mode == ReplayToolMode::Recorded
    }

    pub(crate) fn capture(&self, call: &mut ReplayCall, entry: CacheEntry, request: Json) {
        self.capture_stream(call, entry, request, None);
    }

    pub(crate) fn stream_limit_exceeded(&self, call: &ReplayCall) {
        let mut state = self.state.lock().expect("replay lock poisoned");
        state.report.as_mut().unwrap().failed_writes += 1;
        call.failure_locked(&mut state, "persistence_error");
    }

    pub(crate) fn capture_stream(
        &self,
        call: &mut ReplayCall,
        entry: CacheEntry,
        request: Json,
        response_stream: Option<Vec<Json>>,
    ) {
        if self.strict() {
            return;
        }
        let entry = FixtureEntry {
            surface: call.surface.clone(),
            key_hash: entry.key_hash,
            response: entry.response,
            response_stream,
            recorded_unix_ms: entry.created_unix_ms,
            identity: if call.surface == "tool" {
                json!({"tool": call.identity, "version": entry.model_name})
            } else {
                json!({"provider": entry.provider_name, "model": entry.model_name})
            },
            request: self.config.capture_requests.then_some(request),
        };
        let mut state = self.state.lock().expect("replay lock poisoned");
        if let Some(previous) = state.entries.get(&entry.key_hash) {
            if previous.response != entry.response
                || previous.surface != entry.surface
                || (previous.response_stream.is_some()
                    && entry.response_stream.is_some()
                    && previous.response_stream != entry.response_stream)
            {
                state.report.as_mut().unwrap().conflicts += 1;
                call.failure_locked(&mut state, "recording_conflict");
                return;
            }
            if previous.response_stream.is_none() && entry.response_stream.is_some() {
                let previous_size = entry_bytes(previous).unwrap_or(usize::MAX);
                let size = entry_bytes(&entry).unwrap_or(usize::MAX);
                let additional = size.saturating_sub(previous_size);
                if additional > self.max_bytes.saturating_sub(state.bytes) {
                    state.report.as_mut().unwrap().failed_writes += 1;
                    call.failure_locked(&mut state, "persistence_error");
                    return;
                }
                state.bytes += additional;
                state.entries.insert(entry.key_hash.clone(), entry);
            }
        } else {
            let size = entry_bytes(&entry).unwrap_or(usize::MAX);
            if size > self.max_bytes.saturating_sub(state.bytes) {
                state.report.as_mut().unwrap().failed_writes += 1;
                call.failure_locked(&mut state, "persistence_error");
                return;
            }
            state.bytes += size;
            state.entries.insert(entry.key_hash.clone(), entry);
        }
        call.captured = true;
        counts(state.report.as_mut().unwrap(), &call.surface).captured += 1;
    }

    /// Stop accepting calls, drain active calls and stream commits, and atomically
    /// publish a complete artifact. Consume or close streams before awaiting this.
    /// Saving failures preserve the previous destination and permit retry.
    pub async fn finalize(&self) -> Result<ReplayReport> {
        let _finalize = self.finalize_lock.lock().await;
        self.state.lock().expect("replay lock poisoned").accepting = false;
        loop {
            let notified = self.idle.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.state.lock().expect("replay lock poisoned").active == 0 {
                break;
            }
            notified.await;
        }
        if self.report().finalized {
            return Ok(self.report());
        }
        let (entries, mut coverage) = {
            let state = self.state.lock().expect("replay lock poisoned");
            (
                state.entries.values().cloned().collect::<Vec<_>>(),
                state.report.as_ref().unwrap().clone(),
            )
        };
        coverage.finalized = true;
        if let Some(path) = &self.config.output_path {
            let fixture = Fixture {
                manifest: Manifest {
                    format_version: FORMAT_VERSION,
                    key_version: CACHE_SCHEMA_VERSION,
                    recording_id: coverage.recording_id.clone(),
                    created_unix_ms: now_unix_ms(),
                    relay_version: env!("CARGO_PKG_VERSION").into(),
                    harness_revision: self.config.harness_revision.clone(),
                    key_policy: self.key_policy.clone(),
                    tool_policy: self.tool_policy.clone(),
                    capture_requests: self.config.capture_requests,
                    middleware: self.middleware.clone(),
                    entry_count: entries.len(),
                    checksum: checksum(&entries)?,
                    coverage,
                },
                entries,
            };
            let bytes = serde_json::to_vec(&fixture).map_err(AdaptiveError::Serialization)?;
            let path = path.clone();
            let input = self.config.input_path.clone();
            let result = if bytes.len() > self.max_bytes {
                Err(fixture_error(
                    "persistence_error: serialized fixture exceeds memory budget",
                ))
            } else {
                tokio::task::spawn_blocking(move || atomic_save(&path, input.as_deref(), &bytes))
                    .await
                    .map_err(|e| fixture_error(&format!("persistence_error: {e}")))?
            };
            if let Err(error) = result {
                let mut state = self.state.lock().expect("replay lock poisoned");
                let report = state.report.as_mut().unwrap();
                report.persistence_errors += 1;
                *report
                    .reasons
                    .entry("persistence_error".into())
                    .or_default() += 1;
                return Err(error);
            }
        }
        self.state
            .lock()
            .expect("replay lock poisoned")
            .report
            .as_mut()
            .unwrap()
            .finalized = true;
        Ok(self.report())
    }
}

fn entry_bytes(entry: &FixtureEntry) -> Result<usize> {
    Ok(serde_json::to_vec(entry)
        .map_err(AdaptiveError::Serialization)?
        .len()
        .saturating_add(256))
}

fn counts<'a>(report: &'a mut ReplayReport, surface: &str) -> &'a mut ReplayCounts {
    if surface == "tool" {
        &mut report.tool
    } else {
        &mut report.llm
    }
}

/// Active call guard remains with a stream until its commit or cancellation.
pub(crate) struct ReplayCall {
    state: Arc<Mutex<State>>,
    idle: Arc<Notify>,
    surface: String,
    identity: String,
    position: u64,
    key: Option<String>,
    live: bool,
    captured: bool,
    model: Option<String>,
    failure_reported: std::sync::atomic::AtomicBool,
    uncaptured_reason: &'static str,
}

impl ReplayCall {
    pub(crate) fn model(&mut self, model: Option<String>) {
        self.model = model;
    }
    pub(crate) fn streaming(&mut self) {
        self.uncaptured_reason = "incomplete_stream";
    }
    pub(crate) fn hit(&self) {
        counts(
            self.state
                .lock()
                .expect("replay lock poisoned")
                .report
                .as_mut()
                .unwrap(),
            &self.surface,
        )
        .hits += 1;
    }

    pub(crate) fn live(&mut self) {
        self.live = true;
        counts(
            self.state
                .lock()
                .expect("replay lock poisoned")
                .report
                .as_mut()
                .unwrap(),
            &self.surface,
        )
        .live_calls += 1;
    }

    fn failure_locked(&self, state: &mut State, reason: &str) {
        self.failure_reported
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let report = state.report.as_mut().unwrap();
        *report.reasons.entry(reason.into()).or_default() += 1;
        report.last_failure = Some(ReplayFailure {
            reason: reason.into(),
            call_position: self.position,
            surface: self.surface.clone(),
            identity: self.identity.clone(),
            model: self.model.clone(),
            key_hash: self.key.clone(),
        });
    }

    pub(crate) fn failure(&self, reason: &str) {
        self.failure_locked(
            &mut self.state.lock().expect("replay lock poisoned"),
            reason,
        );
    }

    pub(crate) fn error(&self, reason: &str) -> FlowError {
        self.failure(reason);
        FlowError::Internal(format!(
            "replay: {reason}; surface={}; call_position={}; identity={}; key={}",
            self.surface,
            self.position,
            self.identity,
            self.key.as_deref().unwrap_or("unavailable")
        ))
    }
}

impl Drop for ReplayCall {
    fn drop(&mut self) {
        let mut state = self.state.lock().expect("replay lock poisoned");
        if self.live
            && !self.captured
            && !(self.surface == "tool"
                && state.report.as_ref().unwrap().tool_mode == ReplayToolMode::Live)
        {
            counts(state.report.as_mut().unwrap(), &self.surface).uncaptured += 1;
            if !self
                .failure_reported
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                self.failure_locked(&mut state, self.uncaptured_reason);
            }
        }
        state.active -= 1;
        if state.active == 0 {
            self.idle.notify_waiters();
        }
    }
}

fn atomic_save(path: &str, input: Option<&str>, bytes: &[u8]) -> Result<()> {
    if let Some(input) = input
        && resolved_path(input).ok() == resolved_path(path).ok()
    {
        return Err(fixture_error("persistence_error: output aliases input"));
    }
    let path = Path::new(path);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let temporary = parent.join(format!(".relay-replay-{}.tmp", Uuid::new_v4()));
    let result = (|| -> std::io::Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result.map_err(|e| fixture_error(&format!("persistence_error: {e}")))
}

impl CacheStore for ReplaySession {
    fn get<'a>(&'a self, _key: &'a str) -> BoxCacheFuture<'a, Option<Arc<CacheEntry>>> {
        Box::pin(async { Err(fixture_error("replay lookup requires an active call")) })
    }
    fn set<'a>(
        &'a self,
        _key: &'a str,
        _entry: CacheEntry,
        _ttl: Duration,
    ) -> BoxCacheFuture<'a, ()> {
        Box::pin(async { Err(fixture_error("recording requires an active call")) })
    }
    fn health<'a>(&'a self) -> BoxCacheFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
    fn backend_kind(&self) -> &'static str {
        "replay"
    }
    fn replay_session(&self) -> Option<&ReplaySession> {
        Some(self)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/response_cache/fixture_tests.rs"]
mod tests;

static SESSIONS: std::sync::LazyLock<Mutex<BTreeMap<Uuid, std::sync::Weak<dyn CacheStore>>>> =
    std::sync::LazyLock::new(|| Mutex::new(BTreeMap::new()));

pub(crate) fn register_session(id: Uuid, store: &Arc<dyn CacheStore>) {
    SESSIONS
        .lock()
        .expect("session registry lock poisoned")
        .insert(id, Arc::downgrade(store));
}

pub(crate) fn unregister_session(id: Uuid) {
    SESSIONS
        .lock()
        .expect("session registry lock poisoned")
        .remove(&id);
}

fn active_sessions() -> Vec<Arc<dyn CacheStore>> {
    SESSIONS
        .lock()
        .expect("session registry lock poisoned")
        .values()
        .filter_map(std::sync::Weak::upgrade)
        .collect()
}

/// Snapshot all active adaptive replay sessions in this process.
/// Generic plugin activations and directly registered runtimes share this API.
pub fn replay_reports() -> Vec<ReplayReport> {
    active_sessions()
        .iter()
        .filter_map(|store| store.replay_session().map(ReplaySession::report))
        .collect()
}

/// Finalize all active adaptive replay sessions. Complete or close every live
/// stream before awaiting. Returns an error if no replay session is registered.
pub async fn finalize_replay() -> Result<Vec<ReplayReport>> {
    let stores = active_sessions();
    if stores.is_empty() {
        return Err(AdaptiveError::InvalidConfig(
            "replay is not registered".into(),
        ));
    }
    let mut reports = Vec::with_capacity(stores.len());
    for store in stores {
        if let Some(session) = store.replay_session() {
            reports.push(session.finalize().await?);
        }
    }
    Ok(reports)
}
