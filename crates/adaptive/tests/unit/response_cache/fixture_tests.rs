// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::response_cache::config::{ReplayToolsConfig, ToolCacheConfig, ToolOverride};
use crate::response_cache::{make_intercept, make_stream_intercept, make_tool_intercept};
use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::{
    LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn, ToolExecutionContext,
    ToolExecutionNextFn,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio_stream::StreamExt;

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("relay-replay-test-{}", Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn file(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().into_owned()
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn config(path: String, mode: ReplayMode) -> ResponseCacheConfig {
    let replay = ReplayConfig {
        mode,
        input_path: (mode != ReplayMode::Record).then(|| path.clone()),
        output_path: (mode == ReplayMode::Record).then_some(path),
        ..Default::default()
    };
    ResponseCacheConfig {
        namespace: "fixture-test".into(),
        bypass_rate: 1.0,
        replay: Some(replay),
        ..Default::default()
    }
}
fn request(text: &str) -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::from_iter([("authorization".into(), json!("transport-secret"))]),
        content: json!({"model": "test", "messages": [{"role": "user", "content": text}], "temperature": 0}),
    }
}
fn response(text: &str) -> Json {
    json!({"id":"chatcmpl-test", "object":"chat.completion", "created":1, "model":"test",
        "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}]})
}
fn provider(calls: Arc<AtomicUsize>, text: &str) -> LlmExecutionNextFn {
    let answer = response(text);
    Arc::new(move |_| {
        let calls = calls.clone();
        let answer = answer.clone();
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(answer)
        })
    })
}
async fn buffered(
    session: Arc<ReplaySession>,
    cfg: &ResponseCacheConfig,
    req: LlmRequest,
    next: LlmExecutionNextFn,
) -> nemo_relay::error::Result<Json> {
    make_intercept(session, Arc::new(cfg.clone()))("openai", req, Default::default(), next).await
}
async fn stream(
    session: Arc<ReplaySession>,
    cfg: &ResponseCacheConfig,
    req: LlmRequest,
    next: LlmStreamExecutionNextFn,
) -> nemo_relay::error::Result<LlmJsonStream> {
    make_stream_intercept(session, Arc::new(cfg.clone()))("openai", req, Default::default(), next)
        .await
}
fn stream_provider(calls: Arc<AtomicUsize>, terminal: bool) -> LlmStreamExecutionNextFn {
    Arc::new(move |_| {
        let calls = calls.clone();
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            let mut chunks = vec![Ok(
                json!({"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"test",
            "choices":[{"index":0,"delta":{"role":"assistant","content":"answer"},"finish_reason":null}]}),
            )];
            if terminal {
                chunks.push(Ok(
                    json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}),
                ));
            }
            Ok(LlmJsonStream::new(tokio_stream::iter(chunks)))
        })
    })
}

#[tokio::test]
async fn disk_roundtrip_is_strict_and_independent_of_expiry_and_sampling() {
    let dir = Directory::new();
    let path = dir.file("fixture.json");
    let mut cfg = config(path.clone(), ReplayMode::Record);
    cfg.replay.as_mut().unwrap().capture_requests = true;
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    for _ in 0..2 {
        assert_eq!(
            buffered(
                session.clone(),
                &cfg,
                request("prompt"),
                provider(calls.clone(), "answer")
            )
            .await
            .unwrap(),
            response("answer")
        );
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "record must always run live"
    );
    let report = session.finalize().await.unwrap();
    assert_eq!(report.llm.captured, 2);
    assert!(report.finalized);
    assert!(
        buffered(
            session.clone(),
            &cfg,
            request("prompt"),
            provider(calls.clone(), "answer")
        )
        .await
        .is_err()
    );
    let bytes = std::fs::read(&path).unwrap();
    assert!(!String::from_utf8_lossy(&bytes).contains("transport-secret"));
    let mut fixture: Fixture = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(fixture.entries.len(), 1);
    assert_eq!(
        fixture.entries[0].request.as_ref().unwrap(),
        &request("prompt").content
    );
    fixture.entries[0].recorded_unix_ms = 1;
    fixture.manifest.checksum = checksum(&fixture.entries).unwrap();
    std::fs::write(&path, serde_json::to_vec(&fixture).unwrap()).unwrap();
    // A much shorter ordinary TTL cannot expire a frozen fixture.
    let mut cfg = config(path, ReplayMode::ReplayOnly);
    cfg.ttl_seconds = 1;
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    assert_eq!(
        buffered(
            session.clone(),
            &cfg,
            request("prompt"),
            provider(calls.clone(), "wrong")
        )
        .await
        .unwrap(),
        response("answer")
    );
    assert!(
        buffered(
            session.clone(),
            &cfg,
            request("changed"),
            provider(calls.clone(), "wrong")
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("missing_entry")
    );
    let mut ineligible = request("prompt");
    ineligible.content["previous_response_id"] = json!("stateful");
    assert!(
        buffered(
            session.clone(),
            &cfg,
            ineligible,
            provider(calls.clone(), "wrong")
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("ineligible_request")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(session.report().llm.live_calls, 0);
}

#[tokio::test]
async fn stream_finalization_captures_completed_streams_and_excludes_truncation() {
    let dir = Directory::new();
    let path = dir.file("stream.json");
    let cfg = config(path.clone(), ReplayMode::Record);
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let mut live = stream(
        session.clone(),
        &cfg,
        request("prompt"),
        stream_provider(calls.clone(), true),
    )
    .await
    .unwrap();
    while let Some(chunk) = live.next().await {
        chunk.unwrap();
    }
    let report = session.finalize().await.unwrap();
    assert_eq!(report.llm.captured, 1);
    let cfg = config(path, ReplayMode::ReplayOnly);
    let replay = Arc::new(ReplaySession::load(&cfg).unwrap());
    let mut saved = stream(
        replay.clone(),
        &cfg,
        request("prompt"),
        stream_provider(calls.clone(), true),
    )
    .await
    .unwrap();
    let mut text = String::new();
    while let Some(chunk) = saved.next().await {
        if let Some(content) = chunk.unwrap()["choices"][0]["delta"]["content"].as_str() {
            text.push_str(content);
        }
    }
    assert_eq!(text, "answer");
    assert!(
        stream(
            replay.clone(),
            &cfg,
            request("changed"),
            stream_provider(calls.clone(), true)
        )
        .await
        .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(replay.report().delivery["streaming"], 2);

    let cfg = config(dir.file("truncated.json"), ReplayMode::Record);
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    let mut live = stream(
        session.clone(),
        &cfg,
        request("prompt"),
        stream_provider(calls, false),
    )
    .await
    .unwrap();
    while live.next().await.is_some() {}
    assert_eq!(session.finalize().await.unwrap().llm.uncaptured, 1);
    let mut strict = cfg.clone();
    let r = strict.replay.as_mut().unwrap();
    r.mode = ReplayMode::ReplayOnly;
    r.input_path = r.output_path.take();
    assert!(ReplaySession::load(&strict).is_err());
}

#[tokio::test]
async fn finalization_waits_for_an_active_call_and_preserves_previous_file_on_failure() {
    let dir = Directory::new();
    let destination = dir.file("fixture.json");
    let cfg = config(destination.clone(), ReplayMode::Record);
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    let call = session.begin("llm", "buffered", "openai").unwrap();
    let finalizer = {
        let session = session.clone();
        tokio::spawn(async move { session.finalize().await })
    };
    tokio::task::yield_now().await;
    assert!(!finalizer.is_finished());
    drop(call);
    finalizer.await.unwrap().unwrap();
    assert!(session.report().finalized);

    // Atomic rename cannot replace a directory. Its existing contents survive.
    let destination = dir.file("existing");
    std::fs::create_dir(&destination).unwrap();
    std::fs::write(Path::new(&destination).join("previous"), b"preserved").unwrap();
    let cfg = config(destination.clone(), ReplayMode::Record);
    let session = ReplaySession::load(&cfg).unwrap();
    assert!(session.finalize().await.is_err());
    assert!(!session.report().finalized);
    assert_eq!(
        std::fs::read(Path::new(&destination).join("previous")).unwrap(),
        b"preserved"
    );
    std::fs::remove_file(Path::new(&destination).join("previous")).unwrap();
    std::fs::remove_dir(&destination).unwrap();
    session.finalize().await.unwrap();
    assert_eq!(session.report().persistence_errors, 1);
    assert!(ReplaySession::load(&config(destination, ReplayMode::ReplayOnly)).is_ok());
}

#[tokio::test]
async fn derivation_preserves_source_and_reports_conflicting_answers() {
    let dir = Directory::new();
    let source = dir.file("source.json");
    let cfg = config(source.clone(), ReplayMode::Record);
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    buffered(
        session.clone(),
        &cfg,
        request("one"),
        provider(calls.clone(), "answer"),
    )
    .await
    .unwrap();
    session.finalize().await.unwrap();
    let original = std::fs::read(&source).unwrap();
    let mut cfg = config(source.clone(), ReplayMode::ReplayOrRecord);
    let output = dir.file("derived.json");
    cfg.replay.as_mut().unwrap().output_path = Some(output.clone());
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    buffered(
        session.clone(),
        &cfg,
        request("one"),
        provider(calls.clone(), "wrong"),
    )
    .await
    .unwrap();
    buffered(
        session.clone(),
        &cfg,
        request("two"),
        provider(calls.clone(), "new"),
    )
    .await
    .unwrap();
    session.finalize().await.unwrap();
    assert_eq!(std::fs::read(&source).unwrap(), original);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        serde_json::from_slice::<Fixture>(&std::fs::read(&output).unwrap())
            .unwrap()
            .entries
            .len(),
        2
    );
    cfg.replay.as_mut().unwrap().output_path = Some(source.clone());
    assert!(ReplaySession::load(&cfg).is_err());

    let cfg = config(dir.file("conflict.json"), ReplayMode::Record);
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    buffered(
        session.clone(),
        &cfg,
        request("one"),
        provider(calls.clone(), "a"),
    )
    .await
    .unwrap();
    buffered(session.clone(), &cfg, request("one"), provider(calls, "b"))
        .await
        .unwrap();
    assert_eq!(session.finalize().await.unwrap().conflicts, 1);
    let mut strict = cfg;
    let r = strict.replay.as_mut().unwrap();
    r.mode = ReplayMode::ReplayOnly;
    r.input_path = r.output_path.take();
    assert!(ReplaySession::load(&strict).is_err());
}

#[tokio::test]
async fn loader_rejects_corruption_policy_versions_and_budget_overflow() {
    let dir = Directory::new();
    let path = dir.file("fixture.json");
    let cfg = config(path.clone(), ReplayMode::Record);
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    buffered(
        session.clone(),
        &cfg,
        request("prompt"),
        provider(Arc::new(AtomicUsize::new(0)), "answer"),
    )
    .await
    .unwrap();
    session.finalize().await.unwrap();
    let original = std::fs::read(&path).unwrap();
    let strict = config(path.clone(), ReplayMode::ReplayOnly);
    for field in ["format_version", "key_version", "entry_count", "checksum"] {
        let mut fixture: Json = serde_json::from_slice(&original).unwrap();
        fixture["manifest"][field] = if field == "checksum" {
            json!("wrong")
        } else {
            json!(999)
        };
        std::fs::write(&path, serde_json::to_vec(&fixture).unwrap()).unwrap();
        assert!(ReplaySession::load(&strict).is_err(), "{field}");
    }
    std::fs::write(&path, &original).unwrap();
    let mut mismatch = strict.clone();
    mismatch.namespace = "other".into();
    assert!(ReplaySession::load(&mismatch).is_err());
    mismatch = strict;
    mismatch
        .backend
        .config
        .insert("max_bytes".into(), json!(10));
    assert!(ReplaySession::load(&mismatch).is_err());
}

#[tokio::test]
async fn tool_and_delivery_modes_are_independent_and_strict_tools_skip_callbacks() {
    let dir = Directory::new();
    let path = dir.file("tools.json");
    let mut cfg = config(path.clone(), ReplayMode::Record);
    cfg.tools = Some(ToolCacheConfig {
        enabled: true,
        overrides: BTreeMap::from([(
            "lookup".into(),
            ToolOverride {
                cacheable: Some(true),
                tool_version: Some("v1".into()),
                ..Default::default()
            },
        )]),
        ..Default::default()
    });
    cfg.replay.as_mut().unwrap().tools = ReplayToolsConfig {
        mode: ReplayToolMode::Recorded,
    };
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let tools_called = Arc::new(AtomicUsize::new(0));
    let tool: ToolExecutionNextFn = {
        let calls = tools_called.clone();
        Arc::new(move |_| {
            let calls = calls.clone();
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(json!({"value":42}).into())
            })
        })
    };
    buffered(
        session.clone(),
        &cfg,
        request("prompt"),
        provider(calls.clone(), "answer"),
    )
    .await
    .unwrap();
    make_tool_intercept(
        session.clone(),
        Arc::new(cfg.clone()),
        Arc::new(cfg.tools.clone().unwrap()),
    )(
        ToolExecutionContext::new("lookup", json!({"a":1})),
        tool.clone(),
    )
    .await
    .unwrap();
    session.finalize().await.unwrap();
    for tool_mode in [ReplayToolMode::Live, ReplayToolMode::Recorded] {
        for streaming in [false, true] {
            let mut strict = cfg.clone();
            let r = strict.replay.as_mut().unwrap();
            r.mode = ReplayMode::ReplayOnly;
            r.input_path = Some(path.clone());
            r.output_path = None;
            r.tools.mode = tool_mode;
            let session = Arc::new(ReplaySession::load(&strict).unwrap());
            if streaming {
                let mut saved = stream(
                    session.clone(),
                    &strict,
                    request("prompt"),
                    stream_provider(calls.clone(), true),
                )
                .await
                .unwrap();
                while let Some(chunk) = saved.next().await {
                    chunk.unwrap();
                }
            } else {
                buffered(
                    session.clone(),
                    &strict,
                    request("prompt"),
                    provider(calls.clone(), "wrong"),
                )
                .await
                .unwrap();
            }
            let intercept = make_tool_intercept(
                session.clone(),
                Arc::new(strict.clone()),
                Arc::new(strict.tools.clone().unwrap()),
            );
            let before = tools_called.load(Ordering::SeqCst);
            intercept(
                ToolExecutionContext::new("lookup", json!({"a":1})),
                tool.clone(),
            )
            .await
            .unwrap();
            assert_eq!(
                tools_called.load(Ordering::SeqCst) - before,
                usize::from(tool_mode == ReplayToolMode::Live)
            );
            if tool_mode == ReplayToolMode::Recorded {
                for (name, args) in [("lookup", json!({"a":2})), ("unclassified", json!({}))] {
                    assert!(
                        intercept(ToolExecutionContext::new(name, args), tool.clone())
                            .await
                            .is_err()
                    );
                }
                assert_eq!(tools_called.load(Ordering::SeqCst), before);
            }
            assert_eq!(session.report().tool_mode, tool_mode);
            assert_eq!(session.report().llm.live_calls, 0);
        }
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn strict_stream_fallbacks_never_execute_live_and_diagnose_the_model() {
    let dir = Directory::new();
    let path = dir.file("lossy.json");
    let cfg = config(path.clone(), ReplayMode::Record);
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    let mut lossy = response("answer");
    lossy["choices"][0]["logprobs"] = json!({"content":[{"token":"answer","logprob":-0.1}]});
    let next: LlmExecutionNextFn = Arc::new(move |_| {
        let lossy = lossy.clone();
        Box::pin(async move { Ok(lossy) })
    });
    buffered(session.clone(), &cfg, request("prompt"), next)
        .await
        .unwrap();
    session.finalize().await.unwrap();
    let cfg = config(path, ReplayMode::ReplayOnly);
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let result = stream(
        session.clone(),
        &cfg,
        request("prompt"),
        stream_provider(calls.clone(), true),
    )
    .await;
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("stream_not_replayable")
    );
    assert_eq!(session.report().llm.hits, 0);
    assert_eq!(
        session.report().last_failure.unwrap().model.as_deref(),
        Some("test")
    );
    let unknown = LlmRequest {
        headers: Default::default(),
        content: json!({"prompt":"custom", "temperature":0}),
    };
    assert!(
        stream(
            session.clone(),
            &cfg,
            unknown,
            stream_provider(calls.clone(), true)
        )
        .await
        .err()
        .unwrap()
        .to_string()
        .contains("stream_not_replayable")
    );
    let mut sampled = request("prompt");
    sampled.content["temperature"] = json!(1.0);
    assert!(
        stream(
            session.clone(),
            &cfg,
            sampled,
            stream_provider(calls.clone(), true)
        )
        .await
        .err()
        .unwrap()
        .to_string()
        .contains("ineligible_request")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn inherited_coverage_and_recording_budget_cannot_become_strict_ready() {
    let dir = Directory::new();
    let source = dir.file("incomplete.json");
    let cfg = config(source.clone(), ReplayMode::Record);
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    let mut ineligible = request("prompt");
    ineligible.content["temperature"] = json!(1.0);
    buffered(
        session.clone(),
        &cfg,
        ineligible,
        provider(Arc::new(AtomicUsize::new(0)), "answer"),
    )
    .await
    .unwrap();
    let report = session.finalize().await.unwrap();
    assert_eq!(report.llm.uncaptured, 1);
    assert_eq!(report.last_failure.unwrap().reason, "ineligible_request");
    let derived = dir.file("derived.json");
    let mut cfg = config(source, ReplayMode::ReplayOrRecord);
    cfg.replay.as_mut().unwrap().output_path = Some(derived.clone());
    let session = ReplaySession::load(&cfg).unwrap();
    assert_eq!(session.finalize().await.unwrap().llm.uncaptured, 1);
    assert!(ReplaySession::load(&config(derived, ReplayMode::ReplayOnly)).is_err());

    let path = dir.file("budget.json");
    std::fs::write(&path, b"previous fixture").unwrap();
    let mut cfg = config(path.clone(), ReplayMode::Record);
    cfg.backend.config.insert("max_bytes".into(), json!(100));
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    buffered(
        session.clone(),
        &cfg,
        request("prompt"),
        provider(Arc::new(AtomicUsize::new(0)), "answer"),
    )
    .await
    .unwrap();
    assert!(session.finalize().await.is_err());
    assert!(session.report().failed_writes > 0);
    assert_eq!(std::fs::read(path).unwrap(), b"previous fixture");
}

#[tokio::test]
async fn normalized_tool_call_ids_replay_but_changed_results_and_controls_miss() {
    let dir = Directory::new();
    let path = dir.file("normalized.json");
    let cfg = config(path.clone(), ReplayMode::Record);
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    let make = |id: &str, result: &str| LlmRequest {
        headers: Default::default(),
        content: json!({
            "model":"test", "temperature":0,
            "messages":[{"role":"user","content":"lookup"},
                {"role":"assistant","tool_calls":[{"id":id,"type":"function","function":{"name":"lookup","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":id,"content":result}]
        }),
    };
    let calls = Arc::new(AtomicUsize::new(0));
    buffered(
        session.clone(),
        &cfg,
        make("random-1", "stable"),
        provider(calls.clone(), "answer"),
    )
    .await
    .unwrap();
    session.finalize().await.unwrap();
    let cfg = config(path, ReplayMode::ReplayOnly);
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    assert_eq!(
        buffered(
            session.clone(),
            &cfg,
            make("random-2", "stable"),
            provider(calls.clone(), "wrong")
        )
        .await
        .unwrap(),
        response("answer")
    );
    assert!(
        buffered(
            session.clone(),
            &cfg,
            make("random-2", "changed"),
            provider(calls.clone(), "wrong")
        )
        .await
        .is_err()
    );
    let mut changed = make("random-2", "stable");
    changed.content["max_tokens"] = json!(10);
    assert!(
        buffered(session, &cfg, changed, provider(calls.clone(), "wrong"))
            .await
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

struct CleanupGateStream {
    chunks: std::collections::VecDeque<Json>,
    entered: Option<tokio::sync::oneshot::Sender<()>>,
    release: Option<tokio::sync::oneshot::Receiver<()>>,
}
impl tokio_stream::Stream for CleanupGateStream {
    type Item = nemo_relay::error::Result<Json>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::task::Poll::Ready(self.chunks.pop_front().map(Ok))
    }
}
impl nemo_relay::api::runtime::LlmStreamInner for CleanupGateStream {
    fn close(
        self: std::pin::Pin<&mut Self>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = nemo_relay::error::Result<()>> + Send + '_>,
    > {
        let this = self.get_mut();
        if let Some(entered) = this.entered.take() {
            let _ = entered.send(());
        }
        let release = this.release.take();
        Box::pin(async move {
            if let Some(release) = release {
                let _ = release.await;
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn finalization_drains_stream_cleanup_and_records_cancellation() {
    let dir = Directory::new();
    for cancel in [false, true] {
        let cfg = config(
            dir.file(if cancel {
                "canceled.json"
            } else {
                "completed.json"
            }),
            ReplayMode::Record,
        );
        let session = Arc::new(ReplaySession::load(&cfg).unwrap());
        let (entered, waiting_close) = tokio::sync::oneshot::channel();
        let (release, wait_release) = tokio::sync::oneshot::channel();
        let live = CleanupGateStream {
            chunks: std::collections::VecDeque::from([json!({
                "id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"test",
                "choices":[{"index":0,"delta":{"role":"assistant","content":"answer"},"finish_reason":"stop"}]
            })]),
            entered: Some(entered),
            release: Some(wait_release),
        };
        let live = Arc::new(Mutex::new(Some(live)));
        let next: LlmStreamExecutionNextFn = Arc::new(move |_| {
            let live = live.lock().unwrap().take().unwrap();
            Box::pin(async move { Ok(LlmJsonStream::from_closeable(live)) })
        });
        let mut returned = stream(session.clone(), &cfg, request("prompt"), next)
            .await
            .unwrap();
        returned.next().await.unwrap().unwrap();
        waiting_close.await.unwrap();
        let finalizer = {
            let session = session.clone();
            tokio::spawn(async move { session.finalize().await })
        };
        tokio::task::yield_now().await;
        assert!(
            !finalizer.is_finished(),
            "finalization must await upstream cleanup"
        );
        if cancel {
            // A complete stream remains recordable when the consumer drops after
            // the producer reached EOF while waiting for upstream cleanup.
            drop(returned);
            release.send(()).unwrap();
        } else {
            release.send(()).unwrap();
            assert!(returned.next().await.is_none());
        }
        let report = finalizer.await.unwrap().unwrap();
        assert_eq!(report.llm.captured, 1);
        assert_eq!(report.llm.uncaptured, 0);
    }
    let cfg = config(dir.file("dropped.json"), ReplayMode::Record);
    let session = Arc::new(ReplaySession::load(&cfg).unwrap());
    let next: LlmStreamExecutionNextFn =
        Arc::new(move |_| Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::pending())) }));
    let returned = stream(session.clone(), &cfg, request("prompt"), next)
        .await
        .unwrap();
    drop(returned);
    let report = tokio::time::timeout(Duration::from_secs(2), session.finalize())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(report.llm.uncaptured, 1);
    assert_eq!(report.reasons["incomplete_stream"], 1);
}
