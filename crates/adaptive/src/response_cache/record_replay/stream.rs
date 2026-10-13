// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    cpu,
    session::{Run, Session, error, lock, mono_ns},
    trace::{Call, Chunk},
};
use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::{
    LlmExecutionFn, LlmJsonStream, LlmStreamExecutionFn, LlmStreamInner,
    stream_consumer_clock::{StreamConsumerClock, StreamConsumerReport},
};
use nemo_relay::error::Result;
use serde_json::{Value as Json, json};
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::time::{Instant, Sleep};
use tokio_stream::Stream;

pub fn make_buffered(session: Arc<Session>) -> LlmExecutionFn {
    Arc::new(move |operation, request, _context, next| {
        let session = session.clone();
        let operation = operation.to_string();
        Box::pin(async move {
            let run = session.enter(&operation, &request, false).await?;
            if session.config.mode == "record" {
                let result = next(request.clone()).await;
                if result.is_ok() {
                    session.completed_model(&run);
                }
                let total = run.t0.elapsed().as_micros() as u64;
                session.append(
                    "calls",
                    call_line(
                        &operation,
                        &request,
                        &run,
                        RecordedAnswer {
                            stream: false,
                            outcome: if result.is_ok() {
                                "ok"
                            } else {
                                "provider_error"
                            },
                            total,
                            first: None,
                            chunks: None,
                            end_paced: Json::Null,
                            response: result.as_ref().ok().cloned().unwrap_or(Json::Null),
                            cpu_end: cpu::snapshot(),
                        },
                    ),
                    Some(run.seq),
                );
                result
            } else {
                let entry = run.entry.as_ref().expect("matched entry");
                let deadline = deadline(run.t0, entry.total_us, session.config.latency_scale)?;
                tokio::time::sleep_until(deadline).await;
                let actual = run.t0.elapsed().as_micros() as u64;
                let scheduled = scaled_us(entry.total_us, session.config.latency_scale);
                session.completed_model(&run);
                session.output("timing",json!({"seq":run.seq,"key_hash":run.key.hash,"repeat_index":run.repeat,"stream":false,
                    "outcome":"complete","t0_mono_ns":mono_ns().saturating_sub(run.t0.elapsed().as_nanos() as u64),
                    "scheduled_first_us":null,"actual_first_us":null,"scheduled_end_us":scheduled,"actual_end_us":actual,
                    "lateness_us":{"p50":actual.saturating_sub(scheduled),"p99":actual.saturating_sub(scheduled),"max":actual.saturating_sub(scheduled)},
                    "chunks":0,"gate_wait_ms":run.gate_wait_ms,"cpu":{"start":run.cpu_start,"end":cpu::snapshot()}}));
                Ok(entry.response.clone())
            }
        })
    })
}

pub fn make_stream(session: Arc<Session>) -> LlmStreamExecutionFn {
    Arc::new(move |operation, request, context, next| {
        let session = session.clone();
        let operation = operation.to_string();
        Box::pin(async move {
            let run = session.enter(&operation, &request, true).await?;
            let clock = context.consumer_clock();
            if session.config.mode == "record" {
                let upstream = match next(request.clone()).await {
                    Ok(stream) => stream,
                    Err(error) => {
                        session.append(
                            "calls",
                            call_line(
                                &operation,
                                &request,
                                &run,
                                RecordedAnswer {
                                    stream: true,
                                    outcome: "provider_error",
                                    total: run.t0.elapsed().as_micros() as u64,
                                    first: None,
                                    chunks: Some(vec![]),
                                    end_paced: Json::Null,
                                    response: Json::Null,
                                    cpu_end: cpu::snapshot(),
                                },
                            ),
                            Some(run.seq),
                        );
                        return Err(error);
                    }
                };
                let state = Arc::new(Mutex::new(Recording {
                    session,
                    operation,
                    request,
                    run: Some(run),
                    chunks: vec![],
                    error: false,
                    producer_ended: false,
                    total_us: None,
                    end_paced: None,
                    cpu_end: None,
                }));
                if let Some(clock) = &clock {
                    let state = state.clone();
                    clock.observe(move |receipt| finalize_record(&state, Some(receipt)));
                }
                Ok(LlmJsonStream::from_closeable(RecordStream {
                    upstream,
                    state,
                    clock,
                    finished: false,
                }))
            } else {
                let entry = run.entry.as_ref().expect("matched entry").clone();
                let deadlines: Vec<_> = entry
                    .chunks
                    .as_ref()
                    .expect("stream entry")
                    .iter()
                    .map(|c| deadline(run.t0, c.offset_us, session.config.latency_scale))
                    .chain(std::iter::once(deadline(
                        run.t0,
                        entry.total_us,
                        session.config.latency_scale,
                    )))
                    .collect::<Result<_>>()?;
                let t0_mono_ns = mono_ns().saturating_sub(run.t0.elapsed().as_nanos() as u64);
                let state = Arc::new(Mutex::new(Replaying {
                    session,
                    run: Some(run),
                    entry,
                    actual: vec![],
                    producer_ended: false,
                    t0_mono_ns,
                }));
                if let Some(clock) = &clock {
                    let state = state.clone();
                    clock.observe(move |receipt| finalize_replay(&state, Some(receipt)));
                }
                Ok(LlmJsonStream::from_closeable(ReplayStream {
                    state,
                    clock,
                    deadlines,
                    index: 0,
                    timer: None,
                    finished: false,
                }))
            }
        })
    })
}

struct RecordedAnswer<'a> {
    stream: bool,
    outcome: &'a str,
    total: u64,
    first: Option<u64>,
    chunks: Option<Vec<Chunk>>,
    end_paced: Json,
    response: Json,
    cpu_end: cpu::CpuSnapshot,
}
fn call_line(operation: &str, request: &LlmRequest, run: &Run, answer: RecordedAnswer<'_>) -> Json {
    let RecordedAnswer {
        stream,
        outcome,
        total,
        first,
        chunks,
        end_paced,
        response,
        cpu_end,
    } = answer;
    let usage = response.get("usage").cloned().filter(Json::is_object);
    json!({"seq":run.seq,"key_hash":run.key.hash,"repeat_index":run.repeat,"keyless_reason":run.key.reason,
        "operation":operation,"stream":stream,"model":request.content.get("model"),"outcome":outcome,
        "error":if matches!(outcome,"provider_error"|"stream_error"){json!({"class":outcome,"message":"upstream call did not complete"})}else{Json::Null},
        "gate_wait_ms":run.gate_wait_ms,"ttft_us":first,"total_us":total,"chunks":chunks,"end_hermes_paced":end_paced,
        "response":response,"usage_chunk":usage.is_some(),"usage":usage,"cpu":{"start":run.cpu_start,"end":cpu_end}})
}

struct Recording {
    session: Arc<Session>,
    operation: String,
    request: LlmRequest,
    run: Option<Run>,
    chunks: Vec<Chunk>,
    error: bool,
    producer_ended: bool,
    total_us: Option<u64>,
    end_paced: Option<bool>,
    cpu_end: Option<cpu::CpuSnapshot>,
}
struct RecordStream {
    upstream: LlmJsonStream,
    state: Arc<Mutex<Recording>>,
    clock: Option<StreamConsumerClock>,
    finished: bool,
}
impl Stream for RecordStream {
    type Item = Result<Json>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.finished {
            return Poll::Ready(None);
        }
        match Pin::new(&mut this.upstream).poll_next(cx) {
            Poll::Ready(Some(Ok(data))) => {
                let mut state = lock(&this.state);
                if let Some(run) = &state.run {
                    let offset = run.t0.elapsed().as_micros() as u64;
                    let paced = this.clock.as_ref().and_then(|clock| {
                        clock
                            .was_prefetched(state.chunks.len(), std::time::Instant::now())
                            .then_some(true)
                    });
                    state.error |= data.get("error").is_some_and(|e| !e.is_null())
                        || matches!(
                            data.get("type").and_then(Json::as_str),
                            Some("response.failed" | "response.incomplete" | "error")
                        );
                    state.chunks.push(Chunk {
                        offset_us: offset,
                        hermes_paced: paced,
                        data: data.clone(),
                    });
                }
                Poll::Ready(Some(Ok(data)))
            }
            Poll::Ready(Some(Err(error))) => {
                lock(&this.state).error = true;
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                this.finished = true;
                {
                    let mut state = lock(&this.state);
                    if let Some(run) = &state.run {
                        state.total_us = Some(run.t0.elapsed().as_micros() as u64);
                        state.cpu_end = Some(cpu::snapshot());
                        state.end_paced = this.clock.as_ref().and_then(|clock| {
                            clock
                                .was_prefetched(state.chunks.len(), std::time::Instant::now())
                                .then_some(true)
                        });
                        state.producer_ended = true;
                    }
                }
                if this.clock.is_none() {
                    finalize_record(&this.state, None);
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
impl LlmStreamInner for RecordStream {
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let this = self.get_mut();
        Box::pin(async move {
            let result = this.upstream.close().await;
            if !this.finished {
                this.finished = true;
                finalize_record(&this.state, None);
            }
            result
        })
    }
}
impl Drop for RecordStream {
    fn drop(&mut self) {
        if !self.finished {
            finalize_record(&self.state, None);
        }
    }
}

fn finalize_record(state: &Arc<Mutex<Recording>>, receipt: Option<StreamConsumerReport>) {
    let mut state = lock(state);
    let Some(run) = state.run.take() else {
        return;
    };
    let consumer_complete = receipt
        .as_ref()
        .is_none_or(|r| r.complete && r.delivered.len() == state.chunks.len());
    let (response, terminal) = aggregate(&state.chunks);
    let outcome = if state.error {
        "stream_error"
    } else if !state.producer_ended || !consumer_complete {
        "consumer_dropped"
    } else if terminal {
        "ok"
    } else {
        "stream_error"
    };
    let total = state
        .total_us
        .unwrap_or_else(|| run.t0.elapsed().as_micros() as u64);
    if outcome == "ok" {
        state.session.completed_model(&run);
    }
    let first = state.chunks.first().map(|c| c.offset_us);
    let chunks = std::mem::take(&mut state.chunks);
    state.session.append(
        "calls",
        call_line(
            &state.operation,
            &state.request,
            &run,
            RecordedAnswer {
                stream: true,
                outcome,
                total,
                first,
                chunks: Some(chunks),
                end_paced: json!(state.end_paced),
                response,
                cpu_end: state.cpu_end.clone().unwrap_or_else(cpu::snapshot),
            },
        ),
        Some(run.seq),
    );
}

struct Replaying {
    session: Arc<Session>,
    run: Option<Run>,
    entry: Call,
    actual: Vec<u64>,
    producer_ended: bool,
    t0_mono_ns: u64,
}
struct ReplayStream {
    state: Arc<Mutex<Replaying>>,
    clock: Option<StreamConsumerClock>,
    deadlines: Vec<Instant>,
    index: usize,
    timer: Option<Pin<Box<Sleep>>>,
    finished: bool,
}
impl Stream for ReplayStream {
    type Item = Result<Json>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.finished {
            return Poll::Ready(None);
        }
        let timer = this
            .timer
            .get_or_insert_with(|| Box::pin(tokio::time::sleep_until(this.deadlines[this.index])));
        if timer.as_mut().poll(cx).is_pending() {
            return Poll::Pending;
        }
        this.timer = None;
        let result = {
            let mut state = lock(&this.state);
            let actual = state
                .run
                .as_ref()
                .map(|r| r.t0.elapsed().as_micros() as u64)
                .unwrap_or_default();
            state.actual.push(actual);
            state
                .entry
                .chunks
                .as_ref()
                .and_then(|chunks| chunks.get(this.index))
                .map(|c| c.data.clone())
        };
        this.index += 1;
        if let Some(data) = result {
            Poll::Ready(Some(Ok(data)))
        } else {
            this.finished = true;
            lock(&this.state).producer_ended = true;
            if this.clock.is_none() {
                finalize_replay(&this.state, None);
            }
            Poll::Ready(None)
        }
    }
}
impl LlmStreamInner for ReplayStream {
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let this = self.get_mut();
        if !this.finished {
            this.finished = true;
            finalize_replay(&this.state, None);
        }
        Box::pin(async { Ok(()) })
    }
}
impl Drop for ReplayStream {
    fn drop(&mut self) {
        if !self.finished {
            finalize_replay(&self.state, None);
        }
    }
}

fn finalize_replay(state: &Arc<Mutex<Replaying>>, receipt: Option<StreamConsumerReport>) {
    let mut state = lock(state);
    let Some(run) = state.run.take() else {
        return;
    };
    let chunks = state.entry.chunks.as_ref().expect("stream entry");
    let complete = state.producer_ended
        && receipt
            .as_ref()
            .is_none_or(|r| r.complete && r.delivered.len() == chunks.len());
    if complete {
        state.session.completed_model(&run);
    }
    let actual = if let Some(receipt) = &receipt {
        let mut actual: Vec<_> = receipt
            .delivered
            .iter()
            .map(|stamp| {
                Instant::from_std(*stamp)
                    .saturating_duration_since(run.t0)
                    .as_micros() as u64
            })
            .collect();
        actual.push(
            Instant::from_std(receipt.ended)
                .saturating_duration_since(run.t0)
                .as_micros() as u64,
        );
        actual
    } else {
        state.actual.clone()
    };
    let scale = state.session.config.latency_scale;
    let delivered = receipt
        .as_ref()
        .map(|r| r.delivered.len())
        .unwrap_or_else(|| {
            state
                .actual
                .len()
                .saturating_sub(usize::from(state.producer_ended))
        });
    let mut lateness: Vec<_> = actual
        .iter()
        .take(delivered.min(chunks.len()))
        .zip(chunks)
        .map(|(actual, chunk)| actual.saturating_sub(scaled_us(chunk.offset_us, scale)))
        .collect();
    let end = if complete {
        actual.last().copied()
    } else {
        Some(run.t0.elapsed().as_micros() as u64)
    };
    if let Some(end) = end {
        lateness.push(end.saturating_sub(scaled_us(state.entry.total_us, scale)));
    }
    lateness.sort_unstable();
    let quantile = |q: f64| {
        lateness
            .get(((lateness.len().saturating_sub(1)) as f64 * q) as usize)
            .copied()
            .unwrap_or(0)
    };
    state.session.output("timing",json!({"seq":run.seq,"key_hash":run.key.hash,"repeat_index":run.repeat,"stream":true,
        "outcome":if complete{"complete"}else{"consumer_dropped"},"t0_mono_ns":state.t0_mono_ns,
        "scheduled_first_us":state.entry.ttft_us.map(|v|scaled_us(v,scale)),"actual_first_us":actual.first().filter(|_|delivered>0),
        "scheduled_end_us":scaled_us(state.entry.total_us,scale),"actual_end_us":end,
        "lateness_us":{"p50":quantile(0.5),"p99":quantile(0.99),"max":lateness.last().copied().unwrap_or(0)},
        "chunks":delivered,"gate_wait_ms":run.gate_wait_ms,"cpu":{"start":run.cpu_start,"end":cpu::snapshot()}}));
}

fn deadline(t0: Instant, us: u64, scale: f64) -> Result<Instant> {
    let duration = Duration::try_from_secs_f64(us as f64 * scale / 1_000_000.0)
        .map_err(|_| error("RR-UNAVAILABLE"))?;
    t0.checked_add(duration)
        .ok_or_else(|| error("RR-UNAVAILABLE"))
}
fn scaled_us(us: u64, scale: f64) -> u64 {
    (us as f64 * scale).min(u64::MAX as f64) as u64
}

pub fn aggregate(chunks: &[Chunk]) -> (Json, bool) {
    let mut response = json!({"object":"chat.completion"});
    let mut choices: BTreeMap<u64, Json> = BTreeMap::new();
    let mut tools: BTreeMap<(u64, u64), Json> = BTreeMap::new();
    let mut terminal = false;
    for chunk in chunks {
        let data = &chunk.data;
        if data.get("type").and_then(Json::as_str) == Some("response.completed") {
            return (
                data.get("response").cloned().unwrap_or(Json::Null),
                data.get("response").is_some_and(Json::is_object),
            );
        }
        for key in ["id", "model", "created", "system_fingerprint"] {
            if response.get(key).is_none()
                && let Some(value) = data.get(key)
            {
                response[key] = value.clone();
            }
        }
        if let Some(usage) = data.get("usage").filter(|u| u.is_object()) {
            response["usage"] = usage.clone();
        }
        for choice in data
            .get("choices")
            .and_then(Json::as_array)
            .into_iter()
            .flatten()
        {
            let index = choice.get("index").and_then(Json::as_u64).unwrap_or(0);
            let output = choices
                .entry(index)
                .or_insert_with(|| json!({"index":index,"message":{}}));
            if let Some(delta) = choice.get("delta").and_then(Json::as_object) {
                for (key, value) in delta {
                    if key == "tool_calls" {
                        merge_tool_deltas(index, value, &mut tools);
                    } else if key == "role" {
                        if output["message"].get(key).is_none() {
                            output["message"][key] = value.clone();
                        }
                    } else if let Some(text) = value.as_str() {
                        let mut accumulated = output["message"][key]
                            .as_str()
                            .unwrap_or_default()
                            .to_string();
                        accumulated.push_str(text);
                        output["message"][key] = json!(accumulated);
                    } else {
                        output["message"][key] = value.clone();
                    }
                }
            }
            for key in ["finish_reason", "logprobs"] {
                if let Some(value) = choice.get(key) {
                    output[key] = value.clone();
                    if key == "finish_reason" && !value.is_null() {
                        terminal = true;
                    }
                }
            }
        }
    }
    for (index, choice) in choices.iter_mut() {
        let calls: Vec<_> = tools
            .iter()
            .filter(|((choice, _), _)| choice == index)
            .map(|(_, value)| value.clone())
            .collect();
        if !calls.is_empty() {
            choice["message"]["tool_calls"] = json!(calls);
        }
    }
    response["choices"] = json!(choices.into_values().collect::<Vec<_>>());
    (response, terminal)
}

fn merge_tool_deltas(index: u64, value: &Json, tools: &mut BTreeMap<(u64, u64), Json>) {
    for call in value.as_array().into_iter().flatten() {
        let tool_index = call["index"].as_u64().unwrap_or(0);
        let tool = tools
            .entry((index, tool_index))
            .or_insert_with(|| json!({"function":{}}));
        for key in ["id", "type"] {
            if tool.get(key).is_none()
                && let Some(value) = call.get(key)
            {
                tool[key] = value.clone();
            }
        }
        if tool["function"].get("name").is_none()
            && let Some(name) = call.pointer("/function/name")
        {
            tool["function"]["name"] = name.clone();
        }
        if let Some(args) = call.pointer("/function/arguments").and_then(Json::as_str) {
            let mut text = tool["function"]["arguments"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            text.push_str(args);
            tool["function"]["arguments"] = json!(text);
        }
    }
}
