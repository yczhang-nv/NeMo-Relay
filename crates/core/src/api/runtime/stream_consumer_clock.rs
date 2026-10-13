// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional producer/consumer timing shared by the Python bridge and recorders.

use std::future::Future;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Instant;

static RECORDERS: AtomicUsize = AtomicUsize::new(0);
tokio::task_local! { static CLOCK:Option<StreamConsumerClock>; }

/// Lifetime guard enabling timing only while a recorder is registered.
#[doc(hidden)]
pub struct StreamConsumerClockRegistration;
impl Default for StreamConsumerClockRegistration {
    fn default() -> Self {
        RECORDERS.fetch_add(1, Ordering::AcqRel);
        Self
    }
}
impl Drop for StreamConsumerClockRegistration {
    fn drop(&mut self) {
        RECORDERS.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Per-stream caller poll timestamps. This records bridge availability, not socket arrival.
#[doc(hidden)]
#[derive(Clone, Default)]
pub struct StreamConsumerClock(Arc<Mutex<ClockState>>);
type Observer = Box<dyn FnOnce(StreamConsumerReport) + Send>;
#[derive(Default)]
struct ClockState {
    polls: Vec<Instant>,
    delivered: Vec<Instant>,
    finished: Option<StreamConsumerReport>,
    observers: Vec<Observer>,
}
impl std::fmt::Debug for StreamConsumerClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamConsumerClock")
            .finish_non_exhaustive()
    }
}
/// Caller-visible completion receipt, separate from producer exhaustion.
#[doc(hidden)]
#[derive(Clone)]
pub struct StreamConsumerReport {
    /// Caller-visible chunk delivery timestamps.
    pub delivered: Vec<Instant>,
    /// Caller-visible exhaustion or close timestamp.
    pub ended: Instant,
    /// True only when the caller polled the terminal item.
    pub complete: bool,
}
impl StreamConsumerClock {
    /// Allocate only when trace recording has explicitly enabled timing.
    pub fn if_enabled() -> Option<Self> {
        (RECORDERS.load(Ordering::Acquire) > 0).then(Self::default)
    }
    /// Timestamp entry to one caller-visible next operation.
    pub fn requested(&self) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .polls
            .push(Instant::now());
    }
    /// Whether this item was produced before the caller requested it.
    pub fn was_prefetched(&self, index: usize, available: Instant) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .polls
            .get(index)
            .is_none_or(|poll| available < *poll)
    }
    /// Acknowledge one chunk after its Python conversion succeeds.
    pub fn delivered(&self) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .delivered
            .push(Instant::now());
    }
    /// Complete exactly once, including abandoned prefetched streams.
    pub fn finish(&self, complete: bool) {
        let (observers, report) = {
            let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
            if state.finished.is_some() {
                return;
            }
            let report = StreamConsumerReport {
                delivered: state.delivered.clone(),
                ended: Instant::now(),
                complete,
            };
            state.finished = Some(report.clone());
            (std::mem::take(&mut state.observers), report)
        };
        for observer in observers {
            observer(report.clone());
        }
    }
    /// Observe consumer completion without keeping the producer stream alive.
    pub fn observe(&self, observer: impl FnOnce(StreamConsumerReport) + Send + 'static) {
        let ready = {
            let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(report) = &state.finished {
                Some(report.clone())
            } else {
                state.observers.push(Box::new(observer));
                return;
            }
        };
        observer(ready.expect("finished receipt"));
    }
    /// Obtain the clock attached to the current streaming invocation.
    pub(crate) fn current() -> Option<Self> {
        CLOCK.try_with(Clone::clone).ok().flatten()
    }
}

/// Attach timing to stream construction without adding a public Python parameter.
#[doc(hidden)]
pub async fn with_stream_consumer_clock<F: Future>(
    clock: Option<StreamConsumerClock>,
    future: F,
) -> F::Output {
    CLOCK.scope(clock, future).await
}
