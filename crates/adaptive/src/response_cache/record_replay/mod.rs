// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless model traces with live tools and fail-closed model interception.

pub mod config;
mod cpu;
mod key;
mod session;
mod stream;
#[cfg(all(test, unix))]
mod tests;
mod tool;
mod trace;

use nemo_relay::api::runtime::{LlmExecutionFn, LlmStreamExecutionFn, ToolExecutionFn};

pub(crate) const PRIORITY: i32 = i32::MAX;

pub(crate) fn activate(
    config: crate::config::ResponseCacheConfig,
    diagnostics: Vec<String>,
) -> (
    LlmExecutionFn,
    LlmStreamExecutionFn,
    ToolExecutionFn,
    nemo_relay::plugin::PluginRegistration,
) {
    let session = session::Session::activate_with_diagnostics(config, diagnostics);
    let llm = stream::make_buffered(session.clone());
    let streaming = stream::make_stream(session.clone());
    let tools = tool::make_tool(session.clone());
    let cleanup = nemo_relay::plugin::PluginRegistration::new(
        "record_replay",
        "record_status",
        Box::new(move || {
            session.close();
            Ok(())
        }),
    );
    (llm, streaming, tools, cleanup)
}
