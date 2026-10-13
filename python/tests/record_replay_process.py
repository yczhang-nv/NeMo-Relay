# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated process exercising the public record/replay Python APIs."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from nemo_relay import LLMRequest, ToolExecutionResult, llm, plugin, tools
from nemo_relay.adaptive import (
    AdaptiveConfig,
    ComponentSpec,
    DivergenceConfig,
    KeyFiltersConfig,
    RecordReplayConfig,
    ResponseCacheConfig,
)


async def main() -> None:
    mode, trace_path, output_path, scenario = sys.argv[1:]
    trace = Path(trace_path)
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    config = plugin.PluginConfig(
        components=[
            ComponentSpec(
                AdaptiveConfig(
                    response_cache=ResponseCacheConfig(
                        namespace="python-rr",
                        record_replay=RecordReplayConfig(
                            mode=mode,
                            trace_directory=str(trace),
                            output_directory=str(out),
                            latency_scale=float(os.environ.get("RR_TEST_SCALE", "0")),
                            key_filters=KeyFiltersConfig(tool_results="placeholder"),
                            divergence=DivergenceConfig(
                                tool_names=["terminal"], tool_fields=["/exit_code"], flag_fields=[]
                            ),
                        ),
                    ),
                )
            )
        ]
    )
    report = plugin.validate_exact(config)
    assert not report["config"]["diagnostics"], report
    model_calls = 0
    tool_calls = 0
    frames = [
        {"type": "response.created", "response": {"id": "r", "object": "response", "status": "in_progress"}},
        {"type": "response.output_text.delta", "delta": "answer", "output_index": 0, "content_index": 0},
        {
            "type": "response.completed",
            "response": {
                "id": "r",
                "object": "response",
                "status": "completed",
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]}
                ],
                "reasoning": {"effort": "medium"},
                "usage": {"input_tokens": 2, "output_tokens": 3, "output_tokens_details": {"reasoning_tokens": 1}},
                "vendor_extension": {"keep": True},
            },
        },
    ]

    async def provider(_request: LLMRequest):
        nonlocal model_calls
        model_calls += 1
        if mode != "record":
            raise AssertionError("replay reached model provider")
        for frame in frames:
            await asyncio.sleep(0.002)
            yield frame

    def execute_tool(_args):
        nonlocal tool_calls
        tool_calls += 1
        return ToolExecutionResult(
            {"exit_code": 0, "output": "old" if mode == "record" else "new"}, annotation={"opaque": True}
        )

    received = []
    progress = []
    first_delivery = None
    start = asyncio.get_running_loop().time()

    async def other_python_work():
        await asyncio.sleep(0.02)
        sum(i * i for i in range(10000))
        progress.append(asyncio.get_running_loop().time() - start)

    background = asyncio.create_task(other_python_work()) if scenario == "timed" else None
    async with plugin.activate(config):
        request = LLMRequest(
            {"Authorization": "sentinel-credential"},
            {
                "model": "m",
                "input": [{"role": "user", "content": "stable"}],
                "store": False,
            },
        )
        if scenario == "miss":
            request = LLMRequest({}, {"model": "m", "input": [{"role": "user", "content": "changed"}], "store": False})
        collected = []
        try:
            stream = await llm.stream_execute(
                "openai", request, provider, collected.append, lambda: frames[-1]["response"]
            )
            if scenario == "drop":
                # Give the producer enough time to prefetch all three chunks.
                await asyncio.sleep(0.05)
                received.append(await anext(stream))
                await stream.aclose()
            else:
                async for chunk in stream:
                    if first_delivery is None:
                        first_delivery = asyncio.get_running_loop().time() - start
                    received.append(chunk)
                if scenario != "miss":
                    result = await tools.execute(
                        "terminal", {"command": "synthetic"}, execute_tool, tool_call_id="tool-1"
                    )
                    assert result.annotation == {"opaque": True}
                    assert result.result["output"] == ("old" if mode == "record" else "new")
        except Exception as error:
            if scenario != "miss" or "RR-MISS" not in str(error):
                raise
    assert model_calls == (1 if mode == "record" else 0)
    if background is not None:
        await background
        assert progress and first_delivery is not None
        assert progress[0] + 0.02 < first_delivery, (progress, first_delivery)
    if scenario in ("complete", "timed"):
        assert received == frames
        assert tool_calls == 1
    (out / "process.json").write_text(
        json.dumps({"model_calls": model_calls, "tool_calls": tool_calls, "received": len(received)}) + "\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
