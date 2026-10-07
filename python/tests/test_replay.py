# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent replay through the public adaptive plugin lifecycle."""

import json
from pathlib import Path

import pytest

from nemo_relay import Json, LLMRequest, adaptive, llm, plugin


def replay_plugin(replay: adaptive.ReplayConfig) -> plugin.PluginConfig:
    return plugin.PluginConfig(
        components=[
            adaptive.ComponentSpec(
                adaptive.AdaptiveConfig(
                    response_cache=adaptive.ResponseCacheConfig(namespace="python-replay-test", replay=replay)
                )
            )
        ]
    )


async def test_record_reload_strict_replay_and_derived_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.json"
    calls = 0

    def provider(_request: LLMRequest) -> Json:
        nonlocal calls
        calls += 1
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
        }

    request = LLMRequest(
        {"authorization": "transport-secret"},
        {"model": "test", "temperature": 0, "messages": [{"role": "user", "content": "prompt"}]},
    )
    async with plugin.activate(replay_plugin(adaptive.ReplayConfig(output_path=str(fixture), capture_requests=True))):
        answer = await llm.execute("openai", request, provider)
        assert calls == 1
        reports = await adaptive.finalize_replay()
        assert reports[0]["finalized"] is True
        assert reports[0]["llm"]["captured"] == 1
        assert adaptive.replay_reports() == reports
    original = fixture.read_bytes()
    assert b"transport-secret" not in original
    assert json.loads(original)["entries"][0]["request"] == request.content
    assert adaptive.replay_reports() == []

    async with plugin.activate(replay_plugin(adaptive.ReplayConfig(mode="replay_only", input_path=str(fixture)))):
        assert await llm.execute("openai", request, provider) == answer
        changed = LLMRequest({}, {**request.content, "model": "different"})
        with pytest.raises(Exception, match="missing_entry"):
            await llm.execute("openai", changed, provider)
        assert calls == 1
        report = (await adaptive.finalize_replay())[0]
        assert report["llm"]["live_calls"] == 0
        assert report["last_failure"]["reason"] == "missing_entry"

    derived = tmp_path / "derived.json"
    async with plugin.activate(
        replay_plugin(adaptive.ReplayConfig(mode="replay_or_record", input_path=str(fixture), output_path=str(derived)))
    ):
        await llm.execute("openai", request, provider)
        await llm.execute("openai", changed, provider)
        assert calls == 2
        await adaptive.finalize_replay()
    assert fixture.read_bytes() == original
    artifact = json.loads(derived.read_bytes())
    assert len(artifact["entries"]) == 2
    assert all("request" not in entry for entry in artifact["entries"])


async def test_invalid_fixture_prevents_activation(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{}")
    with pytest.raises(Exception, match="fixture_incompatible"):
        async with plugin.activate(replay_plugin(adaptive.ReplayConfig(mode="replay_only", input_path=str(path)))):
            pytest.fail("invalid fixture must not activate")
    assert adaptive.replay_reports() == []
