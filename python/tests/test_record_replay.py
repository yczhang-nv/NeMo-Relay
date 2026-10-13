# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record/replay integration through the public Python stream bridge."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="record/replay currently requires Unix")


def invoke(mode: str, trace: Path, out: Path, scenario: str = "complete") -> dict:
    script = Path(__file__).with_name("record_replay_process.py")
    completed = subprocess.run(
        [sys.executable, str(script), mode, str(trace), str(out), scenario],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "XDG_CONFIG_HOME": str(out / "xdg"), "RR_TEST_SCALE": "0.5" if scenario == "timed" else "0"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads((out / "process.json").read_text())


def lines(out: Path, prefix: str) -> list[dict]:
    return [json.loads(line) for file in out.glob(prefix + ".*.jsonl") for line in file.read_text().splitlines()]


def test_record_then_verify_stream_with_live_tools(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    trace.mkdir()
    recorded = invoke("record", trace, tmp_path / "record")
    assert recorded == {"model_calls": 1, "tool_calls": 1, "received": 3}
    calls = [json.loads(line) for line in (trace / "calls.jsonl").read_text().splitlines()]
    assert calls[0]["outcome"] == "ok"
    assert calls[0]["response"]["vendor_extension"] == {"keep": True}
    assert "sentinel-credential" not in "".join(file.read_text() for file in trace.iterdir())
    status = json.loads(next((tmp_path / "record").glob("record_status.*.json")).read_text())
    assert status["model_calls"] == status["lines"]["calls"] == status["lines"]["requests"] == 1
    assert status["tool_calls"] == status["lines"]["tools"] == 1
    assert not status["in_flight"] and status["io_errors"] == 0
    # Only the launcher, after the recording process exits, finalizes the lock.
    (trace / "record.lock").unlink()
    verified = invoke("verify", trace, tmp_path / "verify")
    assert verified == {"model_calls": 0, "tool_calls": 1, "received": 3}
    assert lines(tmp_path / "verify", "tools")[0]["status"] == "ok"
    assert lines(tmp_path / "verify", "timing")[0]["outcome"] == "complete"
    missed = invoke("replay", trace, tmp_path / "miss", "miss")
    assert missed["model_calls"] == missed["tool_calls"] == 0
    assert lines(tmp_path / "miss", "divergence")[0]["type"] == "RR-MISS"
    dropped = invoke("replay", trace, tmp_path / "drop", "drop")
    assert dropped["model_calls"] == 0 and dropped["received"] == 1
    assert lines(tmp_path / "drop", "timing")[0]["outcome"] == "consumer_dropped"
    assert lines(tmp_path / "drop", "timing")[0]["chunks"] == 1


def test_record_drop_after_prefetch_rejects_completion(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    trace.mkdir()
    result = invoke("record", trace, tmp_path / "out", "drop")
    assert result["received"] == 1
    call = json.loads((trace / "calls.jsonl").read_text().strip())
    assert call["outcome"] == "consumer_dropped"


def test_python_work_progresses_during_replay_deadlines(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    trace.mkdir()
    invoke("record", trace, tmp_path / "record")
    call = json.loads((trace / "calls.jsonl").read_text())
    for i, chunk in enumerate(call["chunks"]):
        chunk["offset_us"] = (i + 1) * 200000
    call["ttft_us"] = 200000
    call["total_us"] = 800000
    (trace / "calls.jsonl").write_text(json.dumps(call) + "\n")
    (trace / "record.lock").unlink()
    result = invoke("verify", trace, tmp_path / "timed", "timed")
    assert result == {"model_calls": 0, "tool_calls": 1, "received": 3}
    timing = lines(tmp_path / "timed", "timing")[0]
    assert timing["scheduled_end_us"] == 400000 and timing["outcome"] == "complete"
    assert timing["actual_end_us"] >= timing["scheduled_end_us"]


def test_record_replay_config_accepts_path_objects(tmp_path: Path) -> None:
    from nemo_relay.adaptive import RecordReplayConfig, StartGateConfig

    config = RecordReplayConfig(
        mode="verify",
        trace_directory=tmp_path / "trace",
        output_directory=tmp_path / "out",
        start_gate=StartGateConfig(path=tmp_path / "go"),
    ).to_dict()
    assert config["trace_directory"] == str(tmp_path / "trace")
    assert config["output_directory"] == str(tmp_path / "out")
    assert config["start_gate"]["path"] == str(tmp_path / "go")
