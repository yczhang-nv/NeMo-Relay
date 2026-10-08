<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codex Replay Demo

Record a Codex task that reads a Python file with a shell tool and reviews it.
Repeat the task in a fresh session using strict replay. The shell tool executes
in both runs; every model call in the replay run must be served from the fixture
with zero live managed model calls.

This demo uses your normal Codex authentication. It requires no Dynamo endpoint
or tracing infrastructure. Recording uses your account's quota or API billing.

## Requirements

- macOS or Linux with Git, Rust through rustup, and a native compiler toolchain.
- Codex CLI 0.143.0 or newer, authenticated with a working model. Prototype
  validation used Codex 0.145.0.
- `jq` for the recording and replay checks.

Run all commands in the same terminal, from the repository root. Keep the model,
Codex settings, prompt, and sample file unchanged between runs. Run both sessions
on the same day. Native tool output and environment content can change the next
model request and cause a strict miss.

## 1. Clone and Build

```bash
git clone --single-branch --branch codex/replay-prototype \
  https://github.com/yczhang-nv/NeMo-Relay.git NeMo-Relay-replay
cd NeMo-Relay-replay

cargo build --locked --release -p nemo-relay-cli

relay="$PWD/target/release/nemo-relay"
demo="$PWD/examples/replay-demo"
mkdir -p "$demo/artifacts"
prompt="$(cat "$demo/prompt.txt")"
```

For an existing clone, pull the feature branch and start at the build command.
The configurations, prompt, and sample file are already included. Their relative
paths assume the repository root is the working directory.

## 2. Record the Task

```bash
"$relay" \
  --log-config-path "$demo/logging-record.toml" \
  run \
  --config "$demo/config.toml" \
  --openai-base-url https://api.openai.com/v1 \
  --plugin-config-path "$demo/record.toml" \
  -- codex exec --ephemeral --sandbox read-only "$prompt" \
  > "$demo/artifacts/record-output.txt" 2>&1

record_status=$?
cat "$demo/artifacts/record-output.txt"
printf '\nExit status: %s\n' "$record_status"
```

Stop if the exit status is nonzero or the file-reading tool fails. Confirm Codex
executes `cat examples/replay-demo/weighted_mean.py` and completes the review,
corrected implementation, and five proposed tests. The tests are returned as
text; this task does not execute them or edit the file.

Let the command finish normally. Gateway shutdown finalizes the recording and
publishes `artifacts/recording.json`. Forced termination does not establish that
completion boundary. Request capture is enabled for miss diagnosis; the fixture
includes private prompt data. Generated artifacts are ignored by Git.

## 3. Verify Complete Recording Coverage

Inspect the report:

```bash
jq '.manifest | {format_version, entry_count, coverage}' \
  "$demo/artifacts/recording.json"
```

Require a successful task, at least two captured model calls, and no coverage
gaps:

```bash
jq -e --argjson status "$record_status" '
  .manifest as $m |
  $m.coverage as $c |
  ($status == 0) and
  ($m.format_version == 2) and
  ($m.entry_count >= 2) and
  ($c.finalized == true) and
  ($c.llm.live_calls >= 2) and
  ($c.llm.captured == $c.llm.live_calls) and
  ($c.llm.uncaptured == 0) and
  ($c.conflicts == 0) and
  ($c.failed_writes == 0) and
  ($c.persistence_errors == 0)
' "$demo/artifacts/recording.json"
```

Proceed only if this prints `true` and exits successfully. The model normally
makes one call to request the tool and another to use its result. Native Codex
tools execute outside Relay's managed tool callback boundary, so the fixture's
tool counters do not measure the shell call.

```bash
recorded_calls="$(
  jq -r '.manifest.coverage.llm.live_calls' "$demo/artifacts/recording.json"
)"
printf 'Recorded model calls: %s\n' "$recorded_calls"
```

## 4. Replay the Identical Task

```bash
"$relay" \
  --log-config-path "$demo/logging-replay.toml" \
  run \
  --config "$demo/config.toml" \
  --openai-base-url https://api.openai.com/v1 \
  --plugin-config-path "$demo/replay.toml" \
  -- codex exec --ephemeral --sandbox read-only "$prompt" \
  > "$demo/artifacts/replay-output.txt" 2>&1

replay_status=$?
cat "$demo/artifacts/replay-output.txt"
printf '\nExit status: %s\n' "$replay_status"
```

Confirm exit status `0`, successful execution of the file-reading tool again,
and the same review, implementation, and tests. Session IDs, timing, and usage
messages in the complete terminal logs can differ.

## 5. Verify All Model Calls Were Hits

Display the latest replay counters; log files append across repeated attempts:

```bash
jq -s '
  [.[] | select(.event == "replay_finalized")] |
  last |
  .fields
' "$demo/artifacts/replay.jsonl"
```

Check that every recorded model call became a hit, with no live model calls:

```bash
jq -se --argjson expected "$recorded_calls" --argjson status "$replay_status" '
  [.[] | select(.event == "replay_finalized")] |
  last |
  .fields |
  ($status == 0) and
  (.llm_hits == $expected) and
  (.llm_misses == 0) and
  (.llm_live_calls == 0) and
  (.llm_uncaptured == 0)
' "$demo/artifacts/replay.jsonl"
```

The demo passes when both tasks complete, the tool succeeds in both sessions,
the answers match, and both coverage checks print `true`. A strict miss is a
failure; preserve the logs and fixture for diagnosis rather than changing to a
mode that permits live fallback.

The stream-completion fix has regression coverage. This exact native Codex tool
walkthrough has not yet been validated end to end.

See [Response Cache](../../docs/configure-plugins/adaptive/response-cache.mdx)
for fixture compatibility, eligibility, and replay lifecycle details.
