// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const require = createRequire(import.meta.url);
const lib = require('../index.js');
const adaptive = require('../adaptive.js');
const plugin = require('../plugin.js');

function config(replay) {
  return {
    version: 1,
    components: [
      adaptive.ComponentSpec({
        version: 1,
        responseCache: adaptive.responseCacheConfig({ namespace: 'node-replay-test', replay }),
      }),
    ],
  };
}

it('records, finalizes and strictly reloads through the shared plugin lifecycle', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'relay-replay-node-'));
  let activation;
  let calls = 0;
  const fixture = join(dir, 'fixture.json');
  const request = {
    headers: { authorization: 'transport-secret' },
    content: { model: 'test', temperature: 0, messages: [{ role: 'user', content: 'prompt' }] },
  };
  const provider = () => {
    calls += 1;
    return {
      id: 'chatcmpl-test',
      object: 'chat.completion',
      created: 1,
      model: 'test',
      choices: [{ index: 0, message: { role: 'assistant', content: 'answer' }, finish_reason: 'stop' }],
    };
  };
  try {
    activation = await plugin.initialize(config({ mode: 'record', outputPath: fixture, captureRequests: true }));
    const answer = await lib.llmCallExecute('openai', request, provider);
    const [report] = await adaptive.finalizeReplay();
    assert.equal(report.finalized, true);
    assert.equal(report.llm.captured, 1);
    assert.deepEqual(adaptive.replayReports(), [report]);
    await activation.close();
    assert.deepEqual(adaptive.replayReports(), []);
    const original = readFileSync(fixture);
    assert.equal(original.includes('transport-secret'), false);
    assert.deepEqual(JSON.parse(original).entries[0].request, request.content);

    activation = await plugin.initialize(config({ mode: 'replay_only', inputPath: fixture }));
    assert.deepEqual(await lib.llmCallExecute('openai', request, provider), answer);
    await assert.rejects(
      lib.llmCallExecute('openai', { ...request, content: { ...request.content, model: 'changed' } }, provider),
      /missing_entry/,
    );
    assert.equal(calls, 1);
    assert.equal(adaptive.replayReports()[0].llm.live_calls, 0);
    await adaptive.finalizeReplay();
    await activation.close();
    activation = undefined;
    assert.deepEqual(readFileSync(fixture), original);

    writeFileSync(fixture, '{}');
    await assert.rejects(
      plugin.initialize(config({ mode: 'replay_only', inputPath: fixture })),
      /fixture_incompatible/,
    );
    assert.deepEqual(adaptive.replayReports(), []);
  } finally {
    await activation?.close();
    rmSync(dir, { recursive: true, force: true });
  }
});
