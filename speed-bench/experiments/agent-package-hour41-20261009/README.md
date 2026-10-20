# Item 41: direct ds4-agent integration

`persistent-fast` now works directly with `ds4-agent`. It combines resident
decoder and selected-prefill substitutions with pre-RoPE suffix reuse, including
even right shifts. The same reuse path now participates in built-in automatic
and `/compact` compaction without requiring a live context file.

Runtime revision: `e203b8065022b8762d2f7eae8dcaff96c68bb97c`.
One-hour budget: 2026-10-09 02:40:48–03:40:48 UTC, including implementation,
builds, qualification and analysis. All behavior remains opt-in.

```sh
./ds4-local persistent-fast --model gguf/DeepSeek-V4.1-Flash-Q2.gguf
```

For CLM-style model editing, add
`--live-context-file .workspaces/agent-context.txt`. The model can edit that
conversation mirror with its file tools. Built-in compaction instead generates
a task-state summary and retains recent turns. In the pre-RoPE path, it can keep
an earlier complete turn within a bounded tail so there is useful history before
the latest user to import. Archiving facts into checkable files remains agent
policy; compaction itself does not create a permanent fact store.

## Qualification

`make -j2 ds4-agent ds4_agent_test`, `./ds4_agent_test`, the focused model-free
suffix mock, and eight wrapper argv/environment capture cases passed. The mock
checks repeated moves from unchanged originals, right-shift admission, the
legacy resident-routing restriction, latest-user freshness, rollback and actual
built-in compaction without a mirror file. Commands and binary identity are in
[checks.json](checks.json).

The real CLI workflow has five turns: read inventory, retain useful context,
remove disposable old logs, then grow an old-history replacement. Each query
must freshly read both inventory and its permanent archive. Correct answers
change from `416 / RIVER-73` to `466 / EMBER-29`. Both selected-prefill and decoder
substitution counters were positive in the combined runs. All ten query answers
and required fresh reads passed across five real CLI runs.

The final matched fixture used ctx 8192, prefill chunk 512, a 64 GiB expert cache,
temperature zero and `--nothink`. The application state-cache hot budget was
zero, forcing checkpoint materialization through its SSD-backed files; OS page
caching remained active. The ordinary preset defaults are hot 512 MiB and SSD
8192 MiB.

| Final matched workflow | Whole launcher | Left edit + query | Right edit + query |
|---|---:|---:|---:|
| Native opening | 266.156 s | 72.439 s | 127.230 s |
| Persistent-fast | 143.021 s | 29.630 s | 47.940 s |
| Native closing | 268.028 s | 72.379 s | 126.017 s |

The combined run saved about **46.5%** against the native mean; native whole-time
drift was **0.70%**. This is one favorable, edit-heavy workflow with warmed
bootstrap and cross-run caches, not a general agent latency or coding-quality
claim. A separate compaction probe ran before the closing native control.
The earlier 23.5% HTTP coding-task result remains a different workload.
No new delta-NLL or KL measurement was made in this integration item.

Observed immutable-source grafts were:

| Operation | Source → destination | Reused | Fresh |
|---|---|---:|---:|
| Left edit | `2546:3749 → 2044:3247` | 1203 | 398 |
| Growing edit after the left graft | `2060:4146 → 2568:4654` | 2086 | 623 |
| Built-in compaction, no mirror file | `2654:3350 → 2138:2834` | 696 | 187 |

All three used pre-RoPE originals, SSD-backed sources, and preserved the latest
user marker/message freshness boundary. The growing edit added 1010 tokens to
the current transcript; its source relocation was +508 because the immutable
source predates the earlier −502 edit. The right import occurred while the live
session was already grafted, using an independently ungrafted original source.
“Ungrafted” here does not imply native expert routing: resident substitutions
were enabled for that source's execution.

The real compaction probe generated its own model summary, reduced the context
from 3537 to 3021 tokens, and returned exactly `EMBER-7319` afterward. Its
90.161-second compaction time includes summary generation and rebuilding; it
was not a comparative compaction speed test. The probe calls the common routine
used by automatic compaction and `/compact`, rather than driving the TUI command.

## Retained limitations and failures

The initial native growing edit was odd (+1009). Appending a word after its
marker also changed newline tokenization, producing +1011 in the first combined
run. That combined run correctly recomputed the context and answered both
questions, but its both-directions qualification failed and returned driver
status 3. The agent itself exited zero. These outputs remain in `live-native/`
and `live-persistent/`; they are not included in the final matched timing.
Moving the inserted word inside a disposable sentence produced the supported
+1010 fixture. Each model run used its first answers without a retry.

Generated-summary parity can likewise cause normal compaction fallback.
A long single-user tool loop may have no eligible retained history before its
latest user. Missing checkpoints, unsupported geometry or a staging-budget miss
also recompute under the chosen routing settings. Suffix reuse remains an
approximation: retained activations still carry their original prefix influence.

All six model jobs ran sequentially under the existing pressure guard. All 1365
samples reported normal pressure, with zero swap growth and peak wired plus
compressed RAM of 93.42 GiB. Useful KV files stay under
`.workspaces/agent-package-20261009/`; small raw outputs, traces, commands and
pressure samples are retained here. [clm-results.json](clm-results.json) contains
the measured outcomes and counters; each guard's `command.json` records the
exact invocation. Use the saved fixtures for the two initial alignment cases;
`run_agent_edits.py` now contains the final supported fixture.

Decision: expose the combined opt-in package in the direct agent. Keep normal
recomputation as the fallback and leave autonomous archival policy to the agent.
