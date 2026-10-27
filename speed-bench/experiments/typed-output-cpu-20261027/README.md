# Sampled-output projection CPU check

Checkpoint: 2026-10-27, starting 04:36 UTC. M5 Max, ReleaseFast native fixture,
no model or GPU job. This measures parser CPU work, not serving throughput.
The interaction work is based on hosted-orchestration cleanup `f65fc2c5`.

The first typed-output implementation rebuilt accumulated argument JSON in both
the generation tracker and the protocol projector. Ordinary reasoning had no
meaningful cost, but a long string exposed quadratic copying: the DS41 tracker
took about 289 ms over an 8192-token/41 KB tool string, versus 0.37 ms for the
legacy incremental byte tracker.

The selected change reuses the same parser with argument construction disabled
for generation. It still recognizes actual token boundaries, validates framing
and reports tool/payload/end state. Wire output and completed parsing keep the
full argument projection. There is no second state machine or inference change.

## Matched CPU measurements

One warmup, five alternating legacy/full-projection repeats, five state-only
repeats; medians below. Each synthetic token contributes five bytes, except
atomic special-token spans. Reasoning ends with a real close and `OK`; tool
payloads are one declared string field. Both parsers' final text, reasoning and
decoded tool values are checked before timing. These are deterministic parser
fixtures, not actual model tokenization or a quality evaluation.

| Model / output | Tokens | Legacy tracker, ms | Full projection, ms | Selected state-only tracker, ms |
|---|---:|---:|---:|---:|
| DS41 reasoning | 2048 | 1.143 | 0.037 | 0.017 |
| DS41 reasoning | 8192 | 4.157 | 0.137 | 0.060 |
| DS41 tool string | 2048 | 0.092 | 19.686 | 0.211 |
| DS41 tool string | 8192 | 0.365 | 294.176 | 0.808 |
| Flint reasoning | 2048 | 0.896 | 0.015 | 0.015 |
| Flint reasoning | 8192 | 3.594 | 0.061 | 0.059 |
| Flint tool string | 2048 | 0.064 | 18.762 | 0.057 |
| Flint tool string | 8192 | 0.262 | 294.664 | 0.217 |

The state-only tool tracker scales roughly linearly here and removes over 99.7%
of its previous full-projection CPU time. The streaming projector still performs
one full pass per update and remains quadratic for very long generated argument
strings. At this 8K point its roughly 0.3 seconds would be about 0.7% of the 40.96
seconds required to generate 8192 tokens at a hypothetical 200 tok/s. That is
context for the cost, not a measured end-to-end overhead. Stop this performance
slice here; actual guarded serving measurements remain a separate gate.

## Reproduction and validation

Run from the repository root:

```text
nu tools/zig-build ds41-output-test server-test --prefix .workspaces/interaction-catchup-20261027/output-perf -j4
.workspaces/interaction-catchup-20261027/output-perf/bin/test-ds41-output --bench
.workspaces/interaction-catchup-20261027/output-perf/bin/test-ds41-output
.workspaces/interaction-catchup-20261027/output-perf/bin/ds4_test --server
```

All builds and checks pass. Two pure Zig tests compare state-only/full framing
at partial prefixes, including literal control spellings and UTF-8 boundaries.
The production-linked fixture passes DS41/Flint incremental Responses argument
and ID checks plus KTM2 SSD replay. The retained server suite passes its existing
all-byte-cut Chat/Anthropic streaming checks. No model generation, decoder kernel
or sampling change was made for this optimization.

Raw JSONL and logs remain ignored under
`.workspaces/interaction-catchup-20261027/output-perf-*`.
