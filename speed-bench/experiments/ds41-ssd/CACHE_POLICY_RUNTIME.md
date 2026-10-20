# Native cache-policy experiments (2026-09-20)

**2026-09-21 update:** SLRU runtime support and its dedicated native test were
removed after the interactive regressions. The measurements below are historical;
reproducing SLRU requires experiment commit `89370fba`. The offline simulator
and its tests remain available. Aging interval support remains in the runtime.

Baseline: rebased local stack `07e8e131`, upstream `0aaea5a2`, Apple M5 Max/128 GiB.
All runs explicitly select `gguf/DeepSeek-V4.1-Flash-Q2.gguf`; the default
`ds4flash.gguf` symlink currently points to Qwen. SSD streaming stays enabled.

## Findings

- Aging 256 improves the recorded tool-session inference time 3.66% against its
  matched controls. A separate attribution run shows 7.80% **more** expert pread.
- SLRU improves the same workload 0.73% against its own controls, trading slower
  prefill for faster decode and 3.96% more expert pread.
- Aging 64 at a smaller cache increases reads 30.41% on a short session excerpt.
  Do not adopt the offline study's tentative 64/256 capacity heuristic.
- SLRU regresses actual-agent response time 6.45% fresh and 1.77% restored;
  leave it disabled for mixed interactive/tool use.
- All matched replay logits and final continuation states are byte-identical.
  Default 16-token aging remained unchanged; both policies were opt-in during
  the study. SLRU runtime support has since been removed.

The offline study omitted prefill. Native cache seeding, batched/scalar access
patterns and in-flight exclusions matter: **neither native gain comes from
fewer aggregate cache misses**. The runs establish timing/read-count outcomes,
not their detailed causal explanation. Phase-separated I/O attribution is a
useful next experiment.

## Method and limits

Full session-v1 retains all 14 generation rounds, 20 recorded tool calls' input
boundaries, 3,371 generated tokens and context growth to 8,710. Tokens are forced
through native inference; tool execution and sampling are excluded. Each
comparison freezes its executable, shaders and recording, checks full-vocabulary
logits at every phase boundary, and checks the complete final snapshot.

Runs are serial. Automatic dynamic expert capacity is 7,822 slots/72.51 GiB,
plus 7.12 GiB of prefill reserve. Demand-workspace and other experimental flags
are unset. All read counts are native `miss_pread` bytes, not estimates from
miss counts or measured physical SSD traffic. Cumulative counters include the
initial system prefix. Main turn-time comparisons exclude that prefix; its
cost and startup-inclusive inference totals are shown separately below.

This is one machine and one full recorded task, with two samples per variant,
plus short interactive and small-cache checks. It does not establish vision,
independent-task, long-context or combined-flag performance. The two policy
tables have separate matched controls/binaries: do not add their gains.

## Experiment 1: slower route-frequency decay

`DS4_METAL_STREAM_CACHE_DECAY_TOKENS=N` accepts 1..4096; unset/invalid uses 16.
The existing decode-token clock, frequency updates and eviction ordering remain.

### Full-session ABBA: 256 versus 16

| Metric | Default 16 | Aging 256 | Change |
|---|---:|---:|---:|
| Tool/user append prefill | 92.836 s | 88.394 s | -4.79% |
| Decode | 185.113 s | 179.369 s | -3.10% |
| Combined, excluding initial system prefix | 277.949 s | 267.763 s | **-3.66%** |
| Initial system prefill | 15.275 s | 15.556 s | +1.84% |
| Inference including system prefix | 293.224 s | 283.319 s | -3.38% |

Individual turn-time samples: A 278.308, B 267.453, B 268.074, A 277.591 s.
Both candidates beat both controls; exact phase logits and final state.
Evidence: `cache-runtime-decay256/`.

This first frozen binary lacked a final counter report. The replay now emits
one outside timed phases, after saving the snapshot. A separate aging 256 run
uses the *same binary as the SLRU comparison*, checked against its aging 16
controls for identical model path, recording, shader contents, logits and state:

| Cumulative counter | Aging 16 controls | Aging 256 diagnostic | Change |
|---|---:|---:|---:|
| Native cache misses | 80,226 | 85,873 | +7.04% |
| Native expert pread | 671.37 GiB | 723.71 GiB | +7.80% |

This single-run attribution supplies read counts, not additional balanced
latency evidence. See `cache-runtime-decay256-reads/cross-validation.json`.

### Smaller-cache ABBA: 64 versus 16

First seven rows of session-v1: three generation rounds/343 generated tokens.
Requested cache 45 GiB gives 4,086 dynamic slots/37.88 GiB, after prefill reserve.
The frozen excerpt is retained in `cache-runtime-decay64-small/session.txt`.

- Combined 43.689 -> 44.899 s (**+2.77%**); decode +6.56%.
- Native cache misses 22,128 -> 26,860 (+21.38%).
- Native expert pread 144.20 -> 188.05 GiB (+30.41%), identical across repeats.
- Exact phase logits/final state; zero mlock failures.

Timing samples overlap: controls 42.473/44.905 s, candidates 44.094/45.704 s.
The latency estimate is noisy, whereas extra reads/misses repeat exactly.
This includes actual prefill frequency updates and seeding, absent offline.
It does not establish full long-session behavior at this capacity.
Evidence: `cache-runtime-decay64-small/`.

### Interactive ABBA: 256 versus 16

Actual ds4-agent, hello / 2+2 / hello, generation limit 256. Both fresh system
prefill and saved-system-KV restoration are tested. Input/output tokens match
in every run (59/15/28 generated tokens, complete answers).

Fresh summed response times: controls 10.482/8.809 s, candidates 9.204/8.889 s.
Control drift exceeds the candidate effect; no reliable fresh improvement.
Restored controls 13.030/12.821 s, candidates 12.751/12.681 s: mean **-1.62%**.
No demonstrated interactive regression in this small check.
Evidence: `cache-runtime-decay256-agent/`.

## Experiment 2: segmented LRU

`DS4_METAL_STREAM_CACHE_SLRU=1` enables SLRU independently of aging. New entries
enter probation. Native resident hits promote or refresh protected membership.
Protected capacity is floor(80% of configured capacity), at least one entry.
Protected LRU demotes to probation MRU when over target; a budget reduction is
enforced on the next hit. Entry removal unlinks metadata only after the existing
in-flight check passes. All five victim-selection paths use segment then recency,
retaining native selected-route/in-flight exclusions. Native seeding remains.

Protected membership uses a constant-time intrusive queue. Probation uses the
existing recency stamps/victim scans; there is no extra full-cache scan per hit.
Frequency bookkeeping is retained, but SLRU does not use it to rank victims.
Combining SLRU with aging 256 is not an independently validated optimization.

### Full-session BAAB

| Metric | Default aging 16 | SLRU | Change |
|---|---:|---:|---:|
| Tool/user append prefill | 94.604 s | 97.089 s | +2.63% |
| Decode | 186.899 s | 182.362 s | -2.43% |
| Combined, excluding initial system prefix | 281.503 s | 279.451 s | **-0.73%** |
| Initial system prefill | 15.051 s | 14.558 s | -3.27% |
| Inference including system prefix | 296.554 s | 294.009 s | -0.86% |
| Native cache misses | 80,226 | 83,096 | +3.58% |
| Native expert pread | 671.37 GiB | 697.97 GiB | +3.96% |

Samples in BAAB order: 279.767, 282.069, 280.937, 279.135 s. Both candidates beat
both controls, but the effect is small (2.05 s/session). Miss/read counts repeat
exactly; all phase logits/final states match, including the aging experiment's
reference. Zero mlock failures. Evidence: `cache-runtime-slru/`.

### Interactive check

Same actual-agent protocol as aging: hello / 2+2 / hello, complete 59/15/28-token
answers, exact input/output tokens in every fresh/restored run.

Fresh summed response times: controls 8.886/8.720 s, candidates 9.430/9.312 s:
mean **+6.45%**. Restored controls 12.910/12.725 s, candidates 13.122/12.968 s:
mean **+1.77%**. Both candidates are slower than both controls in both modes.
Evidence: `cache-runtime-slru-agent/`.

For mixed interactive/tool use, this SLRU configuration is not recommended:
its 0.73% recorded-session gain comes with an interactive regression and more
I/O. Its runtime support was subsequently removed. Aging 256 is the stronger opt-in
candidate for the tested large-cache workload; default 16 remains unchanged.

## Validation and reproduction

`make test-slru` compares 100,000 mixed hit/removal/shrink operations with an
independent array model, checking every node/link. All 33 existing Python harness
tests pass. Native replay checks cover prefill, cache seeding and reuse, and
in-flight exclusions. No arithmetic, quantization or session format changes.

```
make speed-bench/agent-session/replay ds4-agent test-slru
python3 -m unittest discover -s speed-bench/agent-session -p 'test_*.py'
python3 speed-bench/agent-session/replay_abba.py /tmp/new-aging \
  --model gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --candidate-env DS4_METAL_STREAM_CACHE_DECAY_TOKENS=256
python3 speed-bench/agent-session/replay_abba.py /tmp/new-slru \
  --model gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --candidate-env DS4_METAL_STREAM_CACHE_SLRU=1 --order BAAB
python3 speed-bench/experiments/ds41-ssd/agent_abba.py /tmp/new-interactive \
  --model gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --candidate-env DS4_METAL_STREAM_CACHE_DECAY_TOKENS=256 --gen-tokens 256
```

For the smaller cache check, pass the retained excerpt with `--replay`,
`--cache-gb 45`, and decay 64. The diagnostic uses `--order B` with decay 256;
unpaired runs require external identity checks and are not speedup evidence.
