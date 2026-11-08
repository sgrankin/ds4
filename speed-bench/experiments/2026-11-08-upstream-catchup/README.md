# Native adaptation through upstream b3f50180

This catch-up brings the surviving functionality between upstream `890898cf`
and `b3f50180` into the native fork. Implementation and qualification are complete. It does not
resume the paused performance campaign, include the separate Kimi branch, or
advance published main.

The four tested frontends (`ds4`, `ds4-agent`, `ds4-server` and `ds4-bench`)
are installed at the repository root with their matching shader sources.
Their hashes match the build outputs and all four help checks pass. Previous
binaries are retained locally in
`.workspaces/upstream-b3f50180/installed-before/`. The temporary implementation
commits are consolidated into one adaptation merge; `catchup-b3f50180` names
that local result. Published main remains at `364ceeab`.

## Scope and implementation

The audit covers all 59 changed upstream files, 96 non-merge commits and the
final choices made by the intervening merges. Native controllers remain in Zig;
the retained full GLM and V4 Pro controllers receive their shared changes too.
Only the previously retired original V4 Flash, GLM 5.2, CUDA and ROCm products
are excluded. Intermediate implementations that upstream itself removed are
not restored. The exact input and disposition inventories remain with the local
qualification artifacts under `.workspaces/upstream-b3f50180/`.

| Area | What changed and where it is explained |
| --- | --- |
| Qwen storage | SSD experts now work through prefill, ordinary decode and the embedded predictor. Static weights remain mapped, selected experts use bounded owned buffers, and cached work overlaps missing-weight reads. Admission includes session graphs and predictor staging. [Runtime](../../../docs/models/qwen38-flash-next/implementation.md), [shared loader](../../../docs/storage/expert-streaming.md). |
| Qwen prediction and kernels | Predictor history can be prepared without computing discarded expert outputs. Pending trunk rows survive checkpoints and rollback. Low-bit matrix kernels, exact reduction order, half-aligned Q2 reads, embedding gathers and device-specific dispatch follow the final upstream implementation. [Apple Silicon](../../../docs/models/qwen38-flash-next/apple-silicon.md). |
| DeepSeek | The existing local scalar queues gain upstream controls and broader regression coverage. Batched verifier work gets bounded separate Engram inputs and table-major reads. Candidate scoring retains original score positions and tie order. Medium-prefill pruning retains the complete terminal raw window; retained Pro indexers and callback overlap are included. [Apple Silicon](../../../docs/models/deepseek/apple-silicon.md). |
| GLM | Flash decode divides indexed attention into four phases; sparse prefill validates selected prefixes before unchecked gathers. Discarded predictor work prepares caches only. Full GLM retains the dense/sparse verifier fallback, aligned SSD sweep and rotated compact-key reuse. [Apple Silicon](../../../docs/models/glm53-flash/apple-silicon.md). |
| Shared runtime | Batched victim planning preserves scalar eviction order. Static-weight locking narrows page-in where safe, image resizing uses independent row workers, and Q8 kernels preserve prior arithmetic. Optional unpacking falls back when allocation fails; the existing scratch lifetime repair remains intact. Device gates retain their tested boundaries. [Metal backend](../../../docs/backends/apple-silicon.md). |

Qwen writes QW8v3 model payloads inside the existing DSV4 envelope. The new runtime
reads old v2 payloads and rebuilds missing predictor history; the old runtime
cannot read new v3 saves. Preserve old snapshots or replay durable conversation
history when rolling back. Responses transcript persistence is separate from
these disposable inference snapshots.

## Qualification

The machine is an M5 Max with 128 GiB unified memory. Every GPU/model job runs
alone under the existing pressure guard. Tests use the actual native owners,
including test-only adapters for upstream fixtures that previously called
private C functions. No obsolete C implementation is substituted for the Zig
implementation under test.

- Host checks pass for cache victim order and accounting, seven Qwen mapping,
  memory and history policies, exact image resizing, page-in policy and device
  selectors. The sanitized compact-matrix oracle passes 1,033 mapping cases,
  3,200 schedules and 32 deliberate source-mutation controls.
- All eight standalone Metal oracle runs and the production-linked indexer,
  GLM attention/KDA, Q8 prefill, Qwen arithmetic/kernel and DeepSeek kernel
  checks pass. The full Qwen SSD fixture passes selected-only read accounting,
  overlap order, high expert IDs, cache reuse, off-size staging and injected
  failure/retry checks.
- Installed DeepSeek target weights pass 24-token queued-versus-serialized
  continuation with exact logits, Engram history and live cache spans. Initial
  4,096-row prefill, 6,144-row aligned append, 4,353-row ragged append and later
  decode match the unpruned reference state and logits. Installed DSpark passes
  batched verification, every accepted-prefix rollback, checkpoint restore,
  ordinary/exact sampling and cancellation checks.
- Installed Qwen Q4 passes SSD n-gram failure/recovery and ordered ordinary and
  speculative continuation. Predictor tests cover prepared/cold history,
  pending rows, checkpoint reuse, v2 restore, MTP-on/off exchange and rollback.
  Their independent target-logit comparisons use the fixture's numerical
  tolerances where execution schedules differ; they are not an all-bit-identical
  claim.
- Qwen resident versus SSD execution matches all nine complete vocabulary
  frontiers after a 575-token prompt in five chunks and eight continuation
  tokens. One-shot and session execution also agree. The startup plan for this
  fixture is 70.33 GiB resident versus 12.10 GiB with a 1,024-expert SSD cache;
  these are component estimates, not measured process peaks or a speed claim.
- GLM and Qwen each pass all 25 Responses WebSocket checks and every answer
  check: incremental histories, overlapping conversations, tool results,
  interrupted generation, reconnects and durable restart. These bounded tests
  use non-reasoning requests; they do not qualify autonomous agent reasoning.
- Server/agent host regressions, snapshot envelopes, serving boundaries and
  local web recovery pass.

The installed-model run found one production omission: Qwen SSD startup still
selected only the embedding span. It now maps every non-expert static tensor,
including predictor weights, through the native span owner. A focused host
regression checks those inclusions and excludes expert and n-gram payloads.
Other failures were fixture adaptations: missing engine context around private
graph calls, a read probe that did not intercept native calls, cache-reset
assumptions incompatible with local trimming, and pre-scope vocabulary sizing.
Their original logs remain with the successful reruns.

Full GLM and V4 Pro model weights, older Apple GPUs, M6 and physical two-Mac
execution are unavailable here. Their functionality is ported and relevant
shared primitives are checked; whole-model or physical-hardware validation is
not claimed.

## Before/after measurements

Baseline source is `df94063f`; its benchmark binary is from the runtime-identical
`5e8fafb5` build. Candidate runtime was built at `1de70866` and is unchanged by
later fixture, documentation and commit consolidation work. Its `src` Git tree
is `2020a8e38e9eeb8dc6dc5fe247335e3d95406eaf`. Both use Zig 0.17.0 ReleaseFast;
the runner uses Nushell 0.116.1. Benchmark SHA-256 values are:

- Baseline: `f579099b36d498f1ed2cd71fa4634e8fc86e2cb4d992adbd55996b11ffaad881`.
- Candidate: `622b59bcfe62db92f5114e32ae0e4822e9df46438382310185c2a1e5f4393fc4`.

Each workload has five fresh-process pairs, ordered AB, BA, AB, BA, AB, where A
is baseline and B is candidate. The same prompt file and fixed teacher tokens
are used on both sides. Each invocation performs fresh prefill, decode, a
512-token append after state restore and a second decode. DeepSeek and Qwen
use 128 decode tokens per frontier; GLM uses 32 to bound the slower SSD case.
All installed checkpoints are unchanged: DeepSeek Flash-1017 IQ2/Q2,
Qwen Flash Next Q4 and GLM 5.3 Flash Q2. The two DeepSeek rows use the same
checkpoint at different lengths.

DeepSeek uses the existing fast-preset compute/routing settings, context 40960,
chunk 8192 and a 64 GiB expert-cache target on both sides. Qwen is resident with
context 4096 and chunk 512; its 95.37 GiB n-gram table stays on disk. GLM uses
context 40960, a 64 GiB cache target and the existing per-layer decode mapping
fallback. Its chunk is graph-selected; the CLI prefill-chunk override is not
supported for this model. These GLM rates do not describe resident inference.

[run.nu](run.nu) records the complete settings and launches each binary from its
own checkout so that it loads its own shaders. There is no OS cache flush;
expert state survives the KV restore. Inference phases exclude loading and
snapshot save/restore; process wall includes them. Steady decode omits only the
first token. Complete decode durations can be reconstructed from the local CSVs, which also
retain first-token latency and every raw sample. [summarize.py](summarize.py) computes the tables below
without dropping samples. Durations are reconstructed from CSV rates rounded to
two decimals. Every table shows mean ± sample standard deviation; arrows are
baseline → candidate.

| Workload | Fresh prefill, seconds | 512-token append, seconds |
| --- | ---: | ---: |
| DeepSeek, 32K | 47.145 ± 3.468 → 45.986 ± 2.841 | 9.328 ± 4.007 → 7.513 ± 0.394 |
| DeepSeek, 6K | 18.409 ± 0.184 → 18.313 ± 0.207 | 7.736 ± 0.541 → 8.998 ± 3.236 |
| Qwen resident, 2K | 1.827 ± 0.059 → 1.808 ± 0.059 | 0.585 ± 0.024 → 0.600 ± 0.020 |
| GLM SSD, 8K | 24.193 ± 0.121 → 23.200 ± 0.089 | 2.983 ± 0.056 → 2.897 ± 0.038 |

| Workload | Initial steady decode, tokens/s | After append, tokens/s |
| --- | ---: | ---: |
| DeepSeek, 32K | 33.578 ± 0.586 → 33.846 ± 0.759 | 33.404 ± 1.779 → 33.726 ± 1.103 |
| DeepSeek, 6K | 34.748 ± 0.905 → 34.632 ± 1.598 | 33.546 ± 0.635 → 33.742 ± 0.597 |
| Qwen resident, 2K | 50.804 ± 1.761 → 53.506 ± 1.376 | 46.442 ± 0.261 → 49.916 ± 0.920 |
| GLM SSD, 8K | 0.854 ± 0.030 → 0.870 ± 0.022 | 0.844 ± 0.021 → 0.880 ± 0.014 |

| Workload | Total inference, seconds | Process wall, seconds | Paired inference-time changes, % |
| --- | ---: | ---: | --- |
| DeepSeek, 32K | 65.234 ± 7.255 → 62.027 ± 3.402 | 67.514 ± 7.323 → 64.330 ± 3.401 | +0.03, -2.80, +1.30, -8.64, -12.10 |
| DeepSeek, 6K | 34.516 ± 0.700 → 35.764 ± 3.735 | 36.698 ± 0.659 → 37.983 ± 3.730 | +0.50, +19.73, -1.72, -0.81, -0.13 |
| Qwen resident, 2K | 7.700 ± 0.169 → 7.374 ± 0.134 | 8.909 ± 0.301 → 8.782 ± 0.178 | -6.18, -3.00, -1.40, -3.96, -6.50 |
| GLM SSD, 8K | 102.718 ± 1.762 → 99.183 ± 1.068 | 103.370 ± 1.768 → 99.942 ± 1.176 | -0.80, -4.70, -5.49, -2.06, -4.04 |

Qwen's steady decode improves in every pair, by 5.3% in the initial-window mean
and 7.5% after append. Its total inference time falls 4.2%, also in every pair;
prefill is effectively unchanged at this precision. GLM fresh prefill takes
4.1% less time and total inference 3.4% less, each improving in every pair.
GLM remains slow in this mapping configuration; this catch-up does not reopen
the separately deferred investigation of its absolute performance.

DeepSeek does not establish a consistent speed change. The 32K inference mean
falls 4.9%, but the final baseline append takes 16.48 seconds against 7.91 for
its candidate; that pair supplies much of the apparent improvement. At 6K the
mean instead rises 3.6%, dominated by the second candidate append at 14.77
seconds against 8.18 for baseline. The other four 6K paired inference changes
range from −1.72% to +0.50%. Both steady-decode means move by less than 1%.
Neither noisy total is a reason to claim a general speedup or reject the port
as a persistent slowdown.

### Numerical observations

Separate untimed pairs compare every vocabulary logit at 32 teacher positions
before and after the append: one prefill-control position and 31 decoder
positions at each frontier. DeepSeek 32K, Qwen and GLM have exact arrays at all
64 positions apiece: delta-NLL, KL and maximum logit difference are zero.
DeepSeek 6K with the approximate resident-routing preset gives:

| Frontier | Mean delta-NLL, nats/token | Mean KL, nats | Max KL | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| 6144 | +0.008192 | 0.000982 | 0.022830 | 32/32 |
| 6656 | +0.029113 | 0.004345 | 0.042166 | 32/32 |

The pooled mean delta-NLL is +0.018653, or about 1.9% higher teacher perplexity.
It is not a task failure percentage. Repeating the numerical pair with
`--exact-routing` disables both prefill and decode substitutions and their
native-weight override. All 64 arrays then match exactly, including with the
packed-cache and other fast-preset settings retained. In this check, the drift
depends on the interaction with approximate resident routing. Pruning can change
which experts remain cached; individual substitution choices were not traced.
The exact unpruned/pruned state oracle above remains a separate check from
this approximate fast-preset comparison. The first attribution attempt retained
the native-weight override with zero gaps and was correctly rejected; its failed
log is retained, and the corrected pair is in `deepseek-exact-quality-6144-rerun`.

Raw results are under `.workspaces/upstream-b3f50180/measurements/` in the
`deepseek-timing-32k`, `deepseek-timing-6144`, `qwen-timing-2048` and
`glm-timing-8192` folders; matching `quality` folders hold full-vocabulary
arrays and comparisons. Logs, failed attempts and guard records are retained
alongside them and remain ignored. No build or competing model job ran during
final timing. Both timing guards completed normally: every pressure sample
was level 1, swap ranged from 26.854 to 26.865 GiB, and no adverse sample or
guard interruption occurred. The numerical runs are not comparative performance
samples.

For example, from the candidate checkout:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/catchup-32k-guard -- \
  nu speed-bench/experiments/2026-11-08-upstream-catchup/run.nu \
  .workspaces/catchup-32k /path/to/baseline /path/to/candidate --frontier 32768
```

Use `--stage quality` for untimed numerical diagnostics and add `--exact-routing`
to disable DeepSeek resident substitutions; use `--model qwen
--frontier 2048` for resident Q4 or `--model glm --frontier 8192` for GLM Flash
SSD streaming. DeepSeek uses the existing fast-preset compute and routing
settings on both sides. The script is the complete settings record.
