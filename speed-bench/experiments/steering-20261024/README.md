# Residual steering: implementation and measurements

## Current residual results

The residual-steering implementation is complete. Both public modes work on
the installed Flint, DS4.1 Flash 1017, Qwen3.8 Flash Next and GLM 5.3 Flash
checkpoints. Project normalizes each nonzero bank row and
uses per-layer strength; add preserves raw magnitude and divides strength by
the number of nonzero rows. The default edit is the post-FFN/full-block residual,
with attention residual editing independently selectable. See the
[bank and CLI conventions](../../../dir-steering/README.md).

Public checks pass for both modes on all four models: exact zero-scale logits,
independent session scales and future-only live changes. DS41/Flint batch replay
has zero logit gap. Qwen/GLM mixed batch versus scalar has zero gap; their MTP
replay's maximum absolute logit gap is **2.07e-4**. Final Flint checks cover three
full eight-token packet cycles in each mode, all exact against scalar replay,
plus live scales and batching. The shared GPU oracle passes **342 cases plus
18 norm subcases with zero drift**. Flint's final three profile fixtures pass
on the available M5 Max; physical M4/M5 Pro hardware remains unqualified. All host and
script checks pass, including the builder contract. Localhost recovery passed
with the required sandbox escalation. Final Flint evidence is local in
`hoist-smoke.log` and `final-flint-session.log` under the experiment workspace.

Fresh post-block residual banks are local at
`.workspaces/steering-20261024/residual-directions/{flint,ds41,qwen,glm}.{json,f32}`.
Flint/DS41 use four succinct/verbose prompt pairs; Qwen/GLM use two. Every trunk
layer is selected. They were extracted with `--mode add` to retain raw mean
differences, then used by both runtime modes; project normalizes on loading.
These small banks qualify extraction and runtime behavior, not general style
control or task-quality improvement.

### Fixed-input latency

Reference executables are qualified `main` **2e1502a3**, frozen under
`.workspaces/steering-20261024/baseline/bin`. Final binaries are in
`.workspaces/steering-20261024/residual-final/bin`. The final Flint additive
screen uses those binaries; the other rows use the preserved
`.workspaces/steering-20261024/residual/bin` candidate, before the Flint-only
additive hoist. Each screen runs baseline off, candidate off, loaded zero and
active, then reverses that order: **two
observations per state**, with one model job under the pressure guard.
The table compares active against candidate off, using mean time for equal
input counts. Positive percentages mean slower.

| Model / mode | Tokens: prefill + fixed decode | FFN / attention strength | Prefill time change | Decode time change |
| --- | ---: | ---: | ---: | ---: |
| Flint / project | 256 + 256 | 0.1 / 0 | +0.41% | +2.43% |
| DS41 / project | 256 + 64 | 0.1 / 0 | +4.73% | +5.12% |
| DS41 / add | 256 + 64 | 1 / 0 | -2.13% | -0.41% |
| Flint / add, final | 256 + 256 | 1 / 0 | +7.42% | +1.52% |

Actual off/active observations in seconds retain the variation:

| Model / mode / phase | Candidate off (two samples) | Active (two samples) |
| --- | --- | --- |
| Flint project / prefill | 0.300713, 0.293181 | 0.300434, 0.295885 |
| Flint project / decode | 1.138436, 1.160261 | 1.151286, 1.203347 |
| DS41 project / prefill | 15.147929, 15.657492 | 16.710183, 15.552855 |
| DS41 project / decode | 4.007514, 4.027690 | 4.260985, 4.185742 |
| DS41 add / prefill | 16.431322, 15.910503 | 15.871048, 15.782984 |
| DS41 add / decode | 4.058339, 4.081633 | 4.058339, 4.048071 |
| Flint add, final / prefill | 0.344693, 0.344874 | 0.371429, 0.369323 |
| Flint add, final / decode | 1.189094, 1.123103 | 1.180866, 1.166446 |

Flint uses chunk 4096; DS41 uses chunk 2048 and a 32GB streaming expert cache,
with replay/resident approximations disabled. Both allocate context 8192 and
teacher-force the existing `zig-metal-f32-hour26-20261022/prompt.txt` corpus.
The active setting can change expert choices and SSD/cache work. Two samples
are a screen, not a universal speed promise or an established DS41 additive
speedup. Candidate-off/baseline-off and loaded-zero controls remain in each
`comparison.json`; off/zero frontier hashes match exactly. Initial cold-prefill
variation is visible in the controls.

The first residual Flint additive point (R0) was **+2.50% prefill / +5.47%
decode**. One bounded optimization hoists its constant addition in the exporter
and three resident libraries. The final point is **+7.42% prefill / +1.52%
decode**; both figures come from the same final screen. Prefill source was
unchanged, yet this screen measured about 26ms extra prefill. Its off decode
observations span about 6%, so a causal optimization gain is not established.
The small hoist is retained for simpler constant handling, correct math and
packet replay, and the modest final decode cost. R0 is historical; no further
optimization is included.

New-hook append/context latency and Qwen/GLM latency are not measured here.
Retained V4 Pro/full GLM source is adapted, but their checkpoints have not been
qualified. Behavioral strength/layer sweeps and held-out task quality remain
necessary before recommending a bank. A few percent overhead is acceptable;
keep speed and useful behavioral effects visible together.

Concise results are kept here; commands, individual timings and full
logits remain ignored under `.workspaces/steering-20261024/residual-*-speed/`.
Use the driver below to reproduce a screen or recompute its summary.

## Historical module-output checkpoint, 18:20 UTC

These results used the earlier module-output hooks and projection convention;
they do not qualify the current residual banks. Eight-pair style banks were
captured for DS41/Flint from the bundled succinct/verbose lists. The shared GPU
oracle passed 162 zero-drift cases; three Flint profiles passed math checks on
the M5 Max. Public session checks passed for Flint, DS41 and Qwen, including
zero identity, live scales, batching and Qwen MTP. Full GLM had primitive/shape
coverage, without full-checkpoint inference.

| Intermediate point | Prefill time versus candidate off | Fixed decode time versus candidate off |
| --- | ---: | ---: |
| Flint R0, 256+256 | +2.33% | +5.82% |
| Flint R0, 2047 then append2048 | -0.49% initial / +0.90% append | no decode |
| Flint R0, 4095+256 | +2.70% | +1.69% |
| Flint R2, 256+256 | -2.92% | +2.59% |
| DS41 first screen, 256+32 | -0.10% | +3.73% |

R1 separates steering scratch from RMS and removes one barrier per projection.
It also exposed a 268ms cold PSO creation inside first decode. R2 reuses the
FFN accumulation array and prepares active PSOs during vector loading. Its
zero/off frontiers remained identical; the decode cost failed the former gate.
R3 derived auxiliary weight rows at startup and ran their scalar products in
otherwise idle resident groups. It was not adopted: its decode screen was
+7.86% slower than candidate off. Against exact R2 at 32 teacher-forced decoder
positions, mean KL was 0.3633 nats, mean delta-NLL -0.1202 nats and top1 agreement
87.5%; prefill remained exact. Zero-scale R2/R3 decoder logits were exact on a
16-position control. The lower NLL on this small continuation does not establish
better task quality. Raw evidence is local in `flint-r3-short/`,
`flint-derived-quality.json` and `flint-derived-zero.json`.

DS41's single-group active HC is the demonstrated bottleneck: fixed-input
ABBA means are 88.30us active versus 17.55us unsteered expand+BF16 for one row.
At 256 rows, fusion was 136.07 versus 155.90us (-12.72%). This motivated
scalar output tiling. Both actual model timing and isolated math cost are necessary;
different expert choices can also change SSD traffic.

Raw commands, per-run values and logits live in the ignored workspace. Recover
a driver's existing summary without rerunning inference with
`nu speed-bench/experiments/steering-20261024/measure.nu summarize RUN_DIRECTORY`.

The original zero-slowdown gate was superseded by the user's acceptance of a
few percent overhead. Keep fixed tokens so answer length cannot hide slower
inference, and measure whether the behavioral effect justifies the cost.
Use the existing pressure guard and run one model job at a time. Pause unrelated
work for final latency measurements.

## Fixed workloads

`measure.nu` reuses `ds4-bench` and its existing text fixture, CSV timing and full
FP32 frontier-logit dumps. It adds no new timing harness or binary format.

| Workload | Prefill | Decode |
| --- | --- | --- |
| `short` | Fresh 256 tokens | 32 fixed continuation tokens |
| `append` | Fresh 2047, then append 2048 to 4095 | None |
| `context` | Fresh 4095 tokens | 32 fixed continuation tokens |

The append case uses the same live session without intervening decoding or
snapshot restoration. The two decode cases teacher-force the same corpus
continuation; a premature stop is a failure. Prefill and decode timers exclude
model loading and frontier-logit output. Process wall time also contains those
costs and is recorded separately; it is not a startup-only measurement.

Each workload runs baseline-off, candidate-off, candidate-loaded-zero and
candidate-active, then reverses that order. Repeat only where variation needs
resolving. Positive time changes in `comparison.json` mean slower. Keep the
individual times; a single mirrored screen does not establish statistical
equivalence. Disabled and loaded-zero frontier floats should match exactly.
Active differences are expected and measured, rather than rejected for being
different.

```bash
nu speed-bench/experiments/steering-20261024/measure.nu qwen \
  dir-steering/out/qwen-residual-project.f32 .workspaces/steering-20261024/candidate/bin \
  .workspaces/steering-20261024/qwen-ffn-01 --mode project --ffn 0.1 --dry-run
# Remove --dry-run only when this family owns the heavy-job slot.
```

Use `--workload short|append|context`, `--tokens`, `--repeat`, `--attn` and
`--ffn` to select a bounded experiment. `--model` overrides the installed model.
`--mode project|add` selects the public runtime mode and is recorded in every
run. Candidate commands pass `--dir-steering-mode`; the frozen baseline
receives no new flags. The old Flint-only derived/additive environment arms
are historical and have been removed from this driver.
Do not alter scales and kernel implementation together when diagnosing cost.
All raw logs, logits and result JSON remain in the supplied ignored workspace.
Recompute `comparison.json` without inference with
`nu speed-bench/experiments/steering-20261024/measure.nu summarize WORKSPACE`.

The driver preserves the preceding qualification's safe settings: DS4.1 32GB
expert cache, GLM 24GB cache with static decode mapping disabled, Qwen without
residency/warmup, and native Flint. It clears inherited DS4/PT preset variables.
GLM receives no explicit prefill chunk. The baseline receives no new flags.

## Correctness and behavioral boundary

Current residual banks must be extracted at the new capture points. Earlier
Qwen refusal and DS41/Flint style banks were learned at other points and are
historical. Required native shapes are DS4.1 **40×5120**, Flint **49×2048**,
GLM Flash **45×4096** and Qwen **48×2560**.

Project normalizes every nonzero row at loading and keeps strength per layer.
Add preserves raw mean-difference magnitude and divides strength by the number
of nonzero rows. Zero rows select no layer. Sweep both strength and layer
selection; record the actual convention alongside quality and timing.

Synthetic banks can establish arithmetic correctness and zero identity. They
cannot establish useful semantic control. Capture new residual banks, then
grade held-out prompts and actual task behavior before choosing defaults.
The builder supports all four native profiles and retains useful workspaces
outside `/tmp`. It uses prefill-only extraction, averages HC residual streams,
and writes hidden-width rows. See the [capture recipes](../../../dir-steering/README.md)
for the exact modes, layer masks and qualification boundaries.

Fixed tokens hold input constant, but active steering can change expert choices
and SSD misses. Pair these end-to-end measurements with the model agents' fused
kernel fixtures at identical shapes/inputs. Report projection overhead and
changed route/cache cost separately. Scalar teacher-forced decode does not
qualify Flint's packet kernels or Qwen/GLM MTP acceptance: use the existing
packet/MTP correctness fixtures for those paths. Test loaded-zero, nonzero,
live-scale invalidation and cache exclusions before enabling combinations.

## Decoder distribution probe

Choose a fixed reference/candidate pair: a kernel change at the same mode and
strength, or an explicitly different steering setting whose quality tradeoff
you want to quantify. Keep model, corpus, frontier and forced token IDs fixed.
The earlier R3 derived-row probe used an exact R2 control in the same binary;
that experiment concerned the historical module-output hook.

Use the existing `ds4-bench --teacher-forced-decode` workload with
`--dump-decode-logits-dir DIR`. Create that directory first. Run once with each
setting, under the pressure guard and in separate output directories. For
example, a 256-token frontier with `--gen-tokens 33` records the prefill control
plus 32 decoder distributions. Repeat at a longer frontier if the first probe
justifies it. Keep steering settings identical when isolating a kernel change.

The diagnostic writes `decode_000256.json` and raw little-endian FP32 files.
Each row is the distribution **before** the corresponding forced evaluation;
its target is that forced token. Step zero comes from prefill. Steps 1 onward
observe the decoder after the preceding fixed inputs. Prompt IDs, targets,
positions and vocabulary size are recorded so the comparison checks that its
inputs match. Decode timing columns are zero in this diagnostic mode; use the
ordinary benchmark for latency.

```bash
python3 -B speed-bench/experiments/steering-20261024/compare_decode.py \
  .workspaces/steering-20261024/quality-reference/decode_000256.json \
  .workspaces/steering-20261024/quality-candidate/decode_000256.json \
  .workspaces/steering-20261024/decoder-comparison
```

The NumPy comparison emits CSV rows and separate JSON summaries for prefill
and actual decoder positions. Positive delta-NLL means the candidate assigned
less probability to the natural continuation. Forward KL uses the chosen
reference distribution. These small probes quantify drift; they do not alone
establish task quality or useful behavioral steering.

For real scalar-versus-eight-token greedy packet checks, the ordinary `ds4`
CLI already uses packets when available. Set `DS4_MTP_SPEC_DISABLE=1` to force
scalar evaluation; leave the variable **unset** for packets (`=0` also disables
them because this switch tests presence). Run identical active steering,
`--temp 0`, prompt and `-n 64`, then compare outputs and verify both actually
generated 64 tokens. `--dump-logprobs` is a separate scalar diagnostic and
cannot qualify packet execution.
