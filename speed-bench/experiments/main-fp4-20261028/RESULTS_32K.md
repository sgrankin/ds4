# Main FP4: five-pair 32K result

The favorable 32,768-token case runs at ctx=40,960, chunk=8192, 48 GiB configured
expert cache and high reasoning, using the same frozen `aee06ac9` binary and
shaders as the [8K campaign](RESULTS_8K.md). Only physical main storage changes:
A=F32 control, B=FP4 packed. Packed index/SIMD, normal routed-compute tile and
expert routing remain fixed. The unchanged pressure guard exits 0; every fresh
sample is retained in AB/BA/AB/BA/AB order.

**A small, consistent prefill cost buys much smaller state.** Answer prefill
plus normal generation averages **47.600→48.540 s (+1.976%)**, slower in every
pair. Paired increases are 0.650–3.613%, median 2.368%, sample SD 1.221 percentage
points. The order groups both increase: AB +2.213%, BA +1.622%. Answer prefill
is +2.257%; generation is −1.234%; all measured phases are +1.885%.
This 32K result is not speed-neutral. It is also not an automatic rejection:
main owners save 171.882 MiB at the allocated context, canonical prompt state
saves 137.500 MiB, and observed outputs/teacher logits remain exact. The matching
encoder profile locates the added work in indexed batch attention below.
**Decision: enable main FP4 in the opt-in DeepSeek fast presets**, accepting
this measured capacity/latency tradeoff. Ordinary engine defaults stay unchanged.

Initialization is excluded; both prefills have fresh KV, with experts warmed
by the forced teacher branch before answer prefill. OS file caches are not
flushed. This direct inference workload does not measure SSD cache-hit latency
or use saved memory to enlarge the fixed expert cache. The recorded manifest,
raw outputs and local `summary.json` stay under
`.workspaces/main-fp4-20261028/full-32k/`.

## Five primary samples

Seconds, including answer prefill and normal-reasoning generation:

| Arm | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 |
|---|---:|---:|---:|---:|---:|
| control | 48.109830 | 47.506767 | 47.271862 | 47.552341 | 47.559122 |
| packed | 49.253653 | 48.631533 | 48.979724 | 47.969005 | 47.868280 |

## All phase distributions

| Phase | Arm | Mean (s) | Median (s) | Range (s) | Sample SD (s) |
|---|---|---:|---:|---:|---:|
| Teacher prefill | control | 43.564 | 43.895 | 41.856–44.424 | 0.991 |
| Teacher prefill | packed | 44.354 | 44.394 | 42.986–45.720 | 0.973 |
| Forced teacher continuation | control | 1.407 | 1.401 | 1.366–1.488 | 0.050 |
| Forced teacher continuation | packed | 1.421 | 1.394 | 1.360–1.548 | 0.076 |
| Answer prefill | control | 43.769 | 43.696 | 43.517–44.199 | 0.257 |
| Answer prefill | packed | 44.757 | 44.819 | 44.065–45.539 | 0.635 |
| Normal-reasoning generation | control | 3.831 | 3.833 | 3.755–3.911 | 0.058 |
| Normal-reasoning generation | packed | 3.784 | 3.796 | 3.715–3.813 | 0.039 |
| Answer prefill + generation | control | 47.600 | 47.552 | 47.272–48.110 | 0.308 |
| Answer prefill + generation | packed | 48.540 | 48.632 | 47.868–49.254 | 0.610 |
| Teacher prefill + continuation | control | 44.971 | 45.260 | 43.345–45.826 | 0.951 |
| Teacher prefill + continuation | packed | 45.775 | 45.755 | 44.418–47.114 | 0.970 |
| All measured phases | control | 92.571 | 92.820 | 91.455–93.332 | 0.717 |
| All measured phases | packed | 94.315 | 93.724 | 93.412–95.746 | 1.016 |

Positive percentages mean packed slower. Order groups have three AB and two
BA pairs and are descriptive, not a precise estimate of a causal order effect.

| Phase | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean-time change | AB mean paired | BA mean paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher prefill | +2.700% | +2.916% | +1.823% | +1.023% | +0.633% | +1.813% | +1.719% | +1.970% |
| Forced teacher continuation | -3.814% | -0.499% | +13.167% | -3.644% | +0.415% | +1.004% | +3.256% | -2.071% |
| Answer prefill | +3.032% | +2.622% | +3.831% | +1.097% | +0.702% | +2.257% | +2.521% | +1.860% |
| Normal-reasoning generation | -5.016% | -0.532% | +1.084% | -1.626% | +0.058% | -1.234% | -1.292% | -1.079% |
| Answer prefill + generation | +2.378% | +2.368% | +3.613% | +0.876% | +0.650% | +1.976% | +2.213% | +1.622% |
| Teacher prefill + continuation | +2.476% | +2.812% | +2.167% | +0.878% | +0.626% | +1.788% | +1.757% | +1.845% |
| All measured phases | +2.424% | +2.586% | +2.907% | +0.877% | +0.638% | +1.885% | +1.990% | +1.731% |

The prefill direction is consistent in both independent branches: teacher
prefill and answer prefill are slower in all five pairs. The shorter normal
generation branch varies in sign and is modestly faster on average. The
diagnostic profile below locates added work in the packed attention consumer;
byte savings alone do not establish faster attention.

## Exactness and practical answers

All 10 answers complete correctly as `CODE=CEDAR-4826`. Both arms generate the
same 42 token IDs, including normal reasoning, in every process. The eight
forced teacher positions per process are expected-answer tokens plus EOS at
the reasoning frontier. All 40 matched rows have byte-identical full-vocabulary
logits, ΔNLL=0, KL=0 and 0 argmax flips. Repeated controls also match the opening
control exactly. Mean teacher NLL is 1.345159935 nats/token in both arms.
These are narrow answer-position probes, not a broad reasoning-quality score.

## Actual allocation and canonical payload

Every allocation/payload counter is identical across five processes per arm
(SD 0). The shape uses context capacity 40,960 but a live prefix of 32,768 tokens.

| Quantity | Control bytes | Packed bytes | Saved bytes |
|---|---:|---:|---:|
| Actual main owners | 209,723,392 | 29,492,352 | 180,231,040 |
| Actual packed index owners | 6,963,472 | 6,963,472 | 0 |
| Actual raw rings | 10,485,760 | 10,485,760 | 0 |
| Actual total KV | 227,172,624 | 46,941,584 | 180,231,040 |
| Planned main + index | 216,686,864 | 36,455,824 | 180,231,040 |
| Engine-bound KV + buffer plan | 8,515,680,184 | 8,335,449,144 | 180,231,040 |
| Canonical payload at 32,768 tokens | 184,476,724 | 40,297,524 | 144,179,200 |

Actual total KV is 216.649→44.767 MiB (−79.34%). Main
owners alone shrink 85.9375%. Canonical payload is 175.931→38.431 MiB
(−78.16%); private suffix-capture size reports the same values in this
fresh-state probe. These are actual tensor allocations and serializer byte
counts, not SSD traffic or physical filesystem block allocation.

Rounded static context buffers are 8121.19→7949.30 MiB and runtime tensor peaks
7578.16→7406.28 MiB, with scratch 530.61 MiB unchanged. The whole startup plan,
which additionally includes weights/expert cache, is 65.30→65.13 GiB.
Expert live bytes/hits/misses/pread remain equal across all processes:
40.88 GiB, 14,023 hits, 6,329 misses and 17.81 GiB read. These I/O counters span
both prefills and the teacher/answer branches, not one timed phase.
There is no measured expert-I/O reduction or extra-resident-expert benefit.

## Matching diagnostic attribution

The 32K favorable profile uses the same frozen binary, shape, reasoning and
expert budget. One process per layout records native encoder timestamps,
pipeline creation, prefill wall stages and expert I/O. This is attribution
evidence, not another latency estimate or additional performance samples.
GPU totals span both prefills plus the forced and normal continuations.

| Kernel/work | Calls control→packed | Diagnostic GPU seconds control→packed |
|---|---:|---:|
| Indexed mixed batch attention, heads16 dual | 188→188 | 17.615→19.524 |
| Main FP4 publication | 0→5005 | 0→0.032220 |
| Main FP4 selected F32 gather | 0→1862 | 0→0.024766 |
| Quantization, all modes | 7639→2634 | 0.202324→0.162030 |
| Tensor-copy blits, all uses | 21147→16141 | 0.936266→0.701556 |

The indexed attention consumer adds 1.909 diagnostic GPU seconds at an unchanged
dispatch count. New publication plus selected gather total only 56.986 ms,
while explicit quantization/copy work decreases. The source's format 2 load
path decodes FP4 nibbles and E4M3 scales inside that attention consumer.
This identifies the useful optimization target; the trace does not establish
a specific occupancy, cache or execution-unit bottleneck.

Native prefill drain time rises 70.761→72.495 s, while encode wall falls
1.131→1.088 s. Pipeline creation totals 21.800→62.592 ms; it is much smaller
than the consumer delta. Expert selected/load counts and 17.81 GiB read are
unchanged. Thus added expert SSD work or standalone packing launches do not
explain the observed concentration of work. These phase counters overlap and
are not additive CPU/GPU utilization measures.

Keep the measured ~2% 32K task cost explicit when selecting the fast preset.
The current capacity/persistence benefit is useful without claiming a speedup;
optimizing the inline attention decoder is a separate bounded experiment.
The default FP4 scalar path keeps F32 scratch; its half-scratch variant remains
explicit. Raw evidence and `summary.json` stay local at
`.workspaces/main-fp4-20261028/profile-32k/`.

The source presets enable main FP4 by default unless only F16 is explicitly
requested. Explicit `DS4_V41_COMPRESSED_KV_FP4=0` disables it; explicit main 1
wins over F16. Fake-binary wrapper checks cover default 1, explicit 0, F16-only
main 0 and both 1/main 1. Installed binaries are unchanged until final installation.

## Reproduce and profile

```nu
python3 speed-bench/ideas/pressure_run.py GUARD -- nu speed-bench/experiments/main-fp4-20261028/run.nu FROZEN/bin/decoder-replay-bench FRESH_OUTPUT --tokens 32768 --case favorable
python3 speed-bench/experiments/main-fp4-20261028/analyze.py FRESH_OUTPUT
python3 speed-bench/ideas/pressure_run.py PROFILE_GUARD -- nu speed-bench/experiments/main-fp4-20261028/profile.nu FROZEN/bin/decoder-replay-bench FRESH_PROFILE
python3 speed-bench/experiments/main-fp4-20261028/analyze-profile.py FRESH_PROFILE
```

The [profile driver](profile.nu) holds the same shape/settings and enables
native encoder timestamps, pipeline creation, prefill wall stages and expert
I/O diagnostics. It uses one fresh process per layout to attribute work.
Encoder pass boundaries perturb scheduling; these profiled wall times are not
added to the five-sample performance estimates.
