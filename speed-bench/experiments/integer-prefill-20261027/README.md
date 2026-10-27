# Temporary INT8 routed prefill experiment

**Neither prototype is adopted: both are slower in all five pairs.** V1's three answer-prefill-plus-generation
tasks total73.750s in the control versus99.628s with INT8 (+35.09%). Answer
prefill alone rises34.06%. All30 answers remain correct; this implementation
has no measured speed benefit. V2 reuses activation quantization, but its
separate favorable-only comparison still adds18.49% prefill and34.70% whole
task time, with all10 answers correct. The decision is to retain reports and
reproduction patches rather than runtime flags or kernels. The four runtime/
fixture/build files are restored byte for byte to base `9fdb8bf27dd7`;
the experiment's flag and kernels are absent from the final runtime.
See the [not-adopted recipe](../../../recipes/not-adopted/integer-prefill.md).

Prepared 2026-10-27 at20:59UTC; the session stopped after the probe build and
resumed at23:07UTC. The ten-process campaign ran23:12–23:39UTC. Runtime revision
`f7c93c0c3993`, Apple M5 Max/128GiB. Installed executables and main are unchanged.
Raw results stay ignored under `.workspaces/integer-prefill-20261027/`.

The experiment retains the installed1017 IQ2_XXS/Q2_K model bytes on SSD and in
the expert cache. It tests temporary integer arithmetic in routed prefill;
model-cache expansion or model-weight repacking is outside this comparison.
Only gate/up uses the new kernel: exact signed IQ2 grid values become bounded
INT8 tiles, activations are dynamically quantized in groups of32, and native
MPP INT8xINT8 produces INT32 partials before per-group F32 rescaling. The Q2_K
down projection is unchanged. Activation quantization and different rounding/
accumulation introduce numerical changes; this is not lossless arithmetic.
The native integer-matmul reference is
[Draw Things' kernel](https://github.com/liuliu/ccv/blob/unstable/lib/nnc/mfa/kernels/NAInt8MatMulKernel.cpp).
No ConvRot or on-disk requantization is applied.

## Five-pair V1 measurements

A=control half-tile, B=INT8. Each number sums the three cases' independent
answer-prefill-plus-generation time. All five pairs are retained.

| Pair | Order | Control, s | INT8, s | Time change |
| --- | --- | ---: | ---: | ---: |
|1 |AB |73.461 |101.501 |+38.17% |
|2 |BA |76.322 |101.024 |+32.36% |
|3 |AB |73.536 |99.375 |+35.14% |
|4 |BA |72.881 |98.667 |+35.38% |
|5 |AB |72.549 |97.575 |+34.50% |
|Mean | |**73.750** |**99.628** |**+35.09%** |

Median paired change is+35.14%; control spans72.549–76.322s and INT8 spans
97.575–101.501s. Every pair is slower. Aggregate changes are ratios of means;
the analyzer also retains paired percentages and AB/BA strata.

| Case | Mean control task, s | Mean INT8 task, s | Task change | Answer-prefill change |
| --- | ---: | ---: | ---: | ---: |
|Ordinary |25.739 |31.469 |+22.26% |+29.71% |
|Favorable |24.380 |35.786 |+46.79% |+32.83% |
|Changed fact |23.632 |32.373 |+36.99% |+39.19% |
|Sum |73.750 |99.628 |+35.09% |+34.06% |

The favorable continuation also grows31→98 tokens. Decode time therefore
mixes changed output length with per-token execution; it cannot be attributed
solely to arithmetic cost. The prefill penalty is present independently.

## V1 profiling

After the five-pair slowdown, one diagnostic profile per arm ran the same8K
favorable workload, with both teacher and answer prefills and normal reasoning.
GPU timestamp boundaries change scheduling. These captures explain anatomy;
they are not another performance qualification or an inferred speedup.

| Captured GPU work | Calls per arm | Control, s | V1 INT8, s |
| --- | ---: | ---: | ---: |
|Packed IQ2 gate/up |220 |3.618018 |17.582647 |
|Unchanged packed Q2 down |110 |1.804057 |1.891770 |
|All captured GPU work | |23.582855 |42.753139 |

The changed IQ2 kernels account for4.86 times as much captured GPU time; the
Q2 kernel's captured cost changes much less. Total GPU work includes both
prefills and different generated-trace lengths, so it is not the primary
answer-prefill-plus-generation metric. Native host-stage totals are:

| Stage, both prefills | Control, s | V1 INT8, s |
| --- | ---: | ---: |
|Map |7.194426 |5.326272 |
|Encode wall |0.858148 |0.795235 |
|Drain/wait wall |24.750367 |39.510956 |

The captures point to GPU kernel cost rather than greater host encode time.
They do not isolate every cause. V1 repeats activation quantization for each
output tile and for gate/up separately. It also converts and rescales INT32
partials after every K32 group, rather than accumulating one large integer
product. Repeated quantization and per-group rescaling are plausible causes;
their separate costs have not yet been measured. The source verifies INT8
TensorOps inputs and INT32 accumulation, not a hardware-counter proof of the
specific physical neural-accelerator execution path.

## Five-pair V2 activation-reuse follow-up

V2 (`f6616a4a2d13`) moves activation quantization into the existing RHS packing
step, once per packed activation group, then lets gate/up consume those INT8
codes and scales. Expert-cache bytes and per-K32 weight scaling stay unchanged.
The first V2 shader compile rejected a const INT8 TensorOps element type; the
retained retry uses an unqualified view that the kernel only reads. Builds and
the same finite/tail/zero/sentinel fixture pass with unchanged RMS results.
The fresh five-pair favorable-only campaign started23:45UTC and completed.

| Pair | Order | Control task, s | V2 INT8 task, s | Task change | Prefill change |
| --- | --- | ---: | ---: | ---: | ---: |
|1 |AB |22.123 |29.722 |+34.35% |+17.75% |
|2 |BA |22.703 |31.076 |+36.88% |+20.39% |
|3 |AB |22.178 |30.331 |+36.76% |+21.53% |
|4 |BA |22.607 |30.159 |+33.41% |+17.00% |
|5 |AB |22.663 |29.940 |+32.11% |+15.82% |
|Mean | |**22.455** |**30.246** |**+34.70%** |**+18.49%** |

Mean answer-prefill time is19.331089s control versus22.905462s INT8; median
paired prefill change is+17.75%. Median paired task change is+34.35%. All five
pairs regress in both phases. Task times span22.123–22.703s in the control and
29.722–31.076s in INT8; paired task changes span+32.11–36.88%, and prefill
changes+15.82–21.53%.

All10 answers are correct and stop normally. Control generates31 tokens and
INT8 generates98, repeating V1's favorable continuation. Candidate teacher
delta-NLL is−0.011413nats/token and mean KL0.005585, with no teacher argmax
flips. These are the same measured favorable metrics as V1, not a BF16 quality
comparison. In all five pairs, V1/V2 favorable teacher-logit bytes and generated
IDs match exactly for both arms; this is separate from equal aggregate metrics.

All five pairs record total transient scratch267.59→162.59MiB, a105MiB
reduction. The live expert cache remains40.88GiB in every process, with no automatic growth from freed
scratch. Application miss-read bytes are15.23→32.90GiB, and mean `pread` time
0.674→1.428s. These whole-process counters include the longer continuation and
different routes and remain distinct from physical SSD traffic.

V1's favorable case follows ordinary inside a three-case process; V2 runs
favorable alone. They have separate controls and cache histories. Therefore
their absolute timings do not establish a direct V1→V2 speedup percentage.
V2's own five-pair penalty supports the same **do not adopt** decision.

Profile evidence: `profile-v1-summary.json` and `profile-v1/`. V2's frozen probe
SHA-256 is `a0d54d9f87d0c3adf9b61d72e288cae6667a1dda8ea78ea8beb408bc583cfe56`;
shader SHA-256 is `1b851d6448c3d6cc05e268412500b13f2429e94e49f2d0da3e211a1cadfb37cd`.
These are retained in `candidate-v2/frozen.json`; V2 data is in `samples-v2/`.
The V2 campaign guard records487 normal samples with zero swap growth.

The same-workload V2 diagnostic profile is complete:

| Captured GPU work | Calls per arm | Control, s | V2 INT8, s |
| --- | ---: | ---: | ---: |
|Packed IQ2 gate/up |220 |3.640645 |11.018427 |
|Unchanged packed Q2 down |110 |1.805080 |1.942174 |
|All captured GPU work | |23.681573 |36.206159 |

The captured IQ2 kernels still take3.03 times as long as their matched
control. Host map totals6.962008→6.241095s, encode wall0.800903→0.863999s,
and drain/wait24.739602→32.456320s. Integer GPU work remains the bottleneck in
this diagnostic. Total GPU time also includes the longer generated trace and
is not the primary task timing. These are one profiled process per arm, not
additional unprofiled benchmark samples or proof of a specific hardware path.
The profile guard records101 normal samples and0.63MiB maximum swap growth.
Evidence: `profile-v2-summary.json` and `profile-v2/`.

## Quality and capacity

All15 control and15 INT8 answers are correct and stop normally. The reference
is the quantized Q2 model, not BF16. The following values repeat across all five
candidate processes. All110 candidate teacher rows have no argmax flips, but
their logits differ from the control.

| Case | Teacher rows/run | Mean delta-NLL, nats/token | Mean KL | Generated tokens, control→INT8 |
| --- | ---: | ---: | ---: | ---: |
|Ordinary |7 |−0.112489 |0.006535 |87→86 |
|Favorable |8 |−0.011413 |0.005585 |31→98 |
|Changed fact |7 |−0.131388 |0.012552 |26→28 |

Negative delta-NLL means greater confidence in these forced target tokens,
not established general capability improvement. These short teachers start
at a reasoning frontier; actual generated traces retain normal reasoning.
This is a small fact-retrieval screen, not a coding or broad reasoning eval.

The model-free Zig GPU fixture passes rows512/513/1024, zero inputs, finite
outputs, tails and sentinels. Relative RMS error is0.5307–0.5344% on its nonzero
synthetic cases; zero inputs match exactly. This qualifies the bounded kernel,
not general model accuracy.

| Whole-process native counter | Control, all five | INT8, all five |
| --- | ---: | ---: |
|Live expert cache |40.88GiB |40.88GiB |
|Cache hits |50,690 |63,766 |
|Cache misses |14,198 |17,450 |
|Application miss-read bytes |56.97GiB |74.17GiB |
|Mean pread time |2.443s |3.263s |

Counts and rounded byte totals repeat in all five processes per arm. These
cover all six prefills, teachers and generated answers, not one case or only
primary latency. Changed continuations also change routing/cache activity.
Application `pread` bytes are not physical SSD traffic: OS file caching
intervenes. Persistent expert capacity is unchanged. All1,556 pressure samples
are normal (level1); maximum swap growth is2.44MiB.

## Configuration and reproduction

The final source retains [V1's reproduction patch](v1.patch) and
[V2's activation-reuse patch](activation-reuse.patch), rather than the runtime
experiment. Copy this experiment directory into an isolated checkout at base
`9fdb8bf27dd7`, apply `v1.patch`, and build the probe and fixture. For V2,
apply `activation-reuse.patch` after V1 before building. Neither patch changes
model files or their Q2 expert bytes. Both patches were checked in an isolated
base tree; the resulting V2 `moe.metal` matches the actually measured frozen
source byte for byte.

```nu
git apply speed-bench/experiments/integer-prefill-20261027/v1.patch
# For V2, also apply activation-reuse.patch before the build.
nu tools/zig-build decoder-replay-bench integer-moe-test -j3 --prefix .workspaces/integer-prefill-repro
mkdir .workspaces/integer-prefill-repro/shaders
cp src/metal/*.metal .workspaces/integer-prefill-repro/shaders/
```

| Fixed configuration | Value |
| --- | --- |
| Model | `gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` |
| Prompt/context/chunk |8,192 /16,384 /8,192 tokens |
| Expert cache |48GiB, native expert routing; approximate resident substitutions disabled by the existing probe |
| Index cache | Physical FP4 and SIMD enabled; candidate-only indexing off |
| Compute isolation | `DS4_METAL_DISABLE_V41_PACKED_M32N128=1` in both arms, isolating M64N32 |
| Changed setting | `DS4_METAL_ROUTED_MPP_I8=0/1` only |
| Reasoning/generation | Normal high reasoning, T0, maximum384new tokens |
| Samples | Five fresh processes per arm, AB/BA/AB/BA/AB, no discarded run |

`run.nu` defaults to the existing probe's `all` selector: ordinary, favorable and
changed-fact cases within each of ten processes. This isolates arithmetic at
M64N32; it does not compare V1 with the default8K M32N128 path. Single-case
campaigns are available with `--case favorable` or `--case ordinary`.

Each process performs the existing independent teacher and answer prefills.
Primary latency is `greedy_sync_ms + greedy_ms`: fresh answer KV after the
teacher has warmed expert/file caches, excluding engine initialization. First
prefill, answer prefill, teacher and decode are also retained separately.
`analyze.py` reports each selected case and their sum, all five paired changes,
means/medians/ranges, AB/BA strata and geometric ratios. Positive changes mean
INT8 is slower. It reuses the existing fixed-teacher comparison for ΔNLL/KL and
reports actual generated IDs, token counts, correct answers and stop status.
Output degradation is a measured tradeoff, not an automatic rejection.
Both arms also enable the existing expert timing summary. The analyzer retains
native hits, misses, live cache GiB, miss-read GiB and pread milliseconds for
each process and summarizes all five per arm. These counters cover the complete
process, including both prefills and all generation in every selected case;
they are not per-case or limited to the primary answer latency. GiB values
retain the native logger's rounding.

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/integer-prefill-repro/fixture-guard -- .workspaces/integer-prefill-repro/bin/test-integer-moe
python3 speed-bench/ideas/pressure_run.py .workspaces/integer-prefill-repro/samples-guard -- nu speed-bench/experiments/integer-prefill-20261027/run.nu .workspaces/integer-prefill-repro/bin/decoder-replay-bench .workspaces/integer-prefill-repro/samples
python3 speed-bench/experiments/integer-prefill-20261027/analyze.py .workspaces/integer-prefill-repro/samples
```

For the V2 standalone campaign, append `--case favorable` to `run.nu` and use
fresh output paths. The matching diagnostic scripts are retained too:

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/integer-prefill-repro/profile-guard -- nu speed-bench/experiments/integer-prefill-20261027/profile.nu .workspaces/integer-prefill-repro/bin/decoder-replay-bench .workspaces/integer-prefill-repro/profile
python3 speed-bench/experiments/integer-prefill-20261027/analyze-profile.py .workspaces/integer-prefill-repro/profile
```

The runner requires the binary's adjacent `PREFIX/shaders/*.metal` snapshot
and applies every source override. Both arms use the same executable/sources;
older packing runs do not count as any of these five samples. Preserve first
failures and all sample variation. After each process, the driver requires the
actual INT8 encode marker in candidate stderr and its absence in controls;
silently falling back to the old kernel fails the campaign. Any follow-up profiling should use this
same8K task/settings rather than an unrelated raw-prompt proxy.

Use new output/guard paths for a repeat. The frozen probe SHA-256 is
`a20eca41bec2ba8a49b961621ff8e734cedbbb3301048c6709a1df961190e4dd`;
frozen `moe.metal` SHA-256 is
`da0733b13cbeb822638f79d14725aa52c610d6a3f86d0a2b82148a60150121a4`.
Both are recorded in `candidate/frozen.json`. Full results are in
`samples/summary.json`, per-process logs/CSVs/logits and `samples-guard/`.

The measured five-pair decision is **do not adopt**. The reports and patches
retain both the failed approach and the reuse adjustment for a future design;
there is no installed performance feature from this experiment.
