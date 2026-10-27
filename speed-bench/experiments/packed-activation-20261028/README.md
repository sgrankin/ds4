# Compressed activation storage with floating-point matmul

Authorized **2026-10-28 00:15–01:15 UTC** experiment, including implementation,
builds and analysis. Fixed baseline: `19c63b1b26da`. **Complete; none of the
three variants is adopted.** Staged compression saves a real **105 MiB GPU
scratch** but takes **4.04% longer to prefill and 8.07% longer across the three
tasks**. Direct register decoding and wider N64 staging also passed numerical
checks but slowed all five leaf pairs, so neither received a full-model
campaign. Keep the existing runtime and retain these prototypes as patches.
Raw outputs stay ignored under `.workspaces/packed-activation-20261028/`.

This isolates the storage saving from the slower integer arithmetic in the
[preceding experiment](../integer-prefill-20261027/README.md). Expert weights
retain their original IQ2_XXS/Q2_K bytes on SSD and in the shared Metal cache.
Gate/up activations are stored as INT8 codes plus one F32 scale per 32 values,
then decoded into small F16 compute tiles. Matmul uses **floating-point F16
operands and F32 accumulation**, with the existing weight dequantization and
K-loop ordering. Q2 down projections retain their existing half pack. There
is no full expanded activation shadow or persistent INT8 expert-weight cache.

## Completed staged leaf screen

The staged binary and its exact shader snapshot are under the local
`.workspaces/packed-activation-20261028/staged/` prefix. The guarded fixture
passed finite-output, row-tail, zero-input and output-sentinel checks. Its
small shape uses input 256, middle 512, output 260 and 256 synthetic experts;
the timed shape uses production widths input 5,120, middle 2,304, output 5,120
and six fixed routed experts. These are synthetic MoE checks, not model
answers or a quality benchmark.

| Final-mixture check | Maximum absolute error | RMS error | Relative RMS | Cosine |
| --- | ---: | ---: | ---: | ---: |
| Small, 512 rows |1.98196e-5 |3.40995e-6 |0.540882% |0.999985414 |
| Small, 513 rows |1.58373e-5 |3.46910e-6 |0.543610% |0.999985266 |
| Small, 1,024 rows |1.67551e-5 |3.42861e-6 |0.545158% |0.999985209 |
| Small, 512 zero rows |0 |0 |0% |1 |
| Production widths, 512 rows |7.70293e-4 |1.54799e-4 |0.542629% |0.999985346 |

All five measured pairs at production widths are retained. A means control,
B compressed storage. Timing covers the routed MoE call and GPU
synchronization; allocation, uploads and numerical checks are untimed.

| Pair | Order | Control ms | Compressed ms | Time change |
| ---: | :---: | ---: | ---: | ---: |
|1 |AB |7.982 |9.548 |+19.619% |
|2 |BA |8.022 |9.796 |+22.114% |
|3 |AB |8.026 |9.832 |+22.502% |
|4 |BA |7.994 |9.823 |+22.880% |
|5 |AB |7.999 |9.685 |+21.078% |

Control mean/median: **8.0046 / 7.999 ms**, range 7.982–8.026 ms. Compressed
mean/median: **9.7368 / 9.796 ms**, range 9.548–9.832 ms. The ratio of means
is **+21.640%**, with compressed slower in all five pairs. Paired changes
range +19.619–22.880%; their mean is +21.638%, median +22.114%.

The explicit first-call warmups were **18.092 ms control** and **9.767 ms
compressed**. They are retained separately and excluded from every result
above. Raw evidence: `leaf-staged.log`; numerical summary:
`leaf-staged-summary.json`, both under the ignored experiment workspace.

```nu
python3 speed-bench/experiments/packed-activation-20261028/analyze-leaf.py .workspaces/packed-activation-20261028/leaf-staged.log
```

## Alternate consumers and reproduction patches

The qualified **staged variant** expands small activation tiles into
threadgroup memory. It double-buffers half weight and activation tiles and
requests **12 KiB per threadgroup**. Its full-model campaign uses the frozen
staged sources, regardless of later source edits.

The **register variant** decodes activations directly
into half cooperative registers; four SIMD groups each own eight activation
rows. Only the weight tiles use threadgroup storage, reducing its request to
**8 KiB**. Its first runtime shader compilation failed because `decltype(x)`
carried a Metal `thread` address-space qualifier rejected by the destination
factory. Using MPP's public unqualified cooperative operand type repaired the
API call. The failed log and shader snapshot remain local as
`leaf-register.log` and `register/failed-shaders/`.

After that repair, the numerical fixture passed with the same errors as the
staged variant. Its five alternating leaf pairs were nevertheless much slower:
control **[8.137, 8.064, 7.511, 7.939, 7.794] ms**, compressed **[19.192, 18.922,
19.140, 19.107, 18.936] ms**. Means were **7.889→19.0594 ms (+141.595%)**,
medians 7.939→19.107 ms and ranges 7.511–8.137 / 18.922–19.192 ms. All five
pairs slowed down, with paired changes +134.648–154.826%. Explicit warmups
19.150 / 19.120 ms are separate and excluded. No full-model register campaign
was run after this leaf result. Evidence: `leaf-register-fixed.log` and
`leaf-comparison-summary.json` under the ignored workspace.

The final bounded variant used **N64 staged activation tiles and 16 KiB
threadgroup storage**. Numerical checks passed with the same errors. Its five
AB/BA/AB/BA/AB leaf pairs were control **[7.757, 7.735, 7.509, 7.757, 7.724]
ms**, compressed **[9.614, 9.648, 9.377, 9.627, 9.632] ms**. Means were
**7.6964→9.5796 ms (+24.469%)**, medians 7.735→9.627 ms and ranges
7.509–7.757 / 9.377–9.648 ms. All five slowed down, with paired changes
+23.940–24.877%. Warmups 18.553 / 9.746 ms are separate and excluded. This
variant also stopped after the leaf screen. Evidence: `leaf-n64.log` under
the ignored workspace; `analyze-leaf.py` reproduces its summary.

Archived reproduction patches:

- [staged.patch](staged.patch): complete delta from `19c63b1b26da`, including
  build target, fixture, host and staged M64N32 kernel with 12 KiB staging.
- [register.patch](register.patch): apply **on top of staged**, including the
  repaired cooperative operand API and the host 8 KiB request.
- [n64.patch](n64.patch): apply **on top of staged**, replacing only its
  consumer with N64 staging and the host 16 KiB request. This is an alternative
  to the register patch; do not stack those two alternatives.

All three patches passed `git apply --check` and actual sequential application
to small baseline file copies. Resulting sources matched the tested snapshots
byte for byte. Local evidence remains in the ignored `patches/verification.json`,
`patches/verification-register-fixed.json` and `patches/verification-n64.json`.

## Completed staged full-model measurements

Five fresh processes per arm completed all three 8K cases in alternating
AB/BA/AB/BA/AB order; all **30 answers were correct and stopped normally**.
The pressure-guard campaign completed with exit 0. Prefill below is the
independent answer prefill (`greedy_sync_ms`); primary is that prefill plus
normal reasoning and answer generation. Means are seconds; changes compare
compressed storage with control and ranges retain all five paired changes.

| Case | Prefill mean A→B | Change; paired range | Primary mean A→B | Change; paired range |
| --- | ---: | ---: | ---: | ---: |
| Ordinary |19.210→19.757 |+2.848%; +0.676–4.634% |25.982→24.684 |−4.996%; −6.585–−3.385% |
| Favorable |21.740→22.609 |+3.997%; +1.592–6.117% |24.217→29.899 |+23.465%; +20.522–25.850% |
| Changed facts |21.593→22.702 |+5.136%; −0.819–9.475% |23.904→25.501 |+6.682%; +0.655–10.595% |
| Three-case sum |62.543→65.068 |+4.037%; +2.643–6.292% |74.103→80.084 |+8.072%; +6.428–10.194% |

Every aggregate pair was slower for both prefill and primary latency. Here
are all samples: each cell is **prefill / primary seconds**, A control and B
compressed. Unrounded values, medians, ranges and order strata remain in the
local `samples-staged/summary.json`.

| Pair/order | Ordinary A | Ordinary B | Favorable A | Favorable B | Changed A | Changed B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|1 AB |18.574 / 25.549 |19.336 / 24.309 |21.823 / 24.389 |22.902 / 30.266 |22.320 / 24.723 |22.137 / 24.885 |
|2 BA |19.529 / 26.244 |20.434 / 25.356 |22.156 / 24.674 |22.576 / 29.738 |21.931 / 24.347 |23.202 / 25.965 |
|3 AB |19.657 / 26.424 |19.790 / 24.684 |21.722 / 24.213 |22.923 / 30.362 |21.783 / 23.985 |22.878 / 25.789 |
|4 BA |19.112 / 25.787 |19.688 / 24.624 |21.252 / 23.630 |22.551 / 29.739 |20.843 / 23.196 |22.818 / 25.654 |
|5 AB |19.177 / 25.906 |19.537 / 24.446 |21.748 / 24.176 |22.094 / 29.392 |21.088 / 23.269 |22.474 / 25.214 |

Generation lengths, including reasoning and final answers, changed
**87→61 ordinary, 31→98 favorable, 26→32 changed facts**. Each length repeated
in all five samples of its arm. This explains the different task-level
directions: ordinary generation became shorter while favorable generation
became much longer. Whole-task timing combines prefill cost with these changed
reasoning trajectories.

| Case | Forced teacher positions | Mean ΔNLL, nats/token | Mean KL | Teacher argmax flips |
| --- | ---: | ---: | ---: | ---: |
| Ordinary |7 |+0.228316 |0.0137867 |0 |
| Favorable |8 |+0.197556 |0.00229766 |0 |
| Changed facts |7 |−0.106465 |0.00797018 |0 |

These metrics, answer correctness, stop status and generation lengths were
identical across all five repeats within each arm/case. Generated traces
differed between arms in all three cases. The teacher measurements cover a
few forced final-answer positions at a reasoning frontier; they are a narrow
diagnostic, not a broad model-quality score. Practical answers remained
correct on this small task set despite the changed distributions and traces.

## Staged slowdown: matched diagnostic profile

One diagnostic process per arm used the same favorable 8K case with GPU
timestamps enabled. The packed IQ2 gate/up kernels had **220 launches in
each arm**, taking **3.642922→5.250073 seconds (+44.12%)**. In comparison,
110 gate/up packing launches took **316.063→319.638 ms**, and the unchanged
Q2 down F16 pack took **39.421→38.129 ms** over 110 launches.

This locates the additional GPU time in the compressed-activation matrix
kernel rather than the initial pack. It does not establish a physical cause
such as NAX utilization or occupancy; that would require hardware counters.
These are diagnostic timings with perturbed scheduling, separate from the
five unprofiled task pairs. Raw captures and kernel-level totals remain under
`profile-staged/`, including `profile-staged/summary.json`.

## Memory and I/O accounting across all five pairs

The staged processes logged an actual routed batch of **4,096 rows
× 6 routes**, with input width 5,120 and middle width 2,304. The gate/up RHS pack
therefore falls from **240 MiB F16** to **135 MiB codes plus scales**. Q2 down
still requires its unchanged **108 MiB F16** pack. Both operations reuse one
grow-only owner: its peak is the larger pack, so these sizes are not added.

All five pairs reported total GPU scratch **267.59→162.59 MiB**, a **105 MiB
reduction**, matching the gate/up pack difference. Live expert-cache occupancy
was **40.88 GiB in every process**. Scratch accounting is separate from total
process RAM; the experiment did not grow the expert cache into the saved space.

Application miss-`pread` volume was **56.97→68.05 GiB in every process**. This
includes OS file-cache hits and covers all cases, teachers and generation;
it is not physical SSD traffic. The arms follow different generated traces
and perform different amounts of decoding. Evidence: each
`samples-staged/pair-N/{control,compressed}/stderr.log` and the process-I/O
section of `samples-staged/summary.json`, under the ignored workspace.

## Configuration and reproduction

The completed sequence was build/freeze, guarded numerical checks and five
leaf pairs, then five staged full-model pairs and a matched diagnostic profile.
The two alternate consumers stopped after their five-pair leaf screens.
Runtime/build changes were restored byte for byte to the baseline; the patches
preserve the three implementations.

Copy this experiment directory into a clean isolated checkout at
`19c63b1b26da`. Each variant starts from that baseline. Choose `staged`,
`register` or `n64` below; the optional patch is applied after `staged.patch`.
The model campaign also needs the GGUF path in the table below.

```nu
let experiment = 'speed-bench/experiments/packed-activation-20261028'
let variant = 'staged' # Or 'register' or 'n64', in a fresh baseline checkout.
git apply ($experiment | path join staged.patch)
if $variant != 'staged' {
    git apply ($experiment | path join $'($variant).patch')
}
let prefix = $'.workspaces/packed-activation-repro-($variant)'
nu tools/zig-build decoder-replay-bench packed-activation-test -j3 --prefix $prefix
mkdir ($prefix | path join shaders)
cp src/metal/*.metal ($prefix | path join shaders)

mut sources = {}
for source in (glob ($prefix | path join 'shaders/*.metal')) {
    let key = $'DS4_METAL_($source | path parse | get stem | str uppercase)_SOURCE'
    $sources = ($sources | upsert $key ($source | path expand))
}
with-env $sources {
    python3 speed-bench/ideas/pressure_run.py ($prefix | path join leaf-guard) -- ($prefix | path join bin test-packed-activation) out+err> ($prefix | path join leaf.log)
}
python3 ($experiment | path join analyze-leaf.py) ($prefix | path join leaf.log)
```

The Zig build steps compile artifacts; fixture execution is explicit and guarded. All variant
fixtures record warmups separately and five AB/BA/AB/BA/AB measured pairs.
The binary and shader snapshot share a prefix; preserve that layout for the
full-model and profile drivers below.

| Fixed model configuration | Value |
| --- | --- |
| Model | `gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` |
| Prompt / context / chunk |8,192 /16,384 /8,192 tokens |
| Expert cache |48 GiB requested, native expert routing |
| Index cache |Physical FP4 and SIMD enabled; candidate-only indexing off |
| Compute isolation |M64N32 in both arms, `DS4_METAL_DISABLE_V41_PACKED_M32N128=1` |
| Changed setting |`DS4_METAL_ROUTED_PACKED_ACT_I8=0/1` only |
| Arithmetic |Floating-point MPP; F16 operands, F32 accumulation |
| Candidate activation storage |INT8 + F32 scale per32, 1.125 bytes/value |
| Generation |High reasoning, T0, maximum384 new tokens |
| Samples |Five fresh processes per arm, AB/BA/AB/BA/AB; none discarded |

The existing probe disables approximate resident substitutions, decoder replay
and main-cache F16 in both arms. Its first teacher prefill and independent
answer prefill are both retained. Primary task latency is
`greedy_sync_ms + greedy_ms`: fresh answer KV after teacher work has warmed
expert/file caches, excluding model initialization. This is a warm-model task,
not cold-start latency or the duration of the entire benchmark command.

`run.nu` defaults to `--case all`, covering ordinary, favorable and changed-fact
cases in each process. A shorter campaign can use `--case favorable` or
`--case ordinary`. Every candidate process must log the actual compressed
floating-point prefill invocation; a silent fallback fails the run. Controls
must lack that marker. The two arms use the same binary and adjacent frozen
`PREFIX/shaders/*.metal` snapshot; only the named setting differs.

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/packed-activation-repro-staged/samples-guard -- nu speed-bench/experiments/packed-activation-20261028/run.nu .workspaces/packed-activation-repro-staged/bin/decoder-replay-bench .workspaces/packed-activation-repro-staged/samples
python3 speed-bench/experiments/packed-activation-20261028/analyze.py .workspaces/packed-activation-repro-staged/samples
```

Use fresh output paths for each campaign. `analyze.py` reuses the existing fixed
teacher helper, reporting all five paired changes, means, medians, ranges and
AB/BA strata for each phase and case. Positive changes mean compressed storage
took longer. Generated trace divergence is measured rather than rejected;
teacher ΔNLL/KL reference the current quantized Q2 baseline, not BF16 weights.
Correct answer and normal stop status are recorded independently.

Expert-I/O summaries cover the whole process: both prefills, teacher work and
generation for every selected case. Application `pread` bytes include OS file
cache hits and are not physical SSD traffic. Reported GPU scratch at cleanup
and live expert-cache GiB are retained for all five processes. These are
separate from total process RAM, static graph buffers and the requested cache
budget; no automatic cache growth is assumed from saved scratch.

The matching diagnostic profile remains a separate run, not a substitute for
five unprofiled samples:

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/packed-activation-repro-staged/profile-guard -- nu speed-bench/experiments/packed-activation-20261028/profile.nu .workspaces/packed-activation-repro-staged/bin/decoder-replay-bench .workspaces/packed-activation-repro-staged/profile
python3 speed-bench/experiments/packed-activation-20261028/analyze-profile.py .workspaces/packed-activation-repro-staged/profile
```

It uses the same8K favorable task and captures actual kernel timestamps,
pipeline creation, native prefill stages and expert-I/O counters. Timestamp
capture perturbs scheduling; totals include both prefills and generated
reasoning/answers. The analyzer requires the candidate kernel in compressed
GPU traces and its absence in controls.

The experiment is complete. Numerical degradation was measurable and all
tested answers remained correct; the rejection is an engineering tradeoff:
105 MiB scratch saved without added expert-cache capacity does not justify
the measured extra matrix-kernel time and task latency here. The baseline
runtime remains the selected implementation. A future compressed format or
consumer can reuse these fixtures and patches without rerunning the discarded
register design first.
