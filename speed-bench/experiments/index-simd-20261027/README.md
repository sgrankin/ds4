# Register/SIMD index scoring, 2026-10-27

Follow-up to the [physical FP4 pilot](../deepinfer-alignment-20261027/README.md#3-physical-index-fp4-producer-and-reader).
The storage target follows [DeepInfer's index-attention implementation](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/index_attn.py); this SIMD mapping is a local Metal experiment.

Baseline parent `06fb1a64db95`; local change `nnvssrqo`. Budget 15:30–16:30UTC,
including preparation, build and analysis. M5 Max; one guarded GPU fixture at a
time, with the user's reference Flint server left idle and untouched. Main,
installed executables, model allocations and cache defaults are unchanged.

**Result:** the register layout removes the large packed-reader penalty while
improving both formats. On the measured corpus, scores and selected IDs remain
bit-exact. The packed reader retains 68 B rows versus 512 B F32, **86.72% smaller**.
These are allocation sizes of the pilot buffers; production model caches still
use F32. No whole-model TPS or realized serving-memory saving is claimed.

## What changed and why

The old scorer uses one 128-thread threadgroup per key. It copies 128 key values
into shared memory; four SIMDgroups score four heads at a time, with 17
threadgroup barriers per key. FP4 adds decoding before the same shared tile
and scoring loop. Compression therefore reduces stored bytes while retaining
that work and synchronization.

The new matched F32/FP4 kernels assign **one 32-lane SIMDgroup to each key**, four
keys per 128-thread group. Each lane keeps four key values in a `float4` register.
Each head uses `dot` followed by `simd_sum`; lane 0 accumulates all 32 heads in
original order. There is no shared key tile or threadgroup barrier. Packed
lanes read one aligned 16-bit word containing four nibbles, and decode with the
original signed E2M1/scaling/BF16 rules. The 68 B row only promises 2-byte alignment;
this implementation does not assume aligned 16-byte packed loads.

The local FP pragma prevents contraction/reassociation of the final weighted
contribution and accumulator, matching the rounding formerly forced by the
shared-memory contribution store. The dot/reduction retains control flags.
Sparse scoring keeps original global IDs; the existing filter supplies negative
infinity for unselected rows before top-k. Tail returns are SIMDgroup-uniform.

This establishes that a different work layout helps both readers. It does
**not** isolate barriers, occupancy, decoding and cache residency as individual
hardware bottlenecks: no GPU counters were collected. Retaining values in
registers also changes key/head parallelism. The remaining small FP4 overhead
is consistent with its additional unpack work, rather than proof of a single
limiting instruction.

## Measurements

Each point uses eight calls in one command buffer, one warmup and four measured
repeats, rotating the 16 combinations of dense/sparse, old/SIMD, F32/FP4 and
score-only/full chain. The full chain is score → filter → top-k 512. Sparse
selection retains at most 2048 blocks of eight keys, with reversed/spaced IDs
and a partial final block where applicable. Rows here are **index keys**, not
a claim about equal-sized model prompts.

GPU time per call from existing `DS4_METAL_CB_TIMES=1` timestamps:

| Keys | Scope | Old F32 ms | Old FP4 ms | SIMD F32 ms | SIMD FP4 ms |
|---:|---|---:|---:|---:|---:|
|8,192|Dense score|.04834|.05350|.03350|.03406|
|65,536|Dense score|.37663|.41109|.24669|.24900|
|262,144|Dense score|1.47775|1.64028|.97741|.99716|
|8,192|Dense full chain|.08500|.08678|.06647|.06700|
|65,536|Dense full chain|.46731|.50163|.33713|.34584|
|262,144|Dense full chain|1.84188|1.99638|1.32806|1.35566|
|65,536|Sparse full chain|.18869|.19341|.15072|.15056|
|262,144|Sparse full chain|.39716|.40509|.36391|.36197|

At 256K, dense scoring takes **33.9% less GPU time in F32 and 39.2% less in
FP4**; the full chain takes 27.9%/32.1% less. Matched SIMD FP4 scoring costs
about 1–2% more than SIMD F32 at the three profiled dense sizes. For sparse
256K selection, scoring improves 33–39%, while the full chain improves 8–11%:
scored keys cap at 16K but filtering and top-k still traverse the full width.

Two uninstrumented full sweeps agree: dense 256K old F32 full-chain wall time
1.859–1.868ms versus SIMD F32 1.358ms; old FP4 2.022–2.037ms versus SIMD FP4
1.367–1.383ms. These are warmed batched measurements, not an application request
latency prediction. Instrumented wall timings include diagnostic logging; use
GPU timestamps for that run.

The 256K storage/latency choices are 128 MiB / .977ms F32 and 17 MiB / .997ms
FP4 for the new scorer. The [local comparison plot](../../../.workspaces/index-simd-20261027/storage-latency.png)
shows all four measured layouts at 8K/64K/256K; points are not extrapolations.

**Small-work exception:** at 37 keys, full-chain wall time rises from .079–.085ms
to .107–.115ms for F32, and .079–.086ms to .100–.104ms for FP4. The new mapping
launches only 10 groups instead of 37 and serializes heads per key; reduced
parallelism is a plausible cause, not a measured occupancy diagnosis. Keep the
old path for tiny work when integrating, and measure the crossover rather than
inventing one. A separate 37-key timestamp run confirms the bare-score cost:
20.4→36.2µs F32 and 21.5→37.2µs FP4. The ordinary 8K and larger points favor the new layout.

## Numerical checks and integration boundary

Both full sweeps pass all shapes 37/8192/32768/32771/65536/262144, finite and
all-row-tie cases, dense/sparse and all four readers. Changed score bits=0,
changed selected IDs=0. Original quantizer/packed bytes, signed zero, scale
boundaries, BF16-adjacent rounding ties, nonzero tensor offsets, partial blocks
and output/packed canaries also pass. The separate profiled shapes pass again.
This is measured numerical equivalence on this corpus, not a universal proof
or a new end-to-end task-quality evaluation.

The added `ds4_gpu_dsv41_indexer_score_simd` API remains explicit, alongside the
old controls. Actual capacity integration still needs publication at the
original scale boundary, a batched-prefill consumer, canonical import/export
conversion and removal of the F32 shadow. Free space would then need to reach
an expert-cache budget to improve expert residency. Do not infer that merely
calling this scorer has reduced serving RAM.

All five ReleaseFast executables build and pass `--help`. All four guarded
campaigns exit 0; pressure stays at level 1 and swap remains 21.514 GiB. The final
source/build artifacts are separate from the installed executables.

## Reproduce

Run from the repository root; use a fresh guard-output directory for each run.
Raw outputs are local and ignored under `.workspaces/index-simd-20261027/`.

```sh
nu tools/zig-build metal-test --prefix .workspaces/index-simd-build -j6
python3 speed-bench/ideas/pressure_run.py .workspaces/index-simd-guard -- nu speed-bench/experiments/index-simd-20261027/run.nu .workspaces/index-simd-build/bin/test-deepseek41-metal .workspaces/index-simd-run
python3 speed-bench/ideas/pressure_run.py .workspaces/index-simd-profile-guard -- nu speed-bench/experiments/index-simd-20261027/run.nu .workspaces/index-simd-build/bin/test-deepseek41-metal .workspaces/index-simd-profile --rows 262144 --gpu-times
python3 speed-bench/experiments/index-simd-20261027/analyze.py .workspaces/index-simd-profile
```

The optional shape argument stays within the existing six-shape corpus. It
keeps each GPU timestamp run below the existing 400-command log limit. No new
profiling API or global timing instrumentation was added.

With Matplotlib available, reproduce the storage/latency figure from the
analyzer CSV (combine the three separately profiled sizes for all three panels):

```sh
python3 speed-bench/experiments/index-simd-20261027/analyze.py .workspaces/index-simd-profile > .workspaces/index-simd-summary.csv
python3 speed-bench/experiments/index-simd-20261027/plot.py .workspaces/index-simd-summary.csv .workspaces/index-simd-storage-latency.png
```
