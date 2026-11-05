# Fused BF16 rounding and inverse RoPE in prefill

Retain the exact fused path as an opt-in: the isolated operation was repeatedly
cheaper at 8,192 rows, while the whole-model measurements established no gain.
Do not promote it to a preset or change the installed binary on this evidence.

## Question and mechanism

Can combining BF16 rounding with inverse rotary position embedding (RoPE) in
batched prefill reduce latency while retaining the existing numerical boundary?
The baseline rounds the attention output to BF16 across each 512-column head,
then applies inverse RoPE to its final 64 columns. The candidate calls the
existing fused helper, which still rounds every column before rotating the tail.
It eliminates the separate tail pass; it does not eliminate a full 1 GiB of
traffic. For 8,192 rows and 64 heads, that pass accounts for 128 MiB of reads
and 128 MiB of writes. These are logical transfers, not measured SSD traffic.
The scalar decode step is unchanged.

The branch in [batch attention](../../../src/engine/deepseek41/batch_attention.zig)
selects the helper when `DS4_METAL_V41_PREFILL_ROPE_BF16` parses to a nonzero
integer. An unset flag, zero, or the runtime quality mode keeps the original path.
The quality probe below uses ordinary runtime mode with diagnostic logit dumps;
it is distinct from that runtime quality-mode switch.

## Setup and numerical checks

On November 4, 2026, the one-hour experiment window was 21:05–22:05 Eastern.
Both arms use runtime revision `d904029c` and the same binary, SHA-256
`0fe73113760fd6acb74bc4c1713ee58689fb3f2d6f8516114d1592e4b65f0412`;
the fixture test is at `6e9ece37`. The machine is an Apple M5 Max with 128 GiB
RAM. The model is `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`.
The fixture test binary `zig-out/bin/test-metal-prefill-rope` has SHA-256
`6b646bf9fef9e6c1eae75c6011ed272842d477d272a91a4af6a9ce1c5ca34a41`.

The existing driver clears inherited DS4 settings and applies its recorded
baseline plus the six resident GPU settings below. It compares flag values 0
and 1 in alternating order, five times each at 32,768 and 8,192 tokens. Each
run uses a 64 GB expert cache, an 8,192-token prefill chunk and 40,960-token
context allocation. It measures fresh prefill and a 512-token append, with
128 teacher-forced decode tokens per phase. The phase total excludes model
loading and both snapshot save and restore. KV restore retains the expert cache, and no OS
cache flush was performed; these are not cold-storage measurements.

Reproduce from the repository root using the existing pressure guard and driver.
Use fresh output paths and the same built binary. Each guarded command runs
sequentially and the driver records its complete environment and arguments.

```nu
let raw = '.workspaces/hillclimb-20261104/17-prefill-followup'
mkdir $raw
{
  DS4_METAL_V41_RESIDENT_GPU: '1'
  DS4_METAL_V41_RESIDENT_GPU_SIMD: '1'
  DS4_METAL_V41_RESIDENT_GPU_REUSE_ADDR: '1'
  DS4_METAL_V41_RESIDENT_GPU_PARALLEL: '1'
  DS4_METAL_V41_RESIDENT_GPU_REFRESH: '16'
  DS4_METAL_V41_RESIDENT_GPU_FUSED: '1'
} | to json | save --force ($raw | path join 'settings.json')
for frontier in [32768 8192] {
  let out = ($raw | path join $'reproduce-($frontier)')
  python3 speed-bench/ideas/pressure_run.py $'($out)-guard' -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu $out timing --flag DS4_METAL_V41_PREFILL_ROPE_BF16 --frontier $frontier --extra-settings ($raw | path join 'settings.json')
}
```

The first fixture attempt failed because its separate-dispatch counter was not
supported by the active backend. The corrected test passed eight comparisons
and 24 independent head checks, including compressed and uncompressed cases,
multiple row counts and positions; every compared output was bitwise equal.
The full-vocabulary teacher-logit probe compared 256 rows at each context size
(frontiers 8,192 and 8,704; 32,768 and 33,280). All rows were byte-identical,
with zero delta-NLL, KL and top-1 changes; all 256 post-evaluation hashes per
context size also matched. This is evidence for these fixed histories, not a
practical task-quality evaluation or a proof for every input. Raw logs and
summaries are retained locally under `.workspaces/hillclimb-20261104/17-prefill-followup`.

## Profile evidence

One intrusive fresh 32K prefill trace identified 103 attention-output pairs:
the separate BF16 kernels summed to 324.043 ms and their following inverse
RoPE kernels to 42.399 ms. Source inspection and each pair's immediate attention
predecessor support the attribution. The trace contained 101,542 encoder records,
matching declared counts before the first decode, with no sample-buffer allocation
failures in that window. Encoder timings can overlap and coverage is not perfect;
these sums locate work and are not a measured end-to-end speedup. The trace and
its matching details are retained in `profile-notes.txt`,
`fresh-profile-summary.json` and `output-bf16-rope-inferred-baseline.json` under
the local raw root.

A second intrusive trace with the candidate confirms activation in all 103
matching output pairs: attention → BF16 linear → RoPE became attention → RoPE,
with matching shapes and attention predecessors. After normalizing keepalive
suffixes, the only encoder-count change was 103 fewer BF16 linear encoders
(101,542 to 101,439); all 407 RoPE encoders and the 102 separate query-path
pairs remained. Both fresh-prefill windows span command buffers 1–284, with all
declared records present and no allocation failures in that window. Across the
103 matched pairs, local spans sum to 366.441 ms baseline and 326.864 ms
candidate; the 80 full-size pairs sum to 354.262 and 315.813 ms. Their
attention-to-next-operation intervals track those local spans without an added
aggregate dependency delay. Each side is one intrusive diagnostic, so these
sums neither establish a comparative speedup nor predict a whole-model saving.
Details are in `candidate-profile-notes.txt` and
`candidate-profile-summary.json` under the local raw root.

## Operation benchmark

The extended fixture timed the two operations without a model, using the same
Apple M5 Max. Each sample encodes four calls in one command batch and reports
CPU encode plus GPU completion wall time per call; input reset is outside the
timer. This is not an isolated GPU span. Outputs matched byte-for-byte after
every pair, and the default numerical fixture rerun again passed all eight
comparisons and 24 head checks. The extended fixture binary SHA-256 is
`8b5c329a1a7233ff188aaf3d9c4030202b9036c1032d12cc9afde1ca8b88f6be`;
the model benchmark binary was unchanged.

The warmup pair is retained in the raw log and excluded from the statistics.
Five measured pairs alternate order, starting fused first. Values are
milliseconds per call; SD is sample standard deviation.

| Rows | Arm | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | Separate | 0.54350 | 0.72275 | 0.71200 | 0.70175 | 0.68525 | 0.67305 ± 0.07373 |
| 512 | Fused | 0.70800 | 0.42400 | 0.64200 | 0.66050 | 0.64050 | 0.61500 ± 0.11020 |
| 8,192 | Separate | 4.67700 | 4.66625 | 4.67150 | 4.68150 | 4.69300 | 4.67785 ± 0.01023 |
| 8,192 | Fused | 4.23475 | 4.25050 | 4.23975 | 4.25025 | 4.25900 | 4.24685 ± 0.00961 |

At 8,192 rows all five fused samples were faster, with paired change
−9.213% ± 0.196%. At 512 rows four were faster, but the first two measured
pairs were noisy and remain in the table. These are operation-level results,
not whole-model speedups. Reproduce both sizes with the extended fixture:

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/prefill-rope-operation-guard -- nu -c 'for key in ($env | columns | where {|key| $key | str starts-with "DS4"}) { hide-env $key }; ./zig-out/bin/test-metal-prefill-rope --bench'
```

## Timing and decision

The 32K profile-off comparison completed five alternating pairs. Times below
are seconds derived from the benchmark’s two-decimal throughput output. Total
is fresh prefill and decode plus the 512-token append and its decode; it excludes
loading and snapshot save and restore. The summary uses sample standard deviation.

| Pair | First arm | Fresh baseline | Fresh candidate | Total baseline | Total candidate |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Baseline | 43.415 | 45.946 | 58.514 | 62.646 |
| 2 | Candidate | 49.470 | 48.650 | 65.352 | 64.561 |
| 3 | Baseline | 49.514 | 49.543 | 66.030 | 65.578 |
| 4 | Candidate | 49.600 | 49.797 | 65.661 | 65.886 |
| 5 | Baseline | 49.840 | 49.837 | 65.850 | 65.886 |
| Mean ± SD | — | 48.368 ± 2.772 | 48.754 ± 1.641 | 64.281 ± 3.234 | 64.911 ± 1.378 |

The first baseline and candidate fresh-prefill samples were faster than the
later runs. The first candidate’s fresh steady decode was 33.38 tokens/s,
compared with 36.23 for the first baseline; scalar decode is unchanged and the
cause of that difference is not established. Paired total differences range from
−0.791 to +4.132 seconds. The 32K samples do not establish a gain or regression.
The 8K profile-off comparison also completed five alternating pairs:

| Pair | First arm | Fresh baseline | Fresh candidate | Total baseline | Total candidate |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Baseline | 18.546 | 18.717 | 32.934 | 33.116 |
| 2 | Candidate | 19.051 | 18.878 | 34.037 | 33.330 |
| 3 | Baseline | 19.705 | 20.078 | 35.584 | 35.945 |
| 4 | Candidate | 19.977 | 20.339 | 35.878 | 36.353 |
| 5 | Baseline | 20.621 | 19.916 | 36.412 | 35.548 |
| Mean ± SD | — | 19.580 ± 0.807 | 19.586 ± 0.737 | 34.969 ± 1.440 | 34.858 ± 1.522 |

The 8K paired total differences range from −0.864 to +0.475 seconds; both
arms tended to take longer in later pairs. Neither model comparison establishes
a gain or regression. The candidate trace confirms that the intended dispatch was removed and that
its local span fell in the diagnostic run. Together with the isolated benchmark,
this supports retaining the small, exact opt-in path, but the profile-off model
runs do not establish an end-to-end improvement. No preset or installed binary
is changed. Across the guarded runs, memory pressure remained at level 1 and
swap stayed approximately 28.90 GiB.

The following retains decode measurements for all five pairs in pair order.
Each entry is complete decode seconds / first-token milliseconds / steady
tokens per second. Steady excludes the first token; it does not prove all
initialization has finished. Fresh and append each contain 128 decoded tokens.

| Context | Arm | Phase | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32K | Baseline | Fresh | 4.381 / 875.0 / 36.23 | 4.342 / 821.8 / 36.08 | 4.406 / 871.5 / 35.93 | 4.339 / 815.5 / 36.05 | 4.313 / 802.1 / 36.18 |
| 32K | Baseline | Append | 3.573 / 33.9 / 35.88 | 3.912 / 34.1 / 32.75 | 3.960 / 80.0 / 32.72 | 3.952 / 34.6 / 32.42 | 3.954 / 34.3 / 32.40 |
| 32K | Candidate | Fresh | 4.680 / 876.1 / 33.38 | 4.321 / 811.1 / 36.18 | 4.363 / 844.7 / 36.10 | 4.333 / 817.9 / 36.13 | 4.321 / 803.5 / 36.09 |
| 32K | Candidate | Append | 3.820 / 33.5 / 33.54 | 3.928 / 34.1 / 32.62 | 3.948 / 34.4 / 32.45 | 3.954 / 36.5 / 32.42 | 3.951 / 34.2 / 32.42 |
| 8K | Baseline | Fresh | 4.264 / 805.1 / 36.71 | 4.220 / 798.4 / 37.11 | 4.435 / 858.9 / 35.51 | 4.284 / 845.5 / 36.94 | 4.265 / 824.7 / 36.92 |
| 8K | Baseline | Append | 3.441 / 31.3 / 37.25 | 3.784 / 35.5 / 33.88 | 3.855 / 34.0 / 33.24 | 3.844 / 33.9 / 33.34 | 3.820 / 32.8 / 33.53 |
| 8K | Candidate | Fresh | 4.238 / 812.8 / 37.08 | 4.275 / 847.4 / 37.05 | 4.238 / 807.3 / 37.01 | 4.435 / 871.6 / 35.64 | 4.268 / 840.0 / 37.04 |
| 8K | Candidate | Append | 3.447 / 33.3 / 37.20 | 3.459 / 32.9 / 37.07 | 3.886 / 33.9 / 32.97 | 3.865 / 33.3 / 33.14 | 3.754 / 34.1 / 34.14 |
