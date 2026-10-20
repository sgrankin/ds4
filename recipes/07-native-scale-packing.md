# Native MXFP4 scales: a lossless three-bit dictionary pilot

**Status: Python and portable C codecs passed real-data round trips; no serving
implementation or model-speed result.** This is the MXS3 idea supplied by the user, tested against
actual native MXFP4 expert bytes. The standalone `MXS3P001` pilot container is
not an official or registered GGUF type.

## Algorithm to reproduce

A native MXFP4 block stores 32 weights in 17 bytes: one E8M0 scale byte and 16
packed E2M1 code bytes. Keep all code bytes unchanged. For each group of native
blocks, build a sorted dictionary of scale bytes and encode each scale as a
three-bit dictionary index. If the group needs more than 8 distinct scales,
write an explicit raw-group tag and its original bytes. No floating-point
arithmetic, clipping or requantization is involved.

```text
encode(group):
    scales = every block's byte0
    dictionary = sorted(unique(scales))
    if len(dictionary)>8: write RAW tag + original group
    else:
        write dictionary tag + dictionary
        write 3-bit index for each scale
        write unchanged16-byte code payload for each block
decode(group):
    restore each original scale byte from its index
    interleave it with that block's unchanged code bytes
```

The zero-overhead ceiling is `(8-3)/(17*8) = 3.67647%` of native expert bytes.
Group dictionaries, tags and headers reduce that saving. Retain a raw fallback
instead of discarding uncommon scales. Partial final groups and eight-/nine-
symbol boundaries need explicit round-trip checks.

## Actual data and measurement

Source: [antirez's native1017 GGUF snapshot](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf/tree/9dcee51692e6540f697141fc98b097e3c96492a3),
revision `9dcee51692e6540f697141fc98b097e3c96492a3`, shard
`DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf`.
Six complete type 39 expert matrices were selected before inspecting scales:
layer 0 gate/up/down experts 0/191/383, and layer 5 gate/up/down experts 383/0/191.
Each was 6,266,880 bytes; total 37,601,280 bytes. Sampling covered two early layers
of shard 1, not the whole model. Exact HTTP206 ranges and individual byte/hash
proofs were retained; total download including the directory was 51.86 MiB.

All six matrices used scale bytes `{119,120,121,122}`. Four symbols would fit
two bits for this sample, but the requested fixed THREE-bit scheme was retained
rather than tuned after seeing the data.

| Native blocks per group | Packed bytes, all six matrices | Net saving including overhead | Raw fallback | Python decode |
|---|---:|---:|---:|---:|
|256|36,246,102|3.60407%|0/8640 groups|83.39 MiB/s|
|4096|36,221,161|**3.670404%**|0/540 groups|82.55 MiB/s|
|Whole expert|36,219,030|3.67607%|0/6 experts|82.34 MiB/s|

Every reconstructed byte and SHA256 matched the original, for every sample and
grouping. At4096 blocks, untouched code bytes were 35,389,440; indices 829,440;
dictionaries/tags/headers 2281; net saving 1,380,119 bytes. These are one-pass
in-memory Python timings, excluding network, disk, Metal and inference. The
separate sanity check exercised raw fallback, which the real samples did not.
[Full pilot report](evidence/mxs3/report.md) and
[machine-readable results](evidence/mxs3/results.json).

## Portable C follow-up

The standalone [C codec](code/mxs3/mxs3_codec_bench.c) implements the same format
using libc only. Its first compile and benchmark passed on the six real samples;
packets matched Python byte-for-byte and every expanded native byte matched.
Three boundary checks cover a partial group, eight symbols and raw fallback.
Define `MXS3_NO_MAIN` to reuse `mxs3_pack_bound`, `mxs3_pack` and `mxs3_unpack`
without the benchmark frontend.

| Hot-buffer operation | Aggregate throughput |
|---|---:|
| Native bytes → MXS3 |11.603 GB/s|
| MXS3 → native 17-byte blocks |**29.844 GB/s**|
| Native-byte memcpy |81.122 GB/s|

These are one-thread, preallocated, hot in-memory measurements: 128 repetitions
per sample, 4,812,963,840 original bytes timed per operation. File I/O, allocation,
checksums and final comparisons are outside the timers. Expansion ran at 36.79%
of native memcpy throughput. No model or Metal kernel ran. See the
[C report](evidence/mxs3/c-report.md),
[aggregate](evidence/mxs3/c-attempt1/summary.json) and
[per-sample CSV](evidence/mxs3/c-attempt1/results.csv).

The C codec removes Python's obvious bottleneck, but a separate decode pass
still costs about 33.5 ms per original GB. A simple serial read-plus-decode model,
assuming no otherwise-required native copy, breaks even only below about
1.10 GB/s storage throughput: `saving_fraction * decode_throughput`. That is
arithmetic, not a measured SSD or serving result. An integrated path could
overlap work or replace a copy, but neither effect has been measured here.

## Format and performance boundaries

This does not compress routed IQ2_XXS/Q2_K expert weights by relabelling their bytes.
IQ2_XXS gate/up uses F16 base scales plus packed grid/sign indices; Q2_K down
uses scale/min nibbles and affine `d/dmin`. Neither has the native one-E8M0-byte
per 32-values layout.

A separate bounded census checked the actual 1017 Q2 file at the same pinned
revision. Before inspecting scales, it selected the first 4096 blocks of expert
0 in gate/down tensors at layers 0 and 20. Total HTTP download, including its
6 MiB directory header, was **7,520,256 bytes**, below the 8 MiB bound; every
request required exact HTTP206 range/length checks. No model ran.

| Actual Q2 field | Layer 0 distinct values | Layer 20 distinct values |
|---|---:|---:|
| IQ2_XXS FP16 `d` bits, 4096 blocks |863|884|
| Q2_K FP16 `d` bits, 4096 blocks |1104|986|
| Q2_K FP16 `dmin` bits, 4096 blocks |920|910|
| Q2_K packed scale/min bytes, 65,536 values |209|206|

Even separate scale and min nibbles span 14 and 16 symbols in both down samples.
The tested 4096-value dictionary groups exceed eight entries and would take the
raw fallback. This rules out applying this exact E8M0/three-bit scheme to the
sampled routed-expert Q2 fields; it does not rule out other lossless compression. See the
[Q2 census report](evidence/mxs3/q2-census/report.md),
[field counts](evidence/mxs3/q2-census/results.json), and
[separate nibble counts](evidence/mxs3/q2-census/nibble-census.json).

If the sampled saving generalized to all trunk native experts, about 10.60 GB
would be saved. The ideal ceiling is 10,616,832,000 bytes, only 2.12% of the full
501.60 GB nine-shard native checkpoint because other tensors are unchanged.
This is a shape-based extrapolation, not a full-checkpoint measurement. It
would not shrink native MXFP4 below the existing 365.71 GB Q2 checkpoint.

Inflating into ordinary 17-byte blocks before caching can save storage/miss
transfer bytes, but not expert-cache RAM. A RAM benefit requires the cache to
STAY packed and the kernel/read path to consume the dictionary representation.
The capacity gain would be roughly 3.8% if the measured saving generalized;
unpacking instructions and metadata may offset it. The C decoder expands back
to ordinary blocks, so it provides **no expert-cache RAM saving**. No GGUF type,
serving-loader integration, codebook-aware Metal kernel or model evaluation was
implemented.

## Native Engram inside the actual Q2 file

The routed-expert rejection above does not cover every tensor in the Q2 file.
Its two Engram tables retain native E4M3 codes and E8M0 scales: each row contains
256 code bytes and eight scale bytes. A separate fixed plan selected four
512-row windows per table (start, one-third, two-thirds and end) before reading
payloads. Exactly eight `pread` calls read 1,081,344 bytes, below a 2 MiB bound.
All eight windows used only three or four scale symbols, and every restored
byte/SHA256 matched; the code bytes were unchanged.

Keep each row's 256 code bytes and pack its eight dictionary indices into three
bytes. Include a seven-byte block header plus its sorted dictionary; more than
eight symbols selects raw fallback. The real windows all took the dictionary
path: **1,081,344 → 1,060,949 bytes, saving 1.886079%** including metadata.
The ideal payload-only ceiling is `5/264 = 1.893939%` of Engram. Applied to both
whole tables before overhead, that would save 3,840,114,250 bytes, or
**1.050033% of this entire Q2 checkpoint**. Eight windows do not establish
whole-table eligibility. No full table scan, runtime change or serving gain was
measured. Expanded row/graph memory is unchanged; extra dictionary reads and
filesystem amplification could erase the small logical disk saving.

See the [Engram report](evidence/engram-scale/README.md),
[fixed plan](evidence/engram-scale/sample-plan.json),
[byte/histogram evidence](evidence/engram-scale/results.json) and
[standalone census](code/mxs3/engram_census.py). Raw/packed model payloads are not
bundled. To reproduce against the verified local model, copy the tool and plan
to a fresh workspace; adjust `model_path` in the copied plan if necessary:

```sh
mkdir -p .workspaces/engram-scale-rerun
cp recipes/code/mxs3/engram_census.py .workspaces/engram-scale-rerun/
cp recipes/evidence/engram-scale/sample-plan.json .workspaces/engram-scale-rerun/
python3 .workspaces/engram-scale-rerun/engram_census.py
```

## Reproduce without a full checkpoint download

The [range fetcher](code/mxs3/range_sample.py) pins the remote revision and checks
exact HTTP206 extents before reading bounded responses. The
[Python codec](code/mxs3/codec_pilot.py) creates the reference packets consumed
by the C benchmark; [sanity.py](code/mxs3/sanity.py) runs the direct edge checks.
Use a fresh workspace because the fetcher refuses existing artifacts. Commands
below write samples and binaries outside the versioned recipe directory:

```sh
mkdir -p .workspaces/mxs3-rerun
cp recipes/code/mxs3/* .workspaces/mxs3-rerun/
python3 .workspaces/mxs3-rerun/range_sample.py fetch 0 16777216 header.bin
python3 .workspaces/mxs3-rerun/range_sample.py directory .workspaces/mxs3-rerun/header.bin
python3 .workspaces/mxs3-rerun/range_sample.py samples
python3 .workspaces/mxs3-rerun/codec_pilot.py > .workspaces/mxs3-rerun/python.log
python3 .workspaces/mxs3-rerun/sanity.py
cc -std=c11 -O3 -Wall -Wextra -Werror \
  -o .workspaces/mxs3-rerun/mxs3_codec_bench \
  .workspaces/mxs3-rerun/mxs3_codec_bench.c
.workspaces/mxs3-rerun/mxs3_codec_bench 128 .workspaces/mxs3-rerun/*.mxfp4 \
  > .workspaces/mxs3-rerun/c-results.csv
```

The sample plan and original summary are retained in
[evidence](evidence/mxs3/sample-plan.json). This replicates the native packing
experiment, not an engine performance test. It does not require our Q2 model.

The [Q2 census tool](code/mxs3/q2_census.py) uses the same copied range fetcher,
but creates a separate `q2-census/` directory and fetches at most 8 MiB. Run it
once in that fresh workspace to reproduce the fixed Q2 field counts:

```sh
python3 .workspaces/mxs3-rerun/q2_census.py
```

For the separate nibble check, split each down-block's first 16 bytes into
`byte & 15` scale and `byte >> 4` min values, then count distinct values across
the 4096 blocks. The retained JSON contains those two counts for each layer.

## Next bounded replication step and credit

On another native MXFP4 checkpoint, sample experts across layers and gate/up/down
before tuning the group size. Compare complete bytes, not only dequantized
values; record dictionaries, fallback and all overhead. Then compare a native
codebook-aware miss/dequant path with the ordinary path on the SAME expert
bytes, checking exact values/output and measuring miss time plus cache occupancy.
Only after that measure a whole serving workload. No quality or latency gain
is implied by smaller lossless storage alone.

Credit the user for the MXS3 proposal, the released native DeepSeek/antirez data
for the measured bytes, and [ds4-recipe](https://tangled.org/astrra.space/ds4-recipe)
chapter 03 for the relevant native 17-byte MXFP4 layout and precision-preserving
runtime context. That recipe's native AMD execution is distinct from this
three-bit scale-container pilot and from our Q2 Metal measurements.
