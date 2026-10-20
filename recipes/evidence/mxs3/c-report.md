# Portable C scale codec: first attempt

2026-10-20. Standalone libc implementation: `mxs3_codec_bench.c`. It uses the same `MXS3P001`4096-block format as the Python pilot. Define `MXS3_NO_MAIN` to reuse its `mxs3_pack_bound`, `mxs3_pack` and `mxs3_unpack` functions without the benchmark. No engine, GPU, quantization or runtime integration was added.

The first compile succeeded with no warnings (`/usr/bin/cc`, Apple clang21.0.0, arm64-apple-darwin27.0.0):

```sh
cc -std=c11 -O3 -Wall -Wextra -Werror \
  -o .workspaces/upgrade-20261020/mxs3/c-attempt1/mxs3_codec_bench \
  .workspaces/upgrade-20261020/mxs3/mxs3_codec_bench.c
```

The first benchmark also succeeded. It checked the six retained real samples (37,601,280B), generated packets byte-identical to the existing Python packets, and restored every original byte. Three small direct checks covered a partial final group, eight symbols and a nine-symbol raw fallback. Real samples still had zero raw fallbacks and packed to36,221,161B: **3.670404% net saved**, including dictionaries, tags and headers.

Benchmark:128 repetitions per sample per operation, one thread, preallocated hot in-memory buffers. Timing excludes file loading, allocation, checksum and the final full-byte comparisons. A volatile-observed output and function-pointer `memcpy` prevent discarding the copy loop. Six per-sample measurements are retained in `c-attempt1/results.csv`; commands, source hash, compile/run logs and the first source/binary are all in that directory.

| Operation | Time for4,812,963,840 original bytes | Aggregate output/input throughput |
|---|---:|---:|
|Encode native bytes→MXS3|0.414802s|11.603GB/s =11,065.52MiB/s|
|Decode MXS3→native17B blocks|0.161271s|**29.844GB/s =28,461.41MiB/s**|
|Native-byte `memcpy`|0.059330s|81.122GB/s =77,363.90MiB/s|

Decode throughput is36.79% of the native-copy rate. Per-sample decode ranged27,303.9–29,868.8MiB/s. These are single-attempt CPU hot-buffer measurements, not cold RAM, disk, model or whole-serving throughput. The whole executable including file reads and correctness checks took0.734s. Machine-readable aggregate: `c-attempt1/summary.json`.

## What these numbers imply

The C prototype removes Python's obvious bottleneck, but a separate decode pass still costs about33.5ms per original GB. With compressed fraction0.963296 and no native extra-copy cost, a serial `read packed bytes + decode` path beats `read native bytes` only below roughly **1.10GB/s SSD throughput** (`saving_fraction × decode_throughput`). That simple arithmetic excludes cache overlap and all integration overhead; it is not an SSD measurement. If decoding can replace an otherwise necessary native copy, its incremental cost is lower, but that baseline copy has not been established here.

Expanding into the existing17B-block expert cache saves disk/miss-transfer bytes, **not expert-cache RAM**. Keeping the cache compressed and consuming the dictionary directly could change that tradeoff, but needs a separate codebook-aware kernel experiment. No such port was attempted. Current IQ2_XXS/Q2_K weights remain ineligible for this exact E8M0 packing scheme. Byte-identical transport introduces no intrinsic requantization error; no new model quality or latency result is claimed.

Stopped after the requested real-data roundtrip and throughput measurement. Original Python/source sample evidence is unchanged; `c-attempt1/source.c` SHA256 is `68a2ff4b02fa84920643ccbd462c83883c2726e60d49129a3c345c6557bb2a20`.
