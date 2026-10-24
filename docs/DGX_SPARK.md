# DGX Spark

[README](../README.md) | [Getting started](../README.md#start-here)

Use the NVIDIA driver and CUDA development
toolkit for the machine; the build needs `nvcc` and cuBLAS.
Check that `nvidia-smi` sees the GPU before building.

## Build and run

```sh
make cuda-spark
./download-model.nu ds41f-q2
./ds4 --cuda --ssd-streaming --ctx 32768
```

The build targets GB10 (`sm_121a`) and enables the Blackwell-specific kernels.
Do not use `--cuda-tensor-parallel` on a single Spark.
Stop other inference services before loading a model so they do not compete
for memory. Restore any services you stopped when finished.

## DeepSeek V4.1 Flash

Q2 needs SSD streaming on one 128 GB Spark:

```sh
./download-model.nu ds41f-q2
./ds4 --cuda -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming --ctx 32768
```

The file is 341 GiB, including 189 GiB of disk-only Engram tables. Keep it on
a fast local SSD. The expert cache is sized automatically; leave memory for
other programs and the context. Use the same options with `ds4-agent` or
`ds4-server`.

Two Sparks can instead keep half the experts each, using
[network tensor parallelism](DISTRIBUTED.md#tensor-parallelism-between-two-sparks).
Both need the complete GGUF on their local SSD. Single-host V4.1
[DSpark](SPECULATIVE_DECODING.md#deepseek-v41-flash-dspark) also works with SSD
streaming; TP speculation and CUDA vision are not supported for V4.1.
The [model guide](MODELS.md#deepseek-v41-flash) covers
thinking levels and the separate Metal configurations.

September 13-14, 2026, Q2 SSD with automatic cache sizing: a 3,241-token
`/read README.md` prefill reached 93-96 t/s, up from 49-54 t/s, with 32K
allocated context. No extra flag is needed. A separate 64 GiB cache-budget
run with 64K allocated context reached 384 t/s for a 32K initial prefill and
88 t/s for a 3.2K append. These are SSD-streamed V4.1 results, not resident
V4 Flash numbers; speed depends on prompt length and the expert cache.
The automatic-cache 256-token reply test decoded about 5% slower, but
prefill plus the reply fell from about 87 to 59 seconds.
See the [QA record](../QA_BEFORE_RELEASES.md#cuda-ssd-streaming) for conditions
and longer-context measurements.

On two Sparks over the direct RoCE link, Q2 measured **21.8 t/s** for one
session on October 2, 2026: a 1K prompt, 64K allocated context and 2,048
teacher-forced decode tokens, without speculation.

| Work | Speed |
| --- | ---: |
| Fresh 32K prefill | 411 t/s |
| Add 8K after 32K | 314 t/s |

Both runs used 64K allocated context and 512 decode tokens per frontier;
decode at these longer contexts was about 18 t/s. Engram tables stayed on disk.

Decode batching starts at five ready sessions; smaller groups run in order.
The earlier eight-session reference is about **28 aggregate t/s**, not per
client. See the [network TP QA record](../QA_BEFORE_RELEASES.md#cuda-network-tensor-parallelism)
for the workloads and memory limits.

## Qwen3.8 Flash Next

Both Q2 and Q4 fit on one Spark. After building:

```sh
./download-model.nu qwen38-q2
./ds4 --mtp --ctx 32768
```

Q2 keeps 41.73 GiB of weights resident; Q4 keeps 69.74 GiB. Both GGUFs also
contain 95.37 GiB of BF16 n-grams, read directly from the SSD. Leave room
for context and runtime buffers. Vision, the native agent, server APIs and
text-session checkpoints work on CUDA; see [Qwen setup](QWEN38_FLASH_NEXT.md).

September 15, 2026, Q2 on one Spark: a 16K prefill reached 250 t/s, extending
it to 30K reached 243 t/s, and ordinary decode stayed around 17.3 t/s.
These are single runs with 8,192-token prefill chunks and 32 decode tokens.
A separate 256-token prose run measured 17.1 t/s ordinarily and 19.9 t/s with
MTP, at temperature zero and 8K allocated context. Speculation is opt-in;
its benefit depends on the prompt.

## GLM 5.3 Flash

Q2 is the resident target for one Spark:

```sh
./download-model.nu glm53-q2
./ds4 --cuda -m gguf/GLM-5.3-Flash-Q2.gguf --ctx 16384
```

Q4 does not fit resident. GLM Spark-to-Spark tensor parallelism is not
implemented; the two-Mac RDMA instructions do not apply to GLM on CUDA.

## Vision and speculative decoding

GLM 5.3 Flash and Qwen3.8 Flash Next use matching separate vision encoders.
Use `/read image.png` in the CLI; see [models and vision](MODELS.md#vision).
Both use their built-in MTP block with `--mtp`. V4.1 uses its own matching
DSpark support file as described above. Speculation is disabled by default.

## Larger models and serving

CUDA also has [SSD streaming](SSD_STREAMING.md) paths for larger weights.
Memory fit and speed depend on the model layout; start with Q2 for normal use.

V4.1 Q2 SSD serving batches up to eight decode rows together. An eight-session
test reached 11.0 aggregate t/s versus 8.4 with ordered execution; this is
total throughput, not speed per client. Other single-Spark model paths retain
their existing scheduling.
See [serving](SERVER.md) and the recorded [benchmarks](PERFORMANCE.md).
