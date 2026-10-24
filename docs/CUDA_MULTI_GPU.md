# NVIDIA CUDA GPUs

[README](../README.md) | [Getting started](../README.md#start-here)

**Ada Lovelace is supported, including 48 GB L40S cards.** This is an important
use case: a model whose other inference implementations require newer GPUs
can still run here when DwarfStar supports its GGUF layout. Supported tensor
layouts select an appropriate CUDA path. Support remains model-specific, not
a promise to run arbitrary GGUFs.

## Build

Install the NVIDIA driver and CUDA toolkit, including `nvcc` and cuBLAS.
Use the local GPU architecture, or select Ada explicitly:

```sh
make cuda-generic
# For an Ada build:
make cuda CUDA_ARCH=sm_89
```

For a GB10 machine use the [DGX Spark](DGX_SPARK.md) target instead.

## Normal layer placement

Without tensor parallelism, DwarfStar places complete layers on the selected
GPUs. `--gpu-vram auto` uses reported free VRAM, reserving space for the graph
and context. Explicit budgets are comma-separated GiB values, one per device.

```sh
./download-model.nu glm53-q2
./ds4 --cuda --gpu-devices 0,1,2,3 --gpu-vram auto --ctx 32768
```

This is also the multi-GPU placement mode for GLM. Startup refuses a layout
that would require unsupported CPU execution; reduce context or choose a
smaller model if necessary.

## Serving and network tensor parallelism

Serve supported models using their admitted layer placement and session count;
see [serving](SERVER.md). Original Flash's paired multi-GPU CUDA layout and its
L40S launcher are retired. Those historical throughput measurements do not
qualify another model or layout.

For V4.1 Flash across two Sparks, use the separate
[network tensor-parallel path](DISTRIBUTED.md#tensor-parallelism-between-two-sparks).
