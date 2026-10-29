# Execution backends

The maintained local build targets Apple Silicon and Metal. CUDA, ROCm and
multi-machine sources retain useful upstream implementations and historical
qualification; their guides distinguish that evidence from the native build.

- [Apple Silicon](apple-silicon.md): shared Metal resources and platform setup.
- [CUDA](cuda.md): retained NVIDIA and DGX Spark paths.
- [ROCm](rocm.md): retained AMD and Strix Halo paths.
- [Multi-machine execution](distributed.md): tensor and pipeline parallelism,
  networking and recovery.

Model-specific kernels and storage layouts belong to each
[model's Apple Silicon manual](../models/README.md).
