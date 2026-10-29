# Qwen 3.8 Flash Next

The Q2 GGUF includes the language model, embedded MTP weights and a disk-backed
n-gram table. A conservative starting context and prefill chunk are:

```nu
./tools/zig-build ds4-server
./download-model.nu qwen38-q2
./zig-out/bin/ds4-server --model gguf/Qwen3.8-Flash-Next-Q2.gguf --ctx 8192 --prefill-chunk 1024
```

The file-format and source identifier `qwen4exp` refers to this supported
profile. It does not establish compatibility with another Qwen checkpoint.
The [artifact catalogue](../artifacts.md#qwen38-flash-next) covers quantization
choices and the separate vision encoder.

## Manuals

- [Setup and usage](usage.md): build, launch, client examples and operational controls.

- [Reference manual](reference-manual.md): the model's architecture and contract,
  grounded in inspected reference material.
- [Runtime implementation](implementation.md): ownership and the path from input
  through execution to continuation and saved state.
- [Apple Silicon implementation](apple-silicon.md): data layout, memory movement,
  GPU work and optimization tradeoffs.

See the [model index](../README.md) for the other implementations and the
[serving architecture](../../architecture.md) for shared protocol and session flow.
