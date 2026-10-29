# GLM 5.3 Flash

This is the native hybrid Flash implementation, distinct from the full GLM 5.3
legacy engine. Start the packaged Q2 model with:

```nu
nu tools/zig-build ds4-server
./download-model.nu glm53-q2
./zig-out/bin/ds4-server --model gguf/GLM-5.3-Flash-Q2.gguf --ctx 8192
```

The [artifact catalogue](../../MODELS.md#glm-53-flash) describes the Q4
alternative and matching vision encoder. The native FP8 download package does
not yet have an inference implementation.

## Manuals

- [Reference manual](reference-manual.md): the model's architecture and contract,
  grounded in inspected reference material.
- [Runtime implementation](implementation.md): ownership and the path from input
  through execution to continuation and saved state.
- [Apple Silicon implementation](apple-silicon.md): data layout, memory movement,
  GPU work and optimization tradeoffs.

See the [model index](../README.md) for the other implementations and the
[serving architecture](../../architecture.md) for shared protocol and session flow.
