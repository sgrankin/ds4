# DeepSeek V4.1 Flash

For a 128 GB Mac, the Q2 model needs SSD expert streaming. The existing
`persistent-fast` server preset supplies streaming and context-reuse settings:

```nu
./tools/zig-build ds4-server
./download-model.nu ds41f-q2
./ds4-local --bin-dir ./zig-out/bin --server persistent-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --ctx 32768 --prefill-chunk 8192
```

This preset selects approximations and packed caches; it is not the ordinary
engine baseline. The backend manual describes its choices and override rules.
The [artifact catalogue](../../MODELS.md#deepseek-v41-flash) covers other quants
and checkpoint-matched vision files.

## Manuals

- [Reference manual](reference-manual.md): the model's architecture and contract,
  grounded in inspected reference material.
- [Runtime implementation](implementation.md): ownership and the path from input
  through execution to continuation and saved state.
- [Apple Silicon implementation](apple-silicon.md): data layout, memory movement,
  GPU work and optimization tradeoffs.
- [DeepInfer comparison](deepinfer/comparison.md): official mechanisms, local
  adaptations and remaining differences.

See the [model index](../README.md) for the other implementations and the
[serving architecture](../../architecture.md) for shared protocol and session flow.
