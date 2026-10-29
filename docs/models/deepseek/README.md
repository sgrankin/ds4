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
The [artifact catalogue](../artifacts.md#deepseek-v41-flash) covers other quants
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

The [historical V4 Flash reference](legacy-v4-reference.md) preserves the original
model-card synopsis and published scores. Its geometry does not describe current
V4.1 Flash or establish the retained V4 Pro shape. The
[qualification guide](../../development/qualification/deepseek.md) treats the
current and retained paths separately.

See the [model index](../README.md) for the other implementations and the
[serving architecture](../../architecture.md) for shared protocol and session flow.
