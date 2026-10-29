# Clef decision models

Clef answers bounded questions through learned answer heads, rather than
generating a conversation token by token. Start the Flash model with:

```nu
nu tools/zig-build ds4-server
./download-model.nu clef-flash
./ds4-local --bin-dir ./zig-out/bin --server clef-flash --ctx 8192
```

The build needs FFmpeg development libraries and `pkg-config`, as well as the
[native build prerequisites](../../../src/README.md#build-and-run).
Use `clef` or `clef-omni` for the other bundles. Requests use `/v1/decisions`
or `/v1/systemone`; generative conversations use Responses separately. The
implementation manual explains the request mapping and co-hosting decision
helpers with one generative engine.

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
