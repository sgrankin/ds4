# Flint

Flint loads its safetensors directory rather than a GGUF file. Start a small
context on a supported Metal profile with:

```nu
nu tools/zig-build ds4-server
./download-model.nu flint
./ds4-local --bin-dir ./zig-out/bin --server flint --model gguf/flint-int4 --ctx 8192 --port 8001
```

Connect an external agent through Responses HTTP/WebSocket. The built-in
`ds4-agent` does not support Flint's tool format. The backend manual separates
supported hardware profiles from the physical hardware already tested.

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
