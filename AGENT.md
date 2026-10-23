# Agent Notes

The rewrite branch uses Zig for DeepSeek V4.1 and the shared serving frontends,
with retained C engines and an Objective-C Metal driver. Current branch scope
and experiment guidance are in [AGENTS.md](AGENTS.md) and
[src/JOURNAL.md](src/JOURNAL.md); they supersede the broader upstream backend
assumptions in the notes below.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Always make sure that the SSD streaming, CUDA, distributed inference, Metal default inference are not affected by fixes to other parts of the code.
- Keep model loading mmap-backed for the Metal default case; do not eagerly copy the full GGUF. Keep the model loading for SSD streaming of routed experts explicit: allocated buffers, fast reads from disk, always try to hide loading of missing routed experts by loading them while performing the inference of the shared expert and routed experts already in RAM. Always try to hide loading of layers for prefill in SSD streaming mode using the inference time of the current layer as the next one is loaded.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV checkpoints.

## Quality Rules

- Keep the implementation small, sharp, easy to understand. Try to write elegant code in a state of grace. Don't settle for the first thing that comes to mind, try to find the most minimal and better working design. Don't introduce slop: very fragile code that just patches specific cases, dead code, useless code and code ways more complicated of how it should be.
- Comment important inference code where the model mechanics, cache lifetime, memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is intentional.

## Layout

- `src/`: Zig owners, common C/H interfaces and retained C model implementations.
- `src/engine/`, `src/models/`: DeepSeek execution and model request translation.
- `src/server/`, `src/agent/`, `src/serving/`: frontends and shared protocol types.
- `src/cli.zig`: command line, REPL and interactive transcript handling.
- `src/metal/`: Zig controllers, Objective-C driver and compute shaders.
- `src/cuda/`, `src/rocm/`: retained upstream backend sources, not qualified by
  the macOS Metal build.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments and old planning material.

This list is not complete, check the files for more info.

## Testing

Use `make` for build validation and `make test` for model-free host/regression
tests. Run model/GPU fixtures separately under the existing pressure guard. Use live server tests only when intentionally
testing the API surface.

At every major change where one of the following could be affected, make sure to:

1. Test the normal Metal path and that speed is still at the level it was.
2. Test the SSD streaming path.
3. Test the distributed inference if it could be affected, but ask the user before doing so.
4. Check if CUDA could be broken after the change, and ask the user to give you access to the CUDA machine to actually test if everything is still fine.
