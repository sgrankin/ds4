# Build and test

The supported build is native macOS Metal on Apple Silicon. It uses Zig 0.17,
Apple Clang and the host macOS SDK; project scripts have been tested with
Nushell 0.116.1. The build selects the host deployment target; older targets
and other platforms are not qualified by this build.

```sh
xcode-select --install
./tools/zig-build
zig-out/bin/ds4 --help
make
./tools/zig-build -Doptimize=Debug --prefix .workspaces/zig-debug
```

The wrapper installs `ds4`, `ds4-agent`, `ds4-server`, `ds4-bench` and
`ds4-eval` in `zig-out/bin`. A program name selects one target. `make` delegates
to the same build and copies the five programs to the repository root for
existing launchers. The wrapper keeps Zig caches under `.workspaces/zig-build`;
`-Dsdk=PATH` overrides SDK discovery. Debug applies to native owners and
fixtures, while residual C and Metal objects retain their established release
and numerical flags. Assertions remain enabled. `make clean` removes installed
outputs but retains incremental caches.

Use the installed directory with presets, for example:

```sh
./ds4-local --bin-dir "$PWD/zig-out/bin" persistent-fast --model MODEL
./ds4-local --bin-dir "$PWD/zig-out/bin" --server persistent-fast --model MODEL
```

`persistent-packed` aliases `persistent-fast`; see the [packed index recipe](../../recipes/14-packed-index-cache.md).
Model metadata selects its native implementation. [Source organization](../../src/README.md)
explains the engine and platform boundaries.

## Focused checks

```sh
make test
make test-scripts
make test-frontends
make test-metal
./tools/zig-build -Doptimize=Debug server-test agent-test
```

`make test` runs model-free host, protocol and script checks; it does not load
weights or run GPU latency experiments. `make test-metal` runs focused Metal
kernel checks. Named Zig fixture targets build their binaries; run those binaries
separately. For example `./tools/zig-build session-state-test` builds
`zig-out/bin/test-session-state`. Make's individual aliases build and run their
fixtures. See `make help` and `./tools/zig-build --help` for current targets.

A fixture is not full-model or hardware qualification. Use matching checkpoint
continuations from [test vectors](../../tests/test-vectors/README.md), and the
[quality scorer](../../gguf-tools/quality-testing/README.md). Original Flash and
GLM 5.2 vectors are historical, not gates for V4.1 or GLM 5.3. State changes
need rewind, replay, save/load, images and multiple-session coverage where
relevant; distributed changes need both ranks and failure cases. The
[release QA guide](qualification/README.md) guides selection and reporting of checks; a release-specific QA report owns
results, failures and skipped checks. Run model-opening checks serially under the
[pressure guard](benchmarking.md#guarded-execution).

## Diagnose output

```sh
./ds4 --dump-tokens -p "..."
./ds4 --dump-logprobs .workspaces/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4 --dump-logits .workspaces/logits.json --nothink --prompt-file prompt.txt
./ds4-server --trace .workspaces/ds4-trace.txt
```

Token dumps can reveal template differences; logits and continuations help
separate sampling from graph errors. Server traces include cache and tool-parser
decisions and may contain private conversations. Other tools include
[GGUF conversion](../../gguf-tools/README.md), [imatrix collection](../../gguf-tools/imatrix/README.md),
[steering](../../dir-steering/README.md), and [evaluation data](evaluation-data.md).
