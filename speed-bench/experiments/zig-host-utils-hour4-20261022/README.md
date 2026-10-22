# SSD configuration and prompt-prefix parsing

Source preparation and qualification completed by 2026-10-22 04:04 UTC, inside
the one-hour ceiling. Base is the GGUF cursor commit `343af170` on `zig-modular`.

`zig/ssd.zig` and `zig/prompt_prefix.zig` replace their C counterparts through
the actual public C records and standard generated bindings. KV, SSD and prefix
share one compiler-cached declaration module. No handwritten SDK layouts or
translated C implementation dump is added. Engram remains C at this point.

The SSD module preserves libc numeric grammar, cleared outputs on failure,
saturating model targets, wrapping effective-byte arithmetic, caller-selected
cache percentages and the existing bounded volatile touch/lock/unwind policy.
The original pressure/headroom rationale is retained. The prefix module keeps
the original BOM/NUL/alternating-turn grammar, whitespace and terminal-newline
rules, partial-allocation cleanup and path/error diagnostics. Standard
`c_allocator.dupeSentinel` produces the C-free-compatible turn strings.

All five C/native production programs build. Existing C SSD/prefix fixtures pass
for C/native × Debug/ReleaseFast: **eight successful runs**, with identical
outputs. The SSD fixture now also protects normal GiB/count inputs, uint32
overflow with output clearing and an 8 KiB lock/release ownership roundtrip.
That is a tiny allocation check; no large memory pressure test was added.

The actual CLI then loads a prefix containing an earlier user/assistant exchange,
accepts the 48 GiB streaming-cache argument and answers a new question from that
prefix. Both implementations return exactly **ORCHID42** with no thinking and
greedy sampling on the 1017 IQ2 model. [Recorded direct subprocess loop](qualify.py)
and [results](results.json) preserve the commands and hashes. Both runs execute
serially under the existing pressure guard. Their 7.95/6.80-second times include
warm-up differences and do not establish a language-port speedup.

Build fixtures with `tools/zig-build ssd-test prompt-prefix-test`; use
`-Dnative=false` for the C reference and `-Doptimize=Debug` for checked host code.
Run the installed `test-ssd-cache` and `test-prompt-prefix` binaries directly.
Complete logs, source hashes and build records are retained under
`.workspaces/zig-host-utils-20261022/`.

Decision: **retain** these two bounded modules. CUDA argument policy, unrelated
placement code and the terminal/agent frontend implementations remain C.
