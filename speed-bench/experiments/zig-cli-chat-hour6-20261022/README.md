# CLI transcript and session ownership

Completed 2026-10-22 04:19 UTC, within the one-hour ceiling. Base is Engram
commit `a1cae6f5` on `zig-modular`.

The private `ds4_cli_chat.h` record replaces the identical inline C record.
Six handwritten exports in `zig/cli_chat.zig` own image-vector lifetime,
thinking-prefix replacement, session creation/replacement and final cleanup.
The module imports actual public engine/token/image types through standard
compiler-cached header bindings. Its private token edits use slices, forward/
backward copies and explicit foreign-vector reallocations. Comments preserve
prefix placement, rejection-before-mutation and the distinct session-failure
contracts. C retains initialization/prefill callbacks, terminal handling,
signals, configuration and generation. Two narrow compile guards preserve the
original C helper implementations for `-Dnative=false` and Make.

All five C/native production programs build. The actual existing CLI fixture
passes against the real 1017 vocabulary in C/native × Debug/ReleaseFast:
**four successful model-opening runs**. The existing transitions cover seven
thinking modes with/without system text and rejection when context is full.
Small additions verify zeroed image spans, offset changes, and complete image/
transcript preservation on rejection. No image encoder or fake engine is used.

Two actual piped REPLs then start at context 8192, change the thinking prefix
with `/think 0`, generate `OK`, use `/ctx 4096` to replace the live session, and
recall **PINE-251** from the retained transcript. C and Zig produce identical
stdout. A narrow subprocess sandbox denies writes only to the user's history
file, so fixture commands do not pollute terminal history. All model runs are
serial under the existing pressure guard.

C/Zig in-engine prefill rates are 10.42/10.52 tokens/s for the first turn and
29.13/29.22 for the rebuilt second turn; generation rates also match closely.
Whole-process times vary (4.77/7.37 seconds) and are not a language-port speed
comparison. The earlier Engram ABBA comparison provides the longer timing screen.

Two preparation failures are retained in `.workspaces/zig-cli-chat-20261022/`:
the first build exposed a pointer-type error in the unused future GGUF metadata
draft, corrected before these binaries; the first REPL check expected `none`
while the CLI correctly labels numeric effort zero as `0`. Its conversation
already succeeded. The fixed check and complete rerun pass. The metadata parser
is not activated in these binaries.

Decision: **retain** this CLI boundary. [Recorded loop](qualify.py) and
[results](results.json) include commands and binary hashes. Build the fixture
with `tools/zig-build cli-chat-test`, then run `test-deepseek41-cli MODEL` under
the pressure guard. `-Dnative=false` selects the C reference and
`-Doptimize=Debug` checks native helper arithmetic and bounds. Build/source/
pressure/stdout evidence is under `.workspaces/zig-cli-chat-20261022/`.
