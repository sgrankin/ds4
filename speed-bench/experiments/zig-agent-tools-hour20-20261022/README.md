# Native streaming tool parser and call ownership

Completed 2026-10-22, inside the one-hour ceiling. Base `b9a4c9b9`.
The four-syntax streaming parser and call/argument ownership now use handwritten
Zig. One small shared header defines the actual existing records; the C worker,
renderer, detector, sampler, preflight checks and tool execution keep their
current interfaces. The source remains a cohesive parser rather than a translated
agent/SDK dump. Existing C implementations remain selectable.

Original rules remain explicit: strict openings/tolerant DSML closes; online
partial-close detection; literal control spellings inside values; one-newline
Qwen trimming and its existing strtod/bracket typing heuristic; GLM strings;
move-and-zero call ownership; retained raw diagnostics and syntax after reset.
The parser preserves C NUL-sensitive searches where required, while bounded
byte operations remain bounded. Shared delimiter unescaping stays in its real
existing header; no new strict JSON policy was substituted.

Both arms build all five programs. The existing production-including agent
fixture passes **C/native × Debug/ReleaseFast**, including all four syntaxes,
bytewise and every-two-chunk split cases, partial/EOF behavior, literal control
markers, adjacent calls and malformed argument cases. The same fixture also
checks file tools, compaction boundaries, streaming renderers and terminal
helpers. No new parser-test framework was introduced.

The initial native Debug run exposed a real C/Zig boundary mistake: forming
an empty slice from NULL args/calls traps in Zig although the original zero-trip
C loops never dereference it. Three count guards fix cleanup and empty argument
lookup. The existing fixture catches this without another mirrored test.
The [original failure](initial-debug-failure.txt), initial identities and final
passing results are retained. ReleaseFast parity alone was insufficient here.

Decision: **retain**, pending real combined-agent qualification below.
`tools/zig-build agent-test` builds the existing fixture against the selected
production parser object. [Commands and source/binary hashes](results.json),
[build/test loop](qualify.py). Actual archive retrieval and edited-context suffix
reuse are checked in the combined report; these CPU tests make no agent latency
claim. Full logs remain under `.workspaces/zig-agent-tools-20261022/`.
