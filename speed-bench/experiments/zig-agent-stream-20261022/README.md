# Item74b — Native agent stream projection

The handwritten Stream owner controls terminal projection of live tool calls,
short split-marker buffering, thinking/tool suppression, raw malformed-stanza
evidence, read/write/edit display, final flush and grammar greedy/lookahead
decisions. It composes pure Presentation directly through one shared generated
record binding. The actual Viz/detector/Stream records move once into a header;
Parser and Renderer are borrowed. Sampled transcript bytes and tool execution
remain authoritative in their existing owners. Worker stays opaque behind three
whole actions: trace, read-line policy and edit-old preflight.

Integration/qualification17:10–17:15UTC starts at e818fa47. C Stream controls
retain native Presentation and the qualified Engine. C/native ReleaseFast and
Debug, plus the full C fallback, all pass the unchanged actual Agent fixture,
terminal/hint/PTY/code-screen helpers and suffix/Context transaction fixture.
All14 stable schema/ANSI/screen artifacts match bytewise across all five
selections. Concurrent PTY output is checked through its existing screen
assertions, not byte identity. Exact commands, hashes, stdout/stderr and build
records are retained in results.json and evidence/.

Existing production tests exercise arbitrary byte splits for all tool syntaxes,
split opening/closing tags, hidden-thinking attempts, malformed raw evidence,
immediate long write/edit visualization, preflight order, hint/UTF8 boundaries,
greedy grammar sampling and lookahead. No new test body or alternate Stream
implementation was added. The same terminal helper used in Item74 is reused:

```
tools/zig-build agent-test agent-context-suffix-test -Dnative=true \
  -Doptimize=ReleaseFast --prefix .workspaces/stream-repeat/native
.workspaces/zig-terminal-venv/bin/python -B \
  speed-bench/experiments/zig-agent-presentation-20261022/qualify.py \
  .workspaces/stream-repeat/native/bin/ds4_agent_test \
  .workspaces/stream-repeat/native-terminal-output
mkdir -p .workspaces/stream-repeat/native-suffix-files
.workspaces/stream-repeat/native/bin/test-agent-context-suffix-mock \
  .workspaces/stream-repeat/native-suffix-files
```

First source formatting exposed a C field named error, which requires Zig's
quoted identifier spelling. First compilation then rejected a local shadowing
Stream.start and a vararg pointer branch without a common type. These spellings
were corrected; final ReleaseFast/Debug/fallback builds are clean. Renderer
adapters now compile only when C Stream or direct fixtures need them; two shared
fallback byte helpers are explicitly marked unused. No global warning suppression
was added. Failed compilation and successful corrected records remain.

The native package selects Stream with Presentation/Tools. Make retains the
original C implementation over the same records. Direct composition removes
per-byte C renderer crossings; no runtime speed gain is claimed. Actual whole
Agent archive/edit/retrieval generation follows the remaining tool-owner changes
under the existing single-model pressure guard. Dispatcher, Worker, history,
private-summary transaction and TUI remain subsequent owners.
