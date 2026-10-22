# Item74 — Native agent terminal presentation

Handwritten Zig owns the existing streaming Markdown, bounded inline spans,
think-tag formatting, UTF8 grouping, code fences/highlighting/repaint, hints and
tail capture. The actual Renderer/Capture records have one shared definition;
the output dependency is one byte-output callback with opaque Worker userdata.
There is no Worker/Engine mirror or field-access service. Private syntax tables
and scoped native buffers replace C machinery; the original algorithm comments
and fixed language/keyword data remain. C Stream/worker/history still call the
sole presentation facade. They are separate following owners.

Source integration/qualification ran17:04–17:09UTC on the qualified Session
revision83e9da8521de. `results.json` identifies actual source and binary hashes.
The earlier source-prep manifest retains draft hashes; current source hashes in
the results are authoritative. Main did not advance.

All five selections passed the actual production-linked agent fixture, terminal
screen/hints/PTY helpers and model-free suffix/Context transaction fixture:

| Selection | Renderer | Optimization | Result |
| --- | --- | --- | --- |
| control | C, other owners native | ReleaseFast | PASS |
| control-debug | C, other owners native | Debug | PASS |
| native | Zig | ReleaseFast | PASS |
| native-debug | Zig | Debug | PASS |
| fallback | C, all owners C | ReleaseFast | PASS |

Each selection produced14 byte-identical stable artifacts, including six tool
schema prompts, code-repaint ANSI/screen output, hint/UTF8 cases and footer
layout. Concurrent PTY output has scheduling-dependent frame order; each run
passed its existing fragmented navigation/UTF8, resize, concurrent-output and
submission screen assertions. It is not claimed byte-identical. The added
27-line production-output case verifies actual safe-line repaint, multiline
comment state, keyword/number colors, hint badge and UTF8. Existing cases retain
their original assertions and call the selected production code.

Native ReleaseFast/Debug and full C fallback builds are clean. First native
compilation rejected two `color` locals shadowing Renderer.color; they were
renamed without changing behavior. Source review also corrected the forward
declared syntax handle to a nullable opaque pointer before compilation. A first
suffix invocation omitted its required artifact-directory argument and aborted
at argc validation; the corrected command passed in all five selections. These
setup/compile failures and corrected commands remain in evidence.

Reproduce from the repository root with a local Python environment containing
the two pinned terminal-fixture dependencies. Build an explicit fresh prefix:

```
tools/zig-build agent-test agent-context-suffix-test -Dnative=true \
  -Doptimize=ReleaseFast --prefix .workspaces/presentation-repeat/native
.workspaces/zig-terminal-venv/bin/python -B \
  speed-bench/experiments/zig-agent-presentation-20261022/qualify.py \
  .workspaces/presentation-repeat/native/bin/ds4_agent_test \
  .workspaces/presentation-repeat/native-terminal-output
mkdir -p .workspaces/presentation-repeat/native-suffix-files
.workspaces/presentation-repeat/native/bin/test-agent-context-suffix-mock \
  .workspaces/presentation-repeat/native-suffix-files
```

The native package selects DS4_NATIVE_AGENT_PRESENTATION; Make retains the C
fallback. The presentation object links only into Agent and its fixtures, not
the common Engine archive. No model run, latency improvement or broader Worker
migration is claimed here. The whole-agent model workflow follows the next
Stream/tool-owner integrations under the existing one-model pressure guard.
