# Item75e — Anthropic envelopes and content blocks

Handwritten Zig owns complete nonstream Anthropic messages and the live
message_start → thinking/text/tool block → message_delta/message_stop lifecycle.
It appends eight real exports to the sole protocol facade, replacing689 selected
C function lines. Actual records and the shared byte/UTF8/ID helpers retain one
implementation; OpenAI source stays byte-identical. Block/signature/failure
boundaries remain explicit. Model generation, sampled transcript and durable
publication keep their existing owners, with the C fallback intact.

Integration17:36–17:37UTC starts atce34f9f9. Otherwise-native C controls,
native ReleaseFast/Debug and full-C fallback all build cleanly and pass the
full production server fixture. One existing-suite socket case now directly
checks V4.1 syntax lifetime, a split UTF8 argument, one live tool block, its
preserved ID and terminal message_stop. It passes C-first and both native modes.
No compilation/runtime failures, weakened assertions or model/speed claims.
Exact commands, outputs and source/binary hashes are retained. Aggregate real
HTTP generation/restart follows with all new server owners frozen together.

```
tools/zig-build ds4-server server-test -Dnative=true -Doptimize=ReleaseFast \
  --prefix .workspaces/anthropic-repeat/native
.workspaces/anthropic-repeat/native/bin/ds4_test --server
```

Use -Doptimize=Debug for runtime checking and -Dnative=false for full C.
