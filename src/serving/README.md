# Serving request / model boundary

`types.zig` defines logical messages, tool declarations and reasoning controls.
It imports no C headers, server records, transport or engine implementation.
Input bytes are borrowed for model preparation; the caller keeps their owner
alive until the operation returns.

The first native model operation is `models/ds4.zig`'s `Model.prepare`:

```zig
const types = @import("serving_types");
const models = @import("ds4_model");
const model: models.Model = .{ .family = .deepseek41 };
const messages = [_]types.Message{.{ .role = "user", .content = "Hello" }};
const rendered = try model.prepare(allocator, .{
    .messages = &messages,
    .reasoning = .{ .effort = .none },
}, .{}, 8192);
defer allocator.free(rendered);
```

The model owns reasoning defaults/aliases, effort handling, prompt separators,
tool encoding, sampled-tool replay and parallel result ordering. DS4.1's pure
renderer lives in `models/deepseek41/prompt.zig`. Other model families retain
their existing runtime adapters; this does not claim native implementations of
those models.

The current server path is:

```text
wire decoder / durable conversation resolution
    -> server/canonical.zig (borrowed native values)
    -> models/ds4.zig + models/deepseek41/prompt.zig
    -> server/model.zig (runtime/tokenization compatibility)
    -> existing scheduler / inference sessions
```

`server/materialize.zig` resolves stored Responses/conversations, restores tool
history and validates live continuation before invoking model preparation.
`server/model.zig` contains C enum/engine detection and allocation handoff. The
wire parser and existing execution/session records still use their established
C ABI. HTTP and [Responses WebSocket](RESPONSES.md) now share this preparation
boundary and the generation/output path.

The canonical bridge uses an arena for descriptor arrays and normalized tool
parameters, borrowing message/image/schema bytes from the decoded request.
Parameter normalization retains the existing JSON codec's accepted grammar,
duplicate order and value spelling. A native caller supplies parsed parameters
or the documented raw-argument fallback; sampled raw tool text stays verbatim.
Destroy borrowed views before mutating or releasing their source records.
Prepared prompt bytes use the caller's allocator; the C adapter uses libc to
match the existing request destructor.

The standalone fixture links no engine, C server objects or Metal:

```sh
tools/zig-build serving-boundary-test
zig-out/bin/test-serving-boundary
tools/zig-build server-test
zig-out/bin/ds4_test --server
```

The native fixture exercises the boundary independently; the existing server
suite checks actual wire decoding and compatibility. Real concurrency, restart
and cancellation use the existing HTTP qualification script. Raw run output is
local under `.workspaces/`; only concise results and reproduction commands are
tracked.
