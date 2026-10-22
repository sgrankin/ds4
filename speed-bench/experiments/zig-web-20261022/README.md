# Native Web helper

The complete shared Web helper now selects handwritten Zig: browser/profile and consent flow, navigation/tool JS selection, HTTP/WebSocket/CDP transport, tab lifetime and recovery. Existing private Web/tab records and literal JS programs live once in shared C declarations/data. Four existing public entry points retain their ABI; native builds omit the old C implementation object.

The unchanged existing local recovery fixture passes with C, native ReleaseFast and native Debug. It exercises real HTTP/WebSocket framing, missing-window fallback to the HTTP new-tab endpoint, last-tab retention, close behavior and missing-target error reporting. Initial native compilation found a parameter shadow, a defer-block syntax error and missing standard-library fixture includes. Corrections and first diagnostics are retained. No transport algorithm or expected assertion changed.

This qualification uses a local fake CDP server. It does not claim an actual Chrome process, interactive consent or arbitrary website behavior. All commands, source hashes and logs are in [evidence](evidence/); useful binaries remain under `.workspaces/zig-web-20261022/`.
