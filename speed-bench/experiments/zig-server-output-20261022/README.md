# Native Responses output lifecycle

Item 70 qualifies Responses streaming/nonstream output as a separate owner
from the [HTTP/input/coordinator/tool-memory checkpoint](../zig-server-20261022/README.md).
The C-output control, selected native output and explicit all-C fallback pass
the existing server fixture in ReleaseFast and Debug. The first native build
was clean. The actual HTTP workflow also passed all five generated-answer and
eighteen structural checks. No performance claim is made.

## Boundary and qualification

The [native output owner](../../../src/server/output.zig) owns Responses item
identity/order, reasoning/text visibility, output JSON and streaming lifecycle.
The worker retains generation and socket activation policy. The existing
durable publisher and tolerant tool-argument codec remain complete C library
actions. Actual buffer/stream/tool-item records are shared once through
[the output header](../../../src/ds4_server_output.h); no Engine/Session layout is
copied. Native output must durably publish before reporting successful
completion.

The retained C controller is built after the real shared-record extraction and
three focused scenarios were added to the existing server fixture, before
selecting the native output flag. It uses the previously qualified native
engine/server owners; `-Dnative=true` does not yet imply native Responses output.
The [recorded build](evidence/control/control-build.json) completed in 4.333 s
with empty stdout/stderr. That cached build time is not a clean-build comparison.
Its binary directory is `.workspaces/zig-server-output-20261022/control/bin`.

The existing `ds4_test --server` fixture ran against C/native ReleaseFast and
Debug selections and the explicit all-C fallback. Exact commands, successful
first builds and the selected source snapshot are retained below. The focused output scenarios protect
concrete stream/item behavior rather than introducing a second parser or
replacement server harness.

The model-backed screen reuses the already published
[self-contained HTTP qualifier](../zig-server-20261022/qualify_http.py), including
its corrected conversation-item DELETE contract. It checks overlapping
Responses clients, real SSE cancellation and survivor completion, durable
restart/continuation and CRUD. No new HTTP harness is added. Root owns serial
builds and all guarded model/GPU work; this report author only copies bounded
small commands, wire evidence, diagnostics and outcomes. Cache chunks, backing
store directories, weights and executable binaries remain ignored.


## CPU qualification

All six production-linked `ds4_test --server` invocations passed. Each includes
three focused Responses output scenarios through the actual selected owner:

- A partial UTF-8 reasoning chunk followed by natural `</think>` closure and
  visible `OK` text, with reasoning-summary completion before item completion.
- Unfinished reasoning, which remains an incomplete reasoning item and emits
  no visible output-text part.
- Visible text plus a tool call, preserving its stable call ID, empty initial
  arguments, arguments-delta/done order and original argument-key order.

Every scenario also checks a completed terminal response and contiguous event
sequence numbers. The C-output control passed these same assertions before
native selection. These are actual socket-pair/framing scenarios within the
existing fixture, not a replacement output implementation or model workload.

| Selected output | ReleaseFast build / fixture | Debug build / fixture |
| --- | ---: | ---: |
| C output with previous native owners | 4.333 / 0.140 s | 0.874 / 0.179 s |
| Native output | 5.869 / 0.135 s | 1.517 / 0.182 s |
| Explicit all-C fallback | 4.622 / 0.139 s | 0.596 / 0.164 s |

All builds returned zero on their first attempt; no compile failure was
encountered in this item. These are cached development-build and single
CPU-fixture process times, not matched build performance or server latency.
The exact commands and raw build/test logs are retained under
[evidence/control/](evidence/control/), [evidence/native/](evidence/native/) and
[evidence/fallback/](evidence/fallback/). The
[seven-file native snapshot](evidence/native/native-source.json) records the
qualified output owner, actual shared header, C seam and build integration.
The separate real HTTP workflow completed as recorded below.


## Actual HTTP qualification

The [existing published workflow](../zig-server-20261022/qualify_http.py) passed
all five generated-answer checks and all eighteen structural checks with
native Responses output selected. Both server processes exited normally and
used the same executable, SHA-256
`9812023485dfacf801ecef4922494680b07fb36259224c9fa0bac2b96e1d0801`.
The full [result](evidence/http/result.json), commands/environment, wire
requests/responses and raw server/trace logs are retained under
[evidence/http/](evidence/http/).

The first process served two overlapping Responses clients and logged actual
two-row decode batches. A separate SSE client received a generated text delta
before disconnecting; the trace recorded cancellation after two generated
tokens while the surviving request completed correctly. The second process
restored exact stored response bytes and conversation items, then continued
both `previous_response_id` and conversation references. Its log records exact
disk-prefix hits of 478 and 542 tokens. Response retrieval/deletion,
conversation/item CRUD and a missing-item GET after deletion all passed.

| Generated response | Output tokens | Input / cached tokens | Grade |
| --- | ---: | ---: | --- |
| Initial A: IVY-46, count 37, audit 0..15 | 47 | 478 / 0 | Pass |
| Initial B: CORAL-82, count 231, audit 0..15 | 47 | 478 / 0 | Pass |
| Surviving B after other client's cancellation | 12 | 542 / 478 | Pass |
| A continues after restart | 12 | 542 / 478 | Pass |
| B conversation continues after restart | 12 | 571 / 542 | Pass |

All five responses completed with zero reasoning tokens and below their
128-token caps. The phases took 47.521 and 8.058 s. These ordered native
functionality observations are not a C/native speed comparison. The generated
model screen uses simple non-reasoning JSON facts; the separate CPU scenarios
protect reasoning/tool-call lifecycle behavior without claiming broader model
quality from those synthetic inputs.

The model settings were the same bounded 1017 IQ2_XXS/Q2_K target with a
32 GB SSD expert-cache target, context 8,192, prefill chunk/mixed quantum 512,
two batched-session slots, MTP width 1 and temperature zero. Checkpoint hot
cache was zero and disk budget 512 MB. Suffix reuse, replay, compressed-KV F16
and resident-route approximations were off. Reproduce with a fresh output
using the already published script:

```sh
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-server-output-http-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-server-20261022/qualify_http.py \
    --server .workspaces/zig-server-output-20261022/native/bin/ds4-server \
    --output .workspaces/zig-server-output-http-rerun
```

The [guard](evidence/http/guard/result.json) completed with child return code
zero. All [54 pressure samples](evidence/http/guard/pressure-summary.json) were
normal with zero swap growth; peak wired/compressed memory was
42.990/10.312 GiB. The [seven selected source files](evidence/native/native-source.json)
were [verified unchanged](evidence/native/source-verification.json) at this
checkpoint. Only small synthetic wire/log/result evidence is checked in;
cache/store backing files, model weights and executables remain ignored.

This qualifies the native Responses output lifecycle on the retained engine
and server package. The durable publisher, tolerant tool-argument codec and
other explicitly unported C policy remain libraries; this is not a claim that
the entire server has been rewritten.
