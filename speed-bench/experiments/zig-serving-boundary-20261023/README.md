# Native serving request / model boundary

Base: `60813d03` on `zig-rewrite`; implementation change `tlsqtyyltvut`.
Main remains `ec5424de5562`. This change
separates logical serving input from DS4.1 preparation. It changes no numerical
execution, scheduler policy or persisted checkpoint format.

## Change

- C-free native message/tool/reasoning records in `zig/serving/types.zig`.
- Model-owned aliases, reasoning policy and preparation in `zig/models/ds4.zig`.
- The actual DS4.1 renderer moved to `zig/models/deepseek41/prompt.zig`; the old
  whole renderer was removed, leaving compatibility calls at the server edge.
- Durable history/live continuation validation stays in materialization;
  descriptor ownership, C mode conversion and tokenization handoff stay in the
  compatibility adapters. Raw sampled tool text remains verbatim.
- Five standalone cases call native modules without C server/engine/Metal
  objects. Existing server fixtures exercise wire decoding and materialization.

This is the request/preparation boundary, not a complete model-independent
execution API. Existing execution/session records and checkpoint interpretation
remain DS4-specific. Responses WebSocket transport is not added here.

## Verification

ReleaseFast and Debug builds succeed. Five native boundary cases and the
existing `ds4_test --server` suite pass in both modes. Model-name alias sets are
unchanged. Source review also compared the DS4.1 tool preamble/footer bytes.

The existing guarded HTTP campaign passes C-writer -> refactored-server restart:
22 checks and five graded answers, including durable Responses/conversation
recovery. The refactored-server -> refactored-server campaign separately checks
its own fresh requests, concurrent decode, disconnect cancellation, surviving
request, streaming protocols and restart. Final results are recorded below.

Settings: 1017 IQ2_XXS/Q2_K imatrix weights, two batched sessions, ctx8192,
prefill chunk512, mixed-prefill quantum512, expert cache32GB, F32 KV, draft1.
Suffix approximation, decoder replay and resident substitution are off. SSD
state cache is disk-only with a 512MiB limit. Each campaign uses the existing
pressure guard and runs serially. This is behavior qualification, not a latency
or speedup claim.

```sh
tools/zig-build --prefix .workspaces/zig-serving-boundary-20261023/native \
  serving-boundary-test server-test ds4-server
.workspaces/zig-serving-boundary-20261023/native/bin/test-serving-boundary
.workspaces/zig-serving-boundary-20261023/native/bin/ds4_test --server

tools/zig-build -Doptimize=Debug \
  --prefix .workspaces/zig-serving-boundary-20261023/debug \
  serving-boundary-test server-test ds4-server
.workspaces/zig-serving-boundary-20261023/debug/bin/test-serving-boundary
.workspaces/zig-serving-boundary-20261023/debug/bin/ds4_test --server

# Choose fresh output/guard directories for each rerun. For a cross-version
# check, replace the first --server with the independently built C reference.
python3 -B speed-bench/ideas/pressure_run.py \
  .workspaces/zig-serving-boundary-20261023/guard-native-http -- \
  python3 -B speed-bench/experiments/zig-server-protocols-http-20261022/qualify_http.py \
  --server .workspaces/zig-serving-boundary-20261023/native/bin/ds4-server \
  --restart-server .workspaces/zig-serving-boundary-20261023/native/bin/ds4-server \
  --output .workspaces/zig-serving-boundary-20261023/native-http
```

Raw build/test logs, HTTP transcripts/results, commands, pressure observations and
cache payloads remain local under `.workspaces/zig-serving-boundary-20261023/`.
Only this report and the standalone source fixture are new tracked evidence.

## Final results

- C -> refactored restart: pass, 22/22 checks, 5/5 graded answers.
- Refactored -> refactored restart: pass, 22/22 checks, 5/5 graded answers.
- Pressure guard completes normally in both runs; no model jobs overlap.
- Campaign elapsed times: 72.68 s cross-version and 61.51 s refactored-only.
  These are ordered functional campaigns, not a performance comparison.
