# Conversations: durable items and continuation

The corrected qualification passed **45 mixed protocol/model checks** (29 before restart, 16 after). All seven ordinary generated-turn checks passed: six independently specified JSON answers and one pending function call. The concurrent-edit probe returned the expected HTTP 409 and preserved the manual append; this is a protocol result, not a quality check. Both driver phases and the pressure guard exited 0.

Runtime `3fb450f7`, implementation `0392c260`, driver `20ade780`. The earlier query-stripping failure and interrupted run remain in [first-screen.md](first-screen.md); they are separate from this corrected run.

## What was exercised

- Create/get/delete conversations; list/get/append/delete items, ascending/descending pagination and `after` cursors; response output IDs appended exactly once.
- Responses accepts a conversation string or `{id}` object, mutually exclusive with `previous_response_id`. Current instructions are request-local. Conversation ID provides default session affinity.
- `store:false` still persists conversation items and private replay while Response GET returns 404. Pending function calls survive restart; a subsequent function output completed with the independent count 61.
- Deleting a stored Response file leaves its conversation usable. A second conversation retained its independent CEDAR facts across restart.
- Manual item edits discard private replay. The obsolete log alone was deleted in this corrected run; the active MAPLE facts remained. Approximate suffix reuse is globally opt-in and can retain older conditioning despite the public edit.
- Deleted conversations return GET 404 and reject Responses references with 400; retained item listing returns 200. The latter is a locally documented extension.

Text messages, function-call/output items and native textual reasoning are supported. This effort-none run exercised text and functions, not generated reasoning. Metadata updates, custom item types and multimodal items are unsupported. Only completed responses publish conversation state.

Conversation publication and Response-file publication are two separate atomic writes. A later Response-file I/O failure can return HTTP 500 after advancing the conversation; this experiment does not provide a transaction across both files.

## Observed suffix activation and timing

| Phase | Observed grafts | Reused tokens | Sources |
| --- | ---: | --- | --- |
| Before restart | 2 | 476, 498 | SSD, native source |
| After restart | 2 | 531, 549 | SSD, approximate source |

The [phase suffix records](qualified/before-suffix.json) and [after-restart records](qualified/after-suffix.json) contain exact source/destination ranges and fresh-tail sizes. Successful item deletion is separate from graft activation. Before/after client phases took 54.403858 / 36.539976 seconds; server startup took 0.214051 / 0.416859 seconds. These are unpaired phases with different work and histories, so they establish no speedup.

All **92 pressure samples were normal**, with zero adverse samples and zero swap growth. Swap stayed at 0.012998 GiB; peak wired plus compressor memory was 82.615448 GiB. The [guard result](qualified-guard/result.json) is `complete`, child return code 0.

## Reproduce and inspect

ROOT ran the following child under the existing pressure guard, from the repository root:

```sh
/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python -B \
  speed-bench/experiments/conversations-hour13-20261004/launch.py \
  --output speed-bench/experiments/conversations-hour13-20261004/qualified \
  --cache .workspaces/server-agent-day-20261004/conversations-qualified-cache \
  --store .workspaces/server-agent-day-20261004/conversations-qualified-store
```

The launcher uses SSD hot RAM 0, two slots, context 8192, native routing, suffix reuse enabled with fresh tail 128, temperature 0 and no reasoning. It starts two owned server lifetimes; the same durable store/cache serve both. Seed is neither set nor claimed.

Exact server argv/settings are in [before-execution.json](qualified/before-execution.json) and [after-execution.json](qualified/after-execution.json). Full requests, replies, text, IDs, caps and checks are in [before.json](qualified/http/before.json) and [after.json](qualified/http/after.json). [summary.json](summary.json) retains calculated counts, phase times, observed grafts, the exact guard command and pressure metrics. This is one semantic qualification, with no native timing pair or general accuracy claim.
