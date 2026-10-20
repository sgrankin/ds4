# Responses storage and continuation, hour 11

The first guarded run demonstrated durable public response retrieval and private history replay across a server restart. Two generated answers failed their instruction-marker golds; those failures remain part of the result. This is protocol and storage qualification on a small fixture, not a general model-quality or exact KV-cache guarantee.

Revision: `ceab3478a59c` (MAIN runtime `b57dd59f28d9`). The [launcher](launch.py) ran the server, [HTTP checks](run.py), a server stop/restart using the same response directory, and the after-restart checks. Exact server commands and environments are in [before-execution.json](first/before-execution.json) and [after-execution.json](first/after-execution.json).

Settings: Q2 DS4.1 Flash, context 8192, prefill chunk 512, two batched sessions, requested expert cache 64GB, selected tile 64/limit 1024, resident routing disabled. The exact state cache was disk-only (`HOT_MB=0`, `SSD_MB=1024`). `DS4_SERVER_CONTEXT_SUFFIX_REUSE=0` throughout this run. Response storage used `.workspaces/server-agent-day-20261004/responses`.

## First result

| Phase | PASS | FAIL | ERROR | SKIP | HTTP-check time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before restart | 25 | 2 | 0 | 0 | 84.422 s |
| After restart | 13 | 0 | 0 | 0 | 37.354 s |

These counts mix protocol assertions, stored-object comparisons, and generated-answer checks; they are not a task-accuracy benchmark. The times exclude server startup/restart. The phase runner returned 3 before restart and 0 afterward. The outer [guard result](first-guard/result.json) is `complete`, child return code 1 because the launcher retained the two failed gold markers, not because of a crash or pressure stop.

The actual checks and answers are retained in [before.json](first/http/before.json) and [after.json](first/http/after.json), with the authored facts in [fixtures.json](first/http/fixtures.json).

The two failures were:

- `fork-new-count`: memory `MEM_1aa28a40735f5edd` and the updated count **82** were correct, but the answer used `OLD_8b944ca176` instead of `FORK_8b944ca176`.
- `tool-output-continuation`: the same memory and tool-provided count **61** were correct, but the answer used `OLD_8b944ca176` instead of `TOOL_FINAL_8b944ca176`.

Both responses were HTTP 200, completed, and uncapped. GET returned those exact failed answers. A wrong marker by itself does not distinguish a model copying an earlier assistant answer from a replay/cache defect; the control follow-up below addresses that question. Parent and sibling stored objects remained unchanged after the fork.

## Storage and restart checks

The run passed exact GET comparisons for nonstream and SSE responses. SSE's created and terminal response IDs agreed, and the retrieved terminal output matched. `store:false` produced no retrievable record. DELETE was followed by 404. A deliberately capped response was stored and retrievable as `incomplete`; using that response as a parent was visibly rejected.

The low-effort private parent had **323 characters of actual private reasoning** in its stored replay, while its public output contained no reasoning item. This check passed both before and after restart. After restart, a child correctly returned `PRIVATE_1d1930821cc6fc85`, count **19**, and its new marker.

A function call pending at shutdown was retrieved unchanged after restart. Supplying its matching tool output then produced the correct remembered memory, count **73**, and `AFTER_TOOL_8b944ca176`. The existing child was still retrievable after deleting its parent; continuing from that child also passed. Each child stores its complete replay history, so deleting the parent does not remove the child's history.

The guard recorded **122 normal pressure samples**, no adverse samples, and **zero swap growth** (swap stayed 0.012998 GiB). Peak wired plus compressed memory was **84.982 GiB**. The [pressure log](first-guard/pressure.jsonl) retains individual samples. Large memory use did not trigger a stop.

## Implemented API boundary

Set `DS4_SERVER_RESPONSE_STORE_DIR` to enable the local file store; its parent directory must already exist. With storage configured, omitted `store` defaults to true. Without it, the legacy default is false and an explicit `store:true` or durable parent request returns a configuration error. Files contain the actual public response JSON plus private replay; GET exposes only the public object. Files are atomically replaced, mode 0600 in a mode 0700 directory, and limited to 64 MiB.

Supported durable inputs are text and function tools. `previous_response_id` prepends the saved logical history to new input, supports forks, and does not inherit the parent's top-level `instructions`. Prior assistant answers and input-authored system/developer messages remain history. Native reasoning and raw DSML tool-call text are retained privately. Conversations objects, encrypted reasoning, and stored image replay are outside this implementation. GET may retrieve incomplete objects; parent chaining currently requires a completed response.

Durable history survives model-session and KV-cache eviction, but a restart may require fresh model evaluation. This run does not establish byte-identical hidden state or generated tokens after a restart.

Suffix edits use the separate opt-in full-input path from hour 10. A standard durable parent request appends/forks history and skips the suffix edit selector. Approximate live reuse is restricted to the recorded parent lineage and scoped client hint; durable requests bypass the call-ID-only continuation shortcut so current instructions/tools are rendered. A cache hint alone does not enable approximation.

## Full-input versus durable-parent control

The completed [control record](control/http) compares greedy (`temperature=0`) full-history input with `previous_response_id`, using the same current instructions within each pair. ROOT cloned the saved child, pending-tool, and private-parent records into a separate control store. The original parent remains deleted; the first run and its failed answers were not repaired.

| Control | Full-input tokens | Parent-ID input tokens | Generated text agreement | Gold result |
| --- | ---: | ---: | --- | --- |
| Changed instructions/count, saved-child anchor | 158 | 158 | Exact | Both wrong marker |
| Changed instructions/tool output, pending-tool anchor | 396 | 396 | Exact | Both correct |

In the count pair, both paths returned the correct memory and count **82**, but copied `NEW_8b944ca176` rather than the requested `FORK_8b944ca176`. Thus this marker error occurs through the ordinary full-input path as well as durable replay. In the tool pair, both returned the correct memory, count **61**, and `TOOL_FINAL_8b944ca176`. All four HTTP responses were 200 and completed.

These controls give no evidence of inference API corruption from the first wrong markers. They support baseline model behavior as the explanation for the count-marker error; sampling could explain the earlier tool-marker failure, but these two pairs do not prove that cause. The count control uses the saved-child history, rather than repeating the original parent-anchored fork. Exact text and matching token counts here do not establish full hidden-state identity or general answer quality. Control timings remain diagnostic.

The [control guard](control-guard/result.json) also completed with child return code **1**, retaining the failed count golds. Both guards' nonzero returns reflect quality checks, not runtime crashes or pressure stops. Its [pressure log](control-guard/pressure.jsonl) has **33 normal samples**, zero adverse samples, zero swap growth, and peak wired plus compressed memory **84.550 GiB**.
