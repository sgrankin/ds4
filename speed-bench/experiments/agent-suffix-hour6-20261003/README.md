# Item 6: storage-backed idle context edits

The actual agent now imports an externally edited `--live-context-file` before
starting the next user turn. On the longer two-edit workflow, suffix reuse
reduced complete time from **170.522 s to 128.494 s (24.65%)**, including an
exact reconstruction after the second edit. All **13 measured workflows**
passed, with **19 correct graded answers** and actual retrieval from storage.
This is a small lookup/arithmetic screen, not a general accuracy claim.

Budget: 2026-10-03 **15:35:50–16:35:50 UTC**, including preparation, builds,
model runs, analysis and the local commit. The pilot is excluded from timing
comparisons. All runtime approximations remain opt-in.

## Complete workflow results

Each native reference is the mean of two bracket controls. Candidates have one
fixed-seed run each. Whole time includes process startup, earlier turns, context
editing, tools and final answers. Edit-cycle time includes editing through the
next answer; two-edit rows sum both cycles.

| Workflow | Requested fresh-tail minimum | Native whole time | Reuse whole time | Whole saving | Edit-cycle saving | Passes: native / reuse |
|---|---:|---:|---:|---:|---:|---:|
| Short, one edit | 128 | 96.272 s | 80.446 s | 16.44% | 53.87% | 2/2 / 1/1 |
| Long, one edit | 128 | 149.924 s | 102.237 s | 31.81% | 72.59% | 2/2 / 1/1 |
| Long, one edit | 256 | 149.924 s | 140.097 s | 6.55% | 14.18% | 2/2 / 1/1 |
| Short, two edits | 128 | 129.362 s | 110.327 s | 14.71% | 30.51% | 2/2 / 1/1 |
| Long, two edits | 128 | 170.522 s | 128.494 s | 24.65% | 48.23% | 2/2 / 1/1 |

The longer repeated case also saves **23.77% against the faster native control**.
Short cases fall below the 20% whole-time target. These measurements are
separate from item 5's 23.90% private resident-affinity/WRITE-budget result;
they cannot be added together.

[Tradeoff plot](tradeoffs.png), [CSV](results.csv), [complete results](results.json).
The plotted quality axis is strict workflow success; all these points passed.
On that axis, tail 128 dominates tail 256 for the tested long case. That does
not establish equal distribution drift or performance on other tasks.

## What the agent actually did

1. Receive an old audit log containing `RIVER-73`.
2. Read active inventory from JSON with the real `read` tool: 17 batches,
   25 items each, 9 reserved. Short/long variants retain 24/64 irrelevant notes.
3. Complete a readiness turn that retains the calculation instructions.
4. While idle, an external organizer moves the audit fact into `archive.json`
   and replaces the old log with a breadcrumb. The next fresh question asks
   for the active total and the archived code. The agent actually reads the
   permanent file and returns `416` and `RIVER-73`.
5. In repeated variants, remove the earlier inventory notes while preserving
   its three numeric facts, change the permanent file to `COVE-22`, and ask
   again. The agent rereads storage and returns `416` and `COVE-22`.

The removed fact is absent from the edited context before the incoming query.
The driver checks actual tool retrieval, independent expected JSON fields,
normal completion and retention of the incoming question. Fourth and fifth
answers are graded separately. Archival decisions are made by the caller;
this experiment does not implement an autonomous context organizer.

First deletion shifts the suffix by **662 tokens**. Tail 128 reused **740**
tokens in the short case and **1,540** in the long case, replaying **177** old
tokens before the entirely fresh incoming question. Tail 256 reused only
**203**, replaying **1,514**. Available immutable checkpoints make the setting
a minimum, not an exact replay count; this explains the coarse timing step.

Both repeated suffix runs have **one approximate graft**. The second edit
reconstructs exactly from the **1,986-token cached prefix**, evaluating another
**794 tokens**. Its measured next-answer times, 31.373 s short and 31.192 s
long, are included above. Repeated approximation and canonical persistence
after approximation remain limitations of the existing implementation.

Differing generated histories do not support paired teacher-forced ΔNLL/KL.
Those scores are deliberately absent here; the earlier fixed-input screens
measure distribution changes. This screen measures practical agent outcomes.

## Integration and verification

`worker_run_turn()` now invokes the existing live-context importer before
compaction, appending the user request or refreshing the mirror. Previously,
idle edits could be overwritten; import happened only after tool execution.
The existing suffix transaction and rejection behavior are reused. Rejection
is visible and occurs before the new user request mutates the transcript.
No new runtime knob or configuration parser was added.

The existing agent unit suite passes, including one focused test of rejection
ordering. Only the agent frontend and test frontend were compiled; all 14
core/auxiliary objects were reused. [build.json](build.json) records revision,
source/binary hashes, commands and timings. The measured private binary is
`.workspaces/idea-agent-suffix-hour6/build/ds4-agent`; the main `ds4-agent` and
frontend object were updated to that tested build after every model job ended.
The previous binary is retained in the workspace; hashes are recorded in
[main-binary-update.json](main-binary-update.json).

## Settings and reproduction

Q2 GGUF, SSD streaming with **64 GB expert cache**, context 8192, prefill chunk
512, `--nothink --temp 0 --tokens 512 --seed 20261003`. Native resident routing
(`DS4_METAL_V41_RESIDENT_ROUTE_GAP=0`) and selected prefill tile/limit 64/1024.
Shared absolute tool paths and system-prefix cache are identical across arms.

Suffix settings: `DS4_AGENT_COMPACT_PREFIX_CACHE=1`,
`DS4_AGENT_CONTEXT_FRONTIERS=4`, `DS4_AGENT_CONTEXT_FRONTIER_MB=512`,
`DS4_AGENT_CONTEXT_SUFFIX_REUSE=1`,
`DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL=128` or `256`, plus `--live-context-file`.
Native arms change reuse to zero. Every run retains its full command and
environment in `runs/<arm>/execution.json`.

`campaign.py`, `repeat_campaign.py` and `repeat_long_campaign.py` record the
exact orders. Their historical deadline intentionally prevents rerunning them
unchanged. For another screen, run `run.py` or `run_repeat.py` with a **new**
output directory under the existing pressure guard. Analyze retained results:

```sh
.workspaces/bench-plots-venv/bin/python -B speed-bench/experiments/agent-suffix-hour6-20261003/analyze.py
```

All four guard jobs completed normally: **1,762 samples, zero adverse samples**,
peak wired-plus-compressor 86.43 GiB, maximum measured swap growth about
**0.87 MiB**. High RAM usage was expected. See [pressure summary](pressure-summary.json).
Initial sandbox pressure-probe failures occurred before a model started and are
retained in [preflight observations](preflight-observations.json).

The excluded pilot warmed the system-prefix cache and checked edit geometry;
its startup and an earlier selected-prefill environment spelling differ from
the measured campaign. [Pilot note](pilot-note.json) retains that distinction.

The next useful step is a small named preset interface, with the missing
resident/thinking code integrated first where required. See the
[packaging inventory](packaging-inventory.md).
