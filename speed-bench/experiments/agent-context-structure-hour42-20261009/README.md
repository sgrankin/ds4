# Item 42: stable live-context editing handles

`ds4-agent --live-context-file PATH` now exports stable turn IDs and gives the
model short recipes for deleting, archiving and shortening old turns without
dumping the conversation back into its context. Retained IDs do not change;
deleting ID 1 leaves a gap. New turns get larger IDs within the running worker.

Runtime revision: `d29522004f4b7d7405e25416890fc0711a8a8212`.
Budget: 2026-10-09 14:54–15:54 UTC, including implementation, build, checks and
the actual-agent screen. Measurements completed by 15:19 UTC.

```sh
./ds4-local persistent-fast --model gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --live-context-file .workspaces/agent-context.txt --prefill-chunk 2048
```

Restart the agent to receive the new instructions. The chunk override was used
for this qualification; this item does not change the preset's chunk-512 default
or measure a comparative prefill speedup.

## Format and instructions

The mirror keeps its protected system/tool/session prefix and adds file-only
headers before actual user/assistant role tokens:

```text
[[DS4_TURN 7 role=assistant]]
<｜Assistant｜>...native conversation span...
```

These labels and the display escape for literal header-shaped body lines are
removed before evaluation. Unchanged bodies retain their original token IDs,
including ordinary text that spells native control markers. Tool calls/results
may occur within an assistant span. Structured edits preserve explicit IDs;
IDs are worker-local and can change on restart. Raw older mirrors still use
the existing import path.

The model is told that it already sees the conversation: do not cat the mirror
or retype old text into tool arguments. Code may read the file internally while
printing only headers, a small search/range or a short count. Archive useful
evidence, batch edits to specific older IDs once, retain the current request,
then continue after the success receipt. A literal header line has a backslash
display escape, so it cannot masquerade as a boundary. The recipes use Python's
standard `re.split` and file operations.

## Qualification

The model itself identified, archived and deleted an older audit user turn.
The driver did not modify the mirror or give the model its target ID. A later
request required a fresh archive lookup and recall of a current fact.

| Check | Result |
|---|---|
| Focused strict-C11 suffix/mock regression | Pass, no compiler warnings |
| Main agent build and `ds4_agent_test` | Pass |
| Exact embedded deletion recipe | Kept IDs 8/10, dropped 7/9, printed zero bytes |
| Actual model cleanup | Two bounded searches, one archive/delete edit |
| Stable IDs | Before 1/2/3/4; retained 2/3/4 unchanged after deleting 1 |
| Archive | Complete original block byte-for-byte, including 48 audit lines |
| Fresh follow-up answer | `EMBER-7319` from archive; retained `COVE-3917` |
| Read-back loop / automatic compaction / edit rejection | None in completed screen |

The receipt was `Live context edit accepted: 3618 -> 3187 tokens; suffix
reuse=none.` This odd 431-token deletion used ordinary recomputation; it is
not an additional suffix-speed result. Whole four-turn launcher time was
96.47 seconds, without a matched latency baseline. This one cooperative editing
screen demonstrates the intended workflow, not a general guarantee against
loops or lost facts. No new delta-NLL or KL measurement was made.

The first real run exposed a concrete importer bug before cleanup: session-date
text follows the generated system tokens before the first user turn. The
importer initially rejected its own exported prefix on the second request.
`live/` retains that failed run. The fix preserves and validates the complete
unlabelled prefix, with a focused regression; `live-fixed/` reruns the same
scenario, with no generation retry or prompt tuning. The initial driver's
`no_edit_rejection` check looked only at the trace and missed the stderr error;
the corrected driver checks stderr/stdout as well. The original result is kept.

Across both runs, all 129 pressure samples were normal. Swap varied by
0.25 MiB; peak wired plus compressed RAM was 89.22 GiB. Model jobs ran
sequentially under the existing guard. KV/cache files stay in `.workspaces/`.

Commands/settings are in each `execution.json` and guard `command.json`;
source/binary hashes and the independent byte/tool-output audit are in
[audit.json](audit.json). [live-fixed/result.json](live-fixed/result.json),
raw outputs, mirror snapshots, trace, build logs and focused-check commands
are retained alongside this report.

Decision: keep the new structure/instructions behind the existing mirror flag.
Stop this item after the concrete workflow passes; leave archive policy to the
agent instead of adding an automatic storage framework.
