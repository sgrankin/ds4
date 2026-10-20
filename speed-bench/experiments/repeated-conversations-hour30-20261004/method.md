# Repeated edits through durable Conversations

Compare native-before → reuse → native-after if the hour permits. The candidate alone enables suffix64; canonical1/PAD1 stay fixed across arms. The canonical full-input policy excludes Conversation references; padding remains available when suffix reuse is enabled. Q2/cache64GB, ctx8192/chunk512/two slots, GAP0, selected tile64/limit1024, hot0/SSD4096, temperature0, effort none and stable instructions/read-tool schema are unchanged. Each arm receives a new state cache and response store, but all advertise the same A/B paths. Conversation IDs supply the default identity hint; there is no explicit prompt_cache_key or previous_response_id.

Two independent Conversations each begin with three obsolete User note items of64 lines plus an unchanged inventory contract. Warm A then B by actually reading active.json and answering READY. The retained inventory tool result contains24 meaningful notes and the arithmetic inputs: A23×14−9=313 and B31×8−17=231. Source capture precedes any numeric task answer.

For each of three rounds, delete exactly one original obsolete item by its actual ID, keep every other public item, update the real archive.json and ask a newest User question requiring a fresh read. A remains BEACON-46; B becomes ORBIT-57, ORBIT-58, then ORBIT-59. Grade JSON active_total/current archived_code independently and require exactly one actual read. Store:false Responses still append to the durable Conversation. Tool-call IDs and outputs are replayed by that API, not synthesized by the driver.

The before phase warms both histories and executes edit1 for both. The launcher then shuts down and restarts the same selected server against the same arm-owned cache/store. The after phase executes edits2/3 for each Conversation. Requests are serial despite two configured slots; this is not a GPU-parallel benchmark. Eight total replies and eight fresh file reads are measured. Later rounds may choose original exact sources or persisted approximate sources, or rebuild exactly; source_approximate=1 and source_ssd=1 are credited only when actual graft log lines show them. No forced source, alignment tuning, or rescue of a failed reply is allowed.

Save every request/response, public item listing before/after deletion, actual item IDs and owned-store version, first answer/grade/cap/error, per-turn log slice and restart wall. Version is read from the server's owned flat record because the public API omits it; private-replay presence is recorded, not exported as client input. Both tool and answer requests cap128; reaching the cap fails the workflow while preserving the reply. Actual final text and grade remain separate from cache eligibility. Different generated histories do not support paired KL/NLL or a general forgetting/quality claim.

Example ROOT command, repeated with --arm reuse/native-after and fresh output/cache/store names:

```sh
python3 -B speed-bench/ideas/pressure_run.py speed-bench/experiments/repeated-conversations-hour30-20261004/guard-native-before -- python3 -B speed-bench/experiments/repeated-conversations-hour30-20261004/launch.py --arm native-before --output speed-bench/experiments/repeated-conversations-hour30-20261004/native-before --cache .workspaces/server-agent-day-20261004/repeat30-cache-native-before --store .workspaces/server-agent-day-20261004/repeat30-store-native-before --shared .workspaces/server-agent-day-20261004/repeat30-visible --timeout 1200
```

ROOT owns server/model launches and the existing pressure guard. Preparation is source-only and counts toward the item budget; the author performed AST syntax checks only.
