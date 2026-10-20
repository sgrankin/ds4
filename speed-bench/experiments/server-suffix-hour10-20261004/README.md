# Server suffix reuse: ordinary full-input clients

Item10 completed two matched native→reuse screens within its one-hour budget
(deadline 13:53:40 UTC). All twelve responses were correct and completed with
no caps. These repeat the same three questions; they are not twelve independent
tasks. Server reuse is opt-in; defaults remain native.

Each screen sends full Responses input with a stable `prompt_cache_key`,
`store:false`, temperature 0, seed 20261004, effort none and cap 32. The long
reference has obsolete notes followed by active facts and 96 unchanged handbook
entries. A source request asks READY; fresh edited-turn questions ask for
`17*25-9=416` and the retained archive code MAPLE-42/reserve 73. Actual prior
assistant replies are replayed. Golds come from arithmetic and literal facts,
not a native model answer. No `previous_response_id` or remote conversation
store is needed.

Both server policies use context 8192, cache 64GB, chunk 512, selected tile 64/limit 1024,
native routing, two resident session slots and separate disk-cache directories.
The state cache is disk-only (`HOT_MB=0`, `SSD_MB=1024`). Reuse enables
`DS4_SERVER_CONTEXT_SUFFIX_REUSE=1` and `DS4_SERVER_CONTEXT_SUFFIX_FRESH_TAIL=128`.
ROOT ran one server job at a time under the existing pressure guard. The HTTP
runner never launches a server or opens the model.

| Cohort / metric | Native seconds | Reuse seconds | Saving |
| --- | ---: | ---: | ---: |
| Cold48: source warmup | 85.101 | 87.449 | — |
| Cold48: edited turns | 121.596 | 77.713 | 36.09% |
| Cold48: all three requests | 206.696 | 165.161 | 20.09% |
| Warm47: source warmup | 0.651 | 0.758 | — |
| Warm47: edited turns | 118.580 | 44.818 | 62.20% |
| Warm47: all three requests | 119.231 | 45.576 | 61.78% |

Totals sum HTTP request times and exclude server startup. Source warmups are
shown separately and excluded from each edited-turn ratio. Exact values,
guard commands and pressure statistics are in [summary.json](summary.json).

The cold48 screen at `a892cf0957f0` removed 48 arrival lines then 32 maintenance
lines. The first deletion shifted the anchor by 809 tokens (odd), so it safely
fell back. The second edit grafted 1568 tokens from a native SSD source. Its
20.1% total saving remains a separate result; [first-screen.md](first-screen.md)
retains its fallback and timings.

The warm47 screen at `a56347a8a9a3` changed the first deletion to 47 lines,
keeping both policies' inputs matched. Both source warmups restored an existing
3114-token disk endpoint (confirmed by server logs). Those pre-tail-policy
endpoints could not retroactively capture an original archive at 2986; the first
graft still used the available source ending 2048. The source responses' usage
reports cached_tokens 0 despite those logged exact hits; original fields are
retained unchanged.

| Warm47 phase | Native seconds | Reuse seconds | Actual reuse |
| --- | ---: | ---: | --- |
| First edit | 66.045 | 38.043 | 1220 tokens, source 828:2048→destination 36:1256 |
| Second edit | 52.534 | 6.775 | 1732 tokens, source 496:2228→destination 52:1784 |

The first source was native (`source_approximate=0`); its 1100 fresh tokens were
972 during the graft stage plus the final 128. The second source was already
approximate (`source_approximate=1`), loaded from SSD; its 165 fresh tokens were
37 during the graft stage plus the final 128. Both newest user questions
remained fresh. [Native47 results](native47/http/result.json),
[reuse47 results](reuse47/http/result.json).

All four guards completed with child 0, pressure level 1 throughout, no bad
samples and zero swap growth. Swap stayed 0.012998046875 GiB.

| Arm | Normal samples | Peak wired+compressor GiB |
| --- | ---: | ---: |
| Native48 | 203 | 87.4444 |
| Reuse48 | 161 | 87.4756 |
| Native47 | 117 | 84.8284 |
| Reuse47 | 46 | 80.4976 |

Each cohort is one native→reuse order on this small fixed workload. The revised
geometry and warm cache differ between cohorts, so do not pool them into a
balanced speed estimate. Task answers establish this screen's practical result;
raw-logit/NLL/KL or hidden-state equality were not measured here. Fallback and
approximate-source continuation remain visible rather than being pass gates.

The complete guard child commands below ran from `/Volumes/Models/ds4`.
The launcher records actual server argv/environment in each arm's
`server-execution.json`; each guard retains every sample in `pressure.jsonl`.

```sh
/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python -B speed-bench/experiments/server-suffix-hour10-20261004/launch.py --mode native --output speed-bench/experiments/server-suffix-hour10-20261004/native --cache .workspaces/server-agent-day-20261004/native-cache
/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python -B speed-bench/experiments/server-suffix-hour10-20261004/launch.py --mode reuse --output speed-bench/experiments/server-suffix-hour10-20261004/reuse --cache .workspaces/server-agent-day-20261004/reuse-cache
/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python -B speed-bench/experiments/server-suffix-hour10-20261004/launch.py --mode native --output speed-bench/experiments/server-suffix-hour10-20261004/native47 --cache .workspaces/server-agent-day-20261004/native-cache --delete-a 47
/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python -B speed-bench/experiments/server-suffix-hour10-20261004/launch.py --mode reuse --output speed-bench/experiments/server-suffix-hour10-20261004/reuse47 --cache .workspaces/server-agent-day-20261004/reuse-cache --delete-a 47
```
