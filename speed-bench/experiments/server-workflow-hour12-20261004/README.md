# File-backed HTTP workflow, turnover and SSD restart

Runtime `ceab3478a59c`; driver `5badc887c301`. Both native and reuse completed
**7/7 steps correctly**, including **6 actual fresh file reads per arm**.
Client C's turnover request used no tool. Reuse reduced the ordinary edited-turn
sum by **24.32%**; the updated-file, after-restart sum by **15.59%**. This is one
native→reuse timing order on two small inventory/retrieval tasks.

The client sends full Responses input with `store:false`, separate stable
`prompt_cache_key` values for A/B, temperature 0, seed 20261004 and effort none.
There is one client-side `read` function: it actually reads the requested file
with whole/raw true, and replays the real output items and call IDs. Its caps
are 128 for the tool request and 64 for the final answer. The server has two
resident slots, context 8192, chunk 512, cache 64GB, selected tile 64/limit 1024 and
native routing. State caching uses hot 0/SSD 1024MiB. Reuse adds fresh-tail 128.

Both arms advertise the same files under
`.workspaces/server-agent-day-20261004/workflow-shared/{A,B}`. Two disposable
blocks precede the unchanged active inventory tool result. A's independent
gold is `17*25-9=416`; B's is `19*11-7=202`.

1. A and B each read their real `active.json` once, then answer READY.
2. Archive the old code to `archive.json`, remove only the first marked block,
   and alternate A/B fresh archive reads: 416/RIVER-73 and 202/CEDAR-91.
3. Client C asks READY. Both logs show slot 0 evicting a 1790-token checkpoint;
   this is a third client using two slots, not a third resident graph.
4. ROOT shuts down and restarts the server using the same SSD directory.
5. Update each archive file and remove the second block. A/B must freshly read
   the new codes RIVER-74/CEDAR-92 while retaining totals 416/202. These changed
   codes are introduced through actual file reads, not the new question.

The after-restart cases intentionally stress updated external facts and are
reported separately from the ordinary fixed-fact deletions. Actual prior
answers and complete tool histories remain in `http/histories.json`.

| Measurement | Native seconds | Reuse seconds | Saving |
| --- | ---: | ---: | ---: |
| Cold source warmup, A+B | 105.324 | 106.643 | −1.25% |
| Ordinary edits, A+B | 106.186 | 80.364 | 24.32% |
| Updated files after restart, A+B | 85.290 | 71.992 | 15.59% |
| All four edited turns | 191.476 | 152.357 | 20.43% |
| All 7 steps including cold sources and C | 299.082 | 261.234 | 12.65% |
| Steps excluding cold source warmup, including C | 193.758 | 154.590 | 20.21% |

These step sums include archive writes and context edits for edited turns;
they exclude server startup/shutdown and restart. Summed driver-phase wall time
is 299.085/261.237 seconds, differing by only bookkeeping. Server startups total
0.630/0.843 seconds and are recorded separately in [summary.json](summary.json).

Three grafts activated. Before restart, A-edit1 reused 563 tokens from a native
SSD source 808:1371→306:869, fresh 784 (656 staged + 128 final); B-edit1 reused 463
from 1466:1929→964:1427, fresh 227 (99+128). After restart, **A-edit2 reused an
already approximate SSD source** (`source_approximate=1`, `source_ssd=1`):841
tokens 812:1653→310:1151, fresh 231 (103+128). **B-edit2 fell back** with
`reason=no-aligned-source`; its 43.137-second result is retained and still
answered the updated file correctly. All actual deletions were 2469 bytes;
activated grafts shifted 502 tokens. Geometry and activation are observations,
not task pass requirements.

Both guards completed with child 0. Native/reuse had 293/257 samples, all pressure
level 1, zero bad samples and zero swap growth. Swap stayed 0.012998046875GiB;
peak wired+compressor memory was 84.9883/84.6733GiB. One ROOT-owned server/model
job ran at a time, with the existing pressure guard and its stop rule retained.

Commands: [native guard](native-guard/command.json),
[reuse guard](reuse-guard/command.json). Each arm's before/after execution files
record server argv/environment; `http/{before,after}/result.json` retains all
answers, reads, timings, fallback/graft lines and caps/completion status.
The native and reuse phase launchers both exited 0.

This demonstrates actual file retrieval, session alternation, observed slot
turnover and persisted approximate-source reuse after process restart. It does
not establish general accuracy, balanced throughput, physical SSD read savings
or raw-logit/NLL/KL equality. All fallback points remain included; defaults stay
native and suffix reuse remains opt-in.
