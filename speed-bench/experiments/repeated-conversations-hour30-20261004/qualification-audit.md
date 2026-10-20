# Item 30 independent qualification audit

No concrete blocker found in the completed native-before → reuse → native-after screen. All 24 logical turns made one actual file read, completed without a cap, and passed their independent expected answer. These turns contain 48 Responses HTTP replies. This audit read the saved artifacts and source; it did not rerun a model, grader, test or build.

Runtime: `91c9cf212d2e6a35ae558d1c96f37607becbde3f`, as recorded in [build.json](build.json). Settings and commands are in [launch.py](launch.py), each arm's `before-execution.json` / `after-execution.json`, and `guard-*/command.json`. GAP0, context8192, two slots, cache64GB, chunk512, tile64/limit1024, hot0/SSD4096MiB, temperature0, effortnone and cap128 were fixed. The client ran serially; this is not a concurrent-client screen.

## Conversation edits and actual answers

Each arm performed six real DELETE operations: three original obsolete User items in each of two Conversations. For all 18 deletions, the entire ordered list of remaining item dictionaries equals the previous list with only the requested item removed. The owned-store version advanced exactly once (3→4, 6→7, 9→10), private replay changed from present to absent, and pagination reported no hidden remainder. Other IDs and contents were retained. Both Conversations finished at version12 with18 public items and private replay present after the new responses.

The restart retained the post-edit1 Conversation and its history; edits2/3 then removed the next original items rather than rebuilding a fresh conversation. All turns read the advertised absolute active/archive file with `whole=true, raw=true`. A returned `23*14-9 = 313` and BEACON-46 each time. B returned `31*8-17 = 231` and the current ORBIT-57, ORBIT-58, then ORBIT-59. All six warm turns read their active file and returned READY. There were no stale-file answers, missing reads, caps or recorded errors.

Requests have `store:false`, no explicit `prompt_cache_key`, and no `previous_response_id`. Conversations still persist public items/private replay independently of response storage. Source inspection confirms that a missing cache key defaults to the Conversation ID (`ds4_server.c:5694`); approximate sources therefore use that conversation hint plus the forward profile. `request_canonical_suffix()` excludes Conversation requests (`ds4_server.c:11968`), even though the launcher's global canonical flag is1. The valid visible-prefix tool-result continuation remains available.

## Actual suffix activation

Native arms made no grafts. The reuse arm made exactly six first-response grafts and six subsequent visible-prefix continuations marked `approximate=1`, with `ids=0`; there were no extra final-response grafts.

| Turn | Reused tokens | Fresh total / staged | Destination | Latest User start | Source kind / padding |
| --- | ---: | ---: | --- | ---: | --- |
| A/B edit1 | 764 | 729 / 665 | 286:1050 | 1684 | native SSD / +1 |
| A edit2 | 1161 | 236 / 172 | 286:1447 | 1588 | approximate SSD / 0 |
| B edit2 | 1161 | 234 / 170 | 286:1447 | 1586 | approximate SSD / 0 |
| A edit3 | 1067 | 236 / 172 | 286:1353 | 1494 | approximate SSD / +1 |
| B edit3 | 1065 | 234 / 170 | 286:1351 | 1490 | approximate SSD / +1 |

Every graft reports `pre_rope=1, source_ssd=1`. Reused slices end before the latest User begins, so the entire new query is fresh. Edit2's `source_approximate=1` occurs after process restart and demonstrates reuse of the persisted approximate source; edit3 relocates an approximate source again. Source slices are 616:1380 for edit1, 618:1779 for edit2, and 616:1683 / 616:1681 for edit3. Padding accounts for the extra input token on edits1/3; it is an explicit approximation, not identical input tokens.

B's reuse warm turn also reports a **1,316-token exact state-cache hit** after the ordinary no-aligned-source fallback. Its public `cached_tokens` is0. Both observations are retained: public usage does not expose this internal exact hit. This cross-conversation exact native prefix sharing contributes to the warmup gain; it is not an approximate suffix graft.

## Controls and timing scope

Native-before and native-after match all eight turns' normalized public output fields, tool argument bytes, actual read-result bytes, usage dictionaries and completion statuses. Generated IDs differ as expected. This is not a raw-logit, token-ID or RNG parity claim. Reuse has the same answers and generated output counts, with different input/cache usage caused by grafting and padding.

| Seconds | Native-before | Reuse | Native-after | Reuse saving vs bracket mean |
| --- | ---: | ---: | ---: | ---: |
| Source warm turns | 105.993951 | 75.296372 | 106.196054 | 29.03% |
| Six edited turns, including DELETE | 325.636564 | 199.211742 | 333.536105 | 39.56% |
| All eight turns | 431.630516 | 274.508114 | 439.732159 | 36.99% |
| Launcher, including startup/restart/shutdown | 435.670636 | 278.558514 | 443.888919 | 36.66% |

Edited control drift is2.43%; launcher drift is1.89%. Reuse still saves38.82% edited and36.06% launcher time against the faster native-before control. These are one serial ordering of two fixed fact families, with shared exact-prefix warmup effects and approximate conditioning. No paired NLL, broader accuracy or general speed guarantee follows.

## Guards and cache ownership

All three guards completed with child0. Every pressure row has level1, reasonNone and consecutive_bad0. Samples:420 /269 /427; peak wired+compressor:84.464 /84.120 /84.429GiB. Initial swap was0.018125GiB in each arm and measured growth was0. There are no server warning/error lines in the inspected phase logs.

Each arm has a fresh distinct cache/store namespace, the same advertised `repeat30-visible` paths, and the same base profile `c012d28b049720b89667ba27d6cded52e414816f` before/after restart. Cache footers stay below4096MiB with hot_bytes0, pinned_endpoints0 and endpoint_evictions0. Final native SSD usage is352,628,008 bytes (16 endpoints); reuse is589,687,556 bytes (32 endpoints). Reuse's logical disk_reads counters are232 before and370 after, not physical SSD I/O measurements. No cache-pressure or pin-leak finding was observed.

Primary evidence: arm `http/{before,after}/result.json`, request/reply and item snapshots, `before-server.stderr.log` / `after-server.stderr.log`, `launcher-summary.json`, and `guard-*/{result.json,pressure.jsonl}`. Original artifacts remain unchanged.
