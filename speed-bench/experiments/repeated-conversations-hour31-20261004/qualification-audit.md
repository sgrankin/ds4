# Item 31 independent qualification audit

No concrete blocker found. All 27 logical turns made one actual file read and passed their expected answer without caps or recorded errors. These comprise 54 completed Responses HTTP replies. All 24 one-item deletions preserved every other public item. This audit read saved artifacts and source only; no model, grader, test or build was rerun.

Runtime `91c9cf212d2e6a35ae558d1c96f37607becbde3f` and settings are recorded in [build.json](build.json), [method.md](method.md), each arm's execution JSON and `guard-*/command.json`. Native-before → reuse → native-after use distinct fresh cache/store namespaces and the same `repeat31-visible` paths. The runtime change is suffix0 versus suffix64; GAP0, hot0/SSD4096MiB, ctx8192, two slots, chunk512, cache64GB, tile64/limit1024, temperature0, effortnone and cap128 remain fixed. Clients are serial.

## Durable edits and current-file answers

Each arm has one Conversation, eight original obsolete User items, one warm inventory read and eight archive reads. For every deletion, the saved DELETE route names the original selected item ID. Comparing the entire ordered item dictionaries before/after confirms that only that item was removed, with no hidden pagination remainder. Versions advance 3→4, 6→7, …, 24→25; private replay is present before each deletion and absent afterward. Each new response then appends its current input/output once. Final snapshots have version27, 38 items and private replay present.

The process restart follows edit4. Edit5 starts from persisted version15, retaining the first four completed exchanges. Actual client reads use the advertised file with `whole=true, raw=true`; the returned bytes are supplied through the actual call ID as `function_call_output`. All three warm turns returned READY. Every edited turn returned `31*8-17 = 231` and the current code ORBIT-57 through ORBIT-64, including the four post-restart changes. There were no stale-code answers.

All 54 request bodies use `store:false`, Conversation identity and no explicit `prompt_cache_key` or `previous_response_id`. Storefalse does not disable Conversation persistence. The server defaults the missing hint to the Conversation ID (`ds4_server.c:5694`); canonical full-input mode excludes Conversation references (`ds4_server.c:11968`). Consequently, the global canonical1 setting does not convert these requests to the full-input experiment. Ordinary validated visible-prefix tool-result continuations remain available.

## Eight actual grafts

The candidate has one initial no-aligned-source fallback, then eight successful grafts. Every graft reports `pre_rope=1, source_ssd=1`; edit1 uses a native source and edits2–8 use approximate sources. All eight final tool-result responses use visible-prefix continuation with `ids=0, approximate=1`, rather than another graft. Native arms have no grafts.

| Edit | Source | Destination | Reused | Fresh / staged | Latest User start | Source approximate / padding |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | 616:3035 | 286:2705 | 2419 | 729 / 665 | 3339 | 0 / +1 |
| 2 | 618:3434 | 286:3102 | 2816 | 234 / 170 | 3241 | 1 / 0 |
| 3 | 616:3336 | 286:3006 | 2720 | 234 / 170 | 3145 | 1 / +1 |
| 4 | 618:3240 | 286:2908 | 2622 | 234 / 170 | 3047 | 1 / 0 |
| 5 | 616:3142 | 286:2812 | 2526 | 234 / 170 | 2951 | 1 / +1 |
| 6 | 618:3046 | 286:2714 | 2428 | 234 / 170 | 2853 | 1 / 0 |
| 7 | 616:2948 | 286:2618 | 2332 | 234 / 170 | 2757 | 1 / +1 |
| 8 | 618:2852 | 286:2520 | 2234 | 234 / 170 | 2659 | 1 / 0 |

Every reused destination ends before the latest User starts, preserving the entire current query as fresh input. Successive source endpoints match the preceding edited READ prompt endpoints. Edit5's approximate SSD source follows process restart; the logs demonstrate repeated relocation of approximate sources, without proving raw-state numerical equivalence. The one-token padding is reflected in input usage for odd-numbered edits.

There is no second Conversation and **no exact state-cache-hit log in any arm**. Candidate warmup takes97.793s versus native97.328/97.616s, so this gain does not rely on Item30's cross-conversation 1,316-token warm hit.

## Controls and timing

Native-before/after match normalized public output fields, tool argument bytes, actual file-result bytes, completion statuses and usage dictionaries for all nine turns. IDs differ as expected. Candidate answers and output-token counts also match; input/cache usage differs due to padding and grafting. These checks do not establish raw logits, generated token IDs or RNG parity, and no generated-history paired NLL is claimed.

| Seconds | Native-before | Reuse | Native-after | Saving vs bracket mean |
| --- | ---: | ---: | ---: | ---: |
| Warm source | 97.327657 | 97.793450 | 97.615970 | −0.33% |
| Eight edited turns including DELETE | 742.557886 | 249.790287 | 741.242330 | 66.33% |
| All nine turns | 839.885543 | 347.583737 | 838.858299 | 58.59% |
| Launcher including restart/startup/shutdown | 844.317552 | 352.273276 | 842.994900 | 58.24% |

Edited native drift is−0.177%; launcher drift is−0.157%. Reuse saves66.30% edited and58.21% launcher time against the faster native-after control. This is one fixed inventory/archive family in one serial ordering, not a general eight-edit accuracy guarantee or proof that approximation error cannot accumulate.

## Pressure and cache ownership

All guards completed with child0; every sample has level1, reasonNone and consecutive_bad0. Native-before / reuse / native-after have812 /340 /811 samples and peak wired+compressor87.473 /84.835 /87.791GiB. **Native-before swap grew0.003662109375GiB (3.75MiB)** from initial0.018125GiB; reuse and native-after start at0.021787109375GiB and have zero measured growth. The small native-control growth is retained, not rewritten as flat swap.

All phase logs use base profile `c012d28b049720b89667ba27d6cded52e414816f`; each restart reopens its own cache/store. Final native SSD bytes are545,206,500 with27 endpoints; candidate711,001,486 with34 endpoints, below the4096MiB quota. Every footer reports hot_bytes0, pinned_endpoints0 and endpoint_evictions0. Candidate logical disk_reads are490 before restart and570 afterward; they are not physical SSD read counts. No warning/error or pin-leak finding was observed.

Evidence remains in arm `http/{before,after}/result.json`, actual request/reply files, full item snapshots, server stderr logs, `launcher-summary.json`, and `guard-*/{result.json,pressure.jsonl}`. Original results were not changed.
