# Automatic hot RAM and SSD context cache

Item 9 follows pre-RoPE qualification. Source preparation overlapped item 8;
about 30 minutes of it is charged to this item. Runtime qualification and
integration receive the remaining 30 minutes, including builds and analysis.

This implementation uses immutable byte-hashed chunks, 2048-token pages of
cumulative native compressed-attention KV/index state, and separate endpoint
headers, tokens, logits, raw windows and pending pairs. Optional DPR1 originals
are stored as immutable chunks too. It follows the architecture lesson from
[oMLX's V4.1 delta cache](https://github.com/jundot/omlx/blob/main/omlx/cache/deepseek_v41_delta.py):
compressed attention needs both cumulative arrays and endpoint recurrent state.
It reassembles the existing native DS4 payload rather than adding a second GPU
layout. Content deduplication is exact byte sharing.

Each put synchronously reaches SSD; manifests are published last. The hot LRU
caches immutable host chunks, and the SSD LRU evicts unpinned endpoints and
unreferenced chunks. Returned leases pin sources until release. A directory has
one process owner. The agent stores exact and private suffix endpoints
automatically and drops duplicate full frontier blobs after successful spill.
Startup scans manifests, so prefix lookup and suffix-source discovery work after
restart without a named session-save command. Exact lookup excludes private
approximate sources. Text/session history still belongs to the caller.

Environment: `DS4_STATE_CACHE_DIR` enables the tier,
`DS4_STATE_CACHE_HOT_MB` defaults to 512 (0 disk-only),
`DS4_STATE_CACHE_SSD_MB` defaults to 8192. The preset wrapper supplies these
and enables pre-RoPE in context-fast. Native text V4.1/Metal only; model and
forward-policy identity protect exact prefix reuse.

Server requests share the cache under the existing inference lock. Active
slots retain separate mutable GPU graphs; `--batched-session N` supplies the
existing interleaved-prefill/batched-decode scheduler. Storage does not provide
GPU paged attention or eliminate per-active-session graph memory. Synchronous
stores happen at evaluated endpoints, which may cost time; this screen will
report that cost rather than claim oMLX throughput parity.

The focused CPU module test passed deduplication, hot spill, restart reassembly,
source immutability, exact/approximate separation, pinned ownership, smaller-SSD
eviction and interrupted publication cleanup. Native/session/front-end results
follow below. Useful logs, commands and small data stay in this directory.

## Native and server qualification

All twelve native materializations passed: original bytes and pre-RoPE tapes
survived SSD reopen, longest prefix lookup selected 4097 tokens, session B
recaptured identical state/originals, and remained unchanged while A advanced.
Two full sources totalled 63.58 MiB; shared disk files used 51.08 MiB, saving
19.65%. Metadata used 32,048 bytes.

| Hot limit MiB | First post-reopen materialization mean ms | Warm mean ms |
| ---: | ---: | ---: |
| 0 | 81.45 | 79.45 |
| 1 | 78.65 | 77.50 |
| 64 | 64.36 | 0.60 |

These are byte reassembly timings for the two endpoints, not network/task
latency. Hot 64 MiB retained all 51.07 MiB of distinct chunks; put of both
sources cost about 192–199 ms including synchronous SSD publication.
Canonical GPU restore and recapture is reported separately in native.csv.
The original outer guard stdout is preserved in native.stdout; native.csv
contains only its standard CSV output, with the final guard JSON separated.

Two simultaneous server clients returned READY before and after an owned
restart. Both had 2128 prompt tokens. With hot size 0, the restarted server
logged exact-prefix hits [{'slot': 1, 'tokens': 2128}, {'slot': 0, 'tokens': 2128}].
Concurrent-pair time was 61.51 seconds initially and
0.86 seconds after restart, excluding startup.
This is a repeated-prefix best case on two resident slots, not a general
server throughput estimate. Each server exited cleanly. Commands and answers
are in server/execution.json and server/result.json.

## Agent SSD qualification and discovered selector failure

The first disk-only agent workflow passed all three answer/fresh-read grades,
but activated only one graft. The selector ranked a cached native prefix
matching from token zero above the smaller genuinely shifted suffix. Its
destination prefix was empty; the existing synchronizer rejected that, and
the agent safely rebuilt. See agent-ssd/result.json and its trace. This is
retained as a feature-activation failure rather than hidden as a speed result.

The fix excludes destination zero from disk graft candidates, leaving exact
prefix lookup available. A focused existing-mock regression checks that a
longer exact prefix does not displace a valid shifted suffix, remains findable
as a native prefix, and releases its source lease. It passed with the existing
transaction tests; rebuilt agent tests also passed. Fix revision
`bf0d20cb613cf7acfaeb17eb742c374ad7d61496`. The corrected workflow reopens the
same persisted directory after the first process exits, with hot size 0.
Its results follow below.

[Host-memory / reassembly curve](hot-cache-curve.svg), with its input in
native-summary.json. This plots the first pass after reopen and the second
pass over two endpoints. Filesystem page cache was not flushed; the hot0
points describe SSD-backed reads/reassembly, not forced physical SSD misses.
Plot command: `.workspaces/bench-plots-venv/bin/python
speed-bench/experiments/state-cache-hour9-20261003/plot_cache.py`. Useful font
caches stay under `.workspaces/plot-cache`.

The corrected disk-only run passed all three answers and fresh archive reads.
All **3/3** grafts logged `source_ssd=1`; two began from an already approximate
live graph. The third source was private approximate endpoint4521, confirming that
SSD-backed approximate checkpoints can supply later relocations. Full process time was 139.84 seconds,
with 50.54 seconds across the three edited turns. The first failed
selector run remains in agent-results.csv. These runs include different system
warmup histories and diagnose integration; the matching warm performance pair
is the [item8 report](../pre-rope-hour8-20261003/README.md).

Every cache9 guard completed with normal pressure and no bad samples. The
first native job observed 3.25 MiB of swap growth; the following three guards
observed none. See pressure-summary.json. One heavy model job ran at a time. Item9
completed at 00:48:16 UTC, within its charged preparation/qualification budget.
