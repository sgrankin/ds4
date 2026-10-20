# Native-resident admission recheck result

The owner-thread admission recheck avoided 31 and 34 redundant background
expert reads in its two attempts. Combined logical demand and background bytes
remain about 11.7–11.8% above the fixed BF0 control. This does not support adopting
the pool or claiming a latency or physical SSD traffic benefit.

| Arm | Combined requested = returned bytes | Raw ΔNLL | Raw KL | Argmax flips |
| --- | ---: | ---: | ---: | ---: |
| Native before / after | 225302446080 | 0 | 0 | 0 |
| BF0 | 196666859520 | +0.0210325 | 0.0126620 | 13 |
| Pool age1600, recheck off | 219758469120 | +0.0162995 | 0.0139066 | 13 |
| Pool age1600, recheck on | 219718656000 | +0.0204600 | 0.0119500 | 14 |
| Pool age1600, recheck on repeat | 219877908480 | +0.0214742 | 0.0111411 | 12 |

All six arms use the frozen 16×32 calibration input, context4096, cache64GiB,
chunk512 and one-slot GAP.30/mass.20/total.35/rank48 policy. Queue age80,
READY age1600, late rollback and the 400MiB ownership cap remain fixed. The
native controls match all 512 original raw distributions and the full SHA1993
reference, with exact demand counters. BF0 matches its previous full raw SHA
ceb6 and exact counters. Approximate asynchronous pool repeats differ; every
attempt and its complete raw array remain retained.

The recheck performs 8924 / 9191 metadata checks and drops 31 / 34 queued reads,
observing 308551680 / 338411520 unstarted logical bytes. These bytes are a
separate admission observation, not subtracted from actual I/O totals. The
combined totals include demand, global staging and pool reads through teardown
exactly once; cache-loader bytes are not added again. They cover the whole
engine's P+D work, not decode-only or measured physical disk traffic.

Of 2614 / 2635 completed pool reads, 428 / 430 are consumed, including 3 / 5
same-token late rollbacks. About 82% are replaced before use. Replacement still
dominates waste. All completed/expiry/queue/read/late balances pass; no read,
allocation, descriptor, installation or owner rejection fails. Allocations,
frees and descriptor opens/closes balance, all worker/payload/queue ownership
is released, and peak application ownership408727552 bytes stays below the
419430400-byte cap. Timings are diagnostic and are not used as evidence here.

The unrun source1a814051 is preserved as immutable source. Review declined
launch because process lookup was unbounded and raw completeness did not
prove finite CSV completeness/alignment. Corrective source7040eef1 adds an
absolute two-second `/bin/ps` lookup and checks exactly 16×32 finite CSV rows
against raw case/step/target metadata. The earlier expected V1 prepared-artifact
path is absent; the archive records that fact and does not invent a run or
alter frozen ancestors. Runtime f4a96514 was independently reviewed; production,
ASan/UBSan, TSan and actual native-buffer/installer fault evidence is retained.

ROOT's original guard returned0. Its release proof records a reacquired MAIN
lock, no live model processes, 307 normal / zero bad pressure samples, zero swap
growth and peak wired plus compressed memory78.3523GiB. The CPU archiver verifies
all 64 prepared assets, literal source/test/build pins, six full raw hashes,
finite CSV alignment, all sixteen paired prompt summaries and all counter
balances. It opens no model, tokenizer or GGUF metadata.

`backfill-pool-native-recheck-evidence` contains the compact result, original
large-artifact path/hash inventory, frozen command/manifest/build/CPU proofs,
preserved unrun V1 source and a verified 225-member compressed archive of small
completed artifacts. Each of the six raw arrays was streamed once and retained
at its original pinned path. Completed artifacts and model/runtime policies
were not changed.
