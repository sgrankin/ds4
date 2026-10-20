# Bounded original pool: more admissions, low reuse, no promotion

The fixed seven-arm study completed under the MAIN pressure guard with return
code0. Native-before, positive-gap MODE/native POOL0, and native-after each
matched all512 raw distributions and full SHA1993aa5f, with identical native
read/cache counters. Deterministic BF0 POOL0 matched the earlier c3 full raw
SHAceb6df37. Global BF1+late1 POOL0 also happened to match the earlier full raw
SHA5c0aa48c; asynchronous equality was not required. Pool repeats differed,
as expected for read-completion-dependent routing.

Settings are the untouched16×32 calibration fixture, cache64GiB, ctx4096,
chunk512, GAP.30/MAX_MASS.20/MAX_TOTAL_MASS.35/MAX_RANK48/MAX_SUBS1. The one-slot
path ignores the total-mass parameter. This screen measures whole-engine P+D
logical requested/returned I/O, including background reads completed during
teardown. It does not establish physical SSD traffic, engine snapshot equality,
held-out quality, or profile-off speed. No runtime default is promoted.

| Arm | Delta NLL | Mean KL | Argmax flips | Demand bytes | Background bytes | Total bytes | Versus BF0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native |0|0|0|225302446080|0|225302446080|+14.56%|
| BF0 |+.02103248|.01266202|13|196666859520|0|196666859520|0|
| Global BF1+late1 |+.01968953|.01182087|12|196248821760|2478366720|198727188480|+1.05%|
| Pool BF1+late1 |+.01676854|.01139528|14|194925035520|26704650240|221629685760|+12.69%|
| Pool repeat |+.01611828|.01210288|11|194905128960|26963435520|221868564480|+12.81%|

Requested and returned byte totals are equal in every arm. Pool total reads
remain1.52–1.63% below native, while BF0 saves12.71%. The pool reduced demand
reads by1.74–1.76GB relative to BF0 but added26.70–26.96GB of background reads.
Demand cache misses include staged consumption: first pool19843 misses equal
19584 actual demand reads plus259 staged consumes. Cache-miss counts therefore
must not be treated as actual demand pread counts.

| Pool counter | First | Repeat |
|---|---:|---:|
| Requests / completed reads |3408 /2683|3447 /2709|
| Future-native consumption |255|254|
| Same-token rollback / late copy |4|1|
| READY expiry / replacement |2021 /395|2024 /422|
| Reset READY discards |8|8|
| Queued expiry / newest-identity updates |473 /226|488 /225|
| Consumed fraction of completed reads |9.65%|9.41%|
| READY / queue peak |21 /10|23 /10|
| Owned staging peak |209498112 B|229421056 B|

The global single-slot arm admitted249 reads and consumed34 (13.65%), with
3250 READY-blocked requests. The pool admitted roughly10.8 times as many
reads, so the former global READY bottleneck was real. However, about75% of
pool reads expired and another15% were replaced before use. Neither the400MiB
cap nor handoff contention constrained this run: budget/handoff defers were0.
Only4/3408 and1/3447 tentative substitutions restored the original within
the same token. More capacity alone did not solve useful original reuse.

Quality remains mixed. Pool prompt-bootstrap95% delta-NLL intervals were
[-.000671,+.039807] and[-.001859,+.037951]; BF0 was[+.005001,+.042382]. Pool
worst-prompt deltas+.14704/+.14605 exceeded BF0+.13606 and global+.10931.
Pool p99/max row KL were.19273/1.09306 and.19440/1.13014. Lower mean NLL/KL
on some comparisons does not establish a general quality improvement.

All completed-read, request, queued-slot, consumed/discarded ownership,
allocation/free-byte, duplicate-fd, worker/join and actual late-copy/install
identities passed. All2683/2709 mappings and FD pairs were released; each
single persistent worker joined. Final payload/READY/queue/job/worker counts
were0; fixed application metadata remained16384 bytes. No invalid request,
read/allocation/unmap/FD/installer failure or cancellation occurred. Legacy
global read/use counters were zero under pool mode; background bytes were
added exactly once to native explicit I/O.

Pressure evidence has362 samples, all level1/no bad samples; swap remained
.01776367GiB, max wired71.89682GiB and compressor6.68764GiB. Guard PID24638
and all owned probe processes were gone after completion; MAIN flock was
reacquired and released. Exec session85381 closed successfully. The model grant
ended with this study; no additional model jobs were launched.

Runtime source0621295d is a corrective child of the preserved d75 prototype;
study source2ba9843e is frozen independently. Production, ASan+UBSan, TSan,
real-read API,24 actual-installer fault cases, paused-reader ABA regressions,
100000 coefficient-bit transactions and actual worker counter-balance checks
passed. Independent review found no remaining concrete lifecycle/ownership
blocker. Model controls now prove native/BF0 default-off raw parity, while
the successful4/1 late copies exercise the actual Metal protected installer.

Compact evidence is in [compact evidence](compact/).
All full raw records, CSVs, stderr/stdout, commands and pressure samples remain
under `.workspaces/idea-resident-backfill-pool-study/speed-bench/ideas/artifacts/resident-pool-prepared-20261002`.
Frozen binaries/source/object proof remain in the sibling pool workspace's
`speed-bench/ideas/artifacts/resident-pool-corrected-build`.
