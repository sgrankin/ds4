# Background-original pool payload retention80 vs1600

Six-arm calibration16×32 completed0 under MAIN guard. Native-before/after full512 raw distributions SHA1993aa5f… and all demand cache/pread counters are exact; deterministic BF0 SHAceb6df37… matches the prior control. Queue admission TTL remains80; only stored/completed payload retention changes. Features remain opt-in.

|Arm|ΔNLL|KL|Argmax flips|Demand GB|Background GB|Total GB|Useful completed reads|
|---|---:|---:|---:|---:|---:|---:|---:|
|Native|0|0|0|225.302|0|225.302|—|
|BF0|+.021032|.012662|13|196.667|0|196.667|—|
|Pool80|+.015297|.012499|11|194.935|26.924|221.859|248/2705 (9.17%)|
|Pool1600|+.018880|.012172|13|193.641|26.137|219.778|418/2626 (15.92%)|
|Pool1600 repeat|+.023547|.011983|14|193.342|25.869|219.211|425/2599 (16.35%)|

All byte values are decimalGB and explicit logical reads over whole-engine prefill+decode plus teardown. Background requested/returned bytes are counted once, cache-loader bytes are not added again. These are not physicalSSD traffic or latency results. Async approximate repeats differed as expected.

Longer retention recovers older demand reuse:168/186 future consumptions were older than80 global layer boundaries, versus none for80. Completed-payload harvest expiry was0 in all arms. Stored READY expiry fell2043→14/18, while replacement rose407→2166/2130. Latest-per-layer replacement remains the dominant loss. Net logical I/O is still11.75%/11.46% above BF0, so this is not an I/O win.

All completed-payload, queue, allocation/free, fd, worker, late-install and read balances passed. Pool1600 owned peak408727552B remains below419430400B. Final stack/guard fields are0 after joined teardown; peak minus simultaneous payload peak408420352B and metadata16384B leaves290816B for charged stack+guard. With16KiB guard this corresponds to Apple reported stack274432B (requested256KiB), including12288B beyond the requested stack. No budget deferral occurred. OS thread bookkeeping is outside the documented application-owned payload/stack/metadata cap.

Guard pressure stayed normal with zero bad samples and no swap growth. Guard/driver/helper exited; MAIN lock was reacquired and released. `backfill-pool-ready-age-v1-evidence/index.json` pins compact files and full raw paths/hashes. Full pressure/raw artifacts remain in the persistent study artifact directory.
