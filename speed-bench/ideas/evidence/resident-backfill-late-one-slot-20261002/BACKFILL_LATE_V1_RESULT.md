# Same-token original rollback works rarely and adds net I/O

The fixed six-arm calibration completed under MAIN pressure guard, exit0,
exec session26715; MAIN job lock was reacquired/released after completion.
Both positive-gap MODE=native and default-off controls exported512/512
bit-equal rows and full reference SHA
1993aa5ff886b3f51401c1d1015e3882550120777809c31ea263936e9d63dda8.
All native demand-read/cache fields repeated exactly, matching prior native.

Late original restoration executed successfully before routed binding:2 and3
rollbacks out of3539 and3529 tentative substitutions (0.057%/0.085%). No
candidate-finish/install/cancellation failure or no-victim decline occurred.
Every transaction counter identity passed. Native defaults remain unchanged;
this experiment stays opt-in and is not promoted.

| Arm | Delta raw NLL | Raw KL | Argmax changes/512 | Demand bytes | Stage bytes | Combined logical bytes |
|---|---:|---:|---:|---:|---:|---:|
| Native before/after |0|0|0|225302446080|0|225302446080|
| BF0 one-slot |+.02103248|.01266202|13|196666859520|0|196666859520|
| Future BF1, late0 |+.02363964|.01249073|12|196258775040|2478366720|198737141760|
| BF1 late1 |+.01968953|.01182087|12|196248821760|2478366720|198727188480|
| BF1 late1 repeat |+.01717642|.01167778|11|196447887360|2488320000|198936207360|

BF0 reduces requested logical I/O by about12.71% versus native. Adding staging
increases total requested bytes by2.070GB for future-only BF1 and2.060/2.269GB
for late-enabled BF1, or1.053%/1.048%/1.154% versus BF0. Requested and returned
bytes agree in each arm. Cache-loader bytes equal the ordinary demand bytes
here and are not added again. These are whole-engine prefill+decode explicit
syscalls, not physical SSD traffic. Route skipped bytes count tentative avoided
original requests, including rolled-back originals, rather than net savings.

Late quality metrics improve slightly in this calibration, while every
approximate arm still has increased NLL. Prompt-bootstrap95% NLL intervals are
BF0[+.005001,+.042382], future BF1[+.007596,+.044540], late1[+.005018,+.037657],
repeat[+.002656,+.035444]. Worst-prompt NLL increases are.13606/.13878/.10931/
.10957 respectively. Max per-row KL remains about1.11–1.13. This is calibration
with16 prompts, not a held-out semantic or generated-answer success claim.

| Stage/late counter | Future-only | Late1 | Late1 repeat |
|---|---:|---:|---:|
| Attempts |3528|3539|3529|
| Started/completed |249|249|250|
| Consumed, including late installs |33|34|34|
| Expired |216|215|216|
| READY admission skips |3241|3257|3245|
| READING busy skips |38|33|34|
| Late original installs |0|2|3|
| Final substitutions |3528|3537|3526|
| Final live bytes/state |0/IDLE|0/IDLE|0/IDLE|

The one global READY payload prevents most admissions while it waits for a
future exact identity. Late not-ready3537/3526 includes unrelated READY data,
not solely slow SSD completion. Consumption is about13–14% of stage reads;
late installs comprise only2/3 of those34 copies. One payload peaks at9953280
bytes (9.49MiB). No400MiB per-layer stage pool was implemented or authorized.
Changing retention or admission is a separate experiment requiring a new plan.

Late repetitions differ in full raw bytes, quality, cache demand counters and
worker/transaction counts. That variation was allowed and is retained rather
than called a parity failure: completion timing affects future state. Both
exact native controls remain byte/cache-I/O identical. The run is instrumented
and root CPU integration/archiving occurred concurrently; timings are diagnostic
and support no speed claim.

Pressure311 samples remained level1, max_bad0, swap unchanged.025576GiB.
Minimum free RAM.119904GiB, maximum wired70.4742GiB and compressor7.6014GiB.
The guard completed normally and no nested model process remains under this
finished study. No additional model job was launched after releasing its grant.

Runtimec3b74256c96f3e643393c3702ab4950621a1cec8 follows reviewed cleanup of the
frozenf87 prototype; studyf5f217c24ad97eac64cb3a4c02b98a395ce9691d follows it.
Frozen IO/raw probe SHAe1aed3b279c05d9c82f1377eccb70de43b225387d39fe028a9c051c9d34cde96,
driver61f5208b6dfa90425d367ae33629d7d309dfe2fc9ba72d9a987b60cd6cb8ca69,
prepared manifestc22c129e979b94acee886239cc3acc9908c9ee4d0ba57e63975092960c69f82b.
All6 raw records, CSVs, original commands, logs and full pressure trace remain
in persistent artifacts at `.workspaces/idea-resident-backfill-late-study/`
`speed-bench/ideas/artifacts/resident-late-prepared-20261002/`.
This evidence-only child adds compact summaries, commands, final footers,
pressure/provenance and checks under `backfill-late-v1-evidence/`; it changes
no runtime, frozen helper, policy setting, shader or shared ancestor.
