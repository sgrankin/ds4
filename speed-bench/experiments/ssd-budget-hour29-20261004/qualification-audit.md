# Independent item29 qualification audit

The quota change behaved visibly:512MiB preserved all four edited grafts while
evicting older endpoints;128MiB lost all four graft sources and recomputed from
short exact prefixes. All32 required file reads and16 edited answers across
four arms are correct. The128 arm retains one strict Cwarm format failure,
so strict totals are31/32. This audit read completed small artifacts, launcher
and Item19 driver; it ran no model, compiler, tests, grader or jj command.

## Actual settings and namespaces

Execution records confirm SSD4096→512→128→4096MiB, hot0, suffix64,
canonical1/PAD1, GAP0, Q2 expert cache64GB, ctx8192/chunk512/two slots,
tile64/limit1024, temperature0 and effortnone. All use revision91c9cf21 and
the same server/unchanged Item19 client. Each launcher creates fresh output,
cache and response-store directories with `exist_ok=False`; recorded roots
are `ssd29-cache-{before,512,128,after}` and corresponding store namespaces.
All advertise the same `concurrent-visible/{A-D}` files. Before/after-restart
environments match within each arm, and every server lifetime reports profile
`c012d28b049720b89667ba27d6cded52e414816f`. SSD capacity is the varied storage
limit, not a different forward-policy profile. No source is imported from
another arm or rescued after eviction.

## Outcomes and control repeats

Each arm executes eight steps and eight actual reads: four warm active-file
reads and four edited archive reads. Every one of64 HTTP responses completes
with no incomplete/cap record. All16 edited attempts return the independent
inventory/archive gold pairs. These are four repeated facts, not16 independent
new problems. C/D archives change after restart and are actually read afresh.

The128 Cwarm output states the correct inventory19/15/18, then READY, rather
than READY alone. It is a strict format failure despite correct facts and a
successful read; its row remains `correct:false, pass:false`, and guard child3
is retained. No answer was repaired, retried or regraded. Its four edited
answers still pass. All other arms are8/8 strict;128 is7/8.

The4096 before/after controls have identical final texts, tool arguments,
returned file contents and per-response public usage for all eight steps.
Different cache footer occupancy therefore does not imply a quality change
in those controls. Generated histories in128 differ; no paired generated-history
KL/NLL or claim that quota caused Cwarm's extra prose is made.

## Grafts, continuation and eviction

4096 controls and512 each retain four native-source SSD grafts, all1034 reused
tokens with95 total fresh/31 staged, destination308:1342 and latest User1345.
A/C use source660:1694/pad0; B/D source658:1692/pad1. Each then uses four
approximate visible-prefix continuations at cached1530; no extra FINAL graft.

128 has zero grafts. All four edited requests log
`suffix exact fallback ... reason=no-aligned-source`, then exact state-cache
hits of256 tokens. It proceeds through ordinary evaluation and correct current
file answers; it does not silently reuse an evicted suffix source. No
approximate visible continuation is logged in this arm. Cold warmup fallback
lines are present in every arm and must be separated from edited fallback.
Concurrent cache/graft logs belong to joined groups, not per-session row arrays.

Shutdown counters below are per server lifetime; the after column starts after
restart. All observed SSD-owned byte totals are below their configured quotas.
Hot bytes/reads and pinned endpoints are zero in every footer.

| Arm | SSD bytes before / after | Endpoints before / after | Evictions before / after | Logical disk reads before / after |
| --- | ---: | ---: | ---: | ---: |
|4096 before|487345286 /619014302|40 /46|0 /0|164 /165|
|512|508137840 /517476117|42 /24|0 /24|165 /164|
|128|123814437 /124332310|11 /11|55 /12|128 /128|
|4096 after|508137840 /639806856|42 /48|0 /0|165 /165|

Final SSD-owned sizes are590.34/493.50/118.57/610.17MiB. In this sequence512
retains useful sources with24 evictions;128 has67 evictions across lifetimes
and no edited graft coverage. These are logical cache counters, not physical
SSD traffic or a complete directory-size/transient-write audit. No cache-write,
admission-failure or warning/error diagnostic appears; the ordinary startup
message that full-model warmup is skipped is expected SSD-streaming behavior.

## Timing and pressure

Edited time is the sum of joined edited-AB/CD makespans. Do not sum overlapping
queue-inclusive per-request times. Launcher wall includes source warmup,
startup, shutdown and restart; all costs remain recorded.

| Arm | Warm groups, s | Edited groups, s | Whole launcher, s | Strict / reads | Edited grafts |
| --- | ---: | ---: | ---: | --- | ---: |
|4096 before|183.129018|102.839942|290.396260|8/8 /8|4|
|512|187.000194|103.891512|295.331169|8/8 /8|4|
|128|190.379534|157.664786|352.087724|7/8 /8|0|
|4096 after|186.487137|102.075258|293.065869|8/8 /8|4|

Against the4096 bracket mean,512 edited/whole wall increases1.400%/1.234%;
128 increases53.883%/20.689%. Control edited/whole drift is−.744%/+.919%.
This supports512 as a smaller workable quota on this working set, with modest
timing cost, and exposes the128 capacity/fallback tradeoff. It does not establish
an optimal quota for arbitrary sessions or a general performance guarantee.

Guards all complete with children0/0/3/0;128 child3 is the retained format
failure, not pressure termination. Normal sample counts are280/285/340/283;
all samples have level1, no reason and zero consecutive bad. Initial swap is
.018125GiB in every arm and growth is zero. Peak wired+compressor is
86.811966/85.326752/84.963974/84.691696GiB.

Evidence: [build.json](build.json), [method.md](method.md), [launch.py](launch.py),
arm `*-execution.json`, `launcher-summary.json`, server logs,
`http/{before,after}/result.json` and joined group logs; guards are
`guard-{control-before,ssd512,ssd128,control-after}`. No runtime or driver was
changed by this audit. Ownership returns to ROOT for publication and next timing.
