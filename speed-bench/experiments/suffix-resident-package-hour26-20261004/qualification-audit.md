# Independent item26 qualification audit

The fixed strong resident policy operated together with suffix64, canonical
source warming, parity padding and approximate visible continuation. All 24
steps passed, including all 24 required fresh file reads; all 48 HTTP responses
completed without cap/incomplete records. This is a small existing-workload
screen, not a general quality or speed result. The audit read completed small
results, execution records, logs and guards; it executed no model or build.

## Policy and profile separation

The one policy change is GAP .90, per-slot mass .25, rank64, at most three
substitutions, total mass .50, replace, native_weights1. Controls use GAP0 and
native_weights0. The actual environment differences are exactly these routing
settings plus each arm's own cache/store paths. Shader settings and other
runtime settings match; each arm's before/after-restart environments are equal.
All use revision `91c9cf21`, the same server binary, Q2 model, ctx8192,
chunk512, two slots, expert cache64GB, tile64/limit1024, suffix64,
canonical1/pad1, hot0/SSD4096, temperature0 and effortnone.

Both resident-native controls and their restarts report profile
`44f656302d6a106059a893439add7bfdf814706e`. The strong arm and its restart
report distinct profile `a5344dab32c9490536d1ca5ecf1859d13982c11b`.
Each arm starts a fresh cache/store and warms its source under its own policy,
then restarts against that same cache. No GAP0 source is reused under GAP.90.
All advertised active/archive file paths remain identical between arms.
Here “native” means native resident routing: suffix reuse is enabled in all
three arms, so their edited graphs are approximate.

## Actual route and suffix activation

The strong before-restart footer records 9,178 substitutions/skipped loads and
91,351,203,840 logical skipped bytes; the after-restart footer records 4,519
and 44,978,872,320 bytes. Total is 13,697 substitutions and 136,330,076,160
logical skipped bytes. Both corresponding native-weight footers report
`requested=1` and exactly the same substitution counts. Drops are zero.
This establishes activation of the previously reviewed ID-only/native-weight
path; the footer is not an independent GPU coefficient-bit measurement or a
physical SSD-read counter. Neither control emits an active resident footer.

All arms have identical suffix coverage: four edited native-source SSD grafts,
each `reused=1034`, `fresh=95`, `graft_fresh=31`, `source_ssd=1`,
`source_approximate=0`. A/C select source660:1694→destination308:1342 with
pad0; B/D select658:1692→308:1342 with pad1. Latest user starts1345.
The four subsequent tool-result requests each retain the live checkpoint via
`match=visible-prefix cached=1530 approximate=1`. There are no extra second
grafts. The no-aligned-source fallback lines belong to cold warmup requests;
no edited request fell back. Concurrent log slices belong to groups and must
be counted once, not inferred from empty per-step log arrays.

All warm answers are READY. Edited gold pairs remain A `(313, BEACON-46)`,
B `(231, ORBIT-57)`, C `(267, CEDAR-70)`, D `(197, DELTA-83)`.
These totals follow the fixed inventory arithmetic; archive values come from
actual fresh file reads, including changed C/D archives after restart.
Native before/after final texts, tool arguments and returned file contents
match exactly across eight steps. Strong B's JSON whitespace differs while
the exact fields pass. No generated-history paired NLL was measured.

## Timing scope

Edited time is the sum of joined AB/CD group makespans, not the sum of
overlapping request times. Whole launcher wall includes source warmup and both
server lifetimes. Queue-inclusive per-request times remain in the originals.

| Arm | Warm groups, s | Edited groups, s | Whole launcher, s | Strict steps / reads |
| --- | ---: | ---: | ---: | --- |
| Resident-native before |181.669484|102.285113|290.226028|8/8 / 8|
| Strong subs3 |184.993350|100.552094|289.840081|8/8 / 8|
| Resident-native after |190.227625|102.922294|297.780041|8/8 / 8|

Strong saves 2.000% edited time against the control mean and 1.694% against
the faster control. Edited control drift is 0.623%. Whole-wall savings are
1.416% against the mean but only 0.133% against the faster control, with
2.603% control drift. Initial server startup is 2.708s for the first control
versus approximately .42s later; source warmup also varies. The measured
combination is compatible and has a modest edited-work improvement here;
this sequence does not establish a robust whole-task speed gain. It is not
the earlier two-code-task combined result, and has no thinking intervention.

## Pressure and evidence

All three guards complete with child0. Normal sample counts are
280/280/288; every sample is level1 with no reason and zero consecutive bad
samples. Swap remains flat at0.012998GiB, so growth is zero. Peak
wired+compressor is85.011719/84.825455/84.475098GiB respectively.

Evidence: [build.json](build.json), [launch.py](launch.py), each arm's
`launcher-summary.json`, `*-execution.json`, `*-server.stderr.log`,
`http/{before,after}/result.json` and group logs; pressure/exit results are in
`guard-{native-before,subs3,native-after}`. No runtime files or driver were
edited by this audit, and no additional workload was selected after outcomes.
