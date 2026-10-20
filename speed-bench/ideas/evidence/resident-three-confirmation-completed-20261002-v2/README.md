# Completed held-out broader-three confirmation

ROOT session 7261 and the MAIN guard completed with rc0. Fresh native-before,
native-after and reference arrays have identical full SHA and all 512 raw rows
are bit-exact; both native whole-engine IO snapshots are identical. The candidate
uses the same helper/runtime/shaders, with the broader-three policy frozen before
these confirmation outcomes.

| Measure | Native | Broader three |
| --- | --- | --- |
| Mean teacher-forced NLL | 0.394008867483 | 0.401234149475 |
| Whole P+D logical requested/returned bytes | 207277056000 | 147995320320 |

Candidate mean deltaNLL is **+0.007225282** (positive means worse target likelihood).
The paired 16-prompt bootstrap 95% interval is [-0.007474102, +0.020921817],
crossing zero. It uses 2000 resamples, seed 20261002, sorted indices 49/1949.
Worst prompt mean deltaNLL is +0.043891409; mean KL is 0.012595252,
KL p99 0.169840251, maximum 0.603620837; TV mean 0.028612725;
argmax flips are 20/512 and only 16/512 rows are bit-identical.
Every row and every per-prompt result is retained; no tail or case was removed.

Whole prefill-plus-decode logical demanded bytes decrease by **28.600240%**
(59281735680 bytes). Requested, returned and cache-loader counters
agree within each arm. Route-footer skipped bytes are a different counter and are
not added to this total. This is not measured physical SSD traffic. Diagnostic
timings remain in original outputs but support no clean speed claim here. The
positive mean and interval crossing zero do not establish equal quality.

The untouched inputs are the first 16 case indices 2 mod 6 from the frozen 96-case
set, distinct from calibration/earlier holdouts. Their exact 32 input files, hashes,
original TSV and selection proof are archived without outcome-based selection.
Settings: ctx4096/cache64 GiB/chunk512, TILE64/LIMIT1024, 32 teacher-forced tokens
per prompt; broader3 gap.30/mass.20/total.40/rank48/MAX_SUBS3/apply/replace.
All inherited DS4 names were cleared by the frozen driver. Policy stays opt-in.

The copied compact evidence includes all summaries, 512-row CSVs, per-prompt CSV,
checks, reference metadata, strict commands/exits/logs, ROOT launch/preflight/release,
MAIN guard result and 140 normal pressure samples with zero adverse samples/swap
growth. Driver/tests/frontend and source/build/binary/object/shader identities are
verified and inspectable. The ambiguously named original CPU test-hash field pins
captured cpu-test.json stdout, not test source. Its clarification and the first
archiver refusal are retained; the archived test is pinned to verified immutable38
SHA. A fresh run of the actual38 focused CPU tests passed all 37 trials and emitted
the identical stdout SHA. Its first invocation, which failed before trials because
the new workspace lacked the ignored temporary-artifacts parent, is preserved;
only that empty directory was created before retry. No test source changed.
The original proof is unchanged. None of the 64 run assets changed.
All four 264777744-byte raw arrays were independently
stream-hashed once and remain at their original exact paths; they are not copied.
The inventory records full hashes, sizes and row layout. The original run remains
unchanged. The archiver opened no model metadata or tokenizer/model payload,
executed no build/helper/GPU work, and acquired no model-job lock. Model identity
is historical run data.
No default promotion or generalization claim is made.
