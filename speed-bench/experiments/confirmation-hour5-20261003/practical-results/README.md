# Two unused practical confirmation tasks

Budget256 plus strong affinity replacement completed both tasks and was 23.90%
faster than the mean of the two native brackets (22.64% faster than the faster
native bracket). Budget256 alone was 8.83% faster than the native mean. All eight
first attempts passed the read/write/final-stop contract and all 161 unit inputs
for their generated code. These are two distinct generated tasks per arm, not
322 independent task attempts.

| Arm | Join seconds | Window seconds | Pair seconds | Strict task passes | Sampled decisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| native-before | 74.800 | 59.026 | 133.825 | 2/2 | 1529 |
| budget256 | 63.805 | 56.249 | 120.054 | 2/2 | 1351 |
| budget256-affinity | 53.038 | 47.180 | 100.218 | 2/2 | 1238 |
| native-after | 71.453 | 58.095 | 129.548 | 2/2 | 1529 |

Native pair times were 133.825 and 129.548 seconds, with mean 131.687 seconds.
The native repeats have identical complete event records, including every token,
piece and RNG-before/after tuple: 849 join samples and 680 window samples. The
combined arm used 641 and 597 samples (1,238 total); budget256 used 1,351 total.
No raw vector identity is claimed or checked. `checks.json` records the exact
comparisons and arithmetic; `results.json`, CSVs and `tradeoffs.png/svg` come from
the existing phase4 metadata summarizer.

All arms used the same combined helper, ctx8192, SSD cache64GiB (6136 configured
slots), chunk512, TILE64/LIMIT1024, seed20261002, temperature0.6, global 2048 decision
cap and PT_LIGHT_AUDIT=1. Budget256 forces the existing close-thinking protocol
after 256 WRITE-phase thinking decisions. The combined arm additionally uses
resident replacement with gap.90, per-slot mass.25, total mass.50, rank64 and up
to 3 slots, retaining the native six coefficient bits. Its actual close footers
report 10,509 join substitutions and 10,954 window substitutions; native-weight
requested=1/substitutions match in both. This proves replacement was active.
Skipped requested loads are not a physical SSD-read measurement.

The linked Metal object contains the admitted-only original-deadline code, but
the old 8a47 core has no deadline activation calls. Original staging remained
disabled. This study tests budget256 plus ordinary affinity replacement.

The MAIN pressure guard completed with child returncode 0: 469 normal level 1
samples, zero bad reasons/consecutive-bad samples and no swap growth. Pre-existing
swap stayed at 0.024600 GiB. Peak wired+compressor was 78.800 GiB; minimum physical RAM
minus those two categories was 49.200 GiB. Detailed actual values and guard result
are in `checks.json`, with the original samples under `../pressure-practical/`.

The fixtures were authored before candidate selection. Both independent gold
methods agree for all 161 inputs per task; the clean reference solutions pass the
reused language, JSON and input-mutation restrictions (`../practical/check-results.json`).
Full projects, first-written solutions, events, stderr and execution records stay
under `runs/`. There were no caps or failed attempts in this practical sequence.

This is one seed and two unused tasks. Process wall time includes model startup,
prefill, tools and close, and routing profiles remain enabled. The result supports
these task outcomes; it does not establish general accuracy, a physical SSD-byte
saving, or a fixed speedup across workloads. Teacher ΔNLL/KL and the separate four
plain-generation cases are recorded outside this practical task study.
