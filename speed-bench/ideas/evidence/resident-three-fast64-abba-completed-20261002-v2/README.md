# Completed native-fast64 versus resident-three ABBA v2

The candidate reduced mean turn model time by 5.1190% on this single fixed
teacher-forced replay: 31.484023s native to 29.872355s candidate. Decode time fell
9.5033%; append prefill rose 0.4707% and startup prefill rose 0.4116%. Total model
phase time including startup fell 1.6551%. This is one ABBA sequence, two fresh
processes per arm, with other CPU work explicitly paused during timing.

| Mean phase time | Native-fast64 A | Resident-three B | B change |
| --- | ---: | ---: | ---: |
| Startup prefill | 52.767869s | 52.985058s | +0.4116% |
| Append prefill | 13.839537s | 13.904681s | +0.4707% |
| Decode | 17.644486s | 15.967674s | -9.5033% |
| Turn append + decode | 31.484023s | 29.872355s | -5.1190% |
| All model phases | 84.251892s | 82.857413s | -1.6551% |

Runtime `b9408c3b8e3ec5b22ad7efff3e365fd81818c317` used the reviewed three-slot objects.
Both arms used cache64/context8192/chunk512/TILE64/LIMIT1024 and 6136 configured
slots at each phase. A used GAP0; B used gap.30/mass.20/total.40/rank48/MAX_SUBS3,
apply/replace. All inherited names beginning with DS4 were cleared. Diagnostic
IO/route/timing profiles were absent. Ordinary untimed cache/route footers are
retained; both candidates report 4516 substitutions, including 21 triple layers.

All run checks passed. Both native repeats match the prior tile64/limit1024
seven phase-end vectors and full final state. Candidate repeats match each
other and differ from native. Those hashes cover seven complete end vectors
and final state; they do not establish semantic correctness or identity of
every scored decode row. This run does not score generated tasks, establish
statistical significance, or promote a default. The policy remains opt-in.
Phase wall timers exclude sampling, tools and later vector/state export/hash.

V1 is preserved under `v1-failed-preflight/`: its MAIN guard completed with
child rc1 because the exact parent argv check rejected the observed macOS
framework Python path before model metadata, replay helper, GPU, or a run
directory. The launcher shim and framework executable are distinct files.
V2 pins the observed framework path and SHA, requires samefile against that
path and exact remaining argv, and retains guard hash/lock checks. Policies,
binary, shaders, replay and prior controls are unchanged. Eight CPU tests
passed including an actual MAIN guarded tiny Python identity/lock child.
The first test suite's sandbox sysctl refusal and later successful CPU-only
override are both retained. No failed evidence was overwritten.

ROOT session 3445 completed with guard rc0. All 332 pressure samples
were normal with zero adverse samples. `plan/root-release-proof.json` records
lock reacquisition, no matching live model process and explicit CPU thaw at
1790981967.6549602. This release record describes that completed run, rather
than asserting the host has no later jobs.

`manifest.json` pins all copied compact evidence, the full prepared asset
hashes, object/source/build proof and binary SHA. Full vectors/state remain
at the original v2 artifact paths in `raw-output-inventory.json`; they were
not copied or rehashed during this archive. Their hashes are the successful
frozen wrapper's recorded hashes. Archive preparation only read compact
evidence and prepared assets; it opened no model, tokenizer or GPU.
