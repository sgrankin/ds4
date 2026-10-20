# Compact stack and DeepSeek V4.1 Flash 1017 upgrade

2026-10-20, Apple M5 Max / 128 GiB. Runtime source is the compatibility commit
`c820ed50` atop upstream `5379c1c9ac31`; publication-only files follow it.
[Machine-readable record](record.json). Historical September performance numbers
remain historical; no earlier percentage is assigned to these new weights.

## Stack and disk cleanup

The original active first-parent stack had 198 commits (248 local ancestors,
including merges). Its archived tip is
`archive/pre-1017-20261020` → `6110237291321b6e93a989a5b706a26d07497ea4`.
The six reconstructed file-group commits reproduced the original final tree
byte-for-byte before rebasing onto current upstream. Shared backend/cache code
stays together to avoid fragile partial-file layers; agent and server consumers,
user documentation, reusable drivers and historical results are separate layers.
The small 1017 compatibility fix and this journal make eight local commits.

112 obsolete empty workspace registrations were forgotten, and 121 verified
empty tips abandoned. Source-bearing private branches and workspace directories
were retained. The original audit is [stack-audit.md](stack-audit.md); its head
counts describe the initial graph, not the resulting graph.

Removed 65,509,171,200 allocated bytes (61.01 GiB) of untracked generated `raw.fp32`,
KV and checkpoint-chunk payloads. Source, commands, reports and scalar metrics
were retained. [Cleanup result](cleanup-result.json). Full selected-file and
workspace inventories remain in ignored `.workspaces/upgrade-20261020/`.

## Current artifacts and compatibility

The pinned files, repositories, revisions, sizes and SHA256 values are recorded
in [record.json](record.json). Both language and matching vision files passed
full local SHA256 reads and expected-size checks. `ds4flash.gguf` now points to the verified 1017 Q2 file;
its previous Qwen target is retained in the record. Actual 1017 model execution
[passed all 16 first-attempt checks](qualification-status.json); scope and
limits are recorded below.

The upstream default branch is `main`. Its latest source supports the 1017 text
layout, but the vision loader still fixed the older source revision. Applied
[nazeshinjite's PR #1225](https://github.com/antirez/ds4/pull/1225), revision
`ef7894bf3c2e03ae828c1a7ef0d553ed5a61d7ef`, to pair language and vision by their
`general.source.revision`. Fixed the downloader's stale Q2 checksum and added an
explicit `ds41f-1017-vision` target; the earlier Q4/vision pair stays available.

```sh
./download_model.sh ds41f-q2
./download_model.sh ds41f-1017-vision
./ds4-local persistent-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
```

## Completed source checks

The main build and listed source checks passed; original logs, including setup
failures followed by their corrections, are under [checks/](checks/).

```sh
make -j2 ds4 ds4-server ds4-agent ds4_test ds4_agent_test tests/test_resident_route tests/test_engram
./ds4_test --server
./ds4_agent_test
./tests/test_resident_route
./tests/test_engram
make -C gguf-tools libds4quants.dylib
python3 tests/test_deepseek41_conversion.py
python3 tests/test_deepseek41_vision_conversion.py
python3 tests/test_model_download.py
```

Also passed standalone SSD-cache, header-only suffix-matcher, and agent suffix
mock checks. The mock covers protected latest-user state, immutable pre-RoPE
left/right repeated imports, cache restart, rollback, opaque turn IDs, complete
unlabelled-prefix preservation, native control spellings and rejected edits.
The first mock run lacked its required existing directory; after creating that
directory, the unchanged test passed. These are source/API checks, not new-model
execution. Rebase notes: [core](resolution-ds4.md),
[build and Engram](resolution-build-engram.md). A subsequent syntax check moved
`shared_here` into common Apple/non-Apple scope before the final build.

Known GPU-test failure: the later SSD-expert memory-release/resize check fails
on both baseline and candidate, with about28MiB RSS growth per reseed; initial
numerical comparisons and address checks pass. The cause remains unresolved;
a command-pool trial was ineffective and reverted. The separate kernel ABBA
showed no benefit (+0.026%), so runtime code is unchanged. See
[kernel notes](kernel/README.md) and [recipe10](../../../recipes/10-q2-kernel-epilogue.md).
The passed checks listed above do not imply that every GPU test passes.

## Actual-model qualification and scheduler experiment

All 16 first-attempt checks passed: six native-server checks (generation,
Responses parent/get/previous ID, Conversations recall and image OCR), then ten
`persistent-fast` agent checks including archival, stable opaque IDs, an actual
fresh tool retrieval and an eight-case first coding implementation. Both exited
cleanly under the sequential pressure guard. Server and agent wall times were
28.37 s and 110.03 s; these are unpaired screen durations, not speed comparisons.
The agent used context 16,384/chunk 2,048, temperature 0/no thinking and
hot state cache 0 MiB/SSD cache 4,096 MiB;
no suffix activation occurred in that agent screen. The screen is small, not a
general quality benchmark. [Results](qualification/result.json),
[grade](qualification/agent/coding-grade.json),
[server command](qualification/server/execution.json),
[agent command](qualification/agent/execution.json).

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/upgrade-20261020/qualification-live-guard -- \
  python3 speed-bench/experiments/1017-upgrade-20261020/qualify_1017.py \
  --model "$PWD/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf" \
  --vision "$PWD/gguf/DeepSeek-V4.1-Flash-1017-vision-BF16.gguf" \
  --output .workspaces/upgrade-20261020/qualification-live

python3 speed-bench/ideas/pressure_run.py .workspaces/upgrade-20261020/quantum-clean-guard -- \
  python3 speed-bench/experiments/1017-upgrade-20261020/serve_quantum.py \
  --model "$PWD/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf" \
  --output .workspaces/upgrade-20261020/quantum-clean
```

The first driver checks native chat, durable Responses/previous IDs,
Conversations, actual image OCR, then `persistent-fast` context archival and a
fresh retrieval receipt, followed by one tiny first coding implementation.
The existing [hour39 repeated-relocation probe](../recipe-rightshift-hour39-20261008/repeat.c)
also compiled against the rebased objects (`checks/right-repeat-build.log`).
Its three moves are right 66, left 128, right 514 from immutable originals. The
prepared binary is `.workspaces/upgrade-20261020/right-repeat-1017`; its existing
arguments are `--model FILE --out FRESH_DIR --moves 3 --prerope 1 --ctx 8192
--chunk 512 --cache-gb 64`. Run under the same guard with inherited DS4/PT
settings cleared. The actual 1017 run completed all three moves from one
immutable source, reusing 513 tokens and computing 128 fresh per move. Summed edit
time was 205.185 s native versus 143.068 s reuse: **30.27% edit-time saving**,
excluding source setup and teacher continuation. This is not whole-agent time.
Across 12 easy fixed-teacher rows, mean ΔNLL +0.00006524/KL 0.00003049; worst
ΔNLL +0.0002171/KL 0.0001108; zero argmax flips, zero byte-identical logit rows.
The state and pre-RoPE hashes remained unchanged. [Summary](repeat/summary.json),
[actual geometry/times](repeat/moves.csv), [quality rows](repeat/rows.csv),
[execution](repeat/execution.json), [stderr](repeat/stderr.log).

The clean scheduler bracket **128 → 512 → 2048 → 128** completed with all eight
answers correct and normal stops. Long A completion was215.83/304.50/96.02/208.39s;
active B maximum content gaps were7.44/19.30/20.39/7.20s. The128 A bracket drift
was−3.45%, mean212.11s. Quantum2048 saved54.73% versus that mean while increasing
the maximum stream gap2.78×. Quantum512 was dominated by both128 controls.
Interactive128 stays the default;2048 is an opt-in bulk-prefill tradeoff, not a
general throughput result or a DeepInfer asynchronous-scheduler port.

![Scheduler completion versus content-gap tradeoff](scheduler/pareto.png)

[Scheduler summary](scheduler/summary.json), [CSV](scheduler/curve.csv),
[original small results/SSE/commands](scheduler/),
[pressure summary](scheduler/guard/pressure-summary.json). The sequential guard
completed child0 with830 normal samples/no adverse samples; peak wired+compressor
98.04GiB and swap growth21.94MiB. Cold server/expert caches reset each arm, but
OS/SSD state carries over. B's initial TTFT is a cold/order diagnostic; SSE gaps
are not device token-latency measurements. The earlier Homebrew-update
interruption and its partial512 arm remain separately retained in
[scheduler-interrupted/](scheduler-interrupted/), excluded from the clean curve.

## Lossless scales and official sources

The [recipe journal](../../../recipes/README.md) contains portable algorithms,
knobs, evidence and rejected directions. The native-scale codec passed real-byte
round trips with 3.670404% net expert-byte saving; hot C expansion was 29.844 GB/s.
This is a transport prototype, not a model-speed result or expert-cache RAM gain.
A separate real Q2 census found the routed-expert scale fields incompatible
with this exact scheme. Native Engram inside that same Q2 file is eligible in
eight fixed windows: **1.886079% sample saving**, byte-exact. The ideal whole-Q2
ceiling from both full Engram tables is 1.050033% before overhead; the sample
does not prove full-table eligibility. No runtime or serving gain is claimed.
See [packing recipe](../../../recipes/07-native-scale-packing.md) and
[Engram evidence](../../../recipes/evidence/engram-scale/README.md).

Official repositories were cloned beside this checkout at
`/Volumes/Models/DeepInfer` and `/Volumes/Models/deepseek-recipe` (SHAs in the
record). [The source lesson](../../../recipes/08-official-inference-lessons.md)
distinguishes approximate bounded decoder replay from exact compressed/index KV,
and identifies bounded mixed-prefill scheduling as the immediate local test.
The full cluster stack was not imported into this Metal runtime.
