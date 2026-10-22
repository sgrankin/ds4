# Native V4.1 speculative-cycle ownership

Item 62 began at 14:46 UTC on 2026-10-22 with a one-hour budget ending at
15:46 UTC. Source integration, ReleaseFast/Debug builds and all three guarded
runtime arms passed. The native cycle
composes the already-qualified DSpark math/state owner, target Executor and
numeric sampling library. It borrows explicit checkpoint/logit/probability,
cancellation, statistics, policy and TP action arguments without importing a
Session or Engine layout. Four graph facades preserve the C action boundary.
C retains public admission, TP receive/dispatch coordination and rewind entry.
No numerical approximation, sampling policy or runtime tuning flag is added.

## Bounded existing runtime fixture

[qualify.py](qualify.py) runs the actual unchanged
`tests/test_deepseek41_dspark.c --runtime` assertions sequentially in three
selected binaries: frozen C cycle/native Drafter, native cycle ReleaseFast and
native cycle Debug. Four logging lines in that existing fixture expose each
accepted-token list and final RNG state, giving sixteen complete records over
eight cycles each in exact-sampling modes 0 and 1. The driver compares the whole
records exactly, rather than parsing or normalizing their token/RNG values.

The fixture also retains causal keep0..6 verification, rollback and eight fresh
continuation steps per keep, checkpoint/token publication, post-verification
payload reload, cancellation preserving the synchronized checkpoint and
context 2,048 clamping. Successful execution and the sixteen ordered records are
required. First errors/logs are retained; there are no retries. Numeric/state
differences will be reported separately from the existing fixture's acceptance
conditions, without turning this into a new quality or malformed-input suite.

All arms use the matching 1017 target and trained MXFP4 DSpark GGUF from the
[previous item](../zig-engine-dspark-20261022/README.md), context 2,048 and the
existing fixture's 40 GiB SSD expert cache. Resident substitutions and bounded
decoder replay are disabled, with main compressed cache F32. The 49 complete
trained tensor outputs were already qualified in Item61; this item tests the
policy/transaction orchestrator, not a different drafter implementation.

The frozen control at revision `60e65cfb` is
`.workspaces/zig-engine-20261022/cycle-control/bin/dspark-native-oracle`, built
before the cycle flag. Its actual fixture selects the native Drafter and C
cycle. Do not substitute the separate full-C oracle target, which deliberately
omits both native DSpark and cycle macros.

## Reproduction

After preserving that control and freezing the candidate source, ROOT owns
all builds and the serialized pressure-guarded model job:

```sh
python3 -B speed-bench/ideas/pressure_run.py \
  .workspaces/zig-engine-20261022/cycle-model-guard -- \
  python3 -B speed-bench/experiments/zig-engine-cycle-20261022/qualify.py \
    --control-bin-dir .workspaces/zig-engine-20261022/cycle-control/bin \
    --native-bin-dir .workspaces/zig-engine-20261022/cycle-native/bin \
    --debug-bin-dir .workspaces/zig-engine-20261022/cycle-debug/bin
```

Default output is a fresh `.workspaces/zig-engine-20261022/cycle-model`.
Commands, binary/fixture hashes, environment, first logs, exact cycle records
and assertion markers are retained. These ordered arms are compatibility
qualification, not a latency bracket or speculative-throughput measurement.
Physical TP is not exercised. No job has been launched by the report author.

## Build status

The frozen control fixture built in 7.978 s. Initial native ReleaseFast/Debug
builds passed in 10.248/1.787 s. A narrow unused attribute retains the C commit
reference; final rebuilds passed in 8.453/0.614 s with empty stderr. The
[initial and final build records](evidence/build/cycle-native-build2.json)
are retained. These cached development builds are not clean-build performance
comparisons.

## Results

All [three runtime arms](evidence/model/results.json) passed. Native ReleaseFast
and Debug match the frozen C-cycle control's sixteen whole cycle records
exactly: accepted counts, each accepted token ID and final RNG state. Every
existing runtime assertion marker also matches, including zero causal-prefix
logit difference for keep values 0..6, seven rollback/eight-fresh-step cases,
both exact-sampling modes and post-verification snapshot reload, cancellation
without checkpoint mutation and context-limit clamping.

| Implementation | Whole fixture time | Cycle records | Existing runtime checks |
| --- | ---: | --- | --- |
| C cycle / native Drafter | 47.255 s | 16 reference records | Pass |
| Native cycle ReleaseFast | 48.326 s | All 16 exact | Pass |
| Native cycle Debug | 48.210 s | All 16 exact | Pass |

The fixture exercises actual multi-token returns. Exact mode 0 returns
`5,6,1,1,1,3,1,1` tokens over its eight cycles; exact mode 1 returns
`4,6,3,2,2,4,2,3`. Those same records occur in all three arms. These are returned
token counts, including first tokens and possible residual corrections, not
draft acceptance rates. Mode 0 retains its initial RNG state; mode 1 ends at
`5003ce6c11748eb5`, identically across implementations. Full records are retained
in the result and first stderr logs, without normalizing token IDs or RNG.

The [guard](evidence/model/guard-result.json) completed with child status zero.
All [139 samples](evidence/pressure-summary.json) had normal pressure, with
58.554 GiB peak wired memory, 10.387 GiB peak compressor and 0.75 MiB measured swap
growth. This small observed growth is retained rather than rounded to zero.
All [48 selected source hashes](evidence/active-source-sha256.json) still match
the measured snapshot. Commands, binary/fixture hashes, environment, first logs
and [summary calculations](summary.json) are retained; full pressure samples
remain in the ignored workspace.

This qualifies native cycle orchestration over the matching trained Drafter
and target executor for the exercised paths. It does not add a generated-answer
quality suite, exercise physical TP or inject each failure boundary. The
ordered control/native/Debug timings do not establish a speed gain or
speculative throughput.
