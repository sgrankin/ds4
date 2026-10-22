# Native resident-expert policy

Baseline `71fe8a94`, one-hour ceiling 05:43–06:43 UTC on 2026-10-22.
Nine policy functions move from the inline C header into 213 handwritten Zig
lines. Cache snapshots, GPU work, metrics and commits remain with their existing
callers. The native module imports the actual plan/change records through a
small header; no engine or cache layout is reproduced.

The two- and three-substitution selectors now share one candidate/commit loop.
Their eligibility and summation steps remain separate. Comments explain the
original-route mass budget, tie ordering, uniqueness, rejection outputs,
one-slot exception and 2^-14 normalization floor. This is a language migration:
the approximation and preset settings are unchanged.

## Existing fixtures

All five programs build in C and native modes without warnings. The three
unchanged resident fixtures pass in C/native × Debug/ReleaseFast: 12 runs.
They include 100,000 one/two-slot comparisons against the frozen C selector and
its exact weight bits, 50,000 triple-plan invariant cases and 50,000 drop cases.
The native fixtures link the same selected object as production. Fixture math
follows the existing strict C target; application flags retain their ordinary
production policy.

Build and run the existing fixtures, choosing a separate installation for each
mode:

```sh
tools/zig-build -Dnative=true -Doptimize=ReleaseFast \
  --prefix .workspaces/route-native install resident-route-test
.workspaces/route-native/bin/test-resident-route
.workspaces/route-native/bin/test-resident-three
.workspaces/route-native/bin/test-resident-drop
```

Repeat with `-Dnative=false` and `-Doptimize=Debug` as needed. Commands, hashes,
all four build modes and first-run logs are recorded in
[results.json](results.json) and `.workspaces/zig-resident-route-20261022/`.

## Actual coding caller

The unchanged ledger and LRU CLM coding workflow passes in C/native/C order:
all six first implementations pass 161 checks each, with one actual read and
write, a context edit, completed final response and no cap. Code bytes, final
texts, token counts and route footers repeat across all three arms. Each arm
performs 32,681 substitutions with native mixture weights. Suffix reuse also
runs through the real server/SSD path; resident prefill is off in this screen.

The control is the previous otherwise-native server, isolating the C versus Zig
routing policy. The frozen settings are gap .90, individual mass .25, rank 64,
three substitutions and total mass .50; expert cache is 48 GiB, context 8192,
chunk 512, hot cache 0, SSD 4096 MiB, canonical suffix plus padding, fresh tail64.
The existing decoder-replay helper disables routing internally and would not
exercise this replacement. The launcher requires actual substitution counters.

Combined task times are 204.955 / 209.342 / 207.134 seconds. Native takes 1.60%
more time than the C bracket mean, or 1.07% more than the slower control; control
drift is 1.06%. This small one-bracket observation is retained as a cost, not a
speedup or proof of zero overhead. All pressure samples remain normal, with no
swap growth.

Decision: **retain** the readable shared policy with this measured limitation.
No settings or runtime defaults change. Reproduce the guarded caller with fresh
result/cache paths and the two executable paths:

```sh
python3 -B speed-bench/ideas/pressure_run.py NEW_GUARD -- \
  python3 -B speed-bench/experiments/zig-resident-route-hour23-20261022/coding.py \
  --c-server CONTROL_SERVER --native-server NATIVE_SERVER --output NEW_RESULTS
```

[Exact results, commands and hashes](results.json), [caller loop](coding.py),
[existing-task launcher](coding_launch.py). Raw requests, continuations, cache
manifests and guard records remain under the workspace directory above.
