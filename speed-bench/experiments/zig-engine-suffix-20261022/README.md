# Native V4.1 suffix state

Item 60 began at 13:37 UTC on 2026-10-22 with a one-hour budget ending at
14:37 UTC. ReleaseFast/Debug builds and four model-free host checks passed.
All six ReleaseFast model arms, actual Debug relocation and the practical
six-turn agent check passed. Qualification completed at about 14:25 UTC.
The change moves
pre-RoPE archive allocation/capture/prefix restoration and graph suffix grafts
into native Zig.
Session provenance, checkpoint/approximation policy, cancellation callback and
fresh-tail evaluation remain with the C facade. The canonical payload and
pointer-free original-tape formats stay unchanged.

## Bounded existing probe

The [existing right/left repeat helper](../recipe-rightshift-hour39-20261008/repeat.c)
already has the concrete checks needed here, including the Item 51 F16 extension.
It captures one exact source at position 1,537, retains 513 suffix tokens and
uses a fresh 128-token tail. Its four moves shift right 66, left 128, right 514
and zero positions, always from the same immutable source. Each move compares
four full-vocabulary teacher rows against full recomputation within that binary.
These are forced continuations, not independently generated answer grades.

The helper also restores the source prefix into a newly created receiving
session and checks canonical bytes plus original tape; rejects legacy right
relocation before mutation; checks three pending compression pairs, ring/key
repair counts and zero-shift repair counts; verifies immutable source hashes;
and recaptures each approximate graft. The separate legacy case exercises one
left-128 move without pre-RoPE originals. It uses one mutable session at a time.

[qualify.py](qualify.py) runs three ordered control/native pairs:

| Pair | Source and destination storage | Moves | Originals |
| --- | --- | ---: | --- |
| Ordinary pre-RoPE | F32 / F32 | 4 | Yes |
| Packed pre-RoPE | F16 / F16 | 4 | Yes |
| Legacy left | F16 / F16 | 1 | No |

All use context 4,096, chunk 512 and a 32 GB SSD expert-cache target. Resident
substitution and bounded decoder replay are disabled. The control is the frozen
[payload-stage engine](../zig-engine-payload-20261022/README.md) with C suffix
ownership; the candidate changes only suffix ownership. `arm=native` inside
`moves.csv` means full recomputation, while `arm=reuse` means grafting; those
labels do not identify which implementation produced the file.

The driver compares complete raw teacher-vector hashes, token IDs, non-timing
geometry/source hashes and existing quality rows across implementations. It
records the helper's delta-NLL/KL/argmax observations without using degradation
as an automatic rejection rule. Valid execution and requested move counts must
pass. Recompute-versus-graft quality measures stale-prefix approximation;
control-versus-native comparison measures this ownership migration. Raw vectors
stay in the ignored workspace. These ordered pairs are compatibility checks,
not a latency bracket or new suffix-quality claim.

The existing six-turn archival/repeated-suffix agent workflow adds a practical
check of fresh fact retrieval and grafts after prior approximate use; its
completed result is reported below.

## Reproduction

Preserve the control helper before activating native suffix ownership. Build
the same existing C source as `suffix-repeat` in both selected prefixes, then run
serially under the existing pressure guard:

```sh
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-suffix-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-engine-suffix-20261022/qualify.py \
    --control-bin-dir .workspaces/zig-engine-20261022/payload-native/bin \
    --native-bin-dir .workspaces/zig-engine-20261022/suffix-native/bin
```

The driver defaults to fresh `.workspaces/zig-engine-suffix-rerun`. It retains
commands, binary/helper hashes, first logs, CSV and vector hashes. Root owns all
builds and guarded model/GPU execution. No job has been run by the report author.


## Build and host checks

The frozen C-suffix control helper built in 2.223 s. The initial candidate build
passed in 10.561 s with two unused C-reference archive helpers; narrow unused
attributes retained those helpers. The final ReleaseFast build passed in
18.603 s and the Debug build in 3.038 s, with empty stderr. These are cached
development builds, not clean-build performance comparisons.

The [four model-free host checks](evidence/host-results.json) passed: real
canonical/diagnostic source parsing and approximate-state export exclusions in
both ReleaseFast and Debug. They open neither a model nor the GPU. The separate model campaign below exercises actual relocation.

## Results

All [six ReleaseFast arms](evidence/model/results.json) completed. Each of the
three control/native pairs has identical complete teacher-vector hashes, token
files, non-timing geometry/source hashes and quality rows. The two pre-RoPE pairs
each cover 16 teacher rows per implementation; the legacy pair covers four.
The control/native comparison therefore contains 36 paired teacher observations
and two complete vectors (recomputation/graft) per observation.

| Case | Graft minus recompute mean NLL | Mean KL | Argmax flips | C/native vectors |
| --- | ---: | ---: | ---: | --- |
| F32 pre-RoPE, four moves | +0.0000372001 nats/token | 0.0000284266 | 0/16 | All identical |
| F16 pre-RoPE, four moves | +0.0000372001 nats/token | 0.0000284266 | 0/16 | All identical |
| F16 legacy left | +0.0000793349 nats/token | 0.0000501003 | 0/4 | All identical |

Grafted vectors differ from full recomputation, as expected for stale-prefix
reuse. The native owner reproduces the control's existing approximation in all
three cases. These high-probability forced code continuations are not a broad
coding-quality screen.

All original-backed arms retained identical source-state and pre-RoPE hashes,
restored and recaptured the exact prefix/tape, rejected a legacy right shift
before mutation, and recaptured all four approximate grafts. The zero move
reported no key repairs. Legacy left had no original tape by construction.
The raw vectors stay in the ignored workspace; their hashes, actual CSV rows,
synthetic token inputs and first logs are retained. Whole-run and edit timings
are recorded in [summary.json](summary.json), but these ordered pairs do not
establish a port latency improvement.

The [actual Debug legacy F16 relocation](evidence/debug-model/result.json) passed
in 145.754 s, with complete vectors and quality rows identical to ReleaseFast.
The [six-turn agent workflow](evidence/agent/result.json) completed in 231.442 s.
All three context edits led to the correct current total `416` and archived code
`RIVER-73`, with three detected fresh archive reads. Its three SSD/pre-RoPE grafts
include two after prior approximate use; the third source was itself approximate.
This verifies the existing workflow, not an immutable-original-source claim or
a latency comparison. Actual [command/overrides](evidence/agent/execution.json),
first outputs and gzip traces are retained; edited context dumps and cache chunks
are omitted.

All three [pressure campaigns](evidence/pressure-summary.json) completed with
child status zero, normal pressure throughout and zero measured swap growth.
Only compact pressure summaries and guard commands/results are published; the
full samples remain in the ignored workspace. All
[36 explicitly selected source hashes](evidence/active-source-sha256.json) still
match the measured snapshot. Future DSpark, vision and server-owner drafts are
excluded. [Summary calculations](summary.json) retain every case's settings,
quality observations, timings and compatibility scope.
