# Item79 — Canonical and private checkpoint transactions

A handwritten pure owner and twelve real actions in the ONE graph facade own
DS41 local canonical sizing/save/load, memory snapshots, staged-file ownership
and DRP1 private replay snapshots. They borrow actual Frontier/Graph/Config and
reuse the qualified numerical Payload owner. No record mirror or format change.
Header/token failures preserve the old frontier; failures after writes cannot
leave it exportable. Canonical versus private close-error/invalidation policy,
old-length versus zero-on-entry, and fmemopen's extra byte retain original C
contracts. Distributed/TP routing and outer legacy invalidation remain explicit.

Integration17:54–18:05UTC starts at114cd84e. C checkpoint control, native
ReleaseFast/Debug and full-C fallback all build and pass model-free state/server
fixtures, plus sixteen actual GPU/model cases. Each selection passes GPU state,
canonical F32/F16 snapshot+continued decode at1/2/3/127/128/129 and the existing
8193-token private replay→SSD close/reopen→fresh F16 receiver probe. Every probe
preserves all63,479,384 payload bytes and private approximate identity, blocks
canonical/suffix export and wrong-profile/suffix lookup, resaves exactly, then
produces all129280 finite fresh logits byte-identical to uninterrupted execution.
results.json also records cross-selection SHA256 of actual persisted chunks;
these are separate from the probe's within-run fresh-logit comparison.

C-first F16 qualification found an OLD fixture bug after all six continuation
frontiers passed: its deterministic fill/erase/check treated canonical wire-byte
counts as F32 backing storage, even for F16 tensors. The existing case now uses
finite exactly representable half bits and the actual half-sized extent, with
an explicit storage-size assertion. Original F32 checks remain. Initial failed
C artifacts are retained; corrected C control passes before native activation.
No runtime implementation was changed to accommodate that fixture.

All final native compilations pass on first attempt; no shader, numerical
threshold or format changed. The active source hashes match across final native/
Debug/fallback runtime qualification. All four final guards remain normal with
fixed swap. No latency claim: these are functional transactions. Cache backing
stores/binaries/models stay in the workspace; published evidence retains only
commands/settings/hashes, raw logs and compact probe summaries.

```
tools/zig-build graph-test session-state-test session-state-gpu-test \
  replay-checkpoint-bench server-test -Dnative=true \
  --prefix .workspaces/checkpoint-repeat/native
python3 -B speed-bench/ideas/pressure_run.py .workspaces/checkpoint-repeat/guard -- \
  python3 -B speed-bench/experiments/zig-session-checkpoint-20261022/qualify.py \
  --prefix .workspaces/checkpoint-repeat/native --out .workspaces/checkpoint-repeat/cases
```

The new build step only exposes the existing speed-bench/replay_checkpoint_bench.c
through Zig; no test implementation was copied. Use -Dnative=false for full C and
-Doptimize=Debug for native runtime checking. Source/caller lifetime and suffix
transaction ownership remain subsequent owners.
