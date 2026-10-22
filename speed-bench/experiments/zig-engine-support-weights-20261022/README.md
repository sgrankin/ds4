# Item85 — Strict support metadata and tensor binding

The pure411-line SupportWeights owner handles ordered metadata aliases/stage discovery, checkpoint-family admission, optional DSpark binding/layout diagnostics and legacy MTP required binding. Core composes it directly; the one input facade serves C/metadata callers. Summary/Kind/MTP records move once to their actual shared header; tensors and strings borrow Model. Four temporary Core C binder adapters are removed. No metadata loosening, allocation, shader or runtime configuration change.

Integration18:37–18:50UTC starts at44d96ef1. The fixed C-binder control reuses that qualified native Core's exact binaries (copied, hashes recorded). Native ReleaseFast/Debug and full-C fallback compile and pass Session-state and sparse GGUF fixtures. Real trained1017 DSpark inspect output matches C exactly, as do causal verify/rollback, accepted-token/RNG, snapshot/continuation, cancellation and bounds decisions across the complete runtime pair. Debug and fallback also match inspect output. Legacy MTP code is implemented but its unavailable legacy model is not runtime-qualified here.

The first integration build over-guarded a shared type predicate used by remaining C drafter controllers. Only its unchanged enum/predicate moved outside the binder-only inverse guard; native binding remains direct. Failed/successful build records and the guard correction are retained. No runtime/numerical correction.

One initial native inspect took2.66s. A C/N/N/C recheck took0.143/0.120/0.105/0.111s, so the isolated delay did not recur. Fresh Debug/fallback executables also had isolated2.5s first inspect runs; no cause or statistical speed claim is inferred. Every final pressure sample is normal and active source hashes are unchanged.

```
tools/zig-build ds4 dspark-native-oracle gguf-model-test session-state-test \
  -Dnative=true --prefix .workspaces/support-repeat/native
python3 -B speed-bench/ideas/pressure_run.py .workspaces/support-repeat/guard -- \
  python3 -B speed-bench/experiments/zig-engine-support-weights-20261022/qualify.py \
  --prefix .workspaces/support-repeat/native --out .workspaces/support-repeat/cases
```

Add --inspect-only for the short metadata path. Use the parent44d96ef1 native binaries as the C-binder baseline; actual graphs still use the same shader/numeric owners. Vision binding remains the next explicit C library boundary.
