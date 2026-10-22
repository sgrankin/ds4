# Item84 — Loaded Engine Core

Handwritten Core385L owns V4.1 Metal post-family startup and reverse cleanup: absolute Engram path, paired vision load, target bindings, profile, shard accounting/admission, Vocab/overthink policy, support load, GPU/cache/residency setup, support/vision maps and diagnostics. It directly composes existing native owners. The actual common fields are embedded once in Engine.core; TP transport/placement and other-family workspace records remain outside. There is no Engine mirror or field getter/callback facade.

Integration18:25–18:37UTC starts atdca1ffe9. The current record move updates1482 production references and seven fixtures, preserving the prior F16 fixture fix and Session policy case. Four build selections (otherwise-native C Core, native ReleaseFast/Debug and full C fallback) pass the Session, server, Agent and sparse GGUF suites. All supported programs build; the separately changed Qwen private-state fixture passes Clang syntax checking. Initial wrong target name and Zig parameter-shadow errors are retained; corrected builds pass.

Eight real C/native arms produce identical complete129280-logit frontiers and startup diagnostics for F32 automatic cache, F16/32GB, active0.75/all penalty (46/50 markers), and paired vision-sidecar open/map/close. The vision arm is text inference with a loaded encoder, not an image-encoding quality claim. C/native inspect output with real trained support weights is byte-identical. Trained DSpark causal/rollback/accepted-token/RNG/snapshot/cancel decisions are byte-identical too. Debug and full-C fallback each match the C F16 frontier.

One concrete invariant extends the EXISTING allocation-accounting fixture: reject unsupported power after target/family load, then immediately open a valid Engine in the SAME process. All four selections pass that recovery followed by Session allocation/free/recreate checks, protecting partial-init map/instance-lock cleanup. No independent harness or copied implementation.

All final pressure samples are normal and active source hashes remain unchanged. Full frontier JSON is compressed here; binaries/models/cache backing remain in the workspace. Ordered functional checks, no speed claim. Model routing/option prologue, TP transport and explicit nonnull GPU placement remain C. Strict support/vision binding and complete profile/thread/warm/OS/diagnostic libraries are explicit temporary C boundaries; this is not a claim that the whole engine rewrite is finished.

```
tools/zig-build ds4 ds4-agent ds4-server ds4-bench graph-test dspark-native-oracle \
  session-state-test gguf-model-test server-test agent-test -Dnative=true \
  --prefix .workspaces/core-repeat/native
python3 -B speed-bench/ideas/pressure_run.py .workspaces/core-repeat/guard -- \
  python3 -B speed-bench/experiments/zig-engine-core-20261022/qualify.py \
  --workspace .workspaces/core-repeat
```

The paired driver expects control/bin and native/bin. Core-only C control omits DS4_NATIVE_ENGINE_CORE and the appended Core facade from the graph object; build-control.py records/restores that temporary selection. Record-only control predates native selection. Exact commands/source hashes/binary hashes and all driver/compilation failures are retained.
