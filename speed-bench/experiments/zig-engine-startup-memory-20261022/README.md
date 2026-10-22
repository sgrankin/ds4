# Item82 — Ordered startup cache sizing

The existing native Memory owner now owns requested-budget capping, automatic cache selection, and prefill reserve/whole-slot conversion in their original order. It uses actual accounting/config/weights, shared SSD planning, and existing GPU recommendation APIs. No Engine mirror, callback or new configuration option. C global compatibility resets and other-family fallback remain explicit.

Integration18:18–18:21UTC starts at6d97f1bc. C control, native ReleaseFast and native Debug build and pass the existing SSD fixture. C/native extended --memory-plan cases preserve one complete expert slot after reducing prefill reserve, cap count targets to cacheable layers and clear obsolete reserve/full-layer state.

Six real-model C/native arms exercise default automatic sizing,32GB byte target and512 expert slots. Each pair chooses identical startup budgets/diagnostics and produces identical complete129280-logit frontiers. Debug automatic sizing also matches the C frontier. Every pressure sample is normal; exact commands, source/binary hashes, timings and pressure samples are retained. Ordered functional comparisons, no speed claim. Full-C fallback was qualified in the preceding sampling step; this item retains/compiles the C startup bodies in the otherwise-native control.

The initial native compilation exposed three divisions where a compile-time u64 maximum was inferred against a u32 denominator. Explicit u64 denominator conversion preserves the original arithmetic. The failure and successful rebuild are retained; no runtime/tolerance change. Source hashes are unchanged through final qualification. Full frontier JSON is gzip-compressed in this report; workspace originals remain.

```
tools/zig-build ds4 graph-test ssd-test -Dnative=true --prefix .workspaces/startup-repeat/native
.workspaces/startup-repeat/native/bin/test-ssd-cache
python3 -B speed-bench/ideas/pressure_run.py .workspaces/startup-repeat/guard -- \
  python3 -B speed-bench/experiments/zig-engine-startup-memory-20261022/qualify.py \
  --workspace .workspaces/startup-repeat
```

qualify.py expects control/bin and native/bin. To isolate this owner, omit DS4_NATIVE_ENGINE_STARTUP_MEMORY only in the control Engine definitions. Actual target-view mapping/static locking, Core startup/cleanup and optional support/vision orchestration remain subsequent owners.
