# Item80 — Session resource ownership

The handwritten 98-line SessionResources owner constructs and retires the V4.1 Metal graph, optional drafter, borrowed TP views and host logits/probability buffers. It admits memory before allocation, publishes only a complete graph, charges once, and frees using the current allocation size. The actual two-field Device and existing TP state each have one shared declaration. Session shell and transport registration/destruction remain C; no Engine/Session layout mirror or field callbacks.

Integration18:05–18:15UTC starts at0a6fbe24. The otherwise-native C control, native ReleaseFast/Debug and full-C fallback build cleanly and pass the existing model-free Session suite. Thirteen GPU/model cases pass: C/native/Debug each run memory planning, allocation accounting with free/recreate under changed carry settings, and scalar snapshot/continued decode; C/native additionally run GPU state and the actual trained DSpark runtime. All DSpark causal verify/rollback, accepted-token/RNG, exact/nonexact sampling, snapshot, cancellation and bounds decisions are byte-identical across the pair. No new test implementation or numerical assertion. The existing F16 fixture correction is preserved while device references move mechanically.

All active source hashes remain unchanged through qualification. Every pressure sample is normal; per-run swap ranges and exact commands are in results.json. No latency claim: this is a functional ownership comparison. F16 and demand-growth math were qualified in preceding owners; this step does not claim two-host TP or repeat those campaigns.

```
tools/zig-build graph-test session-state-test session-state-gpu-test dspark-native-oracle \
  -Dnative=true -Doptimize=ReleaseFast --prefix .workspaces/resources-repeat/native
python3 -B speed-bench/ideas/pressure_run.py .workspaces/resources-repeat/guard -- \
  python3 -B speed-bench/experiments/zig-engine-memory-20261022/qualify.py \
  --prefix .workspaces/resources-repeat/native --out .workspaces/resources-repeat/cases
python3 -B speed-bench/ideas/pressure_run.py .workspaces/resources-repeat/extra-guard -- \
  python3 -B speed-bench/experiments/zig-session-resources-20261022/qualify-extra.py \
  --prefix .workspaces/resources-repeat/native --out .workspaces/resources-repeat/extra
```

Use Debug for Zig safety checking and native=false for full fallback. To isolate only this owner, omit DS4_NATIVE_ENGINE_SESSION_RESOURCES from the native Engine definition list while keeping the same shared records. Useful binaries/model backing stay in the workspace.
