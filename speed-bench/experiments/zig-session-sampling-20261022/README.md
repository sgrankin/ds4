# Item81 — Public Session sampling and diagnostics

A handwritten 195-line owner selects greedy/sampled tokens, applies the optional generated-thinking penalty in a separate lazy buffer, and provides raw logprob/copy/set operations. The actual five-field overthink policy is declared once and borrowed alongside the real Frontier/Vocab. Numeric filtering/RNG stays with the existing native Sampling library; no Engine/Session mirror, graph dependency or shader change.

Integration18:15–18:18UTC starts at1439360d. C-first, native ReleaseFast/Debug and full-C fallback all pass the existing sampling suite and one added production-linked Session case. That case checks the generated threshold, raw versus adjusted selection and copies, raw diagnostics with invalid checkpoint, stop controls, scratch reuse, common-prefix invalidation and exact numeric RNG reuse. No copied owner or weakened assertion.

Six real-model runs compare ordinary seeded generation, active penalty generation, and active-penalty greedy/logprob output. C/native output matches in every pair; the diagnostic JSON (six raw alternatives per step) is identical. The active policy maps46/50 exact vocabulary markers at0.75/all/after0. Full-vocabulary KL is not measured and no speed improvement is claimed. Sources remain unchanged, pressure is normal throughout.

The first mechanical patch missed one include after previous header moves; the failed build is retained. The include was rebased and C control rerun before native selection. All native compilations and final fixtures pass.

```
tools/zig-build ds4 session-state-test sampling-test -Dnative=true \
  -Doptimize=ReleaseFast --prefix .workspaces/sampling-repeat/native
.workspaces/sampling-repeat/native/bin/test-session-state
.workspaces/sampling-repeat/native/bin/test-sampling
```

model.py records the exact six real-model commands/settings and binary/source hashes. It uses its named workspace; use a fresh output directory when repeating. To isolate this owner, omit DS4_NATIVE_SESSION_SAMPLING in the native Engine definitions while retaining shared records. Generic rewrite/sync routing, policy configuration and other-family dispatch remain C.
