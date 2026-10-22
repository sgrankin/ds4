# Remaining DeepSeek Metal controllers

HC split/producer/expand, shared Q8/BF16, scalar/batch/visual router, FP8/QAT/raw-store primitives, carry/pre-RoPE/candidates/BF16 vision projection and TP gate/fold/poll/prefetch now select their native owners together. Actual shader packets and process policy records move once into shared headers. Fixed ARC leaves retain actual Metal resources; MSL and numerical policies are unchanged.

Each selection passes its focused existing GPU fixture in ReleaseFast. The original controllers also pass after the record moves, before these selections. HC, shared/router, primitives and supported-tail first builds succeed. TP required one opaque-pointer type correction; the first compiler failure is retained. No runtime or tolerance fix was needed in this integration. Prior compiler/runtime corrections belong to the earlier [Metal report](../zig-metal-controllers-20261022/README.md).

The existing command-memory fixture now checks real two-stage Q8 FFN admission, abort, incomplete/successful join and retirement, plus deferred KV flush/concurrent transitions with exact q/KV/raw buffers and untouched guards. A narrow observation entry calls the two actual private stage functions, avoiding a fabricated scheduler. Both original and native controllers pass. The existing 32-case shared BF16 fixture is now a build target and passes.

Local TP row/session/batch/big fixtures pass. This is not two-host qualification or a positive-EWMA prefetch measurement. Native TP prefetch owns range filtering, budget, prioritized wrapping and launch geometry; the existing transport remains an explicit library. Full combined Debug/model/performance qualification follows separately. Public model-map lifecycle, bootstrap/retirement and route diagnostics remain pending native owners at this checkpoint.

Commands, initial failures, source hashes and guard results are in [evidence](evidence/). Useful binaries remain under `.workspaces/zig-metal-final-20261022/`.
