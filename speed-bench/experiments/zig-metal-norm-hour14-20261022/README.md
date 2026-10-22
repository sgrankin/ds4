# Native ordinary RMS normalization controllers

Completed 2026-10-22 04:59 UTC, inside the one-hour ceiling. Base `0876d4c1`.
Seven handwritten exports in `zig/metal/norm.zig` own scalar/row plain and
weighted RMS, BF16 output, additive RMS and fused Q/K normalization dispatch.
They use actual144-byte RMS and40-byte QKV packets through a norm-specific
encoder descriptor. Objective-C owns Apple resources, encoders and lexical
autorelease pools; deferred KV/task state stays C. Ordinary V41 scalar/prefill
uses weighted normalization; the generic fused-QKV path is qualified separately.

The mapped-weight retain/release helpers now have shared model-buffer names,
because HC and RMS both use the resource contract. No dispatch framework is
introduced. HC bodies only change internal helper names, and its actual-object
oracle is rerun. Bounds, aliases, mapped offsets, command ownership, failure
cleanup and original threadgroup geometry remain explicit.

All five C/native programs and the GPU fixture build with clean final logs.
Actual native-controller CPU oracles pass in Debug/RF: RMS598cases plus scalar
and failure assertions; HC1152cases/50fused launches. These CPU fixtures link
libSystem only and do not claim GPU execution.

The original40 weighted/BF16 GPU cases remain byte-exact with offset canaries
and aliases. Added18 QKV shape/alias/offset cases compare fused versus separate
normalization. Their first exact-byte oracle failed in C itself: a fused launch
can use a different reduction width/tree. Only this new comparison now records
ordinary F32 error, bounded by2e-6*(1+abs(reference)). C and native both report
**the same maximum absolute4.76837158e-7 and scaled1.52790378e-7**. The initial
failure and unchanged40 exact checks are retained; no shader/tolerance/default
change is hidden as a port fix.

Four serial ordinary1017 model runs use C/Zig/Zig/C controllers,512-token
prompt/chunk,32forced tokens,ctx768 and48GiB expert cache. All full frontier
logit vectors match exactly.

| Controller | Prefill tokens/s | Steady decode tokens/s | Process seconds |
| --- | ---: | ---: | ---: |
| C before | 37.94 | 22.43 | 18.993 |
| Zig | 35.28 | 22.12 | 20.283 |
| Zig | 35.01 | 22.13 | 18.051 |
| C after | 34.80 | 22.10 | 20.890 |

The opening/closing prefill control drifts8.3%; the native samples sit near the
closing control. Steady decode differs less than1% in the means. Primitive
2000-call timings are also close. This establishes no reliable speedup or
material runtime regression; it does not turn startup/process timing into a
kernel performance claim.

Decision: **retain**. `tools/zig-build norm-metal-test native-norm-test
native-hc-projection-test`; CPU oracles run directly, GPU/model checks serially
under the guard. [Commands/hashes/results](results.json), [actual loop](qualify.py).
All source/build attempts, first oracle failure and pressure/stdout records
remain in `.workspaces/zig-metal-norm-20261022/`.
