# Native F16 cache-publication controller

Completed 2026-10-22 04:34 UTC, inside the one-hour ceiling. Base is tokenizer
commit `bec426ca` on `zig-modular`.

The readable handwritten controller from the preserved `zig-rewrite` work is
retained selectively. Zig owns range checks, UINT32-sized chunks, shared shader
arguments, kernel selection and launch geometry. A small copy-specific ObjC
bridge owns Apple objects and encoders. Generated bindings come from its project
header; both C and Zig use the same 136-byte copy packet. There is no restored
SDK dump or generic dispatch framework. All other copy encoders remain C.

A single retained command token spans all chunks. Release drops that retain on
success/failure, finishes only a successful owned command, and leaves a caller's
borrowed batch active. Tensor offsets remain relative until the bridge adds the
underlying buffer-view offset exactly once. Narrow autorelease pools bound the
Apple-object temporaries, preserving the preceding lifetime fixes.

All five C/native production programs and the actual GPU fixture build, with
empty build stderr. The recovered production-controller CPU fixture passes in
Debug and ReleaseFast: **107 cases, 138 descriptors and 2,181 checks** per run.
It covers both kernels, views, bounds, fake counts above UINT32_MAX and ownership
failure cleanup. Linkage inspection confirms this CPU fixture uses libc only;
it does not pretend to execute Metal.

Actual GPU C/native packed-publication/gather checks pass. The native generic
copy override also passes, including offset canaries and exact values. A fixed
real-1017 model run then explicitly enables `DS4_V41_COMPRESSED_KV_F16=1` and
confirms `main=f16` allocation. The preceding native-host build with C publication
is compared to the new native controller: both use the same host/tokenizer
modules, 512-token prompt, chunk 512, 16 forced tokens and 48 GiB expert cache.
**Full frontier logits are identical**. Prefill is 34.18/33.62 tokens/s and steady
decode 22.16/22.36 tokens/s; this short pair establishes no speedup or material
runtime cost. Publication is exercised only when that compressed-F16 feature is
enabled, not claimed as every ordinary decode path.

The broader default-fast-math Metal suite exposed a quantization oracle mismatch
at mode 3/index 5428 (`11.25` versus `7.5`) in **both C and Zig**, before publication.
It is retained as a baseline finding, not hidden by weakening a tolerance.
Both complete suites pass with the existing `DS4_METAL_MATH_SAFE=1` strict-IEEE
diagnostic setting. This implicates the existing numerical policy rather than
the controller port; no shader, oracle or runtime default changes are made.

Decision: **retain**. Build `tools/zig-build metal-test native-f16-publication-test`;
run `test-f16-publication` directly. Run GPU/model checks serially under the guard.
The three recorded loops preserve [targeted/default checks](qualify.py),
[C-baseline/model checks](qualify-model.py), [strict suite checks](qualify-safe.py)
and their [results](results.json). Full source/binary/pressure/stdout records stay
under `.workspaces/zig-metal-publication-20261022/`.
