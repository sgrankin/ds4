# Packed FP4 index caches and register-based scoring

**The actual DS4.1 index-owner tensors are 86.72% smaller, with compact SSD
checkpoints and old-format imports. A packed-cache serving speed gain is not
established.** This changes index-key storage, not model weights or every KV
cache. Actual allocation, checkpoint conversion, SSD restart, repeated suffix
moves and Responses qualification pass. The qualified executables are installed;
the DeepSeek fast profiles enable packed index storage by default.

`context-fast`, `resident-fast` and `persistent-fast` default
`DS4_V41_INDEX_FP4` and `DS4_V41_INDEX_SIMD` to `1`, using the measured
small-work threshold below. `persistent-packed` remains an alias for
`persistent-fast`. Explicit environment values, including `0`, win:

```nu
./ds4-local persistent-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
./ds4-local --server persistent-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
```

Enable the physical layout before creating sessions:

```nu
with-env {DS4_V41_INDEX_FP4: '1', DS4_V41_INDEX_SIMD: '1'} {
    (^./ds4-server --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
        --ssd-streaming --ssd-streaming-cache-experts 32GB
        --ctx 8192 --prefill-chunk 512 --host 127.0.0.1 --port 18343)
}
```

The maintained foreground
[serving launcher](../speed-bench/experiments/index-serving-20261027/serve.nu)
additionally enables DSpark exact eager drafting, DSP1,
and a HOT0 SSD cache. Its caller owns the pressure guard and server lifetime.
`--restart` reuses that launcher's cache and response store. These are
reproduction settings, not a new measured throughput recommendation.

## Store the existing numerical format at its original boundary

DS4.1 already quantizes index keys to signed E2M1 FP4 with one E8M0 scale per
32 values, then publishes decoded BF16 values in F32 containers. Keeping those
128 F32 values costs 512 bytes per key. This implementation retains the codes
and scales instead:

| Offset within one 68-byte row | Contents |
|---|---|
| 0–63 | 128 four-bit codes; even column in the low nibble, odd column in the high nibble |
| 64–67 | Four unsigned E8M0 scale exponent bytes, one per 32 consecutive columns |

For each group, round the original post-RoPE values to BF16 and find their
maximum absolute value `amax`. Use the existing power-of-two scale
`2^ceil(log2(max(amax, 7.052966104933725e-38) / 6))`. Choose the nearest
magnitude from `{0, .5, 1, 1.5, 2, 3, 4, 6}` after dividing by that scale;
ties choose the even code. Bit 3 holds the sign, including negative zero.
The exponent byte is the biased binary32 exponent of the power-of-two scale.
Decoding multiplies the signed codebook value by the scale and rounds to BF16,
as the original publication path does.

Pack at publication **before the old decoded-value quantizer**, from the
original post-RoPE key. Inferring a scale from an already rounded maximum may
produce a different encoding. Old checkpoint imports necessarily have only
decoded values; the bounded repacker preserves their numerical values on the
tested rounding, signed-zero, tiny-scale and large-scale corpus. Do not assume
an arbitrary unquantized F32 vector has this property.

There are four persistent index owners, with context capacities
`ctx/2 + 1`, `ctx/2 + 1`, `ctx/2 + 1`, and `ctx + 1`. Allocate those arrays at
68 bytes per row and remove their full F32 owners. Scalar and batched
publication write the same physical representation. Ordinary F32 activation
scratch remains for projections, normalization and RoPE; it is not an expanded
owner cache.

## Decode and prefill consumers

The register scorer assigns one 32-lane SIMDgroup to a key and four keys to a
128-thread group. Each lane decodes four values into a `float4`; the existing
head reduction and weighted accumulation order are preserved. Candidate
scoring retains global row IDs, then applies the existing mask and top-k.
Candidate-only scoring is a separate policy; keep it fixed while comparing
storage or SIMD layouts.

Use the register layout at **1,024 index keys and above**. Measured 128/256-key
work favors the old layout; measured 1K and larger work favors the new one.
The threshold counts index rows, not prompt tokens. Smaller work retains the
old F32 or packed reader. `DS4_V41_INDEX_SIMD=1` enables this measured threshold
for either storage type; `0` selects the old scalar scorer. With no explicit
SIMD setting, packed storage enables the threshold automatically.

Wide prefill decodes packed keys directly into the **existing transient BF16
query/key preparation buffer**, then uses the existing tensor score tiles.
It never rebuilds a context-sized F32 index owner. The BF16 preparation buffer
and bounded score scratch are unchanged allocations, so the owner reduction
is not the same percentage reduction in the whole graph or process. On devices
without the newer Metal tensor path, including the intended M1 path, the batch
reader uses a causal SIMD fallback. That path is implemented and fixture-tested
on the current M5 machine; M1 performance has not been measured.

## Checkpoints, SSD and suffixes

Canonical DSV4 version 3 keeps the 13 little-endian u32 header words, token IDs,
F32 logits, raw rings, main-cache spans and odd-frontier pooling carries in
their existing order. It changes each index span from 512-byte F32 rows to
68-byte packed rows. The DS41 family tag remains `0x413431`. Version 2 remains
the expanded F32 layout. F16 main-cache storage still uses F32 wire values.

The loader accepts both directions: v2 into a packed graph, and v3 into an F32
graph. Conversion uses the existing 8 MiB host buffer and one bounded GPU wire
tile; no persistent expanded mirror is retained. Ordinary packed v3 save/load
copies its index bytes directly. A receiving graph's allocation choice is
independent of the source wire version.

The optional DSP1 appendix follows the selected base layout without changing
its own format. SSD admission validates that version's exact base size and
optional DSP1 extent; chunk splitting uses the actual 68-byte index regions.
At equal token prefixes, useful DSP1 history wins over a base-only endpoint
regardless of physical layout. Old v2 entries remain usable after restart.

Private DRP1 replay, SWB1 block-only import and DSSD approximate-suffix
diagnostics retain v2 F32 wire values through bounded conversion. Canonical
suffix sources may be v2 or v3. Grafting unpacks a bounded 128-row repair tile,
restores immutable pre-RoPE rotary lanes or applies the existing legacy repair,
then repacks destination rows. The independent pre-RoPE archive is unchanged;
repeated moves do not repeatedly rotate packed codes. All four measured moves
pass with F32-main source state and F16-main/FP4-index destination state;
capture-size assertions pass and the original tapes remain immutable.

## What has actually been measured

All figures are local M5 Max measurements with the 1017 Q2 checkpoint and
normal DeepSeek reasoning. Raw evidence is ignored; the maintained
[integration report](../speed-bench/experiments/index-serving-20261027/README.md)
contains commands and workload details.

| Measurement | F32 control | Packed index result |
|---|---:|---:|
| Four allocated index owners, context 16,384 | 20,973,568 B | 2,785,552 B |
| Matched canonical checkpoint, 8,192-token frontier | 63,464,500 B | 54,371,380 B |
| Matched canonical checkpoint with DSP1, 263-token frontier | 16,627,824 B | 16,336,560 B |
| Fixed-teacher full-vocabulary rows at 8K | 22 rows | 22/22 byte-exact; ΔNLL=KL=0 |
| Ordinary, favorable and adverse free answers at 8K | Three task answers | All pass; generated reasoning/answer IDs match |

The allocation change is real owner storage, not a formula applied to an
unused pilot buffer. It does not automatically enlarge a fixed expert cache.
The checkpoint comparison includes unchanged raw/main/logit/token state, so
its total reduction is smaller than the index-owner percentage.

GPU packing/unpacking and the batch tensor/fallback corpus pass. Real
v2→packed and v3→F32 imports, DSP1, HOT0 SSD restart and immediate constrained
MTP verification pass. Initial paired teacher logits match exactly; subsequent
cross-import cycles observed maximum absolute error `2.861e-6`. Retain that
measured limit rather than claiming universal converted-continuation identity.

The separate **F32 SIMD** same-process decode sweeps reduce mean time by
0.523% at 32K and 1.936% at 64K. Those are whole-model continuing-decode
measurements, distinct from the isolated kernel's roughly 30–40% reduction.
The original 32K bracket recorded +4.46% prefill time, +3.80% prefill-plus-answer
time and +1.58% warm fixed-trace time. Those controls used the older frozen F32
binary with SIMD disabled; they did not isolate packing from the new SIMD path.
Keep these noisy observations as history, not an established packing penalty.

The [five-pair follow-up](../speed-bench/experiments/index-five-samples-20261027/README.md)
uses the same current binary and frozen shaders, holds SIMD on in both arms,
and changes only FP4 storage. All five fresh processes per layout are retained,
alternating AB/BA order. Mean prefill-plus-answer time is 48.7302418 s F32 versus
48.3616122 s packed (−0.7565% aggregate); the paired median is −0.1221%, with
packed faster in 3/5 pairs. Every aggregate phase mean is lower, so the requested
slowdown-triggered profile was not run. Pair signs depend on order and the first
pair moves the mean appreciably: this establishes neither a repeatable penalty
nor a robust speedup. All ten correct, stopped answers have the same 42 generated
reasoning/answer IDs; all 80 teacher rows match exactly. The second prefill
rebuilds KV with already-warmed experts, and engine initialization is excluded.
Following this five-pair check, the user approved packed storage for the
DeepSeek fast profiles. Direct binary launches still need the explicit setting
shown above; `DS4_V41_INDEX_FP4=0` selects F32 storage in any profile.

Repeated suffix relocation passes all four moves, including cross-layout
prefix restoration and immutable pre-RoPE tapes. Against fresh recomputation,
16 teacher positions give mean ΔNLL `+0.0000372` nats/token, mean KL
`0.0000284`, and zero argmax flips. These describe the stale-suffix
approximation; they do not isolate an additional FP4-storage effect.

The actual server passes all seven complete Responses cases and exact
accumulation of 27 streamed argument deltas. A `previous_response_id` recovers
the durable conversation's fact after process restart, without a KV hit in
that case. A separate repeated strict-tool request restores 392/392 prompt
tokens from SSD after restart and retains the same typed arguments. These
qualify conversation durability and numerical prefix reuse separately. Both
owned test servers exit cleanly. The installed root and `zig-out` copies match
the qualified binaries by hash; all five executables and agent/server preset
entry points pass their help checks.

## Sources and implementation map

Credit [DeepInfer's index-attention implementation at
553741557a57df44a46fb3ab72b6aafeaf945337](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/index_attn.py)
for the FP4 index-cache target and explicit producer/consumer boundary. Its
block-wide data/scale layout differs from this local interleaved 68-byte row.
The register work mapping and Metal conversion paths are local adaptations,
not a port of its NVIDIA/Ascend kernels or a claim about those devices.

- [Packing, decoding and scoring](../src/metal/dsv41.metal),
  [publication/scalar selection](../src/engine/deepseek41/attention.zig), and
  [batched prefill](../src/engine/deepseek41/batch_attention.zig).
- [Physical allocation](../src/engine/deepseek41/resources.zig),
  [wire conversion](../src/engine/deepseek41/state_payload.zig),
  [SSD admission/chunks](../src/state_cache.zig), and
  [suffix repair](../src/engine/deepseek41/suffix.zig).
- [Initial packed producer/reader experiment](../speed-bench/experiments/deepinfer-alignment-20261027/README.md#3-physical-index-fp4-producer-and-reader)
  and [matched SIMD measurements](../speed-bench/experiments/index-simd-20261027/README.md).

Build and run the existing fixture under the normal pressure guard; it covers
original quantizer bits, rounded imports, offset guards, ties and partial tails:

```nu
nu tools/zig-build metal-test --prefix .workspaces/index-cache-check -j6
python3 speed-bench/ideas/pressure_run.py .workspaces/index-cache-guard -- .workspaces/index-cache-check/bin/test-deepseek41-metal --index-fp4
```

For performance, reuse the integration report's frozen binary/shader controls
and paired trace loop. Keep prefill, scalar decode, restored requests and
whole-task time separate, and do not turn buffer savings into an inferred TPS.
