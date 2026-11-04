# Packed E2M1/E4M3 main KV

**Main-cache owners are 85.94% smaller, with qualified checkpoint, SSD,
speculative-decoding and repeated-suffix paths. Five paired 8K runs show
neutral latency; five paired 32K runs cost 1.98% more task time, with exact
measured outputs.** This is enabled in the opt-in DeepSeek fast preset package;
ordinary engine defaults remain unchanged. The matching 32K profile locates the
cost in indexed batch attention. The measured capacity/latency tradeoff is
accepted, with explicit main=0 and F16-only overrides retained.

The completed 8K campaign uses the 1017 IQ2_XXS/Q2_K checkpoint on an Apple M5
Max with 128 GiB RAM, frozen revision `aee06ac9`. The
[experiment report](../speed-bench/experiments/main-fp4-20261028/README.md) and
[all five 8K samples per side](../speed-bench/experiments/main-fp4-20261028/RESULTS_8K.md)
and [the 32K follow-up](../speed-bench/experiments/main-fp4-20261028/RESULTS_32K.md)
retain exact settings, failures, corrections and qualification boundaries.

## Retain the format already produced by the quantizer

The existing main compressed-cache publisher rounds its input to BF16, applies
E2M1 FP4 with an E4M3 scale per 16 values, then stores decoded values in F32.
Keeping the codes and scales removes the expanded container without introducing
a new quantization policy. DeepInfer describes this as NVFP4-like main KV
**without a global scale**. It is separate from the per-32 E8M0 format of
[packed index keys](14-packed-index-cache.md) and from expert-weight packing.

Each main row contains 512 values in this local layout:

| Byte offset | Contents |
|---|---|
| 0–255 | 512 E2M1 codes; even column in the low nibble, odd column in the high nibble |
| 256–287 | 32 positive E4M3 scale bytes, one per 16 consecutive columns |

Code bits 0–2 select `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`; bit 3 carries the sign,
including negative zero. E4M3 scale code `e:m` decodes to `m * 2^-9` when
`e=0`, otherwise `(1 + m/8) * 2^(e-7)`. Finite positive scale codes are 1–126.
Decode a value by multiplying the signed codebook value by its decoded scale.
These products fit both BF16 and F16 exactly for the established cache format.

One row costs **288 bytes**, versus 2,048 bytes in F32 or 1,024 bytes in the
[F16 container](13-packed-compressed-kv.md). Allocate the actual owner arrays
at that stride and remove their F32 owners. For this model, the four capacities
are `ctx/2+1`, `ctx/2+1`, `ctx/2+1` and `ctx+1`, so main-owner savings are
`1760 * (3*(ctx/2+1) + ctx+1)` bytes for even context capacities. Raw rings,
index owners, original tapes and attention scratch remain separate allocations.

## Fresh publication and legacy repacking are different operations

For **fresh publication**, retain the scale computed from the original
post-RoPE values at the existing quantization boundary:

```text
v = BF16(original value)
amax = maximum(abs(v)) over the 16-value group
scale = E4M3_RNE(max(amax, 6 * 2^-9) / 6)
code = nearest E2M1 magnitude to abs(v)/scale, ties to even code
store code, original sign and scale
```

The E4M3 conversion saturates at its finite maximum 448. Scalar and batched
publication use the same rule. Keep the original rounding locations and GPU
math: the established fast-math quantizer can choose a different result from
an ideal CPU calculation at a division halfway point. The fresh packed path
is tested against that established GPU result, without changing its math.

For **old F32/F16 cache imports**, the source is already decoded mode3 data.
Running the fresh quantizer again is not generally idempotent. Coarse
subnormal E4M3 rounding can leave a group's maximum at `4*scale`, rather than
`6*scale`. A real checkpoint contained scale `2^-8` and maximum `0.015625`.
Reconstructing the scale from `amax/6` selected `2^-9`, changing that maximum
to `6 * 2^-9 = 0.01171875`. Initial broad random tests missed this group.

The separate legacy repacker instead:

1. Tries the usual clamped E4M3 scale inferred from `amax/6`.
2. Encodes all 16 values and checks that every decoded magnitude equals its
   source exactly. The Metal kernel OR-reduces failures within each 16 lanes.
3. If that representation fails, uses `max(2^-9, amax/4)` to recover the
   coarse-subnormal case, while retaining each original sign bit.

This relies on the established mode3 output alphabet; it is not a compressor
for arbitrary F32 activations. Different packed encodings can represent the
same decoded group, so the required legacy invariant is
**F32 → packed → F32 byte equality**, including signed zero. Reconstructing
the original packed bytes from decoded values is not required. Fresh
publication still preserves its original scale directly.

The first real cross-layout checkpoint test exposed this defect before MTP:
v2 → packed → v2 differed at a main-cache byte. The expanded GPU fixture now
includes that captured group and all seven subnormal scales with max-code
4/max-code 6, mixed fractional code values and signed zero. Keep an actual
model-cache roundtrip alongside the leaf test when porting this technique.

## Consumers and bounded working storage

Scalar attention decodes only selected rows into scratch. The engine default
for packed FP4 main storage uses F32 scratch; `resident-fast` and
`persistent-fast` select half scratch with `DS4_METAL_V41_PACKED_F16_GATHER=1`.
The half path can fuse staging and padding; native V4.1 still uses vector
attention and the ordinary or heads-transform reduction. Set the flag to `0`
to restore F32 gather. This choice does not alter stored KV or checkpoints.
The existing F16-storage default is separate and remains unchanged. The
[FP4-owner gather comparison](../speed-bench/experiments/2026-11-04-decode-pass/f16-gather.md)
found exact tested logits and neutral to slightly favorable latency at 8K and
32K; earlier F16-storage qualification does not establish this combination.

Small compressed batches decode into the existing FlashAttention F16 staging
buffer. Indexed batches decode four values at a time directly into existing
half tiles. There is no persistent context-sized F32 main-cache shadow.
Projection, normalization and rotary activation scratch still exist.

The resident fast presets also enable four-row staging for eligible indexed prefill,
which reuses each staged row across 16 query heads. This changes the temporary
threadgroup tile, not the packed owner or attention order. The operation-level
improvement did not produce a clear full-model total-time change at 8K or 32K.
See [the Metal backend](../docs/models/deepseek/apple-silicon.md#attention-history-and-retrieval)
for eligibility and the [experiment report](../speed-bench/experiments/2026-11-04-attention-four-row/README.md)
for measurements.

The resident fast presets also select an approximate attention shortcut that
skips rescaling by one. It leaves the packed representation unchanged but changes
compiled arithmetic and measured model outputs, so it must not be confused with
lossless cache packing. Set `DS4_METAL_V41_ATTN_UNIT_SCALE=0` to disable it.
[Combined qualification](../speed-bench/experiments/2026-11-04-combined-prefill/README.md)
found a repeated 32K prefill gain, inconclusive recorded-session timing and modest
fixed-history distribution drift. The practical coding check passed its original
oracle but exposed an additional empty-generator failure in the candidate.
The ordinary engine and `context-fast` defaults leave this shortcut off.

Checkpoint conversion uses the existing 8 MiB host I/O buffer and bounded GPU
tiles. Suffix repair uses at most 128 F32 rows, or 256 KiB, plus packed source
scratch. Each standalone F32 destination blit must own a command batch: this
runtime's tensor-copy API only borrows an already-open batch. A reverse suffix
test found the missing ownership; a local begin/copy/end helper now handles
both main and index F32 destinations, closing the batch even on copy failure.

Implementation entry points:

- [Fresh publication and scalar readers](../src/engine/deepseek41/attention.zig),
  [batch publication/attention](../src/engine/deepseek41/batch_attention.zig).
- [Pack/repack/unpack/gather kernels](../src/metal/dsv41.metal),
  [scalar/vector decoding](../src/metal/dsv4_kv.metal),
  [Metal dispatch owner](../src/metal/embedding.zig).
- [Actual allocation](../src/engine/deepseek41/resources.zig),
  [bounded wire conversion](../src/engine/deepseek41/state_payload.zig).

## Checkpoint formats, provenance and suffix limits

The DSV4 payload keeps its 13 little-endian u32 header words, token/logit
offsets, raw rings and numerical span order. The version selects these strides:

| Version | Main row | Index row |
|---:|---:|---:|
| 2 | F32, 2,048 B | F32, 512 B |
| 3 | F32, 2,048 B | FP4, 68 B |
| 4 | FP4, 288 B | F32, 512 B |
| 5 | FP4, 288 B | FP4, 68 B |

New readers accept all four versions in both storage directions. F16 owners
retain F32 main wire values. Native-layout saves copy packed bytes directly;
cross-layout imports/exports convert bounded tiles. SSD admission and chunk
splitting use the selected row strides; the existing outer SSD format remains.
The optional DSP1 draft-feature appendix follows the chosen base unchanged.
See the [format helpers](../src/engine/deepseek41/state_format.zig) and
[state-cache owner](../src/state_cache.zig).

Private **DSSD** captures also retain their actual main/index layout, allowing
repeated relocations to keep the storage benefit. Their approximate provenance
remains explicit: they cannot be published as canonical exact prefixes.
Private DRP1 replay and SWB1 block-only payloads retain their existing v2 wire
representation through bounded conversion. This extends the older DSSD-v2-only
behavior documented in the historical index-cache recipe.

Suffix relocation unpacks a tile, restores immutable **pre-RoPE BF16 rotary
originals**, rotates at the destination positions, applies the existing cache
rounding and stores the destination layout. Repeated moves start from those
originals rather than repeatedly rotating quantized rotary lanes. The tape is
separate from the packed cache; source state/tapes remain immutable.

Current suffix operations require even-aligned source/destination positions,
text-only non-TP sessions and a fresh tail. Originals support left/right moves
and recapture of approximate sources; legacy repair without originals supports
left shifts only. Suffix sessions exclude DSpark/MTP. Canonical MTP restoration
is a separate qualified operation. The
[suffix owner](../src/engine/deepseek41/suffix.zig) and
[transaction owner](../src/engine/deepseek41/suffix_transaction.zig) enforce
these limits.

Packing is lossless relative to the existing quantized values. **Suffix reuse
after changing a prefix remains approximate:** cached activations retain their
old prefix's influence even when relocation is numerically correct. The four
chained layout cases produced identical recorded teacher rows to each other,
but differed from fresh recomputation: mean ΔNLL `+5.624964e-5` nats/token,
mean KL `3.242040e-5`, no argmax flips over 16 positions per case. That is a
narrow suffix probe, not a general quality guarantee for prefix edits.

## Qualified integration and measured benefit

Real model qualification covers canonical v2↔v4 and v2↔v5, every pre-MTP
roundtrip byte, scalar decoding, constrained speculative verification and
HOT0 SSD close/reopen. Initial scalar/verification logit differences are zero;
reverse/import checks differ by at most `2.861e-6` under the existing oracle.
Five four-move suffix cases pass: chained F32→FP4, FP4→F32 including index,
F16→FP4 and FP4→F16, plus fixed-source FP4→FP4. Recaptured DSSD v2/v3/v5
states become subsequent sources. The failed conversion/blit attempts remain
recorded separately.

An actual Responses server stored v5 with main FP4 enabled, exited, then a new
F32-main server reopened the same SSD cache. The strict-tool request reused
392/392 prompt tokens with no cache rewrite; output, reasoning and tool
arguments matched. This is restart correctness, not a latency comparison.

The five fresh alternating 8K pairs hold packed index/SIMD, native expert
routing, 48 GiB configured expert cache, context 16,384, chunk 8,192 and normal
high reasoning fixed. Initialization is excluded. Each answer prefill has
fresh KV and experts warmed by the preceding teacher branch; OS caches are
not flushed. The primary time is the sum of answer prefill plus generation
across ordinary, favorable and changed-fact cases in each process.

| Quantity | F32 main control | Packed main |
|---|---:|---:|
| Actual main owners | 83,894,272 B | 11,797,632 B |
| Actual total KV, including unchanged raw/index | 97,165,584 B | 25,068,944 B |
| Canonical payload at each 8,192-token frontier | 54,371,380 B | 18,326,580 B |
| Three-case task time, mean | 75.601 s | 74.768 s |
| Three-case task time, range | 73.275–81.248 s | 73.705–76.830 s |
| Three-case task time, sample SD | 3.239 s | 1.324 s |

The main-owner reduction is 85.94%; total KV falls about 74.2% and canonical
payloads 66.29%. Mean task time falls 1.10%, but the five paired changes are
`−5.438%, −1.280%, +0.844%, +0.791%, +0.005%`, with median **+0.005%**.
The first control has a slower changed-case prefill and remains included.
Treat latency as neutral. All 30 answers pass, generated token IDs match and
all 110 matched full-vocabulary teacher rows are byte-identical: measured
storage ΔNLL=KL=0, with no argmax flips.

Expert capacity stays fixed. Every process records the same expert hit/miss
and pread counters; this experiment does **not** establish reduced expert SSD
traffic or the gain from spending saved KV RAM on more experts. Context-sized
owners, not the whole process, shrink by the row percentage. The report's
200K source-size arithmetic is not a 200K inference qualification.

The five-pair favorable 32K follow-up uses context 40,960 with the same frozen
binary/settings. Main owners are 209,723,392→29,492,352 B; actual total KV
227,172,624→46,941,584 B; canonical 32,768-token payload 184,476,724→40,297,524 B.
Primary task time rises 47.600→48.540 s (+1.976%), slower in every pair:
`+2.378%, +2.368%, +3.613%, +0.876%, +0.650%`, median +2.368%.
AB/BA mean paired changes are +2.213%/+1.622%. Answer prefill costs 2.257%
more; generation averages 1.234% less. All 10 answers pass with 42 identical
generated token IDs, 40 byte-identical teacher comparisons and ΔNLL=KL=0.
This is a measured small prefill tradeoff, not neutral 32K latency or a speedup.
The matching encoder profile uses the same 188 indexed batch-attention dispatches:
17.615→19.524 diagnostic GPU seconds. New publication and scalar gather total
only 56.986 ms; quantization/copy work falls and expert read counts/bytes match.
This locates the extra work in the inline packed reader, without proving a
specific hardware bottleneck. Profiled wall time is not added to the five-pair
estimate. The fast presets accept the measured ~2% task cost for capacity,
canonical persistence and suffix-source size; ordinary engine policy is unchanged.

## Enable and reproduce

`ds4-local` DeepSeek fast presets now select main FP4 automatically. An explicit
`DS4_V41_COMPRESSED_KV_FP4=0` keeps expanded main storage. An F16-only request
selects F16; when both main FP4 and F16 are explicitly 1, main FP4 takes priority.
Four fake-binary wrapper checks cover default 1, explicit 0, F16-only/main=0 and
both=1/main=1. This source-preset change does not itself replace installed binaries.

For direct engine use, set the storage flags before opening sessions and keep
them stable during ordinary serving:

```nu
with-env {
    DS4_V41_COMPRESSED_KV_FP4: '1'
    DS4_V41_COMPRESSED_KV_F16: '0'
    DS4_V41_INDEX_FP4: '1'
    DS4_V41_INDEX_SIMD: '1'
} {
    (^./ds4-server --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
        --ssd-streaming --ssd-streaming-cache-experts 48GB
        --ctx 16384 --prefill-chunk 8192)
}
```

For comparison, change only `DS4_V41_COMPRESSED_KV_FP4` to `0`. Leave
`DS4_METAL_V41_PACKED_F16_GATHER` unset or `0` in the main-storage comparison.
Preserve the model, routing, reasoning and unrelated feature settings.

Build the chosen source revision into `FROZEN` with
`nu tools/zig-build metal-test dspark-checkpoint-test suffix-test state-cache-test decoder-replay-bench ds4-server -j3 --prefix FROZEN`.
Copy that same revision's `src/metal/*.metal` to `FROZEN/shaders/` and record
its revision in `FROZEN/source-revision.txt`. The maintained drivers require
these adjacent frozen shaders. Use fresh result/guard directories and run one
heavy job at a time under the unchanged pressure guard:

```nu
python3 speed-bench/ideas/pressure_run.py LEAF_GUARD -- nu speed-bench/experiments/main-fp4-20261028/with-shaders.nu FROZEN/bin/test-deepseek41-metal --main-fp4
python3 speed-bench/ideas/pressure_run.py MATRIX_GUARD -- nu speed-bench/experiments/main-fp4-20261028/qualify.nu FROZEN MATRIX_OUTPUT
python3 speed-bench/ideas/pressure_run.py PERF_GUARD -- nu speed-bench/experiments/main-fp4-20261028/run.nu FROZEN/bin/decoder-replay-bench PERF_OUTPUT
python3 speed-bench/experiments/main-fp4-20261028/analyze.py PERF_OUTPUT
```

The [qualification driver](../speed-bench/experiments/main-fp4-20261028/qualify.nu)
includes the existing DSpark support model; the
[performance driver](../speed-bench/experiments/main-fp4-20261028/run.nu) runs
five AB/BA alternating pairs and accepts `--tokens` / `--case` for later
scaling checks. The [analyzer](../speed-bench/experiments/main-fp4-20261028/analyze.py)
retains sample/order variation, actual allocation/payload counters and quality.
See the report's [foreground server helper](../speed-bench/experiments/main-fp4-20261028/serve.nu)
for the actual cross-process Responses restart.

## Credit and portability

The numerical boundary originates in [antirez/ds4](https://github.com/antirez/ds4)'s
DeepSeek V4.1 quantizer and Metal attention implementation. The physical-cache
work follows DeepSeek's published per-16 E4M3 main-KV design:
[DeepInfer deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md#low-precision-computation)
and its [KV quantization primitives](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/kv_quant.py),
verified in the clean local checkout at that revision. Our 288-byte row order,
legacy repacker, Metal readers and DSV4/DSSD persistence integration are local
adaptations; they do not claim wire compatibility with DeepInfer's paged cache.

For another harness, carry over the format, original-scale publication,
explicit legacy conversion, real smaller owners and all consuming/persistence
paths together. Test native decoding and actual cache lifecycle before claiming
a serving benefit. This extends the local [F16 storage](13-packed-compressed-kv.md)
and [packed index](14-packed-index-cache.md) recipes; their historical results
remain separate workloads.

## Next reader experiment — proposed

The matching 32K diagnostic places most added GPU cost inside indexed
attention, where the packed half4 reader executes repeatedly for query/head
sets. Four neighboring loaders also recompute one group's E4M3 scale. The
profile measures that enclosing kernel, so ALU cost, lookup latency and
occupancy remain possible explanations rather than individually proven ones;
see [the source/profile attribution](../speed-bench/experiments/main-fp4-20261028/RESULTS_32K.md).

The next small probe could construct E2M1 half4 bits directly, compute a scale
once per four adjacent vector lanes and broadcast it with SIMD shuffle, then
multiply exact half vectors. It would retain the current row format, no F32
owner, and the same attention order. This proposal is unimplemented: require
the all-scale/signed-zero GPU fixture and five alternating model pairs before
claiming a gain. The existing capacity/latency tradeoff is accepted separately.
