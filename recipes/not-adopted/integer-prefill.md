# Native INT8 arithmetic over IQ2 expert storage

**Tested on M5 Max, not adopted.** Two implementations preserve the existing
IQ2/Q2 expert bytes on SSD and in the shared Metal cache, but both take longer
than the existing half-tile prefill path. Runtime hooks were removed. The
[report, five-pair drivers and complete patches](../../speed-bench/experiments/integer-prefill-20261027/README.md)
retain the experiment for another harness or a better kernel implementation.

## What was tried

Draw Things' [scaled INT8 implementation](https://releases.drawthings.ai/p/metal-quantized-attention-pulling)
and [open Metal kernel generator](https://github.com/liuliu/ccv/blob/unstable/lib/nnc/mfa/kernels/NAInt8MatMulKernel.cpp)
provided the reference for genuine INT8 TensorOps multiplication, INT32
accumulation and floating rescaling. This experiment is not ConvRot, and does
not requantize or enlarge the persistent expert weights.

An IQ2_XXS block stores 256 weights in 66 bytes. Decode its signed codebook
integers, which fit in INT8, into a small matrix tile. Retain the original
weight scale for each group of 32:

```text
weight_scale = float(block.d) * (0.5 + group_subscale) * 0.25
x_half = half(x)
activation_scale = max(abs(x_half[0:32])) / 127
activation_code = clamp(round(x_half / activation_scale), -127, 127)
partial = INT8_MATMUL(activation_codes, signed_weight_codes)  # INT32
output += float(partial) * activation_scale * weight_scale
```

Zero groups use scale 1 and zero codes. Each M64/N32/K32 operation preserves
the expert-major row map and bounds the tail stores. Q2_K down projections
retain their established implementation. Numerical differences include
activation quantization, removal of the old per-weight half rounding, and
groupwise rescaled accumulation.

The first version quantizes activations inside every output tile and repeats
that work for gate and up. The second quantizes each packed activation once,
reusing codes and scales across both projections. Its temporary layout is an
aligned expert-offset header, all INT8 codes, then one F32 scale per 32 values:
**1.125 bytes/value instead of 2**. No whole-expert INT8 shadow is allocated.

## What the measurements say

Both comparisons retain five fresh samples per arm with alternating order,
normal reasoning, native expert routing, packed index storage and fixed M64
tiles. The baseline is the installed quantized Q2 model, not BF16 weights.

| Implementation and workload | Prefill time change | Whole task time change |
| --- | ---: | ---: |
| First version, sum of three 8K cases | +34.06% | +35.09% |
| Activation reuse, standalone 8K favorable case | +18.49% | +34.70% |

Each candidate is slower in all five pairs against its own controls. These
are separate campaigns, not a direct percentage speedup between versions.
All 40 measured task answers are correct. Teacher delta-NLL is slightly
negative, KL is small and no teacher argmax flips occur; numerical identity
with the baseline was not the acceptance criterion. The favorable reasoning
trace nevertheless grows from 31 to 98 tokens, contributing to task latency.

Numerical GPU fixtures pass finite-output, zero-input, uneven-route and
boundary checks, with about 0.53% relative RMS error in the synthetic mixture.
Activation reuse reduces reported GPU scratch by 105 MiB in every pair, while
the live expert cache stays at 40.88 GiB. The fixed cache budget does not grow.

Matching-workload diagnostic captures locate the cost in IQ2 prefill: 220
launches take 3.62 versus 17.58 seconds in the first version, and 3.64 versus
11.02 seconds with activation reuse. The profiles are diagnostics, not extra
unprofiled performance samples. INT8 TensorOps execution is verified; no
hardware counters establish which physical execution unit performed it.

The original group-of-32 scales prevent long integer accumulations with one
final rescale, unlike a simple row-scaled INT8 model. Partial rescaling,
staging, decoding and barriers remain candidates for further optimization;
the captures do not isolate each instruction's contribution. This rejects
these two kernels on these shapes, not low-bit arithmetic generally.

For another attempt, first screen actual production matrix shapes with five
samples per side, then repeat whole-model quality and timing. Keep expert
storage density, transient compute layout and arithmetic precision as separate
choices. A faster arithmetic format that expands the expert cache can lose
on SSD traffic and residency. Application `pread` counts also include OS file
cache hits and must not be labelled physical SSD bandwidth.

## Follow-up: compressed storage with floating-point arithmetic

The [2026-10-28 experiment](../../speed-bench/experiments/packed-activation-20261028/README.md)
separated activation storage from integer arithmetic. It retained INT8 codes
and per-32 F32 scales, decoded bounded F16 tiles, and used the original IQ2
half-weight decoder with long F32 accumulation. Q2 down stayed unchanged.

Five alternating full-model pairs across ordinary, favorable and changed-fact
8K tasks confirmed **105 MiB less GPU scratch**, with **4.04% more prefill time
and 8.07% more combined task time**. All 30 answers passed; reasoning remained
enabled. Reasoning lengths changed, especially 31→98 tokens in the favorable
case. The cache budget stayed fixed, so saved scratch did not add resident
experts. A matched profile located the extra cost in the consuming GEMM:
220 launches took 3.643→5.250 seconds; activation pack time was nearly unchanged.

Two further consumers passed numerical checks but failed five-pair leaf
screens: direct register decoding (+141.59% time) and a wider N64 staged tile
(+24.47%), each against its own controls. The initial staged consumer was
+21.64% on that screen. These are measured kernel outcomes; they do not prove
an occupancy or physical instruction-level cause.

None is adopted. Complete patches, the native Zig fixture, five-pair drivers
and all concise results remain in the linked report. Removing integer math
recovered most of the earlier loss, but this storage/decode layout still does
not provide an attractive speed/memory tradeoff on these shapes.
