# Packed cache and kernel experiments

Physical packing reduced DeepSeek cache capacity substantially, with different latency consequences for index and main KV. Separate integer and compressed-activation prefill attempts saved scratch space but slowed the measured tasks and were removed. A targeted fusion was correct but neutral. The reports linked below retain samples and reproduction details.

## Index keys: pilot to serving

The October 27 index pilot produced 68-byte rows instead of 512-byte mode-2 rows, an 86.72% reduction. Decoded values, scores and selected IDs matched the existing quantizer in tests through 256K, including ties and tails. Dense readers were roughly 3–13% slower in initial chains while candidate readers were near neutral; a bit-based unpack variant did not help and was reverted. This was a capacity pilot, not a whole-model result. Subsequent explicit SIMD work reduced dense 256K FP4 scoring by 39.2% and the full chain by 32.1% against its prior kernel, while tiny 37-key work regressed. See the [alignment](../../../speed-bench/experiments/deepinfer-alignment-20261027/README.md) and [SIMD](../../../speed-bench/experiments/index-simd-20261027/README.md) reports.

Actual integration covered physical index allocations, batched prefill and decode, canonical and SSD conversion, suffix moves, MTP, Responses streaming and restart; it did not keep a full F32 shadow. The [serving report](../../../speed-bench/experiments/index-serving-20261027/README.md) retains the initial noisy 32K whole-reply penalty. A later [five-pair follow-up](../../../speed-bench/experiments/index-five-samples-20261027/README.md) held binary, shaders and SIMD fixed, changing only packed index on/off for a 32K favorable task with normal reasoning, chunk 8192 and 48 GiB expert cache. Mean prefill plus generation was 48.730 s F32 and 48.362 s packed; paired median changed −0.122%, with packed winning three of five. All ten answers and 80 teacher rows matched. Order effects remained, so the earlier apparent 3.80% penalty did not reproduce and the result is roughly neutral. Fast presets adopted packed index with an explicit off override.

## Main KV packing

On October 28, physical FP4 main KV was qualified across canonical conversion, actual SSD/MTP, chained suffix moves, F16 and F32 layouts, and a Responses restart. An early cross-layout test exposed a legacy tiny-scale reconstruction error, and reverse suffix conversion exposed a missing GPU command batch; both were fixed before comparison. The [main FP4 report](../../../speed-bench/experiments/main-fp4-20261028/README.md) records the tests and revisions.

Five alternating 8K three-case pairs at `aee06ac9`, with normal reasoning, native routes, packed index, chunk 8192 and 48 GiB experts, found neutral whole-answer latency: 75.601 to 74.768 s, with paired median +0.005% and mixed variation. All 30 answers, generated IDs and 110 teacher rows matched. At 32K, five favorable pairs increased mean answer time from 47.600 to 48.540 s (+1.976%); every pair was slower. Actual KV allocation fell from 227,172,624 to 46,941,584 bytes. A matching diagnostic attributed most observed added time to indexed batch attention, not the explicit packing and gathering operations, but profiling changed scheduling. The fast presets accepted the capacity tradeoff; the ordinary engine default remained off and explicit layout preferences remained available.

## Prefill representations and fusion

The preceding October 27 discussion prioritized SSD bandwidth and RAM capacity,
then RAM bandwidth. Its IQ2_XXS gate/up blocks used 66 bytes per 256 weights and
Q2_K down blocks 84 bytes, averaging 2.25 bits per weight across equal matrices;
a routed expert occupied 9,953,280 bytes. Those compact bytes remained in the
expert cache while kernels decoded small compute tiles. There was no F32 expert
shadow analogous to the earlier expanded index cache. Lossless repacking might
improve alignment; expanding every cached expert to four or eight bits merely
to use native arithmetic would instead consume the capacity being protected.

Two external examples motivated distinct options. [ConvRot](https://arxiv.org/html/2512.03673v1),
used by the [MiniMax-H3 conversion](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot),
uses grouped Hadamard rotations to change low-bit quantization; it is not
lossless packing or evidence of a Metal speedup. Draw Things'
[8-bit S account](https://releases.drawthings.ai/p/metal-quantized-attention-pulling)
and [INT8 MPP kernel](https://github.com/liuliu/ccv/blob/unstable/lib/nnc/mfa/kernels/NAInt8MatMulKernel.cpp)
combine row-wise INT8 weights, dynamic activation quantization, INT32
accumulation and rescaling. Those media-model speed claims were not local
measurements. The [WWDC26 TensorOps reference](https://developer.apple.com/videos/play/wwdc2026/330/)
was another native-format lead; IQ2 codebooks and Q2 affine scales do not map
directly to those types. These references motivated temporary compute-tile
experiments while retaining compact expert storage. The [MXS3 scale-packing pilot](../../../recipes/07-native-scale-packing.md) rejected a particular sampled Q2 dictionary scheme; that result does not rule out all lossless layouts.


Transient INT8 prefill tiles kept expert SSD and cache storage unchanged. Five alternating fresh pairs per arm showed V1 increasing three-case 8K prefill by 34.06% and whole time by 35.09%; V2, which reused activation quantization, increased favorable prefill by 18.49% and whole time by 34.70%. All 40 answers passed. V2 saved 105 MiB of scratch, but its IQ2 GEMM still took about three times the kernel GPU time. Both runtime hooks were removed; [results and patches](../../../speed-bench/experiments/integer-prefill-20261027/README.md) and the [not-adopted lesson](../../../recipes/not-adopted/integer-prefill.md) remain.

A separate floating-point matmul experiment compressed activations rather than using integer arithmetic, baseline `19c63b1b26da`. Five production-shaped leaf pairs were slower for each of three consumers. Five full-model pairs of the staged variant saved 105 MiB scratch but increased combined prefill 4.04% and task time 8.07%; all 30 answers passed, with changed reasoning lengths. A matching profile showed IQ2 consumers increasing from 3.643 to 5.250 s while packing time barely changed. It did not establish a physical-unit or occupancy cause. Hooks were removed, and [the report](../../../speed-bench/experiments/packed-activation-20261028/README.md) retains the failed paths.

Finally, a fused HC collapse and normalization kernel initially changed signed zero. Preserving the existing accumulation order repaired bitwise leaf correctness. At `b3b6f079`, five alternating 8K three-case pairs gave 73.483 versus 73.556 s (+0.10%), with mixed paired variation. All 30 answers and 110 teacher rows matched. The [fusion report](../../../speed-bench/experiments/hc-collapse-norm-20261028/README.md) treats this as neutral; the experiment remained explicit rather than entering presets. These outcomes show why capacity savings and a plausible kernel fusion need whole-task measurement; neither implies a speedup by itself.
