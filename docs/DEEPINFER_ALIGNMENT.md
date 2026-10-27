# DeepInfer technique inventory and local alignment

Source audit, 2026-10-27. DeepInfer revision
`553741557a57df44a46fb3ab72b6aafeaf945337` matches upstream HEAD, verified with
read-only `git ls-remote`. DS4 baseline is `e2a38f6e2bf4`. No runtime changes,
builds or model experiments were made for this inventory.

The user subsequently authorized its bounded convergence experiments. See the
[execution report](../speed-bench/experiments/deepinfer-alignment-20261027/README.md)
for current outcomes. Candidate-only scalar scoring is implemented opt-in:
exact distributions/outputs on the32K/64K probes, a small warm-decode benefit,
and no reliable first-answer improvement. The table below retains the pinned
source-audit comparison; it is not a claim that these later experiments were
already present in its baseline.
The subsequent block-only128-row prototype also validates bounded recovery and
small measured baseline-trace changes, with a size/latency tradeoff. It rebuilds
all40 rings through a direct diagnostic API; ordinary cache policy and isolated
encoder-only recovery remain absent. See the report for scope and measurements.
The index-FP4 producer/reader pilot now also verifies68B rows against the old
GPU quantizer/scorer. Dense readers cost modestly more; candidate readers are
near neutral. Actual model-cache and SSD packing remain absent.
Canonical DSpark history is now implemented opt-in as DSP1: exact BF16 raw
target features survive SSD restart and enable immediate verified drafting.
Short restored replies improve in the measured eager-MTP cases; full replies
are mixed. The report distinguishes raw-feature persistence from upstream's
projected-KV implementation and records compatibility/capacity costs.

This covers the material techniques in the public engine, its kernel integration
and the separately documented production service. It is not a claim that every
utility kernel is an independent optimization. **Have** means an equivalent
mechanism exists; it does not mean identical kernels, numerics or performance.
**Partial** includes narrower activation conditions and deliberately different
implementations. Production-only components are identified rather than treated
as available source to port.

The useful alignment target is model semantics, state ownership and efficient
dataflow on one Apple GPU. DeepInfer's supported devices are SM100 and Ascend950;
its multi-GPU throughput is not a local performance target. The installed trunk
here is the 1017 IQ2_XXS/Q2_K quant, not their native MXFP4 expert checkpoint.

## The look-back restore mechanism

There are three distinct windows:

| Window | What it reconstructs | Local status |
|---|---|---|
| Encoder replay, 128 tokens | Missing sliding-window attention (SWA) state after a compressed/index block-cache hit | **Missing** |
| Decoder/YOCO retained tail; example 140 | Decoder local state after encoder prefill | **Partial/opt-in** |
| DSpark target-feature history, 128 positions | The drafter's input history after restoring target KV | **Missing from persistence** |

The first is the remembered restore technique. Keep the cached compressed/index
blocks; replay the preceding 128 tokens to rebuild local attention state. At
replay position `i`, visible raw keys begin at `max(replay_start,i-127)`. This
approximates SWA while retaining the global compressed cache. The public engine
has the replay-floor metadata, but its complete cache-hit controller is omitted.
[Deployment description][di-deployment], [public replay floor][di-prepare].

For example, with a compressed prefix ending at 32,768, recovery can begin
at 32,640 and then process the new suffix. This is not suffix relocation or a
logical context edit. Our current loader instead requires a complete checkpoint
with raw rings, cumulative arrays, pending compression state and logits.

Decoder replay is a separate approximation. Our exact shrinking dependency
suffix already avoids unnecessary decoder work. The optional fixed 128/256-row
path skips more, but activates only on wide sweeps of at least 8,192 rows and
excludes DSpark and images. Normal 512/2,048-row chunk configurations do not
activate it. DeepInfer's compact-query path supports DFlash; its retained tail
must cover the 128-token window plus backtracking. With default five drafts,
the minimum is 134;140 is an example setting, not a universal constant.
DeepInfer's truncation also defaults off, with keep 0 until explicitly configured.
[Reference checks][di-yoco], [local gate][ds-prefill-policy].

## Cache and attention

| ID | Technique | DeepInfer and current DS4 comparison |
|---|---|---|
| C1 | Encoder bounded recovery | **Missing locally.** Public `swa_valid_start` metadata can restrict raw visibility, but no public nonzero-setting caller implements the complete documented cache-hit policy. DS4 imports complete endpoints. [DI][di-prepare], [DS4][ds-payload]. |
| C2 | Decoder/YOCO query compaction | **Partial.** DeepInfer can physically remove discarded decoder queries while still publishing full shared compressed KV. DS4 has exact reduced suffixes plus optional wide replay, with the restrictions above. Both keep full encoder/global-cache publication. [DI][di-yoco-code], [DS4][ds-prefill]. |
| C3 | Shared compressed/index owners | **Have for the validated architecture.** Our KV sources 2/8/14/20 and index sources 2/8/14/20/24/28/32/36 implement the trained sharing topology. DeepInfer derives a more general topology from checkpoint metadata. [DI][di-cross-layer], [DS4][ds-config]. |
| C4 | Inherited candidate-block hierarchy | **Have.** Later indexers inherit source-layer candidates. Our layer 20 selects up to 2,048 blocks of 8. This is model structure, not the optional resident-expert approximation. [DI][di-cross-layer], [DS4][ds-attention]. |
| C5 | Score only candidate keys | **Missing optimization.** DeepInfer's consumer score kernels operate on inherited candidates. We score the full compressed prefix, mask excluded blocks, then top-k. Shared storage does not reduce those discarded dot products. [DI][di-index], [DS4][ds-attention]. |
| C6 | Bounded selection scratch | **Have, different implementation.** DeepInfer sizes score chunks from row bytes/memory budget and caps them at 8,192 queries. We retain 32 live score rows and use Metal partition/merge top-k, causal bounds and prefix optimizations. Replacing a supposedly naive sort is not the first gap. [DI][di-index], [DS4][ds-batch-attention]. |
| C7 | Packed low-precision KV | **Numerics present; physical packing missing.** We round through the reference-like low-precision formats but generally store expanded floats. See the byte table below. [DI][di-cache-format], [DS4][ds-kv-layout]. |
| C8 | Bounded speculative rollback | **Have, different design.** DeepInfer enlarges SWA/compressor rings for backtracking. Our verifier saves overwritten raw/feature slots and compressor carries, then restores rejected positions; committed lengths hide unaccepted compressed rows. Larger rings are not inherently required. [DI][di-cache-engine], [DS4][ds-dspark]. |
| C9 | Separate durable blocks from disposable state | **Partial.** DeepInfer separates paged block KV from fixed state slots and can regenerate dropped consumer state. Our content-addressed SSD chunks deduplicate arrays, but endpoint payloads still retain all 40 raw rings and odd-boundary compression carries. [DI][di-cache-engine], [DS4][ds-payload]. |
| C10 | GPU paging and copy-on-write forks | **Missing.** DeepInfer reference-counts paged GPU blocks and clones state slots. We have immutable SSD chunks, pins and shared stored bytes, but each restored graph has private context-sized mutable GPU arrays. SSD chunking is not GPU paging. [DI][di-block-manager], [DS4][ds-resources]. |
| C11 | Asynchronous cache fetching/import | **Partial elsewhere, absent at checkpoint import.** Our expert and Engram I/O can overlap compute; KV materialization currently assembles the whole endpoint synchronously before importing it. Unified memory removes a separate H2D tier, not the SSD/materialization work. [DS4 store][ds-state-cache], [import][ds-persist]. |
| C12 | Usable draft state across restore | **Partial.** DeepInfer projects target features into drafter attention KV during forward. Our DSV4 payload omits the DSpark feature window, resets the drafter on restore and warms 128 future positions. Persisting raw features and persisting projected drafter KV are different implementations. [DI][di-dflash], [DS4][ds-speculative]. |
| C13 | Approximation-aware persistence | **Deliberate local extension.** Our typed DRP1 checkpoints retain replay policy/approximate identity, resume matching prefixes and exclude canonical export/suffix grafts. This is not an accidental exact-cache leak. Upstream's documented decoder replay cache policy is stricter. [Local policy][ds-checkpoint], [recipe](../recipes/12-bounded-decoder-replay.md). |

The physical row formats are particularly concrete:

| Stored row | DeepInfer | Current DS4 GPU representation |
|---|---:|---:|
| Raw/SWA, 512 values, E4M3 + per32 UE8M0 scales | 528 B | 2048 B F32 |
| Main compressed, 512 values, E2M1 + per16 E4M3 scales | 288 B | 2048 B F32;1024 B optional F16 |
| Index, 128 values, E2M1 + per32 UE8M0 scales | 68 B | 512 B F32 |

`DS4_V41_COMPRESSED_KV_F16=1` enables the existing F16 main storage; repository
defaults and `ds4-local` do not currently enable it. Disk payload values remain
F32. Packing the current rounded values is a promising storage experiment, but
losslessness still requires checking scale generation and rounding; do not infer
it from matching dtype names. For 262,144 context, the four cumulative owners
represent about 655,360 rows: main+index arrays are approximately1.5625 GiB F32,
0.9375 GiB with F16 main, or0.2173 GiB at the reference packed sizes, before
alignment/scratch. These are byte calculations, not measured speedups.
An expert cache with an explicit fixed budget does not automatically grow when
KV gets smaller; the freed headroom must actually reach its admission budget.

## Kernels, layouts and weight traffic

| ID | Technique | DeepInfer and current DS4 comparison |
|---|---|---|
| K1 | Sparse attention kernels | **Have, different backend.** FlashMLA is integrated upstream; ours uses raw/gathered/indexed Metal attention and typed F16 readers. CUDA SASS/TMA details do not port literally. [DI interface][di-attention-backend], [DS4][ds-attention]. |
| K2 | Norm/RoPE/quantize/store fusion | **Partial.** DeepInfer combines transforms with final cache publication. Our scalar path generally dispatches normalization, rounding, RoPE and stores separately. Fuse one publication boundary while preserving rounding locations. V4.1 has no indexer Hadamard; do not import a generic transform accidentally. [DI][di-attn], [DS4 quantization](../src/metal/dsv41.metal). |
| K3 | Producer-to-consumer layout and attention-boundary fusion | **Partial.** Upstream can merge Q transforms, sparse attention, inverse RoPE and quantization, with one-time projection-weight permutation. Our staging/gather, transforms and projection are more separate. Preferred output layout and fewer intermediate writes are portable lessons. [DI][di-attention-backend], [DS4][ds-attention]. |
| K4 | Merge independent same-input projections | **Missing fusion opportunity.** Upstream merges compatible Q-A/raw-KV and Q-B/index-Q projections; shared-hidden cases explicitly prevent invalid fusion. Our projections are generally separate. Weight types and publication dependencies determine safe combinations. [DI][di-attn], [DS4][ds-attention]. |
| K5 | Lazy mHC and mega_mhc | **Model semantics present; fusion partial.** We implement four residual streams and Sinkhorn mixing, including some fused batch RMS/project work. Upstream carries prior post-mix lazily and can fuse post+next pre+norm+cast directly into MoE buffers. Engram, steering and capture points constrain fusion. [DI][di-hc], [DS4][ds-ops]. |
| K6 | Fused gate and domain-local Mega MoE | **Partial.** We already have paired gate/up+SwiGLU, summed-down, selected-address/slab/grouped paths and shared-expert work. No equivalent whole resident Mega MoE region exists. Host routing and SSD misses are real boundaries; profile a resident subregion rather than assume one megakernel solves everything. [DI][di-mlp], [DS4][ds-moe]. |
| K7 | Native FP4/FP8 GEMM layouts | **Different quantization target.** Upstream packs weights/scales for its exact tensor-core consumers. The installed trunk here uses IQ2_XXS gate/up and Q2_K down, although the runtime supports other formats. Consumer-oriented repacking is transferable; relabeling Q2 bytes as MXFP4 is not. [DI][di-mlp], [DS4](../src/engine/deepseek41/expert_cache.zig). |
| K8 | Engram host placement and deterministic early fetch | **Have, different transport.** Both hash compressed token n-grams and gate sparse retrieved rows. Upstream uses sharded/host-backed Engram buffers and folds constant weights; we use bounded SSD pread batches, early prefetch and fused gating. RDMA itself is unnecessary on one Mac. [DI][di-engram], [DS4](../src/engine/deepseek41/engram_prefetch.zig). |
| K9 | Reusable buffers and direct publication | **Have in parts.** Both avoid repeated large allocations and reuse scratch. Upstream producer kernels can write activation/scales directly into expert communication/GEMM buffers; our intermediate boundaries leave more fusion opportunities. This is distinct from adopting a tensor runtime. [DI][di-hc], [DS4][ds-ops]. |

Our resident-expert substitution, suffix relocation/pre-RoPE archive, lossless
scale-dictionary pilots and SSD expert cache are local techniques, not features
to attribute to DeepInfer. In particular, EPLB replicates the **same** expert;
it does not substitute a different expert that happens to be resident.

## Scheduling, speculation, sampling and protocols

| ID | Technique | DeepInfer and current DS4 comparison |
|---|---|---|
| S1 | Chunked prefill and admission | **Have, different policy.** Ours has decode priority, round-robin prefill fairness, bounded mixed-prefill quanta and 2 ms decode coalescing. DeepInfer also budgets tokens/sequences/cache, uses image-safe boundaries and distributed ALL/ANY votes. The votes do not belong in a one-device scheduler. [DI][di-scheduler], [DS4][ds-coordinator]. |
| S2 | Continuous batching and pressure preemption | **Partial.** Ours batches pending work, but DS41 streaming/DSpark sessions often use ordered or exclusive turns rather than a shared native GPU batch. Idle-slot checkpoint eviction is not active-sequence block-pressure preemption. DeepInfer can recompute ordinary preempted requests; SO preemption currently terminates the request. [DI][di-scheduler], [DS4](../src/engine/deepseek41/batch_transaction.zig). |
| S3 | Zero-overhead asynchronous scheduling | **Partial analogue only.** Upstream pipelines CPU scheduling/postprocessing with device execution and repairs stale metadata from acceptance results. Our admitted model turns are synchronous; Flint's masked packets overlap one specific CPU/GPU boundary. ZOS defaults on for NORMAL/COMPLETION, off for PROMPT. [DI worker][di-worker], [defaults][di-engine-args], [DS4][ds-coordinator]. |
| S4 | Captured graphs, bucketed shapes and measured cost curves | **Partial/hardware-specific.** CUDA graph buckets amortize launch/setup and provide a measured decode cost curve, optionally cached on disk. We have resident Flint packets and Metal command batching, not the same graph-bucket scheduler. The portable target is reduced launch/idle cost. [DI](../../DeepInfer/deepinfer/worker/cuda_graph_runner.py), [cost cache](../../DeepInfer/deepinfer/worker/decode_perf_cache.py). |
| S5 | DSpark block drafting and verification | **Have.** Our native DS41 supports actual block verification, opportunistic greedy acceptance and opt-in exact stochastic p/q correction. Grammar masks now apply to draft and target distributions with correct rollback. [DI][di-sampler], [DS4][ds-speculative], [qualification](../speed-bench/experiments/constrained-packets-20261027/README.md). |
| S6 | Throughput-costed confidence allocation | **Partial.** We use a confidence cutoff (Metal default 0.6) and bounded depth. DeepInfer optionally ranks cumulative acceptance estimates across requests and picks a verification budget using measured device cost; GPU metadata uses newer confidence under ZOS. Variable-length mode defaults off; maximum draft length defaults to 5. [DI][di-mtp-varlen], [DS4][ds-speculative]. |
| S7 | Per-position grammar-mask reuse/overlap | **Partial.** Upstream saves masks from the tentative walk, copies them asynchronously and reuses them for target verification. DS41 currently fills the same-prefix mask for q and p separately; Flint already overlaps mask work with the next forward. A small profile-driven experiment, not a predicted major TPS win. [DI][di-sampler], [DS4][ds-speculative]. |
| S8 | Async grammar compilation and bounded cache | **Partial.** Both cache immutable compiled grammars and keep per-generation matchers. Upstream uses a builder pool/futures before inference; our preflight compiles synchronously under `tool_mu`. Our compiler cache is 128 MiB, theirs defaults to 10 GiB; copying that budget is inappropriate here. [DI](../../DeepInfer/deepinfer/structure_output/so_matcher_builder.py), [DS4](../src/server/grammar_runtime.zig). |
| S9 | Grammar triggers, bounds and speculative rollback | **Have equivalent model framing; different interface.** Upstream accepts arbitrary trigger sequences and an optional bound token. We build model-specific structural grammars around actual control IDs, keeping reasoning separate. We expose JSON/JSON Schema and Chat regex, not arbitrary EBNF or the `so_args` API directly. [DI](../../DeepInfer/deepinfer/structure_output/matcher.py), [DS4](../src/server/grammar_format.zig). |
| S10 | Fused GPU sampling/verification | **Partial.** Flint's supported native policies sample on GPU; unsupported filters retain CPU/scalar fallback. DS41 copies logits to CPU for sampling and verification. Upstream fuses filters/log-softmax/sampling/rejection and can use a compact force-top-k space. Our DS41 min-p=.05 policy is an intentional local difference; Flint defaults to 0. [DI][di-sampler], [DS4](../src/sampling.zig). |
| S11 | Position-based deterministic RNG | **Different/partial.** Upstream keys draws by sequence seed, token position and namespace. Our DS41 request RNG is mutable XorShift; speculative draws change its future state. Chat/completions parse positive seeds; Responses currently does not. Flint has its own counter-based native sampler. [DI](../../DeepInfer/deepinfer/model_executor/layers/tilelang_ops/random_sampling.py), [DS4](../src/server/input.zig). |
| S12 | Batch-invariant numerics | **Not established locally.** Upstream has an opt-in NVIDIA mode controlling reduction/kernel choices; deterministic sampling additionally requires a fixed seed and variable-length MTP disabled. Separate scalar/batch parity tests are not that guarantee. This may help qualification more than speed. [DI](../../DeepInfer/deepinfer/worker/runtime_config.py), [DS4][ds-ops]. |
| S13 | Numerical safeguards and failure isolation | **Have, different details.** Upstream isolates failed grammar requests and slightly biases rejection residuals to avoid empty numerical distributions. Our exact mode uses FP32 `max(p-q)` and explicit failure/rollback; packet failure invalidates affected state. Neither implies bitwise-equivalent numerical policy. [DI](../../DeepInfer/deepinfer/model_executor/layers/tilelang_ops/common.py), [DS4](../src/engine/token_constraint.zig). |
| S14 | Token provenance and parallel segmented BPE | **Semantics aligned; parallel encoding not established.** Both distinguish inserted control IDs from identical ordinary text. We carry prompt ranges and use ordinary BPE for payload spans; the reference recipe additionally encodes independent segments in parallel. [Local prompt](../src/models/deepseek41/prompt.zig), [recipe reference](https://github.com/deepseek-ai/deepseek-recipe). |
| S15 | Incremental protocol conversion | **Semantics largely aligned; efficiency partial.** Our DS41/Flint streams names and argument fragments with stable IDs and exact final accumulation. The recipe maintains incremental parser/matcher state; parts of our output projection still reparse accumulated text. Full recipe feature parity is not implied. [Local parser](../src/server/ds41_output.zig), [contract](SERVER.md#constrained-output-and-client-tools). |

## Distributed and production-service techniques

| ID | Technique | Applicability to our server |
|---|---|---|
| P1 | Attention DP and wide expert parallelism | Shard model work/weights across devices and route token activations to expert owners. **Cluster-specific**, although local token grouping and shared loaded weights are useful. Our SSD demand-loading is a different response to memory limits. [Public communication](../../DeepInfer/deepinfer/model_executor/parallel_utils/communication_op.py). |
| P2 | Balanced context parallelism | **Public capability, not their stated current production choice.** Mirrored context partitions and gather scheduling spread long prefill across GPUs. No reason to add cross-process CP for one M5. [Implementation](../../DeepInfer/deepinfer/model_executor/layers/tilelang_ops/prepare_cp_mirror.py). |
| P3 | Expert-parallel load balancing | Replicate identical logical experts, count recent demand, reassign physical replicas and shuffle weights. **Not resident-expert substitution.** On one GPU, cache admission/popularity is the relevant analogy; redundant copies do not add compute capacity. [Policy](../../DeepInfer/deepinfer/model_executor/eplb/policy.py). |
| P4 | Symmetric buffers and asynchronous dispatch/combine | **Hardware-specific transport, portable buffer discipline.** DeepEP/NVLink/RDMA and reserved communication SMs do not translate directly to Metal; preallocation and direct consumer writes do. [Buffer contracts](../../DeepInfer/deepinfer/model_executor/parallel_utils/userbuffer.py). |
| P5 | Disaggregated vision/prefill/decode and load balancing | **Production design, integration partly omitted.** Stage-specific resource budgets and latency goals are useful; separate GPU pools and cross-rank load balancers are unnecessary for one computer. [Release boundary][di-readme]. |
| P6 | Long-lived network KV storage and recovery | **Documented infrastructure, not a complete public implementation.** Production stores durable block KV on 3FS and SWA separately, with async fetching and GC. Our bounded local hot/SSD cache already supplies persistence. USRBIO is3FS transport, not prefix matching or a required single-process component. [Local cache](../recipes/03-state-cache.md), [release boundary][di-readme]. |
| P7 | Autoscaling, migration and failover | **Mostly cluster-only.** Capacity pools, request migration and healthy-rank regrouping address service availability. Locally we need sound session shutdown, cancellation and restart, which are qualified; no equivalent cluster migration service is required. |

## Multimodal and qualification techniques

| ID | Technique | DeepInfer and current DS4 comparison |
|---|---|---|
| V1 | Async any-resolution image preprocessing | **Semantics present; pipeline partial.** Our native resize/patch/layout follows the official geometry. DeepInfer uses CPU workers and can prepare inputs off the model critical path; our server encode/cache path holds inference admission. [DI](../../DeepInfer/deepinfer/ve/multimodal_processor.py), [DS4](../src/engine/vision_runtime.zig). |
| V2 | GPU pad/normalize/ViT/downsample/aligner and image batching | **Native vision exists; fusion/batching differs.** We run a Metal vision sidecar and cache exact-byte embeddings. Upstream packs raw images and runs GPU patch preparation plus batched vision work. Copying the multiprocessing pool itself is unnecessary. [DI](../../DeepInfer/deepinfer/model_executor/models/vision_encoder/vl_aligner.py), [DS4](../src/server/image_resources.zig). |
| V3 | Image-aware chunk/cache boundaries | **Have in parts.** Upstream strips fully cached images and slices multimodal metadata to each chunk. We retain image identities/spans, exact-byte embedding caching and image-aware prompt assembly; explicit cache breakpoints currently support DS41 text only. [DI](../../DeepInfer/deepinfer/ve/prepare_tensor_input_vl.py), [DS4](../src/server/image_resources.zig). |
| V4 | Teacher-forced numerical and semantic regressions | **Have, different baselines.** Upstream supplies PPL/logprob/top-k, YOCO mask-vs-compact comparisons and AIME/GPQA/HLE drivers. We have teacher-forced ΔNLL/KL probes, practical tool/coding cases and protocol tests. Compatible subset fixtures are reusable; their model/hardware scores are not local quant scores. [DI](../../DeepInfer/examples/README.md), [local recipes](../recipes/README.md). |
| V5 | Profile-based sweeps and throughput/latency curves | **Have locally.** Both separate profiler traces from steady-state timing and retain fixed-setting sweeps. Continue one-hour baseline/change/measure decisions, including ordinary/favorable/adverse cases, rather than importing a broad benchmark framework. [DI](../../DeepInfer/examples/sweep/plot_pareto.py), [local work order](../speed-bench/ideas/next-experiments.md). |

## Recommended alignment sequence — proposals, not implementation

| Order | Bounded first experiment | What would justify continuing |
|---|---|---|
| 1 | **One hour: candidate-only decode scoring.** Leave layer 20 selection unchanged; fill scores with `-inf`, compute only existing selected blocks and scatter to original IDs before the existing top-k. Keep the current path when all keys survive. | Same selected/global-ID semantics (including causal tails and tie order), measured index-kernel savings at 16K/64K/256K and an actual long-context decode improvement. At 256K at most 16,384 candidate keys gives an approximate 1/16 dot-product work bound, not a TPS prediction. Treat prefill separately. |
| 2 | **One hour: block-only restore prototype.** Keep compressed/index blocks, discard encoder SWA and replay 128 rows with an explicit raw floor. Do not change general cache policy until this can be measured. | Bounded128-row recovery rather than a full miss; per-row global attention/index work can still grow with context; ordinary/retrieved-fact/adverse quality cases plus ΔNLL/KL and restore latency. Suppress writes to old compressed rows, preserve absolute RoPE/Engram history, and retain approximate identity. |
| 3 | **One hour: packed-KV producer/reader pilot.** Choose one format/owner and compare real-row round trips and one consuming kernel. | Actual allocation reduction and acceptable latency/quality; SSD bytes only shrink if their encoding also changes. Match scales and rounding before calling it storage-only or lossless. Do not combine all cache formats and kernel fusion in the first change. |
| 4 | **One hour: restore useful drafter history.** Compare an optional saved feature/projected-KV section against the current 128-step warmup on short continuations. | Real speculative blocks after restore and a whole-request benefit; our recent default-depth results were near neutral, so enabling speculation is not itself a speed result. |
| 5 | **Further bounded checkpoints:** generalize decoder compaction to ordinary chunking and DFlash rollback; profile one norm/RoPE/store fusion; then consider same-input projections, DS41 GPU sampling/mask reuse, position-based seeds and exposed CPU/GPU idle gaps. | Profiled costs and representative end-to-end gains. Async grammar compilation, full ZOS or a large Mega MoE port need evidence that their costs matter here. |

Encoder restore needs special care with shared owner publication counts,
candidate epochs, original positions, pending C2 pairs at odd anchors and the
three-token Engram history at the replay start. Rewinding only `graph.pos` is
not an implementation. Existing approximation-aware persistence is useful, but
its current replay kind excludes DSpark and suffix capture; those invariants
need an explicit decision if the new approximation is made persistent.

[di-readme]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/README.md
[di-deployment]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md#L179-L191
[di-prepare]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/input_metadata/prepare_tensor_input_v4.cpp#L499-L523
[di-yoco]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/examples/yoco/README.md
[di-yoco-code]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/yoco.py#L115-L151
[di-cross-layer]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/cross_layer.py
[di-index]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/index_attn.py#L421-L454
[di-cache-format]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L109-L137
[di-cache-engine]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/worker/cache_engine.py
[di-block-manager]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/block_manager.hpp
[di-dflash]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/mtp/dflash.py#L198-L227
[di-attention-backend]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/attention_backend.py
[di-attn]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/attn.py
[di-hc]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/hc.py#L174-L314
[di-mlp]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/mlp.py
[di-engram]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/engram.py
[di-scheduler]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/scheduler.hpp
[di-worker]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/worker/worker.py
[di-engine-args]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/engine/arg_utils.py#L48-L55
[di-mtp-varlen]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/mtp_varlen.hpp
[di-sampler]: https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/sampler.py#L323-L437
[ds-config]: ../src/engine/config.zig#L54-L94
[ds-attention]: ../src/engine/deepseek41/attention.zig#L123-L176
[ds-batch-attention]: ../src/engine/deepseek41/batch_attention.zig#L90-L150
[ds-kv-layout]: ../src/engine/deepseek41/layout.zig#L186-L217
[ds-payload]: ../src/engine/deepseek41/state_payload.zig#L87-L105
[ds-resources]: ../src/engine/deepseek41/resources.zig#L82-L88
[ds-state-cache]: ../src/state_cache.zig#L801-L825
[ds-persist]: ../src/server/persist.zig#L270-L299
[ds-prefill-policy]: ../src/engine/deepseek41/prefill_policy.zig#L105-L127
[ds-prefill]: ../src/engine/deepseek41/prefill.zig#L262-L269
[ds-dspark]: ../src/engine/deepseek41/dspark.zig#L389-L513
[ds-speculative]: ../src/engine/deepseek41/speculative.zig#L196-L349
[ds-checkpoint]: ../src/engine/deepseek41/checkpoint.zig#L149-L212
[ds-ops]: ../src/engine/deepseek41/ops.zig#L127-L165
[ds-moe]: ../src/engine/deepseek41/moe.zig#L120
[ds-coordinator]: ../src/server/coordinator.zig#L51-L278
