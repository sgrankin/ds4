# DeepInfer technique inventory and local alignment

Source audit, 2026-10-27, pinned to DeepInfer revision
`553741557a57df44a46fb3ab72b6aafeaf945337` (upstream HEAD when audited) and
DS4 baseline `e2a38f6e2bf4`. Local status refreshed 2026-10-28 through the
subsequently qualified `aee06ac9` main-FP4 work and recorded five-pair campaigns;
this document refresh did not fetch upstream, build or run models. Original
measured revisions remain reachable from
`archive/pre-stack-cleanup-20261028`.

The user subsequently authorized its bounded convergence experiments. See the
[execution report](../speed-bench/experiments/deepinfer-alignment-20261027/README.md)
for current outcomes. Candidate-only scalar scoring is implemented opt-in:
exact distributions/outputs on the32K/64K probes, a small warm-decode benefit,
and no reliable first-answer improvement. The source references remain pinned;
status notes below include the subsequently qualified local implementations.
The subsequent block-only128-row prototype also validates bounded recovery and
small measured baseline-trace changes, with a size/latency tradeoff. It rebuilds
all 40 rings through a direct diagnostic API; ordinary cache policy and isolated
encoder-only recovery remain absent. See the report for scope and measurements.
The index-FP4 producer/reader pilot now also verifies68B rows against the old
GPU quantizer/scorer. The initial shared-tile reader costs modestly more.
A [matched register/SIMD follow-up](../speed-bench/experiments/index-simd-20261027/README.md)
keeps exact fixture scores/IDs and cuts dense256K FP4 scoring GPU time39.2%;
its packed/F32 gap is now about 2%. The subsequent
[serving integration](../speed-bench/experiments/index-serving-20261027/README.md)
completes physical index allocation, scalar/batch inference, v3 SSD checkpoints,
v2 imports and suffix conversion. DeepSeek fast profiles enable packed index
storage and SIMD scoring; `persistent-packed` aliases `persistent-fast`.
The subsequent [five-pair serving comparison](../speed-bench/experiments/index-five-samples-20261027/README.md)
found the packed layout roughly speed-neutral at32K, with exact teacher/output
IDs. Its established benefit is86.72% smaller index owners, not a claimed
whole-model speedup from the isolated kernel percentage.
Canonical DSpark history is implemented as DSP1: exact BF16 raw target features
survive SSD restart and enable immediate verified drafting. The subsequent
[ordinary-confidence five-pair campaign](../speed-bench/experiments/draft-features-five-20261028/README.md)
reduces restored three-task HTTP time 18.12% at confidence 0.6, with matching
outputs and a 10.343 MiB stored-file increase across three endpoints. The
DeepSeek fast preset package now enables DSP1 export by default, honoring an
explicit 0 override; it only matters with DSpark. Ordinary engine defaults
remain unchanged. The earlier eager-policy observations are separate evidence;
our raw-feature representation still differs from upstream's projected KV.

[Physical main FP4](../recipes/15-packed-main-kv.md) is now also implemented
and qualified: 288B GPU rows, scalar/batch consumers, canonical v4/v5, old-format
conversion, DSP1/MTP, SSD restart, repeated pre-RoPE suffix recapture and an
actual Responses restart into an F32-main receiver. Five 8K pairs are
latency-neutral; [five 32K pairs](../speed-bench/experiments/main-fp4-20261028/RESULTS_32K.md)
cost 1.98% more task time, slower in every pair, with identical observed output
IDs/teacher logits. The matching profile locates added work in indexed batch
attention, rather than standalone publication. Main FP4 is now enabled through
the opt-in DeepSeek fast presets, accepting that measured capacity/latency
tradeoff; explicit main=0 or an F16-only request opts out. Ordinary engine
defaults remain unchanged. Raw/SWA physical FP8 remains absent.

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
| Encoder replay, 128 tokens | Missing sliding-window attention (SWA) state after a compressed/index block-cache hit | **Diagnostic SWB1 prototype; ordinary cache policy absent** |
| Decoder/YOCO retained tail; example 140 | Decoder local state after encoder prefill | **Partial/opt-in** |
| DSpark target-feature history, 128 positions | The drafter's input history after restoring target KV | **Qualified at ordinary confidence; DSP1 enabled in opt-in fast presets, projected drafter KV rebuilt on use** |

The first is the remembered restore technique. Keep the cached compressed/index
blocks; replay the preceding 128 tokens to rebuild local attention state. At
replay position `i`, visible raw keys begin at `max(replay_start,i-127)`. This
approximates SWA while retaining the global compressed cache. The public engine
has the replay-floor metadata, but its complete cache-hit controller is omitted.
[Deployment description][di-deployment], [public replay floor][di-prepare].

For example, with a compressed prefix ending at 32,768, recovery can begin
at 32,640 and then process the new suffix. This is not suffix relocation or a
logical context edit. Our ordinary loader still requires a complete checkpoint
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
| C1 | Encoder bounded recovery | **Diagnostic prototype; ordinary policy missing.** A direct SWB1 importer rebuilds all 40 raw rings from128 scalar rows while preserving immutable blocks; it is not the isolated encoder-only policy or ordinary SSD-cache admission. Public `swa_valid_start` metadata is narrower than the documented production cache-hit policy. [DI][di-prepare], [DS4][ds-payload]. |
| C2 | Decoder/YOCO query compaction | **Partial.** DeepInfer can physically remove discarded decoder queries while still publishing full shared compressed KV. DS4 has exact reduced suffixes plus optional wide replay, with the restrictions above. Both keep full encoder/global-cache publication. [DI][di-yoco-code], [DS4][ds-prefill]. |
| C3 | Shared compressed/index owners | **Have for the validated architecture.** Our KV sources 2/8/14/20 and index sources 2/8/14/20/24/28/32/36 implement the trained sharing topology. DeepInfer derives a more general topology from checkpoint metadata. [DI][di-cross-layer], [DS4][ds-config]. |
| C4 | Inherited candidate-block hierarchy | **Have.** Later indexers inherit source-layer candidates. Our layer 20 selects up to 2,048 blocks of 8. This is model structure, not the optional resident-expert approximation. [DI][di-cross-layer], [DS4][ds-attention]. |
| C5 | Score only candidate keys | **Partial, opt-in scalar path.** `DS4_V41_CANDIDATE_INDEX=1` skips excluded keys after layer20 once more than16K keys exist, for F32 and packed index storage. Batch prefill still scores the dense prefix before filtering/top-k. DeepInfer also scores inherited candidates in its consumer kernels. [DI][di-index], [DS4][ds-attention]. |
| C6 | Bounded selection scratch | **Have, different implementation.** DeepInfer sizes score chunks from row bytes/memory budget and caps them at 8,192 queries. We retain 32 live score rows and use Metal partition/merge top-k, causal bounds and prefix optimizations. Replacing a supposedly naive sort is not the first gap. [DI][di-index], [DS4][ds-batch-attention]. |
| C7 | Packed low-precision KV | **Main/index packing qualified and fast-preset enabled; raw remains expanded.** Index uses 68 B physical GPU/v3 rows; main uses 288 B native FP4 GPU/v4–v5 rows, scalar/batch consumers, v2/v3 conversion, SSD/DSP1/MTP, repeated suffix grafts and actual Responses restart. Five 8K pairs are latency-neutral; five 32K pairs cost 1.98% more task time with exact measured teacher/output IDs. The profile locates indexed batch-attention consumer cost. Fast presets accept this memory tradeoff, honoring explicit main=0 or F16-only requests; ordinary engine defaults remain unchanged. Raw/SWA remains 2048 B F32 rather than 528 B packed FP8. See the byte table and [main recipe](../recipes/15-packed-main-kv.md). [DI][di-cache-format], [DS4][ds-kv-layout]. |
| C8 | Bounded speculative rollback | **Have, different design.** DeepInfer enlarges SWA/compressor rings for backtracking. Our verifier saves overwritten raw/feature slots and compressor carries, then restores rejected positions; committed lengths hide unaccepted compressed rows. Larger rings are not inherently required. [DI][di-cache-engine], [DS4][ds-dspark]. |
| C9 | Separate durable blocks from disposable state | **Partial.** DeepInfer separates paged block KV from fixed state slots and can regenerate dropped consumer state. Our content-addressed SSD chunks deduplicate arrays, but endpoint payloads still retain all 40 raw rings and odd-boundary compression carries. [DI][di-cache-engine], [DS4][ds-payload]. |
| C10 | GPU paging and copy-on-write forks | **Missing.** DeepInfer reference-counts paged GPU blocks and clones state slots. We have immutable SSD chunks, pins and shared stored bytes, but each restored graph has private context-sized mutable GPU arrays. SSD chunking is not GPU paging. [DI][di-block-manager], [DS4][ds-resources]. |
| C11 | Asynchronous cache fetching/import | **Partial elsewhere, absent at checkpoint import.** Our expert and Engram I/O can overlap compute; KV materialization currently assembles the whole endpoint synchronously before importing it. Unified memory removes a separate H2D tier, not the SSD/materialization work. [DS4 store][ds-state-cache], [import][ds-persist]. |
| C12 | Usable draft state across restore | **Qualified fast-preset option, different representation.** `DS4_V41_CHECKPOINT_DRAFT_FEATURES=1` persists raw BF16 target-feature windows as DSP1 in canonical v2–v5 and restores immediate verified drafting after SSD restart; projected drafter KV is rebuilt on first use. Five ordinary-confidence 0.6 pairs reduce complete restored-task time 18.12%, with matching output and bounded trailer cost; fast presets enable export, explicit 0 overrides it, ordinary engine defaults stay unchanged. DeepInfer stores projected drafter attention KV during forward. Both avoid 128 future scalar warmup tokens through different stored state. [DI][di-dflash], [DS4][ds-speculative], [qualification](../speed-bench/experiments/draft-features-five-20261028/README.md). |
| C13 | Approximation-aware persistence | **Deliberate local extension.** Our typed DRP1 checkpoints retain replay policy/approximate identity, resume matching prefixes and exclude canonical export/suffix grafts. This is not an accidental exact-cache leak. Upstream's documented decoder replay cache policy is stricter. [Local policy][ds-checkpoint], [recipe](../recipes/12-bounded-decoder-replay.md). |

The physical row formats are particularly concrete:

| Stored row | DeepInfer | Current DS4 GPU representation |
|---|---:|---:|
| Raw/SWA, 512 values, E4M3 + per32 UE8M0 scales | 528 B | 2048 B F32 |
| Main compressed, 512 values, E2M1 + per16 E4M3 scales | 288 B | 288 B opt-in physical FP4; 2048 B F32; 1024 B optional F16 |
| Index, 128 values, E2M1 + per32 UE8M0 scales | 68 B | 68 B physical FP4 in fast presets; 512 B legacy F32 |

`DS4_V41_COMPRESSED_KV_FP4=1` enables qualified physical main packing, retaining
the original group16 E4M3 scales at fresh publication. It takes precedence over
`DS4_V41_COMPRESSED_KV_F16=1`; F16 owners still use F32 canonical main bytes.
Main FP4 canonical v4/v5 and private DSSD captures retain physical rows, while
raw/SWA disk values remain F32. Legacy F32/F16 repacking separately preserves
decoded values, including coarse subnormal scales where amax/6 alone fails.
The index producer retains original group32 scales and mode2 rounding;
packed v3/v5 snapshots retain those bytes. Old-format imports convert through
bounded tiles, rather than creating a full expanded shadow. DRP1/SWB1 replay
bodies retain v2 through conversion; private DSSD may use v2–v5 and keeps its
approximate provenance. The fast preset package now enables main/index packing
and DSP1 export. An explicit main=0 disables main packing; an F16-only request
selects F16 unless main FP4 is also explicitly 1. This leaves ordinary engine
defaults unchanged; installed executables are updated separately.
For 262,144 context, the four cumulative owners represent about 655,360 rows.
Main+index arrays are approximately1.5625 GiB with both F32,1.2915 GiB with
F32 main plus the current packed index,0.6665 GiB with F16 main plus packed
index, or0.2173 GiB with the now-qualified physical main+index layouts, before
alignment/scratch.
These are byte calculations, not measured speedups or whole-process RAM.
An expert cache with an explicit fixed budget does not automatically grow when
KV gets smaller; the freed headroom must actually reach its admission budget.

## Kernels, layouts and weight traffic

| ID | Technique | DeepInfer and current DS4 comparison |
|---|---|---|
| K1 | Sparse attention kernels | **Have, different backend.** FlashMLA is integrated upstream; ours uses raw/gathered/indexed Metal attention and typed F16 readers. CUDA SASS/TMA details do not port literally. [DI interface][di-attention-backend], [DS4][ds-attention]. |
| K2 | Norm/RoPE/quantize/store fusion | **Partial.** DeepInfer combines transforms with final cache publication. Physical FP4 main/index publication combines quantization and store; normalization and RoPE remain separate, as do most other cache boundaries. Fuse one publication boundary while preserving rounding locations. V4.1 has no indexer Hadamard; do not import a generic transform accidentally. [DI][di-attn], [DS4 quantization](../src/metal/dsv41.metal). |
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

## Initial convergence sequence — completed

| Item | Current outcome |
|---|---|
| Candidate-only decode scoring | Implemented opt-in, preserving selected/global IDs and distributions on the probes. A five-pair end-to-end promotion comparison remains worthwhile if long-context decode is the target. Batch prefill remains dense. |
| Block-only restore | SWB1 diagnostic import plus128 scalar reconstruction rows works. It rebuilds all 40 rings; batched recovery and an ordinary cache-retention/admission policy remain open. Full-state hits are the faster existing path. |
| Packed index producer/reader | Completed beyond the pilot: physical GPU owners, scalar/batch consumers, SIMD, v3 SSD, old-format imports and suffix/MTP. Enabled in fast presets with five-pair speed-neutral qualification. |
| Useful drafter history | DSP1 raw BF16 target-feature persistence works across canonical/SSD restore. Five ordinary-confidence 0.6 pairs reduce complete restored-task time 18.12%; enabled through fast presets with explicit 0 override. Storage grows 10.343 MiB across three endpoints. |
| Physical main producer/consumer | Qualified beyond the original proposal: 288 B GPU/v4–v5 canonical rows, legacy conversion, SSD/DSP1/MTP, repeated suffix grafts and actual Responses restart. Five 8K pairs are neutral; five 32K pairs cost 1.98% more task time. Consumer cost profiled, fast presets now enabled with overrides; raw FP8 remains separate work. |

## Next bounded candidates — proposals, not execution authorization

The **DSML length-finalization repair is complete**: token/context limits retain
typed partial calls, argument prefixes and already-streamed IDs, without
canonicalizing/executing partial arguments or leaking raw DSML as text.
Malformed output at normal stop remains an error. See the
[qualified wire report](../speed-bench/experiments/dsml-length-20261028/README.md).
Ordinary-confidence DSP1 and physical main FP4 are completed functionality,
not remaining implementation proposals, and both are enabled in the opt-in
fast preset package. Ordinary engine defaults remain unchanged.

After that, choose the objective and give each experiment an initial one-hour
cap, including preparation/builds/analysis. Performance comparisons need five
samples per side, alternating order; keep normal reasoning and actual task quality.

| Objective | Concrete next experiment | Gate for continuing |
|---|---|---|
| More concurrent-session capacity | Physical **raw FP8** publication plus a consumer after the qualified main/index work. Existing mode1 already quantizes then expands into BF16-rounded F32; retain original per32 UE8M0 scales at publication. Raw rows2048→528B. | Preserve existing decoded values where possible, measure any deviations, demonstrate real smaller allocation and measure consumer/whole-inference cost. Main packing's measured 32K cost is not evidence that smaller storage automatically accelerates attention. |
| Ordinary prefill latency | Extend decoder query compaction from wide≥8K sweeps to ordinary2K chunks, initially without DSpark; handle speculative rollback in a separate slice. | Full encoder/global-cache publication, explicit replay floor/approximate identity, practical tasks and ΔNLL/KL, five-pair whole-request savings. Do not infer benefit from the old wide-only branch. |
| Fewer intermediate writes/launches | First profile direct SwiGLU→existing packed F16 down input, eliminating HALF MID→HALF PACKED copy; alternatively fuse V4.1 collapse→BF16→weighted RMSNorm→BF16 using the previous PRE mix. | Preserve rounding and old consumers. The matched activation experiment measured only~39ms total down-pack GPU time, so this one copy alone is a modest target. Legacy HC compound kernels apply the newly computed PRE and are not drop-ins. |
| Better cold-cache retention | Batch the128-row SWB1 reconstruction and test block-only fallback under a bounded SSD working set. | Demonstrate that dropping disposable state actually improves retention/miss cost. Do not replace faster exact hits with recovery. Ordinary cache policy remains a separate integration step. |

Later candidates remain same-input projection fusion, DS41 GPU sampling and
grammar-mask reuse, position-based seeds, asynchronous checkpoint import and
CPU/GPU scheduling overlap. GPU paging/COW and full Mega MoE/ZOS are larger
projects; distributed RDMA/3FS transport has no direct one-process benefit here.

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
