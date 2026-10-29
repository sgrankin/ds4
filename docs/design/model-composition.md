# Composing independently owned models

The intended architecture can keep two or more generative models loaded in one process, with independent engines and sessions, one frontend, and a shared scheduler and memory budget. DeepSeek V4.1 plus Flint is an example: a main agent can delegate a bounded subtask to Flint without unloading either model. This is design direction from October 24–26, 2026, not implemented multi-generator routing or evidence that two engines already compose safely. [Project status](../project.md) owns the current selection of work.

## Ownership and scheduling

A Responses request selects a model; its conversation, continuation state and `previous_response_id` ancestry remain pinned to that model. A model switch must not reuse another engine's KV. The process runtime should own shared Metal device/queue services and the total pressure-aware budget; each engine owns its weights, mappings, model execution and expert residency; each session owns its frontier and continuation state. Shared scratch can be leased for a bounded transaction instead of copied into every session. The interfaces should follow real model components, including decision and extraction models, rather than anticipate every possible plugin.

Initially, serialize GPU transactions: submit and finish one engine's work before another takes the shared resources. Bound prefill chunks and decode packets so other requests and cancellation can progress; host preparation and I/O may overlap. A separate CPU or Neural Engine helper need not wait on an unrelated GPU transaction, although shared bandwidth and pressure still require measurement. Simultaneous GPU execution is a later measured option: overlapping Flint cooperative resident kernels can violate whole-grid progress assumptions.

Flint's native port already owns most model buffers, pipelines and session state while borrowing the shared device and queue. That useful boundary does not prove co-residency is safe. The initial DeepSeek design had process-global model mappings and caches rather than independent owners. The October 25 ownership migration gave native DS4.1 Cores actual Host and Metal contexts and reference-counted shared services. Two DS4.1 engines preserved nine full-vocabulary logit comparisons through interleaving and closing one engine while the other continued; see the [ownership qualification](../../speed-bench/experiments/clef-native-20261024/OWNERSHIP.md). Retained CPU, TP and other GGUF paths keep their previous admission restrictions. That qualification does not implement the server routing and aggregate admission needed for two generative models. Qualification should exercise A → B → A continuation and logits, separate conversations, pinned response references, cancellation, failed startup and closing one engine while the other continues.

Account for all weights, expert caches, sessions and scratch together, charging shared allocations once. An early short-context estimate of 65–70 GiB for the example pair was not a simultaneous measurement; two 262144-token Flint sessions can materially increase it. The October 26 direction favors a resident Flint subagent with multiple sessions and eventual SSD spill for inactive session state while active decode remains resident. Evicting and reloading Flint weights to give DeepSeek more expert cache is lower priority: reload cost and contention with expert reads remain unknown.

## What Flint teaches about execution

Staying loaded, alternating GPU work and simultaneous execution are distinct capabilities. A cross-model megakernel is not required. Megakernels can reduce launch and synchronization costs and retain activations within a pipeline, but unrelated models do not gain same-model weight reuse merely by batching together.

The October 24 review of [`recursive-oss/rec` at `e53663b262af`](https://github.com/recursive-oss/rec/tree/e53663b262af4cb1ca498e596ca8b1a2d79e0800) distinguished load-time packing, batched prefill and persistent decode. Flint's M5 Max decoder uses a fixed resident pool, local expert teams and phase-specific work, reusing activations and avoiding a global routing publication barrier. It can generate up to eight greedy tokens per resident launch; filtered packets chain sampling and handoff on the GPU. Those scheduling ideas are candidates for DeepSeek, not evidence of a DeepSeek speedup: its router arithmetic differs and SSD expert misses create host boundaries that a fully resident decoder does not have. Its ordinary path read full logits to the host at the reviewed revision; a device-side selection experiment is a smaller candidate than immediately pursuing a whole-model persistent kernel. See [native Flint qualification](../../speed-bench/experiments/flint-native-20261023/README.md).

Repacking weights is also a valid design choice. GGUF need not be the hot execution layout: load-time packing or reproducible conversion may improve alignment, metadata, expert reads and activation reuse. Compare load/conversion time, memory and cache capacity, SSD bytes, prefill, decode and whole-task latency; preserve the source checkpoint. Separate lossless packing from requantization or changed arithmetic, and measure practical quality and distribution drift for the latter. Flint's INT4, fixed shapes and BF16 rounding are examples to study, not DeepSeek defaults. Any such candidate requires a separately selected experiment and the project's baseline and measurement loop.

## Earlier research leads

An October 23 source review at runtime `37498248b4cb` compared Flint, Strata
snapshot `7ddc17a38fe7` and eight vLLM proposals. Strata's
[resident-router bonus](https://github.com/Niko1221/Strata/blob/7ddc17a38fe7f31637e895ecfba58c2cd43d65ec/src/kernels/cuda/route_prior.cu)
biases selection toward resident experts while retaining unbiased mixture
weights. A local comparison would need to preserve DeepSeek's transformed
router-affinity semantics and compare the existing bounded substitution policy,
quality and expert reads. It remains an unselected alternative.

The historical [SSD handoff census](../../speed-bench/experiments/ds41-ssd/README.md#62-full-session-route-and-exposed-io-census)
found 75.3% all-resident layer events. Free future routes plus deferred checks
reduced decode time by 30.7%, while a real prefetch predictor was slower. Those
oracle bounds expose a possible host/GPU handoff cost; they are not current Zig
speed measurements or proof that useful routes can be predicted cheaply.

A header-only inspection of the 1017 checkpoint found a Q8_0 output head with
shape 5120 × 129280, occupying 670.703 MiB. Q4_K would occupy 355.078 MiB and
already has a native dense dispatch. That 47.1% reduction is a calculated head
size, not a measured whole-task gain. Profiling head time and then measuring
quality and latency remained proposed work.

The vLLM references covered [attention shapes](https://github.com/vllm-project/vllm/pull/60317),
[KV-sharing metadata](https://github.com/vllm-project/vllm/pull/60445),
[vision/audio](https://github.com/vllm-project/vllm/pull/60478),
[MoE shapes](https://github.com/vllm-project/vllm/pull/60504),
[vision batching](https://github.com/vllm-project/vllm/pull/60643),
[diffusion batching](https://github.com/vllm-project/vllm/pull/60663),
[text fusion](https://github.com/vllm-project/vllm/pull/60674) and
[MTP/head work](https://github.com/vllm-project/vllm/pull/60689).
Only the vision-batching PR was merged at that review. The transferable ideas
were tuning actual M5 shapes, reducing intermediate passes and host dependencies,
and judging speculation by accepted tokens per wall time and unique expert
bytes. No runtime change or benchmark was performed by this source review.
