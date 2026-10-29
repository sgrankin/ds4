# DeepSeek on DS4 and DeepInfer

DeepInfer provides a reference for DeepSeek V4.1 Flash's state, dataflow and
inference techniques. DS4 follows that model contract while adapting execution
to one Apple GPU and local storage. This comparison explains the mechanisms and
the differences: what upstream does, what exists here, why an adaptation is
useful, and what evidence supports it.

The chapters below explain all 49 audited techniques, with the original IDs
retained in section headings for earlier references. Each account gives the
context, mechanism, local differences and evidence; it also identifies remaining
proposals without making them an execution queue.

## Read by topic

- **[Cache and attention](cache-and-attention.md), C1–C13.** Follow attention
  history from publication through row selection, checkpoint storage and restore.
  This chapter distinguishes encoder recovery, decoder query compaction and
  drafter history, then explains packing, rollback, paging, asynchronous import
  and approximation-aware persistence.
- **[Kernels and weight traffic](kernels-and-weight-traffic.md), K1–K9.** Follow
  an activation through projections, normalization, attention and experts. The
  comparison explains the opportunity behind each fusion, the numerical and
  layout constraints on a Metal adaptation, native versus quantized weight
  formats, Engram reads and reusable buffers.
- **[Scheduling and speculation](scheduling-and-speculation.md), S1–S6.** Follow
  requests sharing a model: prefill admission, continuous batching, pressure
  preemption, asynchronous scheduling and graph buckets. Draft verification and
  confidence allocation have separate costs and state contracts.
- **[Sampling, grammars and protocols](sampling-and-protocols.md), S7–S15.**
  Follow a constrained output token through mask preparation, sampling, matcher
  advancement and API conversion. This chapter also explains random-number
  policies, batch invariance, numerical failure handling and token provenance.
- **[Distributed and production serving](distributed-serving.md), P1–P7.**
  Understand attention and expert partitioning, load balancing, communication
  buffers, disaggregated stages, 3FS storage and failover. These techniques still
  deserve explanation when their useful local lesson is a narrower ownership
  or scheduling principle rather than a distributed implementation.
- **[Vision and qualification](vision-and-qualification.md), V1–V5.** Follow an
  image through preprocessing, GPU encoding and cache-aware chunking. The final
  sections explain how numerical, task and performance evidence establish
  different claims in the two implementations.

A practical starting point is the cache chapter's recovery example. Discarding
raw sliding-window state while retaining compressed history creates a different
problem from reducing decoder queries or restoring a drafter. Treating all three
as one "look-back window" hides the state each must reconstruct. The later
chapters build on those ownership and dataflow distinctions.

## Sources and comparison boundaries

The upstream sources are DeepInfer
[`553741557a57df44a46fb3ab72b6aafeaf945337`](https://github.com/deepseek-ai/DeepInfer/tree/553741557a57df44a46fb3ab72b6aafeaf945337)
and deepseek-recipe
[`935817c66f5d6e64838f419692545f06787e485f`](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f).
The initial local audit used DS4 `e2a38f6e2bf4`; subsequent implementations and
measurements retain their own revisions in the linked reports. Original
experiment revisions remain reachable through
`archive/pre-stack-cleanup-20261028`. The chapters reconcile that evidence with
the current source; an older baseline is not silently presented as the current
runtime.

DeepInfer's released engine targets SM100 and Ascend950. Its production-service
description includes infrastructure and integration that are not fully released.
The chapters identify that boundary instead of treating a deployment description
as available code. Locally measured results use the 1017 IQ2_XXS/Q2_K model on
the M5 Max, not the upstream native MXFP4 expert checkpoint. Numerical policy,
allocated capacity, kernel time and whole-task performance are separate claims.

For the underlying model, read the [reference manual](../reference-manual.md).
The [runtime implementation](../implementation.md) explains local state and
transactions; [Apple Silicon execution](../apple-silicon.md) explains the physical
layouts and Metal paths. The comparison owns the upstream-to-local account and
links to those manuals for the full implementation detail.

## Remaining differences

An unimplemented upstream mechanism is not automatically the next development
item. Some differences are local tradeoffs, some are incomplete integration,
and others serve hardware or deployment requirements absent on one Mac. Each
chapter states those distinctions and preserves proposed experiments with their
scope. Selecting further work belongs to the
[project direction](../../../project.md) and
[selected experiment](../../../../speed-bench/ideas/next-experiments.md), not to
an old "next" label in a comparison.
