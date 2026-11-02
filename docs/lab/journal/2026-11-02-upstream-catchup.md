# Adapting upstream through 890898cf

## Question and setup

Could we bring the native runtime forward to upstream's 2 November 2026
`890898cf` boundary while preserving the local serving, cache and model
ownership that accumulated since the last adaptation? The previous upstream
boundary was `ee7e840`; the native pre-merge parent was `d41daf9f`. The merge
has those native and new upstream revisions as its first and second parents,
respectively. The [upstream comparison](https://github.com/antirez/ds4/compare/ee7e840...890898cf)
contains 118 non-merge commits. This is an adaptation to the native tree, not a
claim that upstream's original implementation was copied wholesale. The local
`main` bookmark has not been moved or pushed as part of this work.

We audited the range against the native implementation and kept the current
source organization. The [DeepSeek implementation](../../models/deepseek/implementation.md),
[Apple Silicon backend](../../models/deepseek/apple-silicon.md), and
[serving overview](../../serving/overview.md) describe the resulting mechanisms
and their current boundaries. No new checkpoint release was found in this range.

## What changed and why

The Metal adaptation brings in upstream's revised arithmetic and dispatch
mechanisms while retaining local packed caches and steering. The native
[V4.1 dispatch](../../../src/metal/v41_dispatch.zig) admits fused operations
according to their shapes, and keeps fallbacks for unsupported cases. This
includes active scalar routing, index selection, paired projections and norms,
plus integrated attention and output-to-residual fusions. The
[backend manual](../../models/deepseek/apple-silicon.md#scalar-work-and-command-scheduling)
describes their eligibility and the separate paths needed for steering and tracing.
Primitive parity establishes correctness only for the
exercised shapes; it does not establish a model-level speedup.

Layer-major prefill reuses a layer's streamed weights across a group of prompt
rows. The upstream policy can keep a medium append in one such sweep, then
skip older decoder rows that cannot influence the final result through the
remaining sliding-window layers. Kernel selection retains the original tile
size to preserve its arithmetic plan, and drafter features are captured before
later layers prune the rows they need. Semantic checkpoint collection needs earlier endpoint
activations, so it still requests full rows rather than using decoder pruning.
Packed KV remains a local storage choice; tensor-parallel ownership now follows
upstream's negotiated sharded or replicated policy.
The implementation is in the native [DeepSeek engine](../../../src/engine/deepseek41/),
with cache behavior explained in the [checkpoint recipe](../../../recipes/16-semantic-context-checkpoints.md).

Qwen's [next-chunk prefetch](../../../src/engine/qwen4/prefetch.zig) reads the next
chunk's n-gram rows while Metal executes the current chunk. The caller joins the
worker before publishing a completed frontier or returning on cancellation.
A worker that cannot start or whose read fails leaves the next chunk to
synchronous staging.
Full prompt validation remains in place. Native [Engram](../../../src/engram.zig)
gains asynchronous batch reads and a bounded page-cache warmer with explicit
ownership and join lifetimes.

The original released MXFP4 expert conversion
preserves codes and scales; it does not calibrate routed experts.

Serving adopts the upstream whitespace-only reasoning normalization, reports
image input capability from the loaded engine, raises the image count limit to
256, and bases idle V4.1 prefill quantum on configured capacity. The image body
size limit remains separate. The [serving overview](../../serving/overview.md)
owns the current contract. The retained V4.1 CUDA/HIP header now matches the
final upstream header, including stride-aware masks and small-batch fallbacks.
The previously removed CUDA/ROCm builds remain removed.

## Completion audit

The initial adaptation stopped short of integrating several upstream paths.
The user rejected that scope decision: upstream functionality must be ported
unless it was explicitly excised from this fork. Hardware availability limits
validation, not implementation. We recorded this rule in `AGENTS.md`, audited
all 118 commits, and checked native definitions for all 52 new public Metal
functions. Merge ancestry alone had concealed the missing integration.

The completion adds negotiated replicated experts with balanced three-per-rank
decode, split and concurrent shared FFN, inline TP release gates, and the
opt-in residency-set workaround. It also connects pre-M5 selectors, deferred
hyper-connection work, attention/residual fusions, batched index MMA, K-slice
projections and tiny-batch expert-load reuse. The
[upstream performance account](../../models/deepseek/upstream-performance.md)
explains the bottlenecks and original evidence; the
[distributed manual](../../backends/distributed.md) owns TP behavior.

The upstream next-token pre-encoding
experiment was reverted upstream and was not restored here. Retired root C
files and old documentation were not reintroduced; their surviving behavior
belongs in the native implementation and current manuals.

## Evidence and decision

The production-linked Metal suite, indexer fixture, Qwen kernel fixture, memory
fixture and host suite passed on the assembled tree. The host suite's first
sandboxed run failed on socket access; the permitted rerun passed. The installed
Q4 Qwen model test passed exact chunk and continuation comparisons, including
cancellation and resume and invalid input. These results qualify the exercised
paths; worker and thread failure fallback was not fault-injected. Raw logs stay
local and ignored under `.workspaces/upstream-890898cf/`.

The corrected runtime's reasoning-enabled DeepSeek checkpoint test passed six
capture endpoints, 18 teacher rows and two repeated relocations. State and
immutable pre-RoPE originals survived exact round trips, and the final complete
prompt frontier was byte-identical. Endpoint teacher logits were not all
bit-identical across execution shapes: the largest measured KL was 0.00632
and the largest absolute delta-NLL was 0.0528 nats/token. This was a continuation
check, not a task-success benchmark.

These were the initial adaptation's checks. Completion qualification is
recorded separately below, along with the corrected cache publishers.
These tests do not by themselves establish comparable request latency or cover
physical two-Mac execution.

## Publication mismatch and final screen

An initial five-pair regression screen found lower decode time and neutral
prefill. The subsequent F32/packed drafter continuation test caught a producer
mismatch: upstream's FP4 quantizer uses exact division, while local packed-cache
producers still used reciprocal multiplication. At input −37.5 and scale 7.5,
rounding could select −45 instead of the canonical −30. We aligned every
equivalent publisher and added midpoint coverage to the default Metal suite.

The corrected model test has zero logit difference on the formerly failing
fresh step. Its later constrained-draft cycles differ by at most 3.815e-6,
including both cache-layout conversions and SSD restart. The full Metal suite
and shared dense/Q4, normalization, expert-buffer, Qwen and GLM KDA checks pass.
A separate CLI smoke test kept reasoning enabled and returned `{"sum":42}`.

The corrected five-pair screen finds lower decode time, neutral prefill and
neutral complete-process wall time for the short fixed history. Its
[report](../../../speed-bench/experiments/2026-11-02-upstream-catchup/README.md)
owns all samples, absolute times, variation and the numerical comparison.
The frontier argmax matches the baseline, while the imported quantizer changes
the distribution. This supports the maintenance merge; it does not
establish a whole-agent speedup or a broad quality result.

## Completed port qualification

The completion passes the full host suite and production-linked Metal checks,
including exact K-slice and grouped low projections, tiny-batch MXFP4/Q4,
concurrent shared FFN, deferred KV, steering, Qwen and GLM. Two new fixtures
initially failed because their setup omitted a packet field and a required
command scope. Both now use the production contracts. The TP loopback fixture
explicitly checks inline versus session-batch gate selection and completes
8,192 ordered exchanges with bounded command memory. Paired TCP checks pass
all four optional-residency negotiations and all three required-mode mismatches.

Installed-model checkpoint capture, repeated suffix relocation, packed/F32
cross-import and DSpark SSD restart pass with reasoning retained. The CLI smoke
test returns the requested JSON answer. The
[completion section of the report](../../../speed-bench/experiments/2026-11-02-upstream-catchup/README.md#completion-follow-up)
records the candidate revision, five alternating pairs and numerical limits.
Physical two-Mac/RDMA operation and older-device latency remain unvalidated;
their implementation was retained and adapted. The full-table Engram warmer
also cannot meet its admission condition on this 128 GiB Mac.

The lesson is to separate completeness from qualification: enumerate surviving
upstream behavior, integrate it into the native owners, then state exactly
which hardware and paths the available tests exercised. A second merge parent
or an imported shader alone is insufficient evidence of a completed port.
