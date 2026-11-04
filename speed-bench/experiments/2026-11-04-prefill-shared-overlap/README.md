# Shared-expert overlap during selected prefill

This experiment asks whether the existing batch shared-overlap path shortens
selected prefill at ordinary and larger frontiers. The comparison is in
progress; there is no performance conclusion yet. The first one-hour slice
runs on November 4, 2026, from 08:37 to 09:37 Eastern.

The [previous dependency profile](../2026-11-03-post-rope-profile/README.md)
found long delays before routed-expert GPU commands in a warm append. With
batch overlap off, the selected backend waits for shared-expert work before
preparing selected cache entries. The hypothesis is that preparing those
entries while shared work runs can hide some of that preparation. The earlier
[SSD experiment](../ds41-ssd/README.md#30-overlap-selected-batch-ssd-binding-with-shared-expert-gpu-work)
introduced this path as an opt-in; its historical snapshot checks do not
establish performance for the current baseline.

In [`Moe.batch`](../../../src/engine/deepseek41/moe.zig), the eligible streaming
path completes the router and marks its selected IDs ready before queuing
shared GPU work. The selected backend can then flush the shared work and prepare
expert cache entries on the host while it runs, before submitting routed work
behind it. The final wait protecting shared and routed work remains. This is
cache preparation overlap, distinct from the CPU resident-routing substitution
performed earlier by [`route_services.prefill`](../../../src/engine/deepseek41/route_services.zig).
The feature applies only when the streaming selected-batch backend supports
the shape and the call does not force resident execution.

## Setup

Both arms use published runtime revision `364ceeab`; no runtime change is
required for this first comparison. The baseline omits
`DS4_METAL_V41_BATCH_SHARED_OVERLAP`, while the candidate sets it to `1`.
The flag is presence-based, so setting it to `0` would also enable it. Driver
revision `36f8d24b` extends the existing
[decode-pass runner](../2026-11-04-decode-pass/run.nu) with `--control-unset`
to remove it from the control environment. The runner clears inherited DS4
settings and records the effective environment and arguments for each run.

The workload uses an M5 Max with 128 GiB RAM and the Flash 1017 IQ2_XXS/Q2_K
checkpoint, a 64 GB expert-cache request, 8,192-token prefill chunks and the
adopted routing and decode settings in the runner. It compares fresh prefill,
a 512-token append after restoring the prompt frontier, and complete
teacher-forced decode windows at 8K and 32K frontiers. Fresh processes start
with empty process-local expert caches; the OS file cache is not flushed.
Profile-free timing uses five alternating pairs per frontier under the pressure
guard. Raw logs and outputs are retained locally under
`.workspaces/hillclimb-20261104/01-prefill-overlap/`.

## Observations so far

The 8K quality comparison completed: all 256 paired logit rows were
byte-identical, as were all 256 post-evaluation hashes; delta-NLL and KL were
zero for every compared row. This checks the teacher-forced histories in that
run, not practical task quality or other workloads.

A single intrusive timeline diagnostic per arm recorded 288,572 encoders for
each warm append. Command buffers rose from 1,609 to 1,929, including 320
additional empty readiness drains. The append envelope was 16.737 s versus
16.829 s, and the union of encoder intervals was 6.385 s versus 6.374 s.
Instrumentation perturbs these runs; they locate behavior and are not a
performance comparison.

The five alternating 32K timing pairs are underway. The 8K timing comparison,
resource observations and decision remain pending.
