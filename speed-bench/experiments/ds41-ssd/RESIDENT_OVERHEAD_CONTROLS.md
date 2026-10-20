# Resident routing overhead controls

First neutral instrumentation child based02d094c4; no policy-default change.
The existing apply selector/mass/rank/affinity semantics are retained. GPU
parity and performance are PENDING root's serialized model grant.

## Read-only audit

The native scalar Metal MoE path already ends router commands and copies six
IDs before early loading. resident_route_apply runs after this completion
boundary. tensor_contents, tensor_read and tensor_write are direct shared
memory accesses, with no GPU wait or allocation. Cache snapshots use the
[layer][expert] entry directly, checking model/offset/size/buffer identity;
they neither lock, age, count cache hits, install nor wait. Six selected checks
precede a mass/weak-miss guard; only then is a full384-entry snapshot made.
Candidate score and rank scans then read384 affinities/biases on the CPU.

This adds host work and delays the start of demand reads/shared-expert overlap,
even when it substitutes nothing. The old skipped_bytes statistic measures
logical missing-expert demand avoided, not physical SSD traffic. Cache state
can diverge after replacements, so matched complete phase I/O/end-to-end time
is needed as well as the immediate avoided demand.

## Exact native controls

Use positive DS4_METAL_V41_RESIDENT_ROUTE_GAP and ordinary scalar text SSD mode.
DS4_METAL_V41_RESIDENT_ROUTE_MODE is apply by default:

- native: parse the same policy parameters, execute the original routing path.
- lookup: read the original six cache identities and calculate weak eligibility;
  never take a full snapshot or change routes/weights/cache behavior.
- noop: calculate the entire replacement plan and final affinity coefficients,
  then discard it. A24-byte volatile CPU sink keeps normalization observable to
  the compiler. Original GPU IDs/weights and loader inputs remain untouched.
- apply: commit the existing bounded replacement plan.

Profile clocks are OFF unless DS4_METAL_V41_RESIDENT_ROUTE_PROFILE=1. Enabled
profile data aggregates per layer; nothing prints per token. The public CPU
snapshot ds4_resident_route_metrics_read (ds4_resident_route.h) returns counts
and router_wait, selected_read, lookup, guard, full_snapshot, plan, commit and
begin_load milliseconds. Use before/after snapshots outside timed phases.
The clocks are attribution-only; latency claims use PROFILE unset.

## I/O snapshots

The Metal CPU API ds4_gpu_stream_expert_io_metrics_read (ds4_gpu.h) adds no
GPU wait/cache mutation. DS4_METAL_STREAMING_EXPERT_IO_METRICS=1 enables relaxed
atomic counters in pread workers: actual syscall attempts, requested want
bytes (including EINTR retries/short reads), and returned bytes, including
partial/error reads. Existing successful loader-returned bytes and cache
hits/misses/evictions are reported separately. Profile-off adds no atomic
increments. Cache reset joins outstanding pending reads before changing the
workers' metrics setting.

Darwin proc_pid_rusage(RUSAGE_INFO_V4) supplies process_disk_read_bytes outside
timed phases. It covers ALL disk I/O attributed to the model process, including
mapped weights and Engram, and is not an expert-only or drive-global counter.
Returned pread bytes may have come from the OS file cache. Record cold/warm
conditions and effective cache capacity explicitly (64 and80GiB are distinct).

Root replay.c owns phase hooks guarded by DS4_REPLAY_ROUTE_METRICS. With the
typed build, set DS4_REPLAY_ROUTE_METRICS=1 to emit REPLAY_ROUTE/REPLAY_IO AFTER
the elapsed timestamp. PROFILE/IO_METRICS remain independent attribution
switches. Pure timing has no snapshot calls when the replay flag is absent.

## CPU validation and prepared binaries

Both changed core objects compile cleanly with production Darwin flags and
maximum-j2. Unchanged-source objects were reused; provenance is in the ignored
persistent workspace artifact directory speed-bench/ideas/artifacts/
overhead-build (reused-objects.json, manifest.json). Existing selector tests
pass production flags. tests/test_resident_metrics.c passes off/on and null
snapshot checks, with a valid process I/O counter; no model was opened, no
GPU tensor was allocated and no Metal command was submitted.

Prepared replay-route-metrics uses ROOT replay.c and the new workspace objects.
Before any latency comparison, root must verify native/lookup/noop raw logits,
checkpoint and I/O behavior exact on the same complete replay. Applied-policy
quality uses delta-NLL/KL and independently checked tasks, not exact logits.

## Following experiments (separate children)

First remove redundant weak/metadata scans, or filter eligible score neighbors
before checking residency, if profile data shows material host cost. Compare
cheap weakest-miss dropping with one/two-slot replacement under original mass
budgets. Broader protected-four settings can be screened with existing v4
(gap.2, max_mass.20, total.35, rank24/48), then locked before held-out testing.
Keep all original top contributors and unbiased affinity normalization rules.

Native async backfill is a later experiment. An original expert may be used
only before the layer's routed kernel is enqueued; afterward no selected ID,
weight or buffer-address rewrite is safe. Cache mutation workers must join
before owner-thread demand binding/eviction, and both fallback/native buffers
need existing in-flight protection. The first cheap fallback must show a
quality/latency tradeoff before adding that contention and lifetime machinery.
