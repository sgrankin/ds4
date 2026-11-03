# Overlapping resident routing and shared-expert work

The resident fast presets now queue independent shared-expert work across the
CPU routing decision and poll readiness briefly before blocking. Five paired
32K measurements show a repeatable decode gain; the combined workload gains
less because initial prefill dominates. Polling costs additional CPU time.
The [result report](../../../speed-bench/experiments/2026-11-03-route-overlap/README.md)
owns all samples, absolute rates, variation, CPU accounting and reproduction.

## Context and setup

The [preceding profile](2026-11-03-context32-profile.md) found that expert
substitution had removed most exposed SSD waits, while every layer still
crossed a host routing boundary. Starting from `e95e6d16`, this slice kept the
five-expert policy and IQ2/Q2 arithmetic fixed and moved only independent
shared gate/up work across that boundary. The M5 Max comparison used packed
KV, a 64 GiB expert-cache target, 8K chunks, fresh 32K prose and a warm
512-token append. Each iteration used five alternating profile-off pairs;
diagnostic and full-logit runs were separate.

## Findings and decision

The first blocking-event iteration produced a small decode gain and no clear
combined-time gain. The refined package with bounded polling produced a clear
decode gain.
The GPU still supplies the event's readiness guarantee; the host polls for
up to 2 ms, then falls back to blocking under the original timeout.

A 16 GiB cache check found a separate issue: leaving earlier expert references
pinned until whole-command completion changed eviction choices and later
substitutions. The router event proves those earlier reads are complete, so
the final implementation retires only their cache references at that boundary.
Pending command buffers, static views and the new batch remain protected.
The tight-cache full-logit comparison then returned to exact agreement.

Adopt the package in `resident-fast` and `persistent-fast`, with explicit
environment overrides for disabling overlap or polling. Keep its eligibility
limited to the guarded scalar resident-routing path; the work does not qualify
TP, other hardware or autonomous agent success. The owning
[Apple Silicon manual](../../models/deepseek/apple-silicon.md#scalar-work-and-command-scheduling)
explains the dependencies and resource lifetimes.
