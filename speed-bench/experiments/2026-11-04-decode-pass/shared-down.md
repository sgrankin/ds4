# Shared down projection during router wait

Moving the shared expert’s down projection earlier produced modest profile-free decode gains at 8K and 32K, with exact output in the tested fixed histories. The `resident-fast` and `persistent-fast` presets adopt the change; an explicit 0 overrides it. Ordinary runtime and `context-fast` remain unchanged.

## Hypothesis and mechanism

The baseline overlaps shared gate and up projections with the host's wait for the router event. The shared down projection normally remains until after routed expert work, where a fused operation combines it with the routed result and expands the hyper-connection (HC) residual.

The candidate schedules the Q8 shared down projection after gate/up and before the existing router-event commit. It records that the shared projection was enqueued, not that GPU execution has completed, and later uses the existing shared-first add and BF16/HC join. The projection input and output, sum and HC output retain their BF16 boundaries. This schedule may let the GPU execute the projection while the host handles routing and expert selection.

Splitting the fusion writes the intermediate to an existing shared scratch buffer, which must remain live until the join, and adds dispatch and memory traffic. It introduces no new kernels, queues or scratch buffers. The event path retires earlier expert-cache reads while keeping pending static and shared resources live.

The [cadence experiment](flush-cadence.md) established the baseline's every-layer submission setting. Its separate intrusive 32K warm timeline at `.workspaces/overnight-decode-20261104/flush-cadence/timeline1/1-candidate` showed about 1.09 ms per token of exclusive shared-down/HC tail and 6.34 ms per token of uncovered router-to-routed gaps. These are diagnostic observations, not a prediction that either amount can be saved: dependencies, scheduling and instrumentation can change the overlap.

## Setup

The one-hour loop budget was 01:21–02:21 Eastern on 2026-11-04. Baseline revision `1efa1175` includes F16 gather and decode flush cadence 1. The candidate revision is `ea71515a`; both arms use its same binary, with `DS4_METAL_V41_ROUTE_SHARED_DOWN=0` as control and the exact value `1` as candidate.

The candidate requires the existing scalar streaming router/shared overlap path and a Q8 shared down projection. Tracing, FFN steering, other weight types and disabled expansion fusion leave the early down path off. Scalar decode also includes eligible one-token prefill tails, so the checks include an 8,193-token frontier.

Both arms use cadence 1 and otherwise identical settings. The timing comparison uses five alternating, profile-free pairs at both 32K and 8K frontiers. Each timing run uses a fresh frontier and 128 teacher-forced decode steps, then restores KV, appends 512 tokens and runs another 128 decode steps. The expert cache survives restore; the OS cache is not flushed. Steady decode excludes the first step; complete decode includes it. Inference phase times exclude model loading and snapshot save and restore.

All completed timing and quality runs exited successfully at pressure level 1, with swap stable at 24.94640625 GiB. Raw evidence remains local under `.workspaces/overnight-decode-20261104/shared-down/`.

The runs used an M5 Max with 128 GiB memory and `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` with SSD streaming. Common settings include packed FP4 main and index KV, short Q8 query, streaming RoPE table, shared route overlap with a 2,000 µs poll, route cap 5 and gap 0.90, 4 GiB expert slabs, a requested 64 GB cache budget and 8,192-token prefill chunks. The driver clears inherited DS4 settings. The 32K timing command was:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/overnight-decode-20261104/shared-down/timing32k-guard -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/overnight-decode-20261104/shared-down/timing32k timing --flag DS4_METAL_V41_ROUTE_SHARED_DOWN --control 0 --candidate 1
```

## Findings and decision

At 32K, fresh steady decode rose from 33.834 ± 0.117 to 34.364 ± 0.218 tokens/s (mean ± sample SD); all five pairs favored the candidate. Warm steady decode rose from 33.152 ± 0.577 to 33.406 ± 0.376 tokens/s, with four favorable pairs and the first slower. Order alternated baseline first in pairs 1, 3 and 5, candidate first in pairs 2 and 4.

| Pair | Fresh baseline → candidate, tokens/s | Warm baseline → candidate, tokens/s |
| --- | --- | --- |
| 1 | 33.73 → 34.40 | 34.17 → 32.97 |
| 2 | 33.76 → 33.98 | 32.84 → 33.04 |
| 3 | 33.98 → 34.48 | 33.00 → 33.80 |
| 4 | 33.76 → 34.45 | 32.77 → 33.65 |
| 5 | 33.94 → 34.51 | 32.98 → 33.57 |

Complete fresh decode took 4.646 ± 0.045 versus 4.584 ± 0.040 s; warm decode took 3.868 ± 0.066 versus 3.836 ± 0.043 s. First steps took 892.3 ± 34.2 versus 888.3 ± 17.2 ms fresh and 36.2 ± 1.7 versus 33.7 ± 0.3 ms warm. Inference phase totals were 64.710 ± 1.883 versus 65.004 ± 1.336 s. Prefill variation dominates that total; these data do not establish a whole-workload improvement. Seconds derived from the driver's two-decimal throughput values are approximate.

At 8K, fresh steady decode rose from 34.030 ± 0.999 to 34.940 ± 0.795 tokens/s; four pairs favored the candidate and the first was slightly slower. Warm steady decode rose from 33.222 ± 0.462 to 34.158 ± 0.388 tokens/s; all five pairs favored the candidate. The order alternated in the same pattern as 32K.

| Pair | Fresh baseline → candidate, tokens/s | Warm baseline → candidate, tokens/s |
| --- | --- | --- |
| 1 | 33.62 → 33.52 | 33.55 → 33.90 |
| 2 | 34.62 → 35.28 | 33.66 → 34.74 |
| 3 | 32.45 → 35.32 | 32.48 → 33.73 |
| 4 | 34.72 → 35.24 | 33.17 → 34.27 |
| 5 | 34.74 → 35.34 | 33.25 → 34.15 |

Complete fresh decode took 4.586 ± 0.106 versus 4.515 ± 0.080 s; warm decode took 3.862 ± 0.054 versus 3.759 ± 0.051 s. First steps took 851.9 ± 16.4 versus 878.7 ± 35.7 ms fresh and 38.6 ± 7.1 versus 40.8 ± 17.4 ms warm. One candidate warm first step took 71.8 ms. Inference phase totals were 35.372 ± 0.457 versus 35.075 ± 0.389 s; these include variable prefill and append work.

The 8,192- and 32,768-token frontiers, an unaligned 8,193-token frontier and an 8K run requesting a 16 GB expert-cache budget each yielded 256 byte-identical full-vocabulary teacher-logit rows and 256 matching post-evaluation hashes, including the warm append: 1,024 of each in total. Delta-NLL and KL were zero, with no top-1 mismatches. These fixed-history checks do not establish parity on other contexts or sampled tasks.

Source inspection shows the split changes one pipeline lookup and dispatch into two, and seven buffer bindings into nine per layer, while retaining the same one model-view lookup and five explicit tensor-view retains. It adds no command buffer, wait, allocation or residency walk. Preparing the standalone projection before the router commit can delay the event and consume overlap. Profile-off scheduling reuses the active encoder; additional encoder counts under timeline instrumentation do not imply additional profile-off encoders.

## Timeline and decision

A separate intrusive 32K timeline pair confirmed activation. Both arms used 80 command buffers per token; diagnostic encoder counts were 892.055 versus 932.055. In warm decode, the old fused down/HC tail summed to 1.301 ms per token, while the candidate's standalone down and late join summed to 1.141 and 0.153 ms. Uncovered router-to-routed time fell from 6.433 to 5.592 ms per token, encoder union from 28.709 to 28.125 ms and token envelope from 35.208 to 33.956 ms. Fresh uncovered time fell from 6.316 to 5.703 ms, but encoder union rose from 27.573 to 28.274 ms and envelope from 33.933 to 34.069 ms. This mixed, instrumented single pair explains scheduling but is not a latency comparison. Its guard passed at pressure level 1; swap rose from 24.9464 to 24.9473 GiB.

The resident fast presets adopt the opt-in setting for the modest profile-free gains and exact tested output. This decision rests on the five-pair timings, not the timeline. The results cover one model and machine; fixed teacher histories do not substitute for broader task-quality testing.

The benchmark, agent and server builds passed, as did the launcher checks for preset defaults and explicit overrides.
