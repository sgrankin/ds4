# Lab journal

These entries explain context, setup, findings and decisions, preserving failures
and limits as well as useful results. Read the [methodology](methodology.md) for
measurement conventions and the preferred entry format; [project direction](../project.md)
owns current goals.

Dates identify the research period or final synthesis. Some entries group related
follow-ups across several days; the text dates important results. A historical
command or speed is evidence at its recorded revision, not a current-build claim.

| Entry | Question and lesson |
| --- | --- |
| [Initial cache and quality probes](journal/2026-10-02-initial-cache-and-quality-probes.md) | Exact frontiers, approximate suffix state, reasoning interventions and operational failures. |
| [Resident experts](journal/2026-10-03-resident-expert-tradeoffs.md) | Route quality versus residency, mixture weights and rejected backfill. |
| [Persistent-agent workflows](journal/2026-10-04-persistent-agent-workflows.md) | Real edits, archives, restarts, retention and package-level attribution. |
| [Recipe integration](journal/2026-10-08-recipe-integration.md) | Combining routing and suffix reuse while separating caller policy and measured scope. |
| [Checkpoint and kernel tradeoffs](journal/2026-10-21-checkpoint-and-kernel-tradeoffs.md) | F16 state, relocation, replay and attempts to reduce execution costs. |
| [Native Zig rewrite](journal/2026-10-22-native-zig-rewrite.md) | Why translation was insufficient, ownership migration and qualification. |
| [Native models and steering](journal/2026-10-24-native-models-and-steering.md) | Shared runtime, model ports, residual interventions and behavioral checks. |
| [Decision models and scheduling](journal/2026-10-25-decision-models-and-scheduling.md) | Clef quality, GPU performance, engine ownership and coexistence costs. |
| [Hosted orchestration](journal/2026-10-26-hosted-orchestration-lessons.md) | Why tools, code mode, MCP and result references moved back to the client. |
| [Interaction and constraints](journal/2026-10-27-interaction-and-constraints.md) | Official templates, tool schemas, truncation and grammar-aware speculation. |
| [DeepInfer convergence](journal/2026-10-28-deepinfer-convergence.md) | Recovery windows, drafter history and the public/production service boundary. |
| [Packed caches and kernels](journal/2026-10-28-packed-cache-and-kernels.md) | Storage, SIMD, compute formats, negative prefill results and neutral fusion. |
| [Semantic checkpoints](journal/2026-10-28-deepseek-semantic-checkpoints.md) | Useful edit endpoints without extra weight sweeps; setup and no-edit costs. |
| [Documentation reorganization](journal/2026-10-28-documentation-reorganization.md) | Replacing running status blobs with maintained manuals and discoverable research. |
| [Root documentation cleanup](journal/2026-10-29-root-documentation-cleanup.md) | Evaluation provenance, scoped release qualification and one canonical agent guide. |
| [Upstream catch-up](journal/2026-11-02-upstream-catchup.md) | Native adaptation through 890898cf, cache quantizer alignment, continuation checks and five-pair regression evidence. |
| [Resident routing through six](journal/2026-11-02-resident-routing-through-six.md) | Seven policies, five samples each: cap five improves warm time beyond the then-current preset; six adds drift without a reliable extra mean saving. |
| [Longer resident-routing quality](journal/2026-11-02-resident-long-quality.md) | Longer text and recorded agent histories retain a small mean cost at five substitutions, with local wording changes and narrow answer checks. |
| [32K throughput and dependencies](journal/2026-11-03-context32-profile.md) | Five-pair chunk comparison and separate profiles identify the per-layer routing handoff as a useful next target after resident substitution. |
| [Routing/shared-expert overlap](journal/2026-11-03-route-overlap.md) | Bounded event polling improves decode; precise expert-cache retirement preserves eviction behavior when work continues after the router event. |
| [Profiling after overlap](journal/2026-11-03-post-overlap-profile.md) | The current path uses fewer command buffers; repeated inverse rotation in attention output is the next bounded decode target. Noisy append timings and intrusive profile limits remain visible. |
| [RoPE reuse in SSD streaming](journal/2026-11-03-stream-rope.md) | Extending the existing attention-head transform gives a modest, consistent 32K decode improvement with exact fixed-history logits; short-context and whole-phase results have narrower limits. |
| [Profiling after RoPE reuse](journal/2026-11-03-post-rope-profile.md) | Slow selected-prefill waits occur before routed expert GPU work begins. This motivated the subsequent slab-size test; the query projection remains a separate decode candidate. |
| [Expert-cache slab size](journal/2026-11-03-expert-slabs.md) | Smaller buffers shorten the first decode step but do not reliably improve append or whole-workload time; five-pair tests retain the 4 GiB default. |
| [Short Q8 query projection](journal/2026-11-03-short-q8-query.md) | Sharing input loads and keeping a rotated output pair within one SIMD group preserves measured logits. Adopted in resident fast presets despite uncertainty about the size of the speed benefit. |
| [Overnight decode optimization](journal/2026-11-04-overnight-decode.md) | Five adopted changes raise measured 32K steady decode by about 7–8%; recorded coding replay inference takes 4.2% less time. Three other candidates remain unadopted. |
| [Decode and prefill hill climbing](journal/2026-11-04-through-06-performance.md) | Paused November 5 after the compressor-pair experiment; four-row staging and approximate rescaling were adopted locally after combined prefill, replay and quality checks. |
| [Upstream catch-up through b3f50180](journal/2026-11-08-upstream-catchup.md) | Complete native adaptation, Qwen SSD and predictor state, exact continuation checks, and five-pair gains for Qwen/GLM with inconclusive DeepSeek timing. |

The [design folder](../design/README.md) preserves unselected ideas separately from
measured conclusions. Original concise reports under `speed-bench/` own complete
samples and reproduction details. Raw output and generated artifacts remain local
and ignored.
