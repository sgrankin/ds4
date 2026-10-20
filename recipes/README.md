# Local DS4 inference and persistent-agent recipes

This journal explains the changes we made, how another DeepSeek harness can
reproduce them, and what we tried without adopting. The algorithms are portable;
the flags and measurements describe this DS4 checkout. Read the evidence for the
specific workload before copying a setting.

Historical runtime chapters 01–06 describe the pre-upgrade 2026-10-20 snapshot,
local working revision `030ae74983d6df9526716d3add86cc05189497a8`. Experiment
revisions are stated beside results. Chapters 07–08 separately record the new
native-byte pilot and official-source research, with their own status and pinned
sources. [The upgrade record](09-upgrade-20261020.md) identifies the compact
stack, matching 1017 artifacts and the first 16-check model qualification.
Current launch examples use 1017 filenames; historical measurements retain their original
September checkpoint and revisions. New upgrade findings belong in their own
result blocks, not relabelled historical measurements.

## Start here

| Recipe | What to carry to another harness | Status in the inspected checkout |
|---|---|---|
| [Resident routing](01-resident-routing.md) | Use eligible resident substitutes while retaining native mixture coefficients; extend to selected prefill | Opt-in `resident-fast` / `persistent-fast` |
| [Suffix reuse](02-suffix-reuse.md) | Reuse an unchanged span after an edit; relocate from immutable pre-RoPE originals | Opt-in; even left/right moves, one source/span per edit |
| [Checkpoint storage](03-state-cache.md) | Immutable deduplicated chunks, endpoint state, bounded hot RAM and SSD storage | Used by agent/server presets; survives restart |
| [Agent and API workflow](04-agent-and-api.md) | Editable context, opaque turn handles, archival/retrieval, stable client histories | Direct agent and a supported Responses/Conversations subset |
| [Reasoning controls](05-reasoning-controls.md) | Keep stable instructions; budget thinking by known task phase | Caller opt-in; not an automatic phase detector |
| [Cache policies and failed directions](06-cache-policy-and-dead-ends.md) | Measure eviction, workspace and scheduling on real appends and growth | Several useful opt-ins; several rejected prototypes |
| [Native scale packing](07-native-scale-packing.md) | Lossless codebook indices for native MXFP4 scale bytes | Native experts save 3.670404%; routed Q2 experts ineligible; native Engram windows save 1.886079%; no serving gain |
| [Official inference lessons](08-official-inference-lessons.md) | Replay, rendering and scheduling boundaries from pinned DeepInfer/official recipe | Measured mixed-quantum tradeoff; no replay feature adopted |
| [2026-10-20 upgrade](09-upgrade-20261020.md) | Compact feature stack, checkpoint pairing and checksum-verified downloads | 16 first-attempt checks pass; three relocations save 30.27% edit time; scheduler latency separate |
| [Q2 kernel epilogue](10-q2-kernel-epilogue.md) | Keep paired gate/up activation in registers; measure before adopting | No useful ABBA gain; patch retained for replication, defaults unchanged |
| [Incremental tool schemas](11-incremental-tool-schemas.md) | Announce discovered schemas at their history position; preserve reusable prefixes | Opt-in; favorable parent workflow 46.30% faster, ordinary full-input 8.17% slower |
| [Bounded decoder replay](12-bounded-decoder-replay.md) | Replay a fixed decoder tail while retaining complete compressed/index state | Opt-in 128/selected1; 7.03% unbatched HTTP workflow saving includes shorter search output; separate direct fixtures save 34.32–38.17%; presets unchanged |

[Not-adopted summary](not-adopted/README.md) collects short reasons and links
for rejected policies and candidates that still lack serving evidence. The
native-scale pilot and official-source audit have compact evidence bundled
under `evidence/`; they do not change the earlier Q2 workload results below.

The combined server package saved **23.52% whole-task time** on two fresh coding
families (21.54% against the faster control), with every first implementation
passing 161 checks. The direct-agent package later saved about **46.5%** on one
favorable, edit-heavy retrieval workflow with warmed bootstrap/cross-run caches.
They are different workloads, not percentages to add or general speed promises.
[Server evidence](../speed-bench/experiments/combined-fresh-hour40-20261008/README.md),
[agent evidence](../speed-bench/experiments/agent-package-hour41-20261009/README.md).

## Local entry points

```sh
# Combined direct agent; add the mirror to let the model edit its context.
./ds4-local persistent-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  --live-context-file .workspaces/agent-context.txt

# The same runtime features behind HTTP for a different agent harness.
./ds4-local --server persistent-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
```

`context-fast` retains native expert routing while enabling context reuse.
`resident-fast` enables resident decoder routing and disables suffix reuse.
`persistent-fast` combines decoder/selected-prefill routing and left/right
suffix reuse. Inspect [ds4-local](../ds4-local): all use context 8192, an explicit
chunk 512 and a requested 64 GB expert cache, with later CLI arguments overriding
CLI defaults. Presets leave thinking policy unchanged. These settings are a
tested combination, not an established startup optimum.

On the reference Apple M5 Max/128 GiB machine, a cold 1,982-token agent system
prefill at chunk 512 took 56.38s (35.16 tokens/s). Most later package runs restored
that system prefix. Keep bootstrap, expert residency, application checkpoint
cache and OS file cache distinct when comparing “cold” and “warm”.

## A short measurement loop

1. Budget one hour including preparation, build, execution and analysis. Freeze
   a baseline, prompt/tool contract, model, revision, settings and timed scope.
   Clear inherited experimental variables so controls really disable them.
2. Change one policy. Run one heavy model job at a time under the existing
   pressure guard; high RAM use is expected, rising pressure/swap is the signal.
3. Use ordinary and favorable edits as well as adverse changed/deleted facts.
   Retain first outputs, protocol failures, caps and fallbacks. Grade actual
   tools and answers, not agreement with baseline wording.
4. Bracket with controls or use ABBA. Report time saved as
   `1 - candidate_time / control_time`; show faster-control comparison when
   drift matters. Separate inference, edit, task and launcher time.
5. For tunable policies, plot settings against speed and practical quality.
   Fixed-teacher ΔNLL/KL describe distribution changes; they are not rejection
   rules or substitutes for task grading. Stop improving the harness once it
   measures the experiment.

For a fixed teacher, `ΔNLL = mean(log p_baseline(y) - log p_candidate(y))`, in
nats/token. `exp(ΔNLL)` is the teacher perplexity ratio: +0.1823 means +20%.
It does not mean 20% more failed tasks. Use `KL(baseline || candidate)` on the
same histories/teacher tokens; no paired KL is assigned to diverging sampled
agent histories. Report worst cases as well as means. Logical skipped loads
and `pread` counters do not measure physical SSD traffic.

Here “native baseline” means the same Q2 weights with the studied routing or
reuse approximation disabled, not an unquantized BF16 reference model.

## Credit and scope

This work builds on [antirez/ds4](https://github.com/antirez/ds4). Its local
experiments predate and then compared notes with
[astrra.space/ds4-recipe](https://tangled.org/astrra.space/ds4-recipe), inspected at
`0316b956f8236d810f50c2ae0a6fda9f77ce35d6`. That recipe helped prioritize warm
prefill routing and broader suffix tests; its AMD/HIP/MXFP4 results are not our
Metal/Q2 results. We implemented bounded adaptations, not its whole router,
multi-source planner, DSpark or memory scheduler.

Original references: [DeepSeek-V4.1-Flash report](https://arxiv.org/abs/2609.19969),
[Context Language Models, Shao et al.](https://arxiv.org/abs/2609.37725) and
[authors' code](https://github.com/facebookresearch/context-language-models),
[Mixture of Cache-Conditional Experts](https://arxiv.org/abs/2412.00099),
[Quantized Reasoning Models Think They Need to Think Longer, but They Do Not](https://arxiv.org/abs/2606.00206),
and [oMLX's V4.1 delta cache](https://github.com/jundot/omlx/blob/main/omlx/cache/deepseek_v41_delta.py).
Each recipe identifies which ideas were adopted, adapted or only tested.
