# Proposed server and persistent-agent day

2026-10-04. Approved by the user. All times Eastern
(UTC-4). Target start 08:45; prepare a results handoff by 18:00; keep 18:00–19:00
for the user's chosen follow-up and final handoff before the stated reset.
Execution began08:53:40 Eastern; shift the sequence and shorten the optional end.
This is an extension of next-experiments.md, not a new harness project.

## Priorities

1. Make existing suffix reuse useful to ordinary HTTP clients.
2. Add useful standard durable-response behavior, keeping response storage
   independent of GPU/checkpoint residency.
3. Measure actual persistent-agent work, including quality loss and possible
   benefits of retained old-prefix influence.
4. Expose already measured options where integration is small and honestly
   scoped; spend the adaptive hour on an observed bottleneck.

## Sequence and budgets

Each numbered item has a 60-minute wall budget including preparation, edits,
build, checks, model work, interpretation and a small jj commit.

| Item | Proposed time | Concrete result |
| --- | --- | --- |
| 10. HTTP suffix reuse | 08:45–09:45 | Enable original-backed capture/reuse in ds4-server for full-input context edits, behind an explicit experimental option. Reuse shared cache storage and core relocation; report accepted grafts, fresh/reused tokens and fallback reasons. Demonstrate one real HTTP edit. |
| 11. Durable response chaining | 09:45–10:45 | Implement a useful text/tool subset of store, previous_response_id and stored-response retrieve/delete. Preserve standard append/fork semantics and current-request instructions; exercise tool continuation and a restart. Use simple records, not a database abstraction. |
| 12. Server workflow qualification | 10:45–11:45 | Run existing storage-backed ordinary edits through HTTP. Alternate two conversations, force slot turnover, reopen disk-only checkpoints, and compare complete edit-to-answer latency against matching fresh controls. Check state isolation and actual archive retrieval. |
| 13. Conversations basics | 11:45–12:45 | If they fit the same item store, add the text/tool conversation reference and essential create/read/item append/list/delete operations needed to manage history. Demonstrate deleting an old log then continuing. Record supported scope; do not chase the full API surface. If integration is not small, use this hour for a measured server gap. |
| 14. Suffix tradeoff curve | 12:45–13:45 | Compare fresh execution with fresh tails 64/128/256 where geometry/query length permits; refine only a useful gap. Include favorable, ordinary and changed-fact cases. Plot wall time versus delta-NLL with independent task outcomes; report KL separately. Include capture and SSD costs. |
| 15. Retained-history experiment | 13:45–14:45 | Compare fresh shortened context, suffix reuse and an explicit summary/retrieval control. Test carried decisions/constraints plus changed facts and superseded instructions. Ask whether latent historical influence helps, harms or is indistinguishable; do not infer reliable memory from one correct answer. |
| 16. Usable reasoning control | 14:45–15:45 | Investigate the small production port of the measured reasoning cutoff and qualify an explicitly scoped opt-in control. The private WRITE256 stage was supplied by a fixed read→write observer: do not label a generic cutoff as equivalent. Compare a native control and two useful settings on existing executable tasks if integration fits. |
| 17. Unused practical confirmation | 15:45–16:45 | Confirm the most useful server suffix point and any newly usable reasoning/resident option on unused executable code/tool and retrieval tasks, with reversed timing order. Keep feature attribution clear; combine only if it actually fits and can be measured. |
| 18. Adaptive measured follow-up | 16:45–17:45 | Pick the largest observed cost or usability gap: insufficient cache hits, recurring geometry fallback, checkpoint selection, request/stream compatibility or an incomplete high-value integration. Right shifts receive time only if observed edits justify them; odd-pair repooling has no automatic priority. |

17:45–18:00: update the small server preset/help and usage examples, consolidate
results and prepare the check-in. Packaging known settings is a short task.
18:00–19:00: reserve for the user's selected next item or, absent new direction,
one previously identified high-value follow-up and the final handoff. Do not
start a new broad implementation at the deadline.

## Compatibility and semantics

- Existing ordinary full-input clients remain useful. Pi's normal Responses
  path replays input with store:false; response-ID chaining is optional value,
  not a prerequisite for server suffix reuse.
- Pi-durable reconstructs its own history and has a persistent provider session
  identity for cache affinity. Pi also offers provider request/header hooks for
  a small DS4-specific opt-in, if a server-wide experimental option is not used.
- A cache/session hint can improve source selection; it does not itself opt
  into approximation or become a durable conversation reference.
- Standard previous_response_id appends/forks stored history. It is not an API
  for replacing earlier history. Edited full-input replay and any explicit
  suffix-source option retain that distinction.
- Conversation/response items persist independently of cache eviction. A cache
  miss rebuilds the relevant visible context; the cache is not the fact store.
- Reuse stays explicit and approximate. Preserve original-backed checkpoints,
  fresh incoming queries and existing unsupported-geometry fallbacks. Keep
  exact-prefix checkpoints distinct from historically conditioned grafts.
- Test the smallest relevant SDK/Pi streaming/tool path. Do not migrate or
  customize a whole agent harness during compatibility investigation.

## Measurements and stop rules

- Reuse existing helpers, fixtures, graders, plots and pressure guard. Run one
  heavy model job at a time; suspend unrelated work for final latency runs.
- Fixed baseline, one change, measure, inspect, decide. Record revision,
  command, settings, wall time, outcomes, reuse/fallback counts and pressure.
- Whole workflow and edit-to-next-answer latency matter. Do not add earlier
  percentage gains together or report cache reassembly as whole-task latency.
- Aim for at least 20% lower relevant wall time. For paired fixed teacher
  tokens, +0.0953 nats/token is a preferred delta-NLL reference and +0.1823
  corresponds to 20% higher perplexity. These are interpretation targets,
  not automatic rejection gates. Keep slower/degraded points and baseline
  mistakes visible. Report families individually and real task pass counts.
- If an item is incomplete at its deadline, retain concrete progress and the
  blocker, then select the next runnable item. Extensions must name a measured
  reason and consume a newly assigned budget, such as the adaptive hour.
- Add only checks protecting a concrete invariant: continuation context,
  instructions, tool IDs, session isolation, cache eligibility and fallback.
  No parser framework, exhaustive malformed-input suite or broad provenance
  scaffolding.
- Keep useful artifacts in speed-bench/ and .workspaces/. Keep changes in small
  jj commits. Proposed experiments are not approval to enable approximations
  by default.

## Primary references checked for the proposal

- [OpenAI conversation state](https://developers.openai.com/api/docs/guides/conversation-state)
- [Responses create parameters](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
- [Conversations create](https://developers.openai.com/api/reference/resources/conversations/methods/create)
- [Pi Responses request construction](https://github.com/earendil-works/pi/blob/200387122ca450d6387f033949423114a270b96c/packages/ai/src/api/openai-responses.ts#L279-L315)
- [Pi-durable provider identity](https://github.com/earendil-works/pi/blob/200387122ca450d6387f033949423114a270b96c/packages/durable/src/harness/provider.ts#L6-L38)
- [Pi provider customization hooks](https://github.com/earendil-works/pi/blob/200387122ca450d6387f033949423114a270b96c/packages/coding-agent/src/core/extensions/types.ts#L811-L824)
