"""Authored review and faithful summary; contracts/config stay outside the block."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTERVAL_DISCUSSION = """Planning review: this helper groups an already supplied batch of intervals. The caller may put records in any order, and the function must retain original zero-based positions in the result. We discussed accepting streams or keeping a stateful object between calls, but both ideas were deferred. Each solve call gets its complete batch and configuration.

The endpoints describe half-open intervals, but the grouping policy is based on separation rather than pure intersection. Touching intervals join even when gap is zero. The accepted comparison includes equality: a new start at the current group maximum end plus the configured gap joins that group. This decision should remain visible because a reviewer might otherwise assume strict overlap from the half-open notation.

An early sketch compared each interval to only the preceding interval's end. That fails when an earlier wide interval surrounds several later narrow intervals. The current group must remember its maximum end. Each joined interval can extend that maximum, while a nested interval leaves it unchanged. The group start is its minimum start, established by the sort order.

We agreed to reason about sorted indices instead of sorting the payload itself. Sorting by start, then end, then original index gives a deterministic sweep without losing provenance. Input records remain untouched. Duplicate endpoint pairs are separate original records, so the index list must include every occurrence. No deduplication or identity-based collapsing belongs in this helper.

The original positions returned within a group are sorted numerically. They are not the order in which the sweep visited the intervals. This was a deliberate interface choice: callers will look those positions up in their own original batch. Groups themselves remain in increasing start order; there is no second ordering by number of intervals, duration, or source position.

The configured gap is a nonnegative integer and comes from the actual config read. A design example used one unit, but the implementation must not bake that example in. Larger gaps can connect several otherwise separate spans transitively. Exact boundary cases need the same comparison as all other cases. Negative endpoints are allowed, and Python integer arithmetic avoids a timestamp-size assumption.

Somebody suggested dividing the time line into fixed buckets. We dropped that proposal because bucket boundaries can split a group or merge intervals whose separation is too large. A simple ordered sweep already expresses the rule. A graph implementation would also work for a small batch, but the production function need not build all pairs just to recover these sorted components.

The output is only a dictionary with groups, each containing start, end, and indices. We decided against adding duration, gap totals, event labels, or debug information. The end is the largest original end in the group; it does not include an extra gap allowance. Likewise, a group span may cover gaps that were tolerated by the policy without pretending those gaps were occupied intervals.

Empty input produces an empty groups list. A single interval produces one group with its lone original index. The task guarantees start is smaller than end, so the implementation does not need an input-repair policy for reversed or empty intervals. It also need not normalize integers or modify config. Those choices would obscure the core behavior and have no caller here.

The handoff is a plain solve function using permitted builtins. No class, import, filesystem access, clock, generator protocol, or command-line wrapper is needed. Read the advertised configuration once, consume that real result, and write the first complete implementation once. The unchanged task contract outside this planning block is authoritative. Complete with a brief final reply, without retries or a local testing stage."""

INTERVAL_SUMMARY = """Settled design: group the arbitrary-order interval batch by a deterministic sweep over indices sorted by (start,end,original index), leaving payload/config unchanged. Compare each start with CURRENT GROUP MAXIMUM end + actual config gap using <=: equality and touching at gap0 join despite half-open interval notation. Update the group's maximum end, so nested intervals do not lose a wide earlier reach and chains merge transitively. Return only {groups}; each group has minimum start, maximum original end, and ALL distinct original zero-based indices sorted numerically. Groups are in increasing start order; duplicate intervals keep separate indices. Empty input gives groups:[]. Use exact integers, including negative endpoints. Extra labels, durations, bucket schemes, deduplication, validation, stateful wrappers and logging were rejected or deferred. Preserve the unchanged task contract and actual config read; make the one required first write, then finish briefly."""

DEPENDENCY_DISCUSSION = """Planning review: this helper takes a whole batch of named tasks and their dependency lists. It is a deterministic layering operation, not an executor, worker pool, or persistent scheduler. We do not perform the tasks, write logs about them, or infer whether any work happened outside the supplied batch. Each call starts with no completed tasks.

The important scheduling decision is simultaneous rounds. Every task selected for a layer must have all its dependencies completed in an EARLIER layer. Selecting one ready task must not unlock another task into that same layer. The straightforward approach collects the entire ready set, orders it, appends it as one layer, and only then updates the completed IDs. This preserves the meaning of depth in a dependency graph.

We debated ordering ready tasks by their names everywhere, but the caller has a configuration choice. With order=input, keep original payload order within each ready layer and in unresolved. With order=lex, use Python ascending string order for those lists. Neither setting changes which tasks are ready or their layer number. The actual config read chooses the policy; a sample named lex does not justify hard-coding it.

Task IDs are unique exact strings. Empty strings, Unicode, and strings that look like numbers are ordinary IDs. We explicitly rejected case folding, whitespace stripping, integer conversion, and Unicode normalization. Dependency IDs use the same exact comparison. Do not substitute a hash or compare only a prefix of the name. A normal set of completed strings is enough for membership checks.

A review example repeated one prerequisite several times in a deps list. The duplicates do not require the prerequisite to finish several times; they impose the same condition repeatedly and must not inflate a degree counter. We can either test all listed dependency strings against the completed set or deduplicate them internally. Payload records and their lists stay unchanged either way.

Unknown dependency IDs need careful handling. A prerequisite absent from the payload is never completed in this call. Its task remains unresolved, and so do any tasks waiting on that blocked task. An early diagram silently treated absent dependencies as previously finished external work. That proposal was rejected because the interface has no external-completion input. The final coding contract states the same rule.

Cycles and self-dependencies are handled by the stopping rule. When a round has no ready tasks, stop and return every remaining task ID in unresolved. We do not need to classify which remaining task is in the cycle, which is downstream of a cycle, or which is blocked by an absent prerequisite. They all belong in the same unresolved list. Independent ready components can still produce layers before the stop.

One proposal used a recursive function to compute a depth for every task. That can express the same result, but requires explicit cycle and missing-name handling. Another proposal used degree counts and reverse adjacency. Both are optional implementation choices. The small task contract is clear with repeated scans of remaining tasks and a completed set, and that avoids treating duplicate dependency entries as distinct edges accidentally.

The result contains only layers and unresolved. Layers is a list of ID lists, with no empty layer inserted at the end. Empty input returns two empty lists. A node with no dependencies is ready in the first round. A dependent of that node becomes ready in the next round, even if its input record appeared first. These are consequences of simultaneous rounds rather than special exceptions.

We discussed returning task objects in place of IDs, emitting a flat topological order, or attaching error explanations. Those ideas were deferred because the caller specifically wants dependency layers and the unprocessed IDs. Config and payload must remain unchanged. No sorting in place, removal from the original batch, or editing of deps is appropriate.

The final handoff is one plain solve(payload, config) function using permitted Python builtins. Read the advertised configuration once and consume its contents, then make exactly one first write of the complete implementation. The unchanged task contract outside this planning discussion remains authoritative. No imports, executor, command-line entry point, extra tools, local tests, retries, or second write belong in this attempt; finish with a short final reply."""

DEPENDENCY_SUMMARY = """Settled design: start each call with no completed IDs. In each round collect ALL remaining tasks whose every exact dependency was completed in an EARLIER round, emit the whole layer, then mark it complete. Never unlock within a layer. Apply actual config order=input (original payload order) or lex (Python ascending strings) to each layer and unresolved. IDs are unique exact strings, including empty/Unicode, without normalization. Duplicate dependencies add no extra work. Absent prerequisites never complete; cycles, self-dependencies and downstream blocked tasks remain unresolved when no task is ready. Independent tasks still produce layers. Return only {layers,unresolved}, without empty terminal layers or error classifications; empty input gives both lists empty. Leave payload/config unchanged. Executors, logs, flat ordering, task objects and external-completion assumptions were rejected or deferred. Preserve the unchanged contract and actual read; write one complete implementation and finish briefly."""


def main():
    rows = {
        'interval_groups': {'edit_kind': 'faithful_compaction', 'discussion': INTERVAL_DISCUSSION,
                            'summary': INTERVAL_SUMMARY, 'before': INTERVAL_DISCUSSION, 'after': INTERVAL_SUMMARY},
        'dependency_layers': {'edit_kind': 'faithful_expansion', 'discussion': DEPENDENCY_DISCUSSION,
                              'summary': DEPENDENCY_SUMMARY, 'before': DEPENDENCY_SUMMARY, 'after': DEPENDENCY_DISCUSSION}}
    for row in rows.values():
        row['source_kind'] = 'authored varied design review; equivalent facts in concise and expanded wording'
        row['contract_position'] = 'Task prompt/system and actual config tool result are outside the edited discussion'
        row['before_bytes'] = len(row['before'].encode())
        row['after_bytes'] = len(row['after'].encode())
    (HERE / 'contexts.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: {'edit_kind': row['edit_kind'], 'before_bytes': row['before_bytes'], 'after_bytes': row['after_bytes']}
                      for key, row in rows.items()}))


if __name__ == '__main__':
    main()
