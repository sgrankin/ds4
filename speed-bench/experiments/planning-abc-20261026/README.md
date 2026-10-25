# Planning A → B → C: one bounded change at a time

Historical experiment: server-hosted tools and `tests/mcp-tools-wire.py` were
later retired. Reproduce using the original revisions recorded in this report,
which remain reachable from `archive/pre-stack-cleanup-20261028`.

The previous [MCP screen](../mcp-tools-20261025/RESULTS.md) works functionally but
its natural code plan spills source data, refetches and returns probabilities.
This sequence retains free JavaScript and tests:

- A: truthful coherent contracts, compact-result guidance and imported MCP output
  schemas. It is not uniformly shorter; ordinary input grew five tokens.
- B: one pure-JS targeted-question constructor, `lib.questionsFor`, retaining
  ordinary JavaScript for layout adaptation and other workflows.
- C: explicit `values.stash`/`values.load` for immutable JSON, scoped ownership
  and durable restoration across programs/Responses. Inspect loaded data using
  ordinary JS keys/counts/samples; there is no separate inspection DSL.

Each item has a one-hour budget including measurement. A/B/C are measured and
C's explicit persistence path is implemented and qualified on Flint/DeepSeek in
[RESULTS.md](RESULTS.md). The natural C screen does not adopt stash/load; its
faster unfamiliar plan skips redundant work instead of reusing references.
References are useful independently of whether a planner adopts them. Keep
native DeepSeek reasoning and the existing pressure guard. Root owns model jobs
and builds.

## Fixed tasks and the unfamiliar-layout case

[fixtures.nu](fixtures.nu) preserves the ordinary/favorable prompts, notes,
schema and oracle from the previous experiment. The unfamiliar case copies the
same eight favorable records and user task, exposing them through `fetch_bundle`
instead of `fetch_queue`. Its result is a truthful nested export with an opaque
dataset key resolved from returned metadata. Ticket IDs, reported charges and
current notes move into nested fields; no labels or summary are added.

The advertised `outputSchema` describes field meanings and the payload dictionary
without revealing its runtime key. A model may inspect keys/metadata in the same
program or emit a bounded observation and continue. Success does not require an
extra model round; a one-program plan is not automatically counted as adaptable.
Review the actual code and evidence access. The original tool retains its old
input schema/result and has no new output schema.

```nu
# Start the chosen frozen baseline/candidate server separately, then:
nu speed-bench/experiments/planning-abc-20261026/run.nu --case all --arm ptc --repeats 1 --output .workspaces/planning-abc-20261026/a-baseline
# Reuse exactly the same fixture under candidate A.
nu speed-bench/experiments/planning-abc-20261026/run.nu --case all --arm ptc --repeats 1 --output .workspaces/planning-abc-20261026/a-candidate
```

The wrapper reuses the existing HTTP fixture/wire leaf; it starts no model.
Record wall/input/output/reasoning/cache, actual programs/fetches/helper calls,
emitted bytes, numeric/JSON grades and grounded handoff quality. Inspect native
labels and whether bulk source/probabilities enter main context. Keep raw records
ignored under `.workspaces/`; [RESULTS.md](RESULTS.md) records actual outcomes and
revision/settings once the sequential runs complete.
