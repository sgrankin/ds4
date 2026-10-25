# Choose the program's small output first

Historical experiment: server-hosted tools and `tests/mcp-tools-wire.py` were
later retired. Reproduce using the original revisions recorded in this report,
which remain reachable from `archive/pre-stack-cleanup-20261028`.

One-hour experiment: **03:33–04:33 UTC, 2026-10-26**. This changes only a generic
recipe appended to the PTC binding description. It does not add a runtime cap,
force a particular program, alter the user's task or require a decision helper.

Use the frozen server from stage C of the preceding experiment at
`.workspaces/planning-abc-20261026/c-build/bin/ds4-server`, revision `aef775c2`.
The later automatic-spill candidate is not this experiment's runtime. Root owns
the single guarded model job; the scripts launch no model or build.

[recipe.txt](recipe.txt) asks the generator to choose its compact handoff before
writing code, preserve selected supporting evidence, reduce before emission, and
retain a reference if a separate inspection round is useful. Helpers/references
remain optional; already-visible small tasks may be answered directly.

The [planning fixtures](../planning-abc-20261026/fixtures.nu) supply unchanged
ordinary, favorable and unfamiliar tasks, source notes, schemas and oracle.
The [existing wire leaf](../../../tests/mcp-tools-wire.py) accepts optional
`--program-recipe PATH`, reads it once and appends it only to the existing
case/default PTC `ds4.description`. Without that argument the request body is
unchanged. `program_recipe` in the result file records the exact appended suffix,
including its two separating newlines, or `null` for control.

```nu
# One optional recipe request against the already-running frozen server:
nu speed-bench/experiments/output-first-20261026/run.nu --recipe --case favorable
# Full screen: each case gets control, recipe, recipe, control.
nu speed-bench/experiments/output-first-20261026/sweep.nu
```

The twelve separate CLI requests write unique directories under
`.workspaces/output-first-20261026/screen/CASE-INDEX-VARIANT/`. Both wrappers
support URL/model/helper/Python/output overrides. They omit `--inline-bytes`
entirely and retain eight tool calls, 4,096 output tokens, temperature zero and
native DeepSeek reasoning. No task/input/schema changes accompany the recipe.

Record wall/input/generated/reasoning/cached tokens, rounds/fetches/helper calls,
emitted bytes, stash/load use, numeric/JSON grades and source-grounded handoffs.
Inspect whether reduced output preserves enough evidence to answer correctly;
fewer bytes alone are not success. Compare each case's two control/two recipe
requests and retain failed plans. See [RESULTS.md](RESULTS.md) for measured
outcomes and limits. Raw logs and responses remain local and ignored.

## API-only follow-up

The separate [api-recipe.txt](api-recipe.txt) replaces the longer workflow suffix
with three sentences about typed choices, the question constructor and the
helper's result shape. It does not append the original recipe. Runtime, tasks,
inputs, schemas, eight tool calls and native reasoning remain unchanged.

```nu
nu speed-bench/experiments/output-first-20261026/api-followup.nu
```

This runs two requests for each of the three cases, writing separate case
directories under `.workspaces/output-first-20261026/api-screen/`. It reuses the
same fixtures and wire leaf, passes only the API recipe, and omits the inline
output cap. Original recipe/run/sweep files remain unchanged.
