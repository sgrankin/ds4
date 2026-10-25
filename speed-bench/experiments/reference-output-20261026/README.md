# Bound inline program output, retain full values by reference

Historical experiment: server-hosted tools and `tests/mcp-tools-wire.py` were
later retired. Reproduce using the original revisions recorded in this report,
which remain reachable from `archive/pre-stack-cleanup-20261028`.

One-hour experiment: **02:20–03:20 UTC, 2026-10-26**, including implementation,
qualification and measurement. Baseline is completed planning candidate C,
`aef775c2`. Root owns builds and the single guarded model job.

The [A/B/C screen](../planning-abc-20261026/RESULTS.md) qualified explicit
`values.stash/load`, but its natural plans did not use them. Programs continued
to expose bulk results or fetch them again. This experiment changes the runtime
output boundary, without changing the task prompt or teaching a task-specific
plan.

The optional PTC binding `ds4.max_inline_bytes` is disabled at `0` or when omitted.
Enabled budgets from 256 through 65,536 bytes make `text(value)` retain an
oversized JSON value in the immutable ledger and emit bounded reference metadata.
The main model can load that value in a later program. Complete receipt values
remain checkable outside logical model chat. This is a proposed speed/quality
tradeoff, not an assumption that fewer visible bytes improve whole-task time.

## Fixed comparison

The wrapper reuses the [planning fixtures](../planning-abc-20261026/fixtures.nu)
and [existing wire leaf](../../../tests/mcp-tools-wire.py). Ordinary, favorable
and unfamiliar cases retain their exact prompts, notes, schemas and external
oracles. The only added request field is `ds4.max_inline_bytes` on the PTC tool.
Direct/main-only arms are untouched. Native DeepSeek reasoning is unchanged.

Run one repeat of each of the three cases at budgets **0, 512 and 2,048**:

```nu
nu speed-bench/experiments/reference-output-20261026/run.nu --inline-bytes 0
nu speed-bench/experiments/reference-output-20261026/run.nu --inline-bytes 512
nu speed-bench/experiments/reference-output-20261026/run.nu --inline-bytes 2048
```

The script launches no model process. It defaults to an existing server at
`http://127.0.0.1:19333`; override `--url`, `--model`, `--helper` or `--python`
when needed. Each budget writes separate ignored artifacts under
`.workspaces/reference-output-20261026/inline-BUDGET/`. Use `--output` for a
separate repeat, or `--case` to repeat one promising or uncertain point.

Nine requests are the initial screen. Repeat a promising point if the hour
permits; do not tune fixtures until it wins. Record revision/build/settings and
actual outcomes in [RESULTS.md](RESULTS.md).

Plot the individual requests once those files exist:

```nu
let work = '.workspaces/reference-output-20261026'
let samples = [ordinary favorable unfamiliar] | each {|case|
    [0 512 2048] | each {|budget|
        ['--sample' ($budget | into string) $'($work)/screen/($case)-($budget)/scenarios.json']
    } | flatten
} | flatten
.workspaces/bench-plots-venv/bin/python speed-bench/experiments/reference-output-20261026/plot.py ...$samples --output ($work | path join initial-tradeoffs)
```

The measured screen uses one file per case/budget under `screen/`; the ordinary
wrapper's default instead puts all three cases in each `inline-BUDGET/` file.
Pass either layout as repeated `--sample BUDGET PATH` arguments. The source
[plot.py](plot.py) writes SVG and PNG under the ignored workspace.
Points are labeled with their budget; failed numeric/JSON outcomes use a red X.
This n=1 screen shows observed points, not a statistical curve or a claim that
one budget is optimal. Zero-byte ordinary points may overlap.

The final control/resource comparison uses the initial disabled controls and the
hinted 2048-byte, 12-call follow-up (two requests per larger case):

```nu
.workspaces/bench-plots-venv/bin/python speed-bench/experiments/reference-output-20261026/plot.py --sample 0 .workspaces/reference-output-20261026/screen/ordinary-0/scenarios.json --sample 0 .workspaces/reference-output-20261026/screen/favorable-0/scenarios.json --sample 0 .workspaces/reference-output-20261026/screen/unfamiliar-0/scenarios.json --sample 2048 .workspaces/reference-output-20261026/resource-screen/favorable/scenarios.json --sample 2048 .workspaces/reference-output-20261026/resource-screen/unfamiliar/scenarios.json --caption 'Control: off / 8 calls, n=1; hinted 2048 / 12 calls, n=2. Observed requests, no fitted curve' --output .workspaces/reference-output-20261026/final-tradeoffs
```

This comparison changes the generic runtime hint and call budget as well as the
inline limit; it does not isolate the inline limit's effect. Red X points remain
failed outcomes even when they emit fewer bytes or finish sooner.

Compare whole-response time, generated/reasoning/input tokens, program/fetch/helper
counts, emitted bytes, numeric/JSON grades and source-grounded handoffs. Inspect
reference creation/loading and actual code: a faster plan that skips work is
different from avoiding a refetch through a retained value. A small inline budget
may hide useful evidence, add rounds or exceed the shared call/output budget.
Keep the observed latency/quality/evidence points instead of selecting a budget
from byte counts alone.
