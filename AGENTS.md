# Project guidance

Start every response with an expressive kaomoji that fits what you are saying.

Favor the simplest implementation that lets us learn whether a change is useful.
Write for the experiment we need now, not hypothetical future callers or mistakes.

Read [docs/project.md](docs/project.md), then the [lab journal](docs/lab/README.md)
when resuming rewrite or serving work. The project page owns current goals and
status; the lab journal records questions, findings and decisions with evidence
links. Historical result reports and proposals do not authorize new work.

For technical documentation, follow [docs/writing-guide.md](docs/writing-guide.md)
and [docs/lab/methodology.md](docs/lab/methodology.md). Update the owning manual
when implementation or understanding changes; keep experimental reasoning in
the lab journal rather than duplicating it across status documents.

- Use a short loop: fixed baseline, one change, measure, inspect, decide.
  Record the revision, command, settings and results; reuse existing tools.
- Performance comparisons require at least five measured samples per side.
  Alternate run order, retain all samples and report variation. One-off runs
  are for smoke checks or diagnostics, not comparative performance claims.
- Work sequentially from `speed-bench/ideas/next-experiments.md`, with a
  one-hour budget per item. Measure ordinary and favorable cases as well as
  stress cases; plot tradeoffs for tunable features.
- Select one outcome before starting work. Side ideas and completed experiments
  do not authorize new experiments. Parallel preparation must serve that outcome.
  Write one result report; update the recipe conclusion and one short lab journal
  entry instead of copying detailed results across status documents.
- Output degradation is acceptable. Measure speed and practical task quality;
  use delta-NLL and KL to describe changes, not as automatic rejection criteria.
- Use Nushell for maintained orchestration and shell test drivers, with plain
  `#!/usr/bin/env nu` shebangs and normal `nu script.nu` invocations. Keep Python
  for numerical/conversion work and focused binary I/O; retain historical
  experiment snapshots in their recorded languages.
- Prefer standard parsers and simple formats. Check essential assumptions and
  make real failures obvious; avoid custom validation frameworks, redundant
  provenance layers and exhaustive malformed-input tests.
- Add tests and abstractions when they protect a concrete invariant or resolve
  a demonstrated problem. Do not require perfect reproduction before trying an
  approximation. Keep experimental runtime behavior opt-in.
- Run one heavy model job at a time and retain the existing pressure guard.
  Large RAM usage is expected; judge pressure and swap behavior, not an arbitrary
  RAM cap. Pause unrelated work during final latency measurements.
- Keep useful temporary workspaces and results under `.workspaces/` and
  `speed-bench/`, rather than `/tmp`. Keep experiments in small jj commits.
- Keep raw benchmark output, logs and generated snapshots local and ignored.
  Commit concise reports, reproduction scripts and necessary fixture inputs.
