# Project guidance

Start every response with an expressive kaomoji that fits what you are saying.

Favor the simplest implementation that lets us learn whether a change is useful.
Write for the experiment we need now, not hypothetical future callers or mistakes.

Read [src/JOURNAL.md](src/JOURNAL.md) when resuming rewrite or serving work.
Update it at meaningful checkpoints with decisions, progress and next steps;
distinguish completed work from proposals and retain relevant evidence links.

- Use a short loop: fixed baseline, one change, measure, inspect, decide.
  Record the revision, command, settings and results; reuse existing tools.
- Work sequentially from `speed-bench/ideas/next-experiments.md`, with a
  one-hour budget per item. Measure ordinary and favorable cases as well as
  stress cases; plot tradeoffs for tunable features.
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
