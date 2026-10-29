# Root documentation and canonical agent guidance

## Context

The first documentation batches left evaluation provenance, a historical model
card and a long release checklist at the repository root. `AGENT.md` also
duplicated the standard `AGENTS.md` and retained conflicting upstream policies.
These remaining files needed owners in the organized documentation tree.

## Findings and decision

Evaluation provenance now belongs to the [development manual](../../development/evaluation-data.md),
including source revisions, case IDs and licenses. The [historical V4 reference](../../models/deepseek/legacy-v4-reference.md)
preserves the original Flash card and published scores without applying its
geometry to V4 Pro or V4.1 Flash. [Release qualification](../../development/qualification/README.md)
is split into shared, model and platform procedures. Reviews preserved useful
inputs and regression boundaries while distinguishing runnable checks, fixtures
that need restoration and historical platform plans.

`AGENTS.md` is the sole root instruction file. Useful ownership and commenting
guidance was retained; obsolete blanket bans on approximation or C++ and
unconditional backend gates were not carried forward. `CONTRIBUTING.md` remains
a short conventional entry point to the maintained guides.

The calibration-data builder consumed the retired document paths. Updating them
also exposed a missing `ds4.c` source key that produced unavailable placeholders
in long-context prompts. The key is restored. A model-free smoke generated 36
records in memory with populated selected sources; no corpus files were
written, and existing generated calibration text remains unchanged.

This was documentation and document-input maintenance, not a model or hardware
qualification. Runtime experiments remain paused.
