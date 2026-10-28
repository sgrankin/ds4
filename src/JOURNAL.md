# Serving project journal

Read [project direction and current status](../docs/PROJECT.md) first.
This journal records short milestones, not the active experiment queue.

## 2026-10-28: source-backed documentation pilot

The user rejected archiving blobs as a substitute for editorial synthesis.
Authorized clean-context GPT 6.1 Terra authors, separate style and source reviews,
then a cross-document editorial pass. The pilot covers architecture, methodology,
DeepSeek on Apple Silicon and a narrative checkpoint lab entry under `docs/`.
Existing documents and experiment folders remain in place for comparison;
the broader migration has not been performed. Runtime experiments remain paused.

Reader feedback found fact-filled paragraphs without a coherent explanation.
The documentation guide now specifies an independent explanatory-flow reviewer
and explicit medium-effort Terra passes. The Apple Silicon page still needs
the structural rewrite identified in that review.

## 2026-10-28: pause and reassess

The user lost track of the goal amid side experiments and repeated status
documents. All implementation and performance experiments are paused.
The latest authorized DSML/DSP1/main-FP4/fusion/checkpoint batch is complete;
further convergence work needs a fresh choice.

Consolidated current goals and capability boundaries in `docs/PROJECT.md`.
Preserved the previous journal and queue intact in dated sibling files so
their relative evidence links still work. Replaced the active queue with a
short paused status. Corrected stale recipe/inventory summaries.

Process decision: one selected outcome, one bounded experiment at a time, one
result report and one short milestone entry. Side ideas do not authorize side
experiments. Retain the five-samples-per-side rule and pressure guard. Do not
duplicate full result tables across the journal, queue and design docs.

## Completed milestones

- **Checkpoint follow-up:** anchor selection and capture during ordinary prefill
  remove avoidable streamed-weight sweeps. Selected for server `persistent-fast`.
  [Recipe](../recipes/16-semantic-context-checkpoints.md).
- **No-edit overhead:** five alternating pairs find no clear penalty on fresh
  requests; 20/20 answers pass. This is not a long append-only conversation test.
  [Report](../speed-bench/experiments/context-capture-20261028/NO_EDITS.md).
- **DeepInfer convergence batch:** typed tool truncation, physical main/index KV
  and useful drafter-history persistence are complete. HC fusion is correct but
  timing-neutral and remains experimental. [Inventory](../docs/DEEPINFER_ALIGNMENT.md).
- **Serving foundation:** native model engines and stateful Responses sessions
  are implemented. Generic hosted orchestration was removed; native decisions
  and client tools remain. Two-generator composition is still proposed.
  [Source map](README.md), [Responses](serving/RESPONSES.md).

## Earlier history

[Journal through 2026-10-28](JOURNAL-history-20261028.md) preserves the original
chronology, decisions, qualifications and source references. Its running/next
instructions describe their historical moment; they are not current work orders.
Use the project page for current status and consult that archive for evidence.
