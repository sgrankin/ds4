# Technical documentation writing guide

This is the preferred style for the documentation rewrite. Improve the guide
when a real review exposes a problem; it is not a document schema.

## Reader and purpose

Write for an engineer who understands inference but has not followed our chat,
experiments or commit history. Explain the system well enough to modify it or
reproduce a technique in another harness. Give the reader a clear answer before
implementation detail. Assume neither familiarity with our abbreviations nor a
need for introductory machine-learning lessons.

## Document responsibilities

- The architecture overview explains component boundaries, ownership and data
  flow. Its block diagrams must describe the current implementation.
- A model reference manual explains what source material establishes about the
  model and its upstream inference. Distinguish documented facts, deductions
  and things we have not verified.
- A model implementation document explains how our runtime realizes that model.
  A separate model backend document explains how Apple Silicon executes its
  operations, including data layout, memory movement, Metal kernels and
  performance optimizations. Link to shared backend mechanisms rather than
  repeating them for every model.
- Shared serving, storage and backend documents own cross-model mechanisms.
  Model pages explain their adaptations and link to the shared explanation.
- DeepSeek's DeepInfer comparison belongs under that model. Use an explanatory
  comparison index with linked breakout documents for mechanisms that need more
  detail; connect those accounts to our implementation and backend documents.
- A date-prefixed lab entry tells the story of a question: the context and
  hypothesis, setup, observations, decision and what remains uncertain. Link
  to detailed results rather than dumping a chronology of commands. Technical
  manuals hold current understanding; the journal retains the reasoning and
  historical evidence. Integrate useful older knowledge into its proper home;
  renaming a blob as history is not an editorial synthesis.
- The lab methodology sits above the journal, in `lab/methodology.md`, and
  defines the currently preferred entry format and measurement conventions.
  Individual journal entries live under `lab/journal/`.

## Explanation and structure

Use descriptive lowercase filenames and meaningful headings. Follow the
reader's questions, rather than the order in which work happened. Use normal
sentences and paragraphs with spaces between words, numbers and units. Explain
an unfamiliar term at its first use. Avoid terse status-code prose.

For a mechanism, explain the problem, how data and control move, who owns the
state, why the design helps, and its limitations. A flag name or source symbol
alone is not an explanation. Include a concrete example when it makes a cache,
protocol or numerical operation easier to understand.

Use tables for real comparisons and mappings. Each cell must carry enough
information to understand the difference. A table may index detailed documents;
it must not substitute labels such as "partial" or "supported" for explanation.
Use Mermaid for small block and data-flow diagrams. Name the data on important
arrows, distinguish control from data where useful, and explain ownership and
the relevant transitions in the surrounding prose. Diagrams should clarify an
explanation, not replace it.

Keep documents focused. Link to the owner of an explanation instead of copying
it into several documents. Avoid arbitrary word counts: split a topic when it
contains distinct reader tasks, not merely because it is long.

## Evidence and capability claims

Read the relevant source, existing documentation and recorded experiment
results. Treat old prose as evidence to reconcile, not an authority to copy.
When claims disagree, trace the implementation or record the uncertainty.

Distinguish implemented, exercised, measured, enabled by a preset and proposed.
Describe activation conditions and unsupported combinations where they matter.
Separate generic engine defaults from launcher presets and explicit overrides.
Distinguish storage size, allocated memory, logical reads, physical SSD traffic,
kernel latency, request latency and whole-task latency.

Performance claims need a workload, baseline, scope and evidence link. Preserve
the measured revision and settings for historical results; explain what they
establish about the current implementation. The [lab methodology](lab/methodology.md)
owns measurement procedures, interpretation and the preferred entry format.

Link directly to useful source files or functions at a relevant explanation.
Use repository-relative paths for local files, and pinned revisions for external
source claims where available. Explain what the cited code establishes; avoid
large unannotated inventories of symbols and links.

## Maintenance and review

Update the owning technical document when code or understanding changes. Mark
historical claims with their date and measured revision; do not present an old
status or planned action as current. Preserve evidence that still explains a
decision, and give readers a route from it to the current conclusion.
Keep one home for detailed results. A later lab entry may supersede a historical
conclusion; add a short cross-link rather than rewriting the original evidence.

Review prose and organization independently from technical correctness, checking
that explanations stand on their own and examples, diagrams and links work. Then
read related documents together for conflicting terminology, duplicated
explanations, missing handoffs
and poor navigation. Correct concrete problems; do not add new abstractions,
tests or research just to make the documentation look comprehensive.
