# Writing, organizing and reviewing documentation

This guide describes how we write, organize and review technical documentation.
Improve it when a real review exposes a problem; it is not a document schema.

## Reader and purpose

Write for an engineer who understands inference but has not followed our chat,
experiments or commit history. Explain the system well enough to modify it or
reproduce a technique in another harness. Give the reader a clear answer before
implementation detail. Assume neither familiarity with our abbreviations nor a
need for introductory machine-learning lessons.

## Document responsibilities

Organize current manuals around the architecture and the reader's tasks, not
the chronology of experiments. Lab narratives preserve the context and history
behind decisions, while manuals explain current understanding.

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
sentences and connected paragraphs, each with one explanatory purpose. Use
bullets for genuinely parallel facts, not as a substitute for explaining how
ideas connect. Put spaces between words, numbers and units. Explain an
unfamiliar term at its first use. Avoid terse status-code prose.

Show executable project launchers as direct relative commands, for example
`./tools/zig-build`. Their shebang selects Nushell; use `nu script.nu` when an
explicit interpreter is needed.

For a mechanism, explain the problem, how data and control move, who owns the
state, why the design helps, and its limitations. A flag name or source symbol
alone is not an explanation. Include a concrete example when it makes a cache,
protocol or numerical operation easier to understand.

One useful order is problem, mechanism, representation, optimization and
tradeoff. Adapt it to the reader's question rather than treating it as a required
schema. Establish a baseline before explaining an optimized variant. Before
giving lane counts or dispatch details for something named a "packed index
scorer," for example, explain its purpose, inputs and outputs, and where it sits
in the dataflow.

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

## Reader effort within sentences

Make relationships explicit when ambiguity or unnecessary inference makes a
sentence hard to follow.
Unpack an opaque stack of nouns or an unfamiliar hyphenated compound with a
verb or preposition. Restore an optional “that” or preposition when it helps
the reader parse the sentence. Keep a subject near its verb and other related
parts near each other when intervening detail makes the connection hard to
follow. Replace a newly coined noun used as a verb, personification or an
unfamiliar pairing of words when the actor or action is unclear. A few extra
words can save inference or rereading; add them where they help, not everywhere.

Keep established terms such as “expert cache,” “prefill” and “frontier,” and
retain exact API and source-symbol names. Noun compounds, omitted “that” and
metaphors are useful when their meaning is clear; none is a blanket error.
These constructed examples illustrate wording choices, not paper quotations or
claims about backend behavior:

- If a budget limits retries of reads after an expert cache miss: “expert cache
  miss retry budget” → “budget for retrying reads after a miss in the expert
  cache.”
- “The runner, after loading the configuration and checking the inputs, starts
  prefill” → “After loading the configuration and checking the inputs, the
  runner starts prefill.”
- Keep “The expert cache eviction policy selects an entry” and “The runner
  snapshots the configuration before prefill” when those are the intended
  meanings: the compound and verb are familiar and clear in context.

These checks are motivated by [*Why is AI-generated language so hard to
understand?*](https://arxiv.org/abs/2610.29162v1) (Clark, Martin and Mahowald,
2026, version 1; sections 3–4 and 7). The study compares English model outputs
with human writing and round-trip translations using features associated with
processing difficulty. Round-trip
translation normalizes style and can change meaning or length, so it is not
a perfect semantic control. The authors do not directly measure human
comprehension, and results vary substantially by model and genre. Their
controlled Qwen2.5-1.5B reinforcement-learning experiment
supports a possible training mechanism: a length penalty can shift grammatical
choices. It does not establish that asking for brevity has the same effect or
that this mechanism causes all frontier-model style.

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

For a substantial document, use fresh contexts and distinct roles:

- A clean-context author, scoped to that document, reads the relevant source
  and documentation and writes the explanation.
- An independent prose and style reviewer checks clarity, readability, examples,
  diagrams and links. Select a dense paragraph and mark noun relationships the
  reader must guess, clauses that require reparsing, long dependencies and odd
  verbs that obscure the action. For each finding, explain the reader effort
  and propose a clearer rewrite that preserves facts, technical terms, actors,
  conditions and sequence. Leave familiar, clear wording alone; adding words
  is useful only when it clarifies. Do not guess AI authorship or assign a
  style score.
- An independent technical and source reviewer checks claims against the
  implementation and evidence. Also compare the document's scope with the
  model's public capabilities and implementation entry points: an omitted
  input path such as vision is a coverage defect even when the text present
  is correct. Give each major path an explanation or a clear link to its owner.
  When replacing an inventory, account for every entry and the supporting
  caveats, evidence and qualifications before retiring the old document.
- An independent explanatory-flow reviewer traces a representative data item
  through the prose and diagrams, first reading as the intended engineer without
  filling gaps from source knowledge. Flag missing explanations even when the
  implementation would let an expert infer them. Check that conceptual prerequisites and the
  baseline precede their use and optimized variants, that sentences and
  paragraphs build on each other, and that lists and prose serve their subjects.
  This is distinct from a style check; the author's self-review does not count
  as independent review.
- After corrections, a clean-context editorial reader reads the related area
  together and proposes improvements to duplication, organization, terminology,
  handoffs and navigation.

For this documentation work, use GPT 6.1 Terra with reasoning effort explicitly
set to medium for these fresh contexts. Scale review to the change: a typo or
small correction does not need multiple review rounds. Correct concrete problems;
do not add process machinery, abstractions, tests or research just to make the
documentation look comprehensive.
