# Replacing running notes with maintained documentation

## Context

After the checkpoint experiments, the user could no longer follow the project
through repeated status blocks, side experiments and a growing benchmark tree.
Renaming the blobs as dated history did not solve that problem. The needed
change was editorial: current technical manuals and a greppable narrative of
why experiments succeeded or failed.

## Approach

This work used source and recorded evidence, without new model or performance
campaigns. The pilot covered architecture, methodology, DeepSeek on Apple
Silicon and semantic checkpoints. Later batches covered native model families,
shared serving and storage, development and retained backends.

Clean-context Terra authors used explicit medium reasoning effort. Independent
reviewers checked source correctness and coverage, prose, explanatory flow and
sentence-level reader effort, followed by an editorial read of related areas.
The [writing guide](../../writing-guide.md) records these roles.

## Findings

Correct sentences did not necessarily form an explanation. Early paragraphs
combined unrelated details and named optimizations before explaining their
purpose, inputs or outputs. The review process gained a distinct flow role;
reader-effort review separately checks sentence parsing without mechanically
expanding clear technical terms.

Coverage needed checking too. Reader follow-up exposed missing DeepSeek vision
coverage and hard-to-find resident-expert substitution. The 49-item DeepInfer
inventory became six explanatory chapters. Retirement reviews recovered usage
details, negative experiments and design references that summaries omitted.
Source review also corrected stale prefill defaults, checkpoint durability
wording and tensor-parallel speculation claims.

## Decision

The [documentation index](../../README.md) follows the architecture. Manuals own
current mechanisms; lab entries own reasoning and qualified historical findings;
original reports own detailed samples. Design ideas remain unselected.
Superseded manuals and running-note blobs are removed once their useful
material is accounted for and links are updated.

Runtime experiments remain paused. The next experiment needs one selected
outcome and the bounded baseline/change/measure/inspect/decide loop.
