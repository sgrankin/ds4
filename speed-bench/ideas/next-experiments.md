# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can modest row blocking help without losing head reuse?**
One-hour slice: November 4, 10:45–11:45 Eastern. Keep the current sixteen heads
per group and stage four KV rows at a time, preserving arithmetic and row order.
This differs from the rejected eight-head/sixteen-row kernel: it retains head
reuse, uses only 4 KiB of threadgroup staging and avoids a private compacted-ID
array in each lane. Start with the focused numerical fixture and five alternating
operation samples. If promising, compare five alternating full-model timing
pairs at 8K and 32K, including decode and full-vocabulary checks.

The cooperative FP4 reader was removed after mixed/neutral whole-model and
operation timings. Its useful packed-versus-expanded correctness fixture remains.
The starting runtime again has the published execution for this workload, with
the readiness cleanup confined to batch overlap's existing opt-in path. Raw
evidence for this question belongs under
`.workspaces/hillclimb-20261104/04-attention-four-row/`; reuse the existing runner
and keep intrusive profiling separate from timing.

Record one result report and a short
[session journal](../../docs/lab/journal/2026-11-04-through-06-performance.md)
entry, update the owning manual if adopted, then select the next target from
measured dependencies. Periodically compare the accumulated package against
its starting runtime; reserve the final session slice for combined qualification.

[Project direction and current status](../../docs/project.md) records the goals,
implemented features and a few possible next milestones. Choose an outcome
with the user before scheduling another experiment. Ideas are not work orders.

When work resumes, record only the selected question, fixed baseline, one-hour
budget and decision criterion here. Keep results in one linked report and move
the completed outcome into a short journal entry rather than growing this queue.
Performance comparisons still require at least five alternating samples per
side and the existing pressure guard.

The [lab journal](../../docs/lab/README.md) records prior experiments and their
outcomes. [Design ideas](../../docs/design/README.md) preserve unselected options;
neither is an active queue.
