# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can attention skip rescaling by one?**
One-hour slice: November 4, 11:22–12:22 Eastern. Baseline `e050fa0b` includes
the adopted four-row staging kernel. Its fresh 32K profile still attributes
9.33 seconds of GPU work to indexed attention. The per-row online softmax
always computes two exponentials and rescales the output, even when the running
maximum does not change and the old-output scale is exactly one.

Try a bounded fast path for that case inside the four-row prefill kernel,
preserving row order and the original path as control. Keep decode and other
attention paths unchanged. Start with the six-shape numerical fixture and five
alternating operation pairs at small and large prefill sizes. If promising,
compare five full-model pairs at 8K and 32K and measure full-vocabulary quality.
Approximate results may be useful; report drift rather than rejecting it solely
for differing bits. Keep profile diagnostics separate from latency comparisons.

Raw evidence belongs under
`.workspaces/hillclimb-20261104/05-attention-unit-rescale/`. Reuse the existing
runner and fixture rather than creating another harness. Four-row staging is
enabled in the local resident fast source presets; whole-model timing did not
establish a total-time change. The root compatibility binaries and published
main remain at the starting package until a later combined qualification.

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
