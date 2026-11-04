# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: what is the next exposed decode dependency?**
One-hour slice: November 4, 13:24–14:24 Eastern. The local baseline is the
[qualified prefill package](../experiments/2026-11-04-combined-prefill/README.md):
runtime `b9c9e77b`, with four-row staging and unit-rescale enabled by its recorded
candidate settings. Its tested binaries are now installed locally; published
main remains `364ceeab`.

Return to decode. Inspect the existing 32K timeline for this runtime and collect
a current profile if the evidence is insufficient. Trace the largest exposed
cost or dependency to source, then select one concrete change. Use the existing
fixture and benchmark infrastructure, five alternating samples per side, and
quality checks suited to any arithmetic change. Keep all timing samples and
interpret variation before accepting or rejecting the change. If no credible
change fits the hour, preserve the narrowed dependency finding and choose a
follow-up slice rather than building a new harness.

Raw evidence belongs under `.workspaces/hillclimb-20261104/08-decode-dependency/`.
Only preparation that serves this selected decode outcome may run in parallel.
Write one result report and a short
[session journal](../../docs/lab/journal/2026-11-04-through-06-performance.md)
entry. Do not publish; continue toward the November 6 deadline after deciding
this bounded result.

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
