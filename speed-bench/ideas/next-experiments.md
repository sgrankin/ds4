# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can four output rows reduce the exposed output-B projection?**
One-hour slice: November 4, 13:24–14:24 Eastern. The local baseline is the
[qualified prefill package](../experiments/2026-11-04-combined-prefill/README.md):
runtime `b9c9e77b`, with four-row staging and unit-rescale enabled by its recorded
candidate settings. Its tested binaries are now installed locally; published
main remains `364ceeab`.

The existing 32K profile identifies attention output-B plus HC expansion as
about 3.7 ms of summed GPU time per steady token, almost entirely unoverlapped
with other encoder intervals. Query projection is larger than its exposed part;
router sorting is almost fully overlapped. These diagnostics rank opportunities
without promising critical-path savings.

Test four output rows per threadgroup instead of two for the literal 8192-to-5120
output-B shape, keeping the same SIMD-group count and each row's K walk and
reduction. This halves threadgroups and reuses activation loads at the cost of
more accumulators and 512 instead of 256 bytes of reduction scratch. The numeric
opt-in is `DS4_METAL_V41_OUTPUT_B_ROWS4`; other shapes and multirow work retain
their existing paths. First use a bounded fixture, then five alternating samples
at 8K and 32K, full-logit checks and a separate activation profile. Retain all
samples and judge any small difference against variation. Before rejecting neutral total timing, compare the before/after profiles for
reduced output-B work, changed overlap or a newly exposed stall. Carry useful
groundwork into a separately measured follow-up if that evidence warrants it.
Do not change other kernel geometries during this experiment.

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
