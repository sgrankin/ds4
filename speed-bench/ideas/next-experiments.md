# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can the large batched F16 projections use a more efficient matrix path?**
One-hour slice: November 4, 21:43–22:43 Eastern. Fix runtime `d904029c`, the
qualified RB4/unit-rescale settings, the opt-in prefill BF16/RoPE fusion,
8,192-token chunks and a 64 GB expert cache. Keep GPU resident routing, SIMD,
router fusion, address reuse, parallel FFN and refresh sixteen fixed.

The previous fresh 32K trace recorded 4.442 seconds of summed F16 matrix-vector
encoder spans, with overlapping calls; that is not an additive latency saving.
Large 8,192-row calls occur beside compressor pooling. Trace their exact caller,
shape and numerical contract, then select one existing matrix kernel or narrow
batch specialization. Establish why the current dispatch chose matrix-vector
execution before changing it. Do not introduce a new model format or loader.

By twenty minutes into the slice, select the concrete implementation or record
why the suspected opportunity is invalid. Check activation and fixed-history
quality, then take at least five alternating profile-free pairs at 32K and 8K.
Retain full phases, first-step latency, steady rates, every sample and variation.
If timing is neutral, inspect the intended operation and dependency before
choosing what to retain. Numerical differences are measured tradeoffs, not an
automatic rejection. A later package comparison must still measure against
published `364ceeab`; installed binaries and presets stay unchanged here.

Keep one guarded model job and pause builds during final timing. Write one
report under `2026-11-04-prefill-f16-projections/`, with raw artifacts under
`.workspaces/hillclimb-20261104/18-prefill-f16-projections/`, then update the
owning manual or recipe conclusion and one short journal entry. Publication
remains separate.

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
