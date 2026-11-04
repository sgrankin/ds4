# Selected experiment

**Completed 2026-11-03: profile the adopted RoPE path and identify the next attempt.**

The [result report](../experiments/2026-11-03-post-rope-profile/README.md) localizes
the slow warm append to a delay before routed expert GPU execution. Recommended
next: compare 4,096 and 512 MiB expert-cache slabs at unchanged actual cache
capacity, with five alternating samples per side and separate diagnostic
profiles. This has not been run or adopted. A standalone short-Q8 query kernel
is a separate later candidate for steady decode TPS.

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
