# Selected experiment

**Completed 2026-11-03: test expert-cache allocation granularity.**

The [result report](../experiments/2026-11-03-expert-slabs/README.md) retains the
4 GiB default after five alternating pairs at 8K and 32K. Smaller slabs shorten
the first decode step, but do not establish an append or complete workload
improvement. Numerical and cache-capacity checks passed. The standalone
short-Q8 query kernel remains a separate, unstarted candidate for decode TPS;
this completed experiment does not authorize starting it.

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
