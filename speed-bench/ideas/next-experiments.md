# Selected experiment

**Completed 2026-11-03: per-token RoPE reuse in SSD streaming decode.**

The [result report](../experiments/2026-11-03-stream-rope/README.md) records
five alternating pairs at both 8K and 32K plus focused and full-model numerical
checks. The resident fast presets adopt the feature; the 32K gain is in decode,
with whole-phase and short-context limits retained. No next experiment is selected.

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
