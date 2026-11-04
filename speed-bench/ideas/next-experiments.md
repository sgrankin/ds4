# Experiment queue

The user authorized a seven-hour decode optimization session, from
2026-11-03 23:55:26 to 2026-11-04 06:55:26 Eastern. Work proceeds through
one-hour profile, target, implementation and measurement loops. This authorization
includes choosing the next bottleneck after each result; it does not authorize
unrelated features or publishing main. Reserve the final slice for combined
qualification and a concise account of adopted changes and unsuccessful attempts.

**Current loop: use a cheaper clock for the selected-event spin deadline.**
Started 2026-11-04 04:50 Eastern. Close by 05:20 if the ordinary and large-context
comparisons answer the question; the one-hour ceiling is 05:50, but final combined
qualification takes priority. The baseline includes five adopted overnight
changes, most recently selected-KV gather reuse.

The earlier CPU sample observed many clock reads while waiting for GPU routing.
Reducing their frequency did not help. This separate candidate keeps readiness
and deadline checks on every poll and uses Darwin's sleep-inclusive raw monotonic
nanosecond API only for the spin interval. The original clock still owns the
60-second blocking timeout; timestamps from the two clocks are never mixed.
Test the exact opt-in flag with five alternating pairs at 8K and 32K, fixed-history
numerical checks, and a short separate CPU sample. Retain the original path if
there is no useful gain. Evidence stays local under
`.workspaces/overnight-decode-20261104/raw-clock/`.

After this experiment, freeze runtime changes. The final comparison uses the
original executable and shaders from `8971dfbd` against the adopted package,
including longer decode and the recorded coding session. Main is not published.

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
