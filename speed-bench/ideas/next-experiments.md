# Experiment queue

The user authorized a seven-hour decode optimization session, from
2026-11-03 23:55:26 to 2026-11-04 06:55:26 Eastern. Work proceeds through
one-hour profile, target, implementation and measurement loops. This authorization
includes choosing the next bottleneck after each result; it does not authorize
unrelated features or publishing main. Reserve the final slice for combined
qualification and a concise account of adopted changes and unsuccessful attempts.

**Current loop: profile host routing gaps and remove one preparation cost.**
Started 2026-11-04 03:32 Eastern, with a one-hour budget ending 04:32. Baseline
`a65c4fa6` includes the four adopted overnight changes. The latest GPU trace
still shows about 5.6–5.9 ms/token uncovered between router completion and
routed work. This interval is not all proven CPU execution time.

Take a short CPU sample during steady 32K teacher decode, outside comparative
timings. Separate event polling from command encoding, resource declarations
and cache preparation. Select one concrete, simple cost from that evidence,
implement it behind a flag, then use at least five alternating samples per side
and appropriate quality/lifetime checks. If no useful host target emerges,
record that outcome instead of expanding instrumentation indefinitely.
Raw evidence belongs under
`.workspaces/overnight-decode-20261104/cpu-host-profile/`.

The [session index](../experiments/2026-11-04-decode-pass/README.md) links four
adopted changes and the unadopted router experiment. Coefficient reuse is a
small low-risk adoption with neutral to slightly favorable timings, not an
established kernel-speed gain. The final combination still needs measurement
against the session's starting configuration.

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
