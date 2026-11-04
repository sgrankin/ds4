# Experiment queue

The user authorized a seven-hour decode optimization session, from
2026-11-03 23:55:26 to 2026-11-04 06:55:26 Eastern. Work proceeds through
one-hour profile, target, implementation and measurement loops. This authorization
includes choosing the next bottleneck after each result; it does not authorize
unrelated features or publishing main. Reserve the final slice for combined
qualification and a concise account of adopted changes and unsuccessful attempts.

**Current loop: reuse selected compressed KV within a scalar layer group.**
Started 2026-11-04 04:20 Eastern, with a one-hour ceiling at 05:20. Baseline
`81632aee` retains the four adopted changes and closes the clock-poll experiment.
The current trace shows 30 compressed gathers whose selection and physical
owner repeat within one token. Most gather duration overlaps query work, so
removing those dispatches is not a prediction of an equivalent latency saving.

Test a small opt-in guard that keeps the publisher gather and reuses its
separate scalar scratch for consumers. Raw-window staging still runs per layer.
Verify the lifetime proof independently, compare fixed-history logits at aligned
and unaligned contexts, and measure five alternating pairs each at 8K and 32K.
Use a separate profile to establish which dispatches disappeared. Adopt or
remove the runtime guard based on the measured tradeoff and implementation cost.
Raw evidence belongs under
`.workspaces/overnight-decode-20261104/selected-kv-reuse/`.

The [session index](../experiments/2026-11-04-decode-pass/README.md) links four
adoptions and two unadopted candidates. The original executable and shaders
have been built in an isolated workspace for final combined qualification.

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
