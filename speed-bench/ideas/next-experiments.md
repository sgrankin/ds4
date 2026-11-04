# Experiment queue

The user authorized a seven-hour decode optimization session, from
2026-11-03 23:55:26 to 2026-11-04 06:55:26 Eastern. Work proceeds through
one-hour profile, target, implementation and measurement loops. This authorization
includes choosing the next bottleneck after each result; it does not authorize
unrelated features or publishing main. Reserve the final slice for combined
qualification and a concise account of adopted changes and unsuccessful attempts.

**Current work: qualify the frozen five-change decode package.**
Started 2026-11-04 05:14 Eastern. Finish within the authorized session ending
06:55:26. No further runtime experiments start during this qualification.

Compare the original `8971dfbd` executable and shaders, built in an isolated
workspace, with the five adopted changes: direct F16 gather, every-layer
submission, early shared down, query RoPE-table reuse and selected-KV reuse.
Use five alternating pairs at 32K with 512-token decode windows, five at 8K,
and five recorded coding-session replays. Separate fixed-history numerical
captures and a reasoning-enabled coding smoke check qualify the result.
The [combined driver](../experiments/2026-11-04-decode-pass/combined.nu) records
both source roots, executable hashes, settings and commands. Raw results stay
under `.workspaces/overnight-decode-20261104/combined/`.

Update the [combined report](../experiments/2026-11-04-decode-pass/combined.md),
manual/status summary and journal, then leave a clean local jj head. Main remains
unchanged and unpublished. Prior experiment percentages are not additive.

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
