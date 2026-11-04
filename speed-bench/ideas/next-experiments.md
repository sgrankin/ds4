# Experiment queue

The user authorized a seven-hour decode optimization session, from
2026-11-03 23:55:26 to 2026-11-04 06:55:26 Eastern. Work proceeds through
one-hour profile, target, implementation and measurement loops. This authorization
includes choosing the next bottleneck after each result; it does not authorize
unrelated features or publishing main. Reserve the final slice for combined
qualification and a concise account of adopted changes and unsuccessful attempts.

**Current loop: reuse RoPE coefficients in the short-query kernel.** Started
2026-11-04 02:52 Eastern, with a one-hour budget ending 03:52. Baseline
`a1900948` closes the router experiment and retains the three adopted decode
changes. The current diagnostic trace shows about 0.9 ms of query work per
token extending beyond compressed KV staging. This is not predicted savings.

Keep the existing short Q8 projection and its reduction, BF16 boundaries and
rotation FMA order. Test reading the existing per-token coefficient table
instead of recomputing sine and cosine for every query head. Enable only with
`DS4_METAL_V41_SHORT_Q8_QUERY_TABLE=1` and a table valid for the current scalar
position; preserve ordinary dispatch and unsupported-shape fallbacks. Check
raw and compressed frequencies, position changes, scalar tails and full-model
logits before five alternating timing pairs at 32K and 8K. Use a separate
profile to see whether any kernel saving reaches the exposed query tail.
Raw evidence belongs under
`.workspaces/overnight-decode-20261104/query-rope-table/`.

The [session index](../experiments/2026-11-04-decode-pass/README.md) links the
three adopted changes and the unadopted router experiment. The router's mixed
small differences are not a demonstrated regression; its prototype remains
recoverable in the recorded experiment revision.

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
