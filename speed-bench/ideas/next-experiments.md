# Selected experiment

**Selected 2026-11-03: standalone short-Q8 query kernel.**

One-hour budget starts at 22:25:16 Eastern. Runtime baseline `ab947bef`
(documented through `05e78fdf`), Flash 1017 IQ2/Q2 on the M5 Max. Reuse the
existing short-Q8 reduction for the scalar 1,280-to-32,768 query projection,
preserving BF16 and RoPE semantics. Keep routing, packed KV, streaming RoPE,
4 GiB expert slabs and total cache capacity fixed. Use an opt-in dispatch with
the generic path as control and fallback. Validate focused GPU equivalence,
then five alternating timing pairs and fixed-history logits at 32K and 8K.
Use a separate profile to confirm activation and explain the measured effect.
Adopt only after assessing decode, prefill and complete phase time together.
Raw artifacts stay ignored under `.workspaces/short-q8-query-20261103/`.

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
