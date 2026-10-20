# FINAL thinking policy

Copy the hour18 natural READ→WRITE→FINAL workflow on the frozen hour17 ledger/LRU tasks and first-write graders. Task instructions, tool definitions, advertised confirmation-visible paths, sampling and continuation protocol stay unchanged across phases and arms. READ uses high effort with budget 0; WRITE uses high effort with budget 128. Only FINAL changes:

| Arm | FINAL effort | FINAL request budget |
|---|---|---:|
| high0-before | high | 0 |
| budget64 | high | 64 |
| none | none | 0 |
| high0-after | high | 0 |

Each arm starts one fresh server/cache/store, Q2/cache64/ctx8192/chunk512/batch1, selected64/limit1024/GAP0, suffix off. There is one actual read and one first-code write; no repairs or retries. The first written code is independently graded against 161 checks per task. FINAL must complete with nonempty text and no new tool call. Its actual text, character/byte length, output/thinking usage, caps and errors remain recorded; there is no predetermined ACK text or semantic final-answer score. All three roles share a 2048-output-token allowance, including forced closure. Budget64 is the existing soft once-per-request thinking control; later reopening or unsafe-state deferral can exceed it.

Compare whole task wall and FINAL request wall against the high0 bracket mean, retaining incomplete/capped outcomes. Per-role usage reports cached input tokens. Saved first-code files and `first_code_bytes` expose implementation changes. Code correctness and workflow completion remain separate fields. Divergent histories have no paired KL/NLL. This two-task screen measures FINAL policy within the existing natural caller, not an automatic server phase detector or general model-quality guarantee.

Proposed ROOT command (not launched by the author):

```sh
python3 -B speed-bench/ideas/pressure_run.py speed-bench/experiments/final-thinking-hour20-20261004/guard -- python3 -B speed-bench/experiments/final-thinking-hour20-20261004/launch.py --output speed-bench/experiments/final-thinking-hour20-20261004/http --max-wall-seconds 1500
```

The launcher accepts `--arm` to select an existing arm if the time budget requires a shorter run. Full four-arm preparation predicts roughly 9–12 minutes from the earlier natural control; observed caps/longer thinking remain measured rather than retried. Preparation time counts toward item20's one-hour budget.

After the four-arm measurement, FINAL effort none was slower because it rebuilt the full prompt, while high/budget64 retained the prefix. A bounded follow-up adds high/budget1 as a lower numeric point, keeping the high-effort prefix unchanged. This was chosen after inspecting those results and is not an independent preregistered confirmation. Original four-arm requests/results stay untouched; ROOT runs only `--arm budget1 --output .../http-lowpoint` in a fresh namespace. The default arm list now appends budget1 after high0-after.
