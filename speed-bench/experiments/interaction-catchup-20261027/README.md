# Official interaction catch-up, 2026-10-27

The cleanup and interaction changes pass the guarded HTTP, WebSocket, SDK,
restart and retained-decision checks below. All five qualified executables are
installed at the root and in `zig-out/bin`; hashes match and their `--help`
checks pass (`installed-binaries.log`). Main is unchanged. The frozen
runtime control is `aef775c29745`; cleanup is `f65fc2c5`, and the first combined
source checkpoint is `e2420dae`. Cache fixes and the final Flint schema adapter
follow that checkpoint. Per-run executable hashes, commands and environments
remain in the local `server/execution.json` files; do not assume every campaign
used the final binary.

Implemented: client-side tool execution replaces hosted QuickJS/PTC/MCP
orchestration; ordinary tools, Clef APIs, durable Responses and conversations
remain. DS41 follows official tool placement and current-tool reasoning policy.
DS41/Flint input, output and replay distinguish real control tokens from literal
spellings. Native XGrammar constrains tools and structured answers using the
shared normalized schema. Incremental Responses arguments preserve final values.
DS41 text cache markers select actual encoded prefix endpoints across replay and
SSD persistence. [Policies and handoff](../../../src/JOURNAL.md),
[bounded sequence](../../ideas/next-experiments.md).

Omitted `strict` remains non-strict. DeepSeek retains `min_p=0.05` and honors
explicit sampling overrides. Normal DeepSeek HTTP cases use high reasoning;
only named-tool conformance explicitly disables it. Flint retains its reference
defaults and autonomous reasoning.

## Qualification

All listed model campaigns completed under the existing pressure guard:
normal pressure throughout, stable swap, clean server shutdowns. A user-owned
reference Flint server remained resident on port 11500; its observed health was
idle. These are correctness runs, **not isolated latency comparisons**.

| Campaign | Result | Local evidence under `.workspaces/interaction-catchup-20261027/` |
| --- | --- | --- |
| DS41-1017 IQ2 HTTP | Ordinary/repeat, strict tool, schema answer and named non-thinking tool pass; literal think/DSML/XML delimiters and UTF-8 round-trip | `deepseek-correctness/`, `deepseek-correctness-guard/` |
| Flint INT4 HTTP | Ordinary/repeat, strict tool and schema answer pass with the same literal payload | `flint-correctness/`, `flint-correctness-guard/` |
| Flint final schema adapter | Referenced string `"true"`, mixed-enum integer `1` and nullable `null` retain their JSON types; ambiguous cross-type pattern rejects with 400 | `flint-coercion/`, `flint-coercion-guard/` |
| DS41 WebSocket/SDK/restart | 26/26 protocol checks; 12/15 answer-quality grades | `deepseek-ws/`, `deepseek-ws-guard/` |
| Frozen DS41 control | Same 26/26 and 12/15, with the identical three failed answers | `deepseek-baseline-ws/`, `deepseek-baseline-ws-guard/` |
| Flint WebSocket/SDK/restart | 29/29 protocol checks and 17/17 answer-quality grades | `flint-ws/`, `flint-ws-guard/` |
| Corrected DS41 cache policy | All read/write-credit and SSD-restart assertions pass | `cache-fix-correctness/`, `cache-fix-guard/` |
| Retained Clef-Flash INT4 | Discovery, both decision APIs, task answers, ordered/typed values and explicit model/type/context errors pass | `clef-retained/`, `clef-retained-guard/` |

DS41's failed optional grades are `disconnect_control`, `disconnect_survivor`
and `disconnect_retry`: each returned `{"surived":false}` instead of
`{"survived":true}`. The frozen control produces the same failure even before
disconnection. Protocol/cancellation recovery passes; the answer-quality screen
is not perfect. The WebSocket suite uses its existing non-thinking DS41 setting,
separately from the high-reasoning HTTP checks.

Flint's long-context parent has 4,855 input tokens. Its post-decode continuation
has 5,011 input tokens and credits the saved 4,855-token prefix. This exercises
prefill beyond the 4,096-token chunk as well as continuation after generation.

HTTP checks also confirm clear errors for unsupported strict-tools-plus-format,
invalid choices/formats, background execution, automatic truncation and custom
tool declarations. DS41 required/named choices with thinking are rejected.

Model-free verification passes: all five executables build; existing full host,
Agent and script checks pass; native grammar 11/11 in ReleaseFast and Debug,
cache/API 11/11, serving boundary 12 cases, native prompt provenance four cases,
and all 47 installed-Flint tokenizer cases retain exact IDs/bytes. The initial
Web recovery check needed its normal loopback permission, then passed. Raw
build/test logs remain local. Flint's final narrow schema-coercion adapter passes
two focused tests and the three-type live probe with native reasoning retained
(139 reasoning tokens). The final five-executable build and fresh Debug
server/output and other Debug fixtures pass.

## Explicit cache result and discovered failures

The corrected run uses DS41-1017
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, 8,192 context, two
sessions, 512-token prefill chunks, 32 GB SSD expert cache, and exact KV with
suffix/replay approximations disabled. State-cache hot capacity is zero and SSD
capacity is 512 MB, so the restored checkpoint is an SSD hit.

| Request, in order | Input tokens | Read credit | Write credit |
| --- | ---: | ---: | ---: |
| Fresh explicit mode, no marker | 49 | 0 | 0 |
| Same prompt, implicit mode | 49 | 0 | 49 |
| Implicit repeat | 49 | 49 | 0 |
| Marked prefix, first request | 613 | 0 | 605 |
| Marked repeat with changed tail | 614 | 605 | 0 |
| Changed earlier prefix | 616 | 0 | 608 |
| Explicit mode, no marker | 613 | 0 | 0 |
| Original marked prefix after server/store restart | 613 | 605 | 0 |

The first run had already demonstrated a 1,469-token SSD hit, but incorrectly
reported freshly computed tokens as explicit writes. It also exposed an unmarked
live-KV frontier that a later implicit request could reuse. The fix counts only
successfully published explicit endpoints and retires unauthorized source state
before the next request can synchronize against it. The fresh no-marker to
implicit pair above directly checks that second failure. Original artifacts and
incorrect counters remain preserved under `deepseek-correctness/`.

The corrected marked repeat took 2.62 s; the earlier 1,469-token pair took about
49.23 s versus 0.46 s in prefill. These ordered observations establish useful
reuse, not a measured speedup against an isolated control. Explicit markers are
currently supported for DS41 text; logical Responses storage remains independent
of KV policy. Historical markers remain lookup candidates, with at most four new
marked endpoints written per request. Moving a boundary may split a BPE merge.

## Reproduction

Run from the repository root. Keep one guarded model job at a time, use fresh
output/store directories, and retain the guard. The first cache-off probe must
start on a fresh store: an older authorized prefix is otherwise a legitimate hit.
The exact HTTP launcher commands and server settings are recorded in each
campaign's guard `command.json` and `server/execution.json`.

Build the combined model-free fixtures:

```nu
nu tools/zig-build ds4-server server-test serving-boundary-test grammar-test prompt-tokens-test constraint-contract-test ds41-output-test server-cache-breakpoints-test --prefix .workspaces/interaction-catchup-20261027/recheck-build -j6
```

The maintained [HTTP driver](qualify.nu) exercises one already-running guarded
server: `nu speed-bench/experiments/interaction-catchup-20261027/qualify.nu BASE
MODEL_ID OUTPUT --family deepseek` (use `--family flint` for Flint, or
`--cache-only` for the DS41 cache sequence, `--coercion-only` for Flint's final
type probe). It saves requests/responses locally
and checks actual parsed tool arguments and structured answers. The local
`qualify_http.py` launcher reuses the maintained WebSocket fixture's `Server`
class and, for `--cache-only`, closes/reopens the same store to check restart.

Repeat the recorded cache-fix command with fresh output paths and the pressure
guard:

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/interaction-catchup-20261027/recheck-cache-guard -- .workspaces/responses-ws-venv/bin/python .workspaces/interaction-catchup-20261027/qualify_http.py --binary .workspaces/interaction-catchup-20261027/cache-fix-build/bin/ds4-server --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --family deepseek --cache-only --output .workspaces/interaction-catchup-20261027/recheck-cache
```

The maintained [WebSocket fixture](../../../tests/test_responses_websocket.py)
starts and restarts the server with recorded settings. Reuse its existing
virtual environment containing `websockets` and the OpenAI SDK:

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/interaction-catchup-20261027/recheck-ws-guard -- .workspaces/responses-ws-venv/bin/python tests/test_responses_websocket.py --server .workspaces/interaction-catchup-20261027/cache-fix-build/bin/ds4-server --output .workspaces/interaction-catchup-20261027/recheck-ws --sdk
```

Use the frozen `baseline/ds4-server` for the control; for Flint add `--model
.workspaces/flint-downloader-live/flint-int4`. The retained Clef campaign launches
`gguf/clef-flash-int4` and runs `nu tests/test_decisions.nu --url BASE --model
clef-flash`. Raw results stay ignored; this report and the driver are tracked.

## Remaining limits

The single-token constraint limit below describes this frozen checkpoint.
The subsequent [constrained-packet qualification](../constrained-packets-20261027/README.md)
restores Flint packets and DS41 verification and records matched measurements.

- Constrained generation currently accepts one token per step and bypasses MTP;
  unconstrained decoding keeps its existing fast path. No isolated constrained
  overhead or no-constraint latency comparison is claimed.
- GLM/Qwen retain their previous Responses streaming path; the new incremental
  tool-argument projection and native grammar work is qualified for DS41/Flint.
- Flint's XML-to-JSON adapter resolves schema types/references and rejects
  ambiguous cross-type constraints it cannot represent faithfully. The focused
  type probe qualifies the concrete cases above, not every JSON Schema keyword.
- No isolated timing campaign ran while the reference Flint server remained
  resident. Pausing it still requires the user's answer.

Primary references: [DeepSeek recipe](https://github.com/deepseek-ai/deepseek-recipe),
[DeepInfer](https://github.com/deepseek-ai/DeepInfer),
[XGrammar v0.2.8](https://github.com/mlc-ai/xgrammar/tree/v0.2.8), and the local
[Flint reference](https://github.com/recursive-oss/rec). Their source contracts
are distinguished from our measured runtime behavior.
