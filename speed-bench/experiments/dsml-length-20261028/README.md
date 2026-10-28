# Typed tool output at generation limits

DeepSeek V4.1 and Flint now retain typed unfinished tool calls when generation
reaches its token/context limit. Responses reports `incomplete` with
`incomplete_details.reason=max_output_tokens`; unfinished function-call items
retain their argument-prefix strings and stable stream identities. Chat keeps
`finish_reason=length` and the same partial argument string. Earlier completed
invocations within a truncated multi-call turn retain their completed status.

This is finalization of already-sampled tokens. It neither supplies missing
closing syntax nor changes model sampling. The ordinary native grammar still
constrains tool generation automatically. Malformed structure, invalid JSON
prefixes and unfinished invocations at ordinary EOS remain errors. A standard
streaming JSON scanner distinguishes an unfinished prefix from invalid bytes.
Incomplete UTF-8 tails are withheld consistently through finalization.

Partial turns acquire no raw DSML replay record, are not remembered as exact
tool calls, and skip tool checkpoint canonicalization and continuation reuse.
The pre-existing `previous_response_id` policy still rejects **every incomplete
parent**, including ordinary reasoning-only truncation, with a clear 400 error.
This change does not expand parent-replay semantics. A client can retry from
safe input with a fresh request and a changed tool set, without submitting a
result for an unfinished call.

Anthropic streaming retains genuine `input_json_delta` prefixes and closes a
partial live block once, with `stop_reason=max_tokens`. Its nonstream
`tool_use.input` requires a complete object, so an unfinished block is omitted
rather than replaced with an invented empty/repaired input. Ordinary Responses
and Chat function arguments are strings and preserve the prefix directly.

## Qualification

The production-linked CPU fixture exercises string/JSON prefixes, split UTF-8,
partially generated tags, malformed prefixes, mixed complete/partial calls,
Responses delta/done/terminal byte and ID consistency, OpenAI argument strings,
Anthropic local-socket streaming, and exclusion from tool memory/replay.
The existing server host regressions and pure token-projector fixture pass.
The malformed-Flint and ordinary-EOS rejection assertions are included in the
final qualification binary. No model or GPU is required for these fixtures.

The guarded DeepSeek 1017 IQ2_XXS/Q2_K + DSpark live campaign keeps high reasoning,
8192 context, 512 prefill chunks, 32 GB expert cache and normal native constraints.
All 12 long-constant requests at caps 64/256/512 pass protocol checks but spend
their entire budgets in reasoning: they establish reasoning-limit behavior,
not partial-call coverage. The retained initial driver's no-partial-call
assertion fails for that coverage gap, not a server failure.

The reused strict-tool fixture then passes all four actual partial-call cases
at cap 96: Responses/Chat, streamed/nonstreamed. Responses records 82 reasoning
and 14 tool tokens and returns an unfinished echo call with argument prefix `{`.
The streamed argument prefix and item/call identities match the terminal
snapshot. An additional 128-token Responses stream reaches the string payload after the
same 82 reasoning tokens. Its 21 argument deltas concatenate exactly to the
unfinished string containing literal `</think>`, DSML/Flint closing spellings,
HTML entities and café. There is one argument-done and one incomplete item-done,
with matching terminal bytes and stable item/call identities.

The first continuation attempt exposes the existing incomplete-parent 400
policy; that rejection is now explicitly checked by the driver. A safe fresh
retry with the replacement recover tool completes with `{"text":"READY"}`
and a new call ID. These are correctness checks, not comparative performance
or completed-answer quality measurements.

Raw evidence remains local under
`.workspaces/convergence-20261028/dsml-live/`, including `http/`, `known96/` and its `payload128.sse`/safe-retry records,
request/response bodies and the unchanged pressure guard's command/settings.

Flint INT4's four cap 64 wire cases also validate correctly: both Responses
paths and nonstream Chat pass the original driver; the original streamed Chat
body passes regrading after fixing a checker that incorrectly rejected a
literal DSML spelling **inside argument data**. Its Responses stream has exact
partial argument/done/terminal bytes and stable IDs. Incomplete-parent rejection
and a safe changed-tool retry also pass, returning `{"text":"READY"}`.

A fresh full-driver repetition did **not** complete: the unchanged pressure
guard stopped the owned Flint server when external CGPDFService memory growth
raised kernel pressure to level 2. The server exited cleanly with code 0. Later
requests received I/O errors because that server was absent. These retained
infrastructure failures are not a fresh qualification pass or evidence of an
inference failure. No runtime change was made for the checker mistake or
pressure stop. The external PDF helper subsequently exited on its own and
pressure returned to normal, without a restart or user intervention. Evidence
is under `flint-live/cap64/`, `cap64-fixed-driver/` and `guard-retry/` beside
the DeepSeek folder above.

## Reproduction

```nu
nu tools/zig-build ds41-output-test server-test --prefix .workspaces/dsml-build -j4
.workspaces/dsml-build/bin/test-ds41-output
.workspaces/dsml-build/bin/ds4_test --server
```

Start one server beneath the existing pressure guard. Use the maintained
[HTTP driver](qualify.nu) against that server. To regenerate the small fixed
Responses requests used by the earlier native-constraint qualification:

```nu
nu speed-bench/experiments/constrained-packets-20261027/qualify.nu unused deepseek-v4.1-flash .workspaces/dsml-fixtures --family deepseek --dry-run
nu speed-bench/experiments/dsml-length-20261028/qualify.nu http://127.0.0.1:18347 deepseek-v4.1-flash .workspaces/dsml-check --family deepseek --caps 96 --fixture .workspaces/dsml-fixtures/T0.0-strict-tool.request.json
```

The optional fixture is preserved verbatim apart from model, cap, streaming and
storage settings. Chat maps that same request into its tool/message wire shape;
DeepSeek reasoning remains high. With no fixture, the driver retains its
long-constant input for reasoning-limit screens. Output directories record the
fixture path and exact requests, so coverage gaps and failed probes remain
inspectable.
