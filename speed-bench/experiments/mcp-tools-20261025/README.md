# Outbound MCP: fixed screen and live proof

The fixed fixture reuses the existing account-takeover policy, records and
external oracle from [the code-mode screen](../ptc-decisions-20261025/README.md).
It retains the four short ordinary records and the first eight long favorable
records. Gold labels are never sent to a model. `fetch_queue` returns the full
queue once in MCP `structuredContent`, plus a short text receipt. Its server is
local, read-only, and started/stopped by the wire leaf; no model is launched.

Three natural arms share the same handoff task: main-only sees the full queue;
direct MCP fetch exposes the source result to the main model; code mode can
fetch, ask one batched decision request and filter/aggregate before emitting a
compact result. Ordinary evidence is visible in all arms, so avoiding an
unnecessary call is allowed. Favorable tool arms see a manifest; that measures
data placement plus composition, not voluntary delegation over equal context.

Start one guarded generator/helper server, retaining native DeepSeek reasoning.
The drivers use temperature zero, a shared 4,096 output-token limit and eight
tool calls. They never add a reasoning override.

```nu
# Explicit direct fetch and one program with MCP fetch→decision→filter.
nu tests/mcp-tools.nu --suite smoke
# Add stored replay without rerun and a WebSocket program.
nu tests/mcp-tools.nu --suite qualify
# Natural task, without naming a tool or supplying code.
nu tests/mcp-tools.nu --suite scenarios --case favorable --repeats 2
# Select one arm for sequential ABBA measurements or a bounded first sample.
nu tests/mcp-tools.nu --suite scenarios --case ordinary --arm ptc --repeats 1
```

`--url`, `--model`, `--helper`, `--output` and `--python` override defaults.
Record wall time, main input/output/reasoning/cached tokens, actual fixture calls,
program/helper counts and time, result/transport bytes and numeric/JSON quality.
Review the two-sentence handoff separately for unsupported claims. Protocol
`pass` is separate from natural-task quality. Keep raw output ignored in
`.workspaces/mcp-tools-20261025/`; commit concise results only.

The first hour should cover transport qualification, one ordinary/favorable
screen, a useful repeat and manual transcript review. Stop once a useful
measurement is possible; do not tune the task until it wins.

## Exa: a separate live integration check

Exa documents [keyless, rate-limited access](https://exa.ai/docs/get-started/exa-mcp)
without signup at `https://mcp.exa.ai/mcp`. The documented defaults are
`web_search_exa` and `web_fetch_exa`; OAuth/API-key modes are separate options.
The [probe notes](RESULTS.md) distinguish official documentation from actual
endpoint observations. Live search results can change and are not the fixed
speed/quality baseline.

```nu
# Sends only a fixed public MCP-spec query and fetch, with no local data/key.
nu speed-bench/experiments/mcp-tools-20261025/probe-exa.nu
# Once the native server is running: one explicit program, two live MCP calls.
nu speed-bench/experiments/mcp-tools-20261025/exa-response.nu
```

This optional probe uses system curl because the initial Python urllib client
was blocked by Exa's edge while system curl succeeded. It stops on an HTTP/MCP
error rather than automatically retrying or changing credentials. The native
serving bridge needs its own live qualification; a curl success alone does not
establish runtime integration. `exa-response.nu` requests normal native reasoning
and emits compact source snippets plus exact string-presence checks. It records
missing terms rather than inventing evidence, and makes no natural-task speed
claim from this explicitly supplied integration program.
