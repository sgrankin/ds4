# Incremental tool schemas preserve a reusable prompt prefix

**Opt-in; useful when discovery otherwise changes an earlier rendered prefix.**
Item46 added `DS4_SERVER_INCREMENTAL_TOOL_SCHEMAS=1` for DeepSeek V4.1 Responses
requests. On one favorable three-request workflow it saved **46.30%** of
whole-workflow time and **66.34%** after discovery. On the ordinary fixture it
was **8.17% slower**: the existing continuation fast path already reused the
prefix. Keep this feature opt-in rather than treating all dynamic tools alike.

## Algorithm and client contract

Initially render the known tools once. When a `tool_search_output` supplies new
`tools`, retain their schemas at the discovery position as a system message
headed `### Updated Tool Schemas`, after the complete tool-result group
containing that discovery. Restore
those announcements in the same chronological position when loading durable
history; restore the discovered call schemas and namespace mapping as well.
Do not hoist newly discovered tools into the initial tool preamble. That would
change tokens before the already-computed history and prevent ordinary prefix
comparison from reusing it.

```text
initial = initial tool definitions + system/user history
on client tool_search_output(tools):
    append the tool-search result
    append system("Updated Tool Schemas", render(tools))
    retain definitions and namespace mapping for subsequent calls/history
on previous_response_id replay:
    preserve that same discovery position and rendered prefix
```

The client still uses the existing Responses discovery exchange: send a
`tool_search_output` with its call ID and `tools`, execute the returned function
call, then supply its real `function_call_output`. This does not execute a
client tool inside the server or create a tool-search service. Source base was
`2639960c`; the changed server used the same binary with flag 0/1 for all four
ABBA arms. The feature changes rendered history and therefore can change model
outputs; no distribution-equivalence guarantee is claimed.

## Actual 1017 measurements

Apple M5 Max/128 GiB; verified 1017 IQ2_XXS/Q2_K weights. Each arm started its own
server, ran ordinary first and favorable second, then exited cleanly. Native
routing, suffix reuse off, application SSD KV cache off; expert SSD streaming
remained enabled. Temperature 0/no thinking, context 16,384, prefill chunk 2,048,
64 GB expert-cache target. Request timings include the actual local file read
in workflow wall time; startup is separate. Post-discovery below means only
R2+R3 request time, excluding the first discovery request and local tool time.

| Case / timed scope | Native bracket mean | Incremental mean | Saving | Native bracket drift |
|---|---:|---:|---:|---:|
| Ordinary full-input workflow |29.470 s|31.878 s|**−8.17%**|+1.78%|
| Ordinary R2+R3 |9.848 s|11.865 s|−20.48%|−1.30%|
| Favorable parent-history workflow |178.838 s|96.045 s|**46.30%**|−0.094%|
| Favorable R2+R3 |125.136 s|42.117 s|**66.34%**|−0.422%|

Against the faster native bracket, favorable workflow savings were 46.27% and
post-discovery savings 66.27%; ordinary workflow was 9.13% slower. The favorable
fixture began with 4,487 input tokens and discovered a namespace containing
13 schemas. Its `previous_response_id` path forced rendered-prefix comparison.
Without the flag, R2/R3 reported zero cached tokens; with it they reused 4,541
and 6,961 tokens, despite slightly MORE input tokens (R2:6880→6887;
R3:5899→6995). The ordinary full-input path already reused 677 then 828 tokens
without the flag; enabling announcements added 73 tokens and extra work.

All eight cases passed on their first outputs. Every discovered function name,
namespace and path was graded before an actual file read; returned archived
code/receipt and current key were checked. All arms returned the same graded
answers and output-token counts. This is practical fixture quality, not broad
coding accuracy, KL or ΔNLL evidence.

## Reproduce and scope

Use the existing [driver](../speed-bench/experiments/incremental-tools-hour46-20261020/run.py)
with a fresh output directory under the existing pressure guard:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/item46-rerun-guard -- \
  python3 speed-bench/experiments/incremental-tools-hour46-20261020/run.py \
  --model "$PWD/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf" \
  --output .workspaces/item46-rerun --order 0110
```

The driver sets the feature flag independently per arm and writes the exact
server command/environment. Preserve first outputs and inspect cached-token
geometry and namespace grades. Ordinary preceded favorable within each server,
so shared initial prefixes may be warm; inspect the retained actual usage
rather than calling all requests cold. There is only one ordered fixture per
case, one ABBA bracket, deterministic sampling and no persistent-agent task.
Do not apply 46.30% to unrelated tools, full-input continuations or the combined
resident/suffix package. Test the target harness's actual rendering path.

[Report and first-output evidence](../speed-bench/experiments/incremental-tools-hour46-20261020/README.md),
[machine-readable comparisons](../speed-bench/experiments/incremental-tools-hour46-20261020/summary.json).
Credit the [official DeepSeek recipe](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f)
for the incremental tool-schema announcement direction; this is a bounded local
Responses adaptation, not its entire renderer or distributed inference stack.
