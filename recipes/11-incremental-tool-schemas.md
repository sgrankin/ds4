# Incremental tool schemas preserve a reusable prompt prefix

**Chronological announcements are the V4.1 Responses default; explicit `0` restores legacy rendering.**
Item49's unset-default bracket saved **57.45%** on the favorable parent-history
workflow and was **4.00% slower** on ordinary full-input continuation. All eight
first workflows and actual reads passed. The current implementation removes
the duplicate schema body within each discovery event. Item46's older opt-in
measurements remain separately labelled below.

## Algorithm and client contract

Initially render the known tools once. When a `tool_search_output` supplies new
`tools`, retain their schemas at the discovery position as a system message
headed `### Updated Tool Schemas`, after the complete tool-result group
containing that discovery. Restore
those announcements in the same chronological position when loading durable
history; restore the discovered call schemas and namespace mapping as well.
When an announcement is generated, an implicit schema-only tool result becomes
`Tool discovery complete.` instead of repeating the raw schema JSON. Caller
`output` or `result` text remains verbatim. The incoming protocol `tools` field
is not mutated, and full definitions remain private trusted replay metadata.
This removes a second rendered copy within an event; it does not compare
schemas or deduplicate repeated names across discovery events.
Do not hoist newly discovered tools into the initial tool preamble. That would
change tokens before the already-computed history and prevent ordinary prefix
comparison from reusing it.

```text
initial = initial tool definitions + system/user history
on client tool_search_output(tools):
    append explicit output/result, or "Tool discovery complete."
    append system("Updated Tool Schemas", render(tools))
    retain definitions and namespace mapping for subsequent calls/history
on previous_response_id replay:
    preserve that same discovery position and rendered prefix
```

The client still uses the existing Responses discovery exchange: send a
`tool_search_output` with its call ID and `tools`, execute the returned function
call, then supply its real `function_call_output`. This does not execute a
client tool inside the server or create a tool-search service. Source base was
`2639960c` for Item46; its changed server used the same binary with flag 0/1 for
all four ABBA arms. Explicit `0` now restores the initial-schema hoist and full
implicit tools-result body. Static tools and other model syntax are unchanged.
The feature changes rendered history and therefore can change model
outputs; no distribution-equivalence guarantee is claimed.

## Current Item49 measurements: unset default

One binary, source base `4e12e4f2`, ran `0,u,u,0`; `u` omitted
`DS4_SERVER_INCREMENTAL_TOOL_SCHEMAS` from the server environment. The same
ordinary/full-input and favorable/parent-history fixtures used native routing,
suffix reuse off, application SSD KV cache off, temperature 0/no thinking,
context 16,384 and chunk 2,048. Workflow time includes the actual local file
read; R2+R3 excludes discovery and tool time.

| Case / timed scope | Legacy mean | Unset default mean | Saving | Legacy bracket drift |
|---|---:|---:|---:|---:|
| Ordinary full-input workflow |29.502 s|30.683 s|**−4.00%**|−0.363%|
| Ordinary R2+R3 |9.744 s|10.157 s|−4.23%|−1.581%|
| Favorable parent-history workflow |188.734 s|80.300 s|**57.45%**|−1.550%|
| Favorable R2+R3 |133.201 s|24.479 s|**81.62%**|−1.716%|

Favorable workflow saving is 57.12% against the faster control. Its default
inputs were 4,485/5,651/5,757 tokens with 0/4,539/5,723 cached, while legacy
R2/R3 reused zero. Ordinary R2 reused 669 tokens in both variants and processed
79 legacy versus 88 default fresh tokens: a remaining nine-token formatting
cost. R3 processed 34 fresh tokens in both. Of the ordinary 1.181 s workflow
gap, 0.769 s occurred in R1, whose prompt was unchanged; the whole difference
cannot be assigned to late-schema formatting alone.

All eight first workflows passed discovered-name/namespace/path checks, read
the real fixture file and returned the expected archived and current facts.
Output lengths were identical across arms: ordinary 48/70/29 tokens and
favorable 54/72/29. All 633 pressure samples were normal, with 3.87 MiB maximum
swap growth. This qualifies the default on these fixtures, favoring persistent
parent-history reuse despite a small ordinary-path cost; it is not broad coding
quality or probability-equivalence evidence. Do not attribute the timing change
from Item46 to deduplication alone: these are separate brackets.

[Item49 report and first outputs](../speed-bench/experiments/incremental-tools-hour49-20261021/README.md),
[comparisons](../speed-bench/experiments/incremental-tools-hour49-20261021/summary.json).

## Historical Item46 measurements: initial announcement implementation

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
These timings include the original schema duplication; they are not Item49
deduplication measurements or a prediction of its ordinary-case cost.

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
The current checkout measures Item49's revised rendering; use the recorded
Item46 binary/revision to reproduce the historical implementation.

## Reproduce the current default

The driver also accepts `u`/`0uu0`; `u` means the feature flag is absent. This
directly tests the default rather than assuming an explicit `1` matches it:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/item49-rerun-guard -- \
  python3 speed-bench/experiments/incremental-tools-hour46-20261020/run.py \
  --model "$PWD/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf" \
  --output .workspaces/item49-rerun --order 0uu0
```

Preserve first outputs, actual reads, namespaces, cached-token geometry, output
lengths and bracket drift. Removing redundant text can change generation;
matching output probabilities is not assumed.

[Item49 method and evidence](../speed-bench/experiments/incremental-tools-hour49-20261021/README.md).

[Report and first-output evidence](../speed-bench/experiments/incremental-tools-hour46-20261020/README.md),
[machine-readable comparisons](../speed-bench/experiments/incremental-tools-hour46-20261020/summary.json).
Credit the [official DeepSeek recipe](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f)
for the incremental tool-schema announcement direction; this is a bounded local
Responses adaptation, not its entire renderer or distributed inference stack.
