# Outbound MCP qualification and first screen — 2026-10-25

Native MCP discovery, direct calls and programmatic fetch→decision→filter work.
The natural-task screen is negative for automatic composition: the eight-record
code-mode plan takes **78.5% longer than direct MCP**, because it spills the full
queue into main context, fetches it again and returns all helper probabilities.
All 12 answers have correct numeric fields and clean JSON. Plumbing success does
not establish that the model chooses an efficient plan.

The independent endpoint probe ran no model or build; subsequent serving checks
used one guarded native DeepSeek/Clef server. Two repeats of two fixed synthetic
cases are a bounded learning screen, not a broad quality or performance benchmark.
Raw requests, headers, bodies and per-call timing stay under ignored
`.workspaces/mcp-tools-20261025/exa-probe/`.

## Anonymous Exa endpoint

The [official Exa documentation](https://exa.ai/docs/get-started/exa-mcp) advertises
keyless rate-limited access without sign-in. This was verified for initialization,
tool listing, one public technical search and one known-page fetch using system
curl. No credentials, local documents or private queries were sent.

The first Python urllib initialize received HTTP403, Cloudflare Error1010
`browser_signature_banned`. A separately authorized single system-curl initialize
succeeded. This is a client/edge-stack difference, not evidence that a signup is
required. The native transport and Responses checks below complete the separate
serving qualification.

| Operation | HTTP | Observed result |
|---|---:|---|
| Initialize | 200 | Negotiated `2025-06-18`; SSE response; `Mcp-Session-Id` returned |
| Initialized notification | 202 | Same session/protocol headers accepted |
| Tools list | 200 | `web_search_exa`, `web_fetch_exa` |
| One web search | 200 | Official MCP-spec results; 1.481s curl wall time |
| One known-page fetch | 200 | Official MCP transport specification content; 0.168s curl wall time |

Server info identifies `exa-search-server` version `3.5.0`. Both listed tools
carry `readOnlyHint:true`, `destructiveHint:false`, `idempotentHint:true` and
`openWorldHint:true`. The actual search schema requires **both `query` and
`objective`** strings; objective is limited to 4,096 characters. Optional
`numResults` is numeric. Fetch requires `urls` as an array of strings and allows
numeric `maxCharacters` of at least one. Both schemas disallow extra properties.

The live required `objective` field was absent from the
[repository's current search registration](https://github.com/exa-labs/exa-mcp-server/blob/main/src/tools/webSearch.ts)
when inspected; the repository package reported `3.4.2`. Discover schemas from
the endpoint instead of assuming a source snapshot matches deployment.
[The fetch source](https://github.com/exa-labs/exa-mcp-server/blob/main/src/tools/webFetch.ts)
also documents batched URLs. These are endpoint/source observations on this
date, not a promise that the free tool surface or limits remain unchanged.

The public query asks for official Model Context Protocol documentation about
Streamable HTTP session IDs and protocol-version headers. The fetch URL is
`https://modelcontextprotocol.io/specification/2025-06-18/basic/transports`.
Returned content is mutable external evidence; it never becomes server policy.
These isolated network times are not native bridge timings or model speedups.

## Native transport and concrete integration fixes

The compiled `mcp-client-probe` independently initializes, lists, searches and
fetches from Exa with protocol `2025-11-25` and a session header. Search/fetch
return ordinary successful MCP content without `isError`. Those native results
remain in `exa-native-{list,search,fetch}.json` beneath the work directory. This
qualifies the implementation's transport separately from system curl.

Live qualification caught two defects before natural-task measurement:

- Empty initialize capabilities and tool-list params serialized as JSON arrays.
  They now use typed JSON objects; a regression check preserves `{}` on the wire.
- Nonstreaming Responses omitted the discovered `mcp_list_tools` item because
  emission depended on the streaming flag. The guard now checks the API kind,
  preserving the catalog in JSON responses as well as streaming transports.

The first catalog failure remains in `qualify/result.json`; the corrected run
in `qualify-fixed/result.json` passes all 11 checks. These cover direct JSON MCP
fetch and its full structured result, the allowed-tool catalog, an SSE MCP fetch
followed by batched decision/filter inside one program, nested audit without
duplicate direct call items, stored replay without a new MCP execution and
WebSocket completion. All five recorded oracle/continuation grades are correct
and all four user-facing answers are clean JSON. Each handoff is two grounded
sentences and makes no invented claim of completed containment or refund.

The explicit program fetches the original four-record queue, gets two positive
and two negative helper labels, filters/joins inside JavaScript and returns only
IDs/count/maximum to the generator. Its MCP fetch takes 2 ms and helper 855 ms;
these are qualification observations, not matched speed claims. Focused native
client/runtime/output suites pass 5/10/33 checks, and the full host verification
passes. Runtime ownership remains native; the Python leaf is test wire I/O and
a local HTTP fixture.

## Fixed fixture and measured setup

[fixtures.nu](fixtures.nu) generates the original four short records and first
eight long records. The independent oracle expects two/three positives and
maximum charges of 78,900/119,900 cents. Fixture generation, Python syntax and
Nushell driver parsing pass. JSON and SSE fixture paths each pass a model-free
initialize/notification/list/fetch check. The two read-only MCP tools use ordinary
JSON-RPC over JSON/SSE and check negotiated session/protocol headers.

The maintained [driver](../../../tests/mcp-tools.nu) compares main-only, direct
MCP results and programmatic composition. Frozen runtime: local jj change
`wkzpplwxsunk`, snapshot `204d027670a0`, parent `be7f820f`. Only reports change
after the snapshot; no runtime, driver or prompt changes occur during measurement.

M5 Max, 128 GiB; DeepSeek V4.1 Flash 1017
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, SSD streaming with a 32 GB
expert cache, and resident `clef-flash-int4`. Context is 16,384, prefill/mixed
quantum 2,048, MTP draft one and two sessions. Every request uses temperature
zero, eight tool calls and the same shared 4,096 output-token budget. Native
DeepSeek HIGH reasoning is retained: no override or extra reasoning cap.
Helper prefix caching, decoder replay and suffix reuse are disabled. All observed
main cached-token counts are zero. Model loading is outside request wall time.

Within each case, arms run main/direct/code then code/direct/main. The ordinary
case exposes full source notes to all three arms; a model may answer without
tools. In the favorable case, the main-only lower bound receives all notes in its
initial prompt, while both tool arms receive a manifest and must fetch notes.
Thus **direct MCP versus code mode is the fetch/composition comparison**;
main-only omits the retrieval step. This is not equal-context voluntary use in
the favorable case. The fixture is local and deterministic; live Exa search is
qualified separately and never used as a mutable latency baseline.

## Natural-task results

Wall values are medians and observed ranges of two runs. Token counts are main
input / all generated output / reasoning included in that output; each count is
identical across its two repeats. MCP counts below are tool invocations, excluding
initialize/list. Discovery still runs for each request declaring MCP, including
the code-available requests that invoke no tool, and is included in wall time.

| Case and arm | Median wall | Range | Input / output / reasoning | Actual MCP / programs / helpers |
|---|---:|---:|---:|---:|
| Ordinary, preprovided-data main-only | 39.791s | 39.477–40.104s | 426 / 297 / 217 | 0 / 0 / 0 |
| Ordinary, direct MCP | 92.241s | 91.532–92.950s | 965 / 790 / 628 | 1 / 0 / 0 |
| Ordinary, code available | 69.689s | 68.721–70.657s | 1,253 / 727 / 642 | 0 / 0 / 0 |
| Favorable, preprovided-data main-only | 57.931s | 57.712–58.151s | 1,742 / 560 / 462 | 0 / 0 / 0 |
| Favorable, direct MCP | 59.211s | 58.813–59.609s | 2,183 / 327 / 184 | 1 / 0 / 0 |
| Favorable, code + helper | 105.714s | 105.465–105.964s | 3,300 / 696 / 416 | 2 / 2 / 1 |

The ordinary direct arm needlessly fetches evidence already in its prompt.
The code-available arm makes no tool invocation, but still takes **75.1% more time than
main-only**. Its larger tool surface accompanies more prompt tokens and 642
reasoning tokens rather than 217. This is an availability/deliberation cost,
not time spent in JavaScript or helper inference. It is 24.4% faster than the
direct arm, but no program executes, so that difference is not a composition win.

The favorable code arm follows the same inefficient plan in both repeats:

1. Fetch the queue and `text(q)`, exposing all notes to main context.
2. Start another program, fetch the same queue again, ask eight targeted helper
   questions and `text(r.answers)`, exposing every probability vector.
3. Let the main model filter IDs and compute the final maximum from its manifest.

Both programs succeed without error or retry. The helper labels are TP3/TN5/FP0/FN0
in each repeat and take only 1.625/1.730 seconds for 2,573 helper input tokens;
local fetches take 2–3 ms. Yet code mode takes **46.503 seconds / 78.5% more time
than direct MCP**, and 82.5% more than the preprovided-data lower bound. Main
input grows 51.2% versus direct MCP and generated output grows 112.8%. The extra
main work and missed compact-result boundary accompany the loss; individual
kernel/host costs were not profiled.
The repeats reuse the same eight records; they are not 16 independent examples.

Main-visible program results total **10,887 bytes**: 9,224 for the full queue
and 1,663 for helper answers. Full HTTP output is 60,903 bytes versus 11,907
direct and 1,574 main-only, including durable native audit. No bulk notes or
probability vectors appear in the final answer, but they do enter the generating
model's continuation. The explicit qualification program proves that fetching,
classifying and aggregating inside one program works; natural planning does not
choose that boundary here. No prompt tuning or new abstraction was added to
turn this screen into a win.

## Task quality and practical interpretation

All 12 requests complete, all numeric ID/count/maximum grades pass, and all 12
final answers are clean JSON. Every handoff has two sentences, with no invented
claim of an account already locked or charge refunded. The favorable main-only
handoff collectively says recovery controls changed; that is explicit for two
selected records but not Q-DV4, whose note establishes ongoing sessions instead.
Keep that minor grounding overgeneralization separate from numeric correctness.
The direct and code favorable handoffs avoid it. No request reaches the output
budget or produces a program/native failure.

This validates an MCP foundation and exposes a model-planning limitation. Keep
the capability request-selected; this screen does not justify enabling automatic
delegation everywhere. Better generic composition examples or a small callable
library are possible later experiments with the same frozen task, not adopted
changes or demonstrated gains in this slice.

## Live Exa through Responses

The separate [live proof](exa-response.nu) completes with one explicitly supplied
program and two successful MCP calls through the native server. Search takes
1.885 seconds and known-page fetch 0.389 seconds, totaling 2.274 seconds inside
the program. The full request takes 106.344 seconds, with 1,956 main input tokens,
898 generated tokens including 28 reasoning tokens, and zero cached tokens.
Native reasoning is retained; an explicit program simply requires little model
deliberation. This is an integration observation, not a matched speed claim or
proof of voluntary tool selection.

Literal checks are independently verified against the audited source text:

| Exact string | In search result | In fetched official page |
|---|---|---|
| `Mcp-Session-Id` | yes | yes |
| `MCP-Protocol-Version` | yes | yes |
| `application/json` | no | yes |
| `text/event-stream` | no | yes |

The program emits only 1,903 bytes of URL, snippets and presence flags from the
4,366-byte search text and 11,056-byte fetched text. Full source stays in the
audit; the HTTP response is 28,240 bytes. All emitted flags and snippets match
the sources. The model's final answer faithfully repeats that object inside a
Markdown JSON fence, so the final raw answer is not clean JSON. No sanitizer or
format-repair loop was added. Raw request, response, audit and summary remain
under `exa-response/` in the ignored work directory.

Both owned guard runs stop cleanly with exit zero, normal pressure throughout
and no swap growth. The first preserves the catalog failure; the corrected run
contains qualification, all 12 natural samples and the live Responses proof.

## Reproduce

From the repository root, use the measured source and one server under the
existing pressure guard. Choose a fresh work directory so prior logs are retained.

```nu
nu tools/zig-build ds4-server -j3
let work = '.workspaces/mcp-tools-reproduce'
mkdir $work ($work | path join responses)
$env.DS4_SERVER_RESPONSE_STORE_DIR = ($env.PWD | path join $work responses)
$env.DS4_LOCK_FILE = ($env.PWD | path join $work server.lock)
$env.DS4_CLEF_PREFIX_CACHE = '0'
$env.DS4_V41_DECODER_REPLAY_ROWS = '0'
$env.DS4_SERVER_CONTEXT_SUFFIX_REUSE = '0'
let server_args = [
    --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
    --decision-model gguf/clef-flash-int4 --ssd-streaming
    --ssd-streaming-cache-experts 32GB --host 127.0.0.1 --port 19333
    --ctx 16384 --prefill-chunk 2048 --mixed-prefill-quantum 2048
    --tokens 4096 --mtp-draft 1 --batched-session 2
    --trace ($work | path join trace.log)
]
^python3 speed-bench/ideas/pressure_run.py ($work | path join guard) -- ./zig-out/bin/ds4-server ...$server_args out+err> ($work | path join server.log)
```

In a second terminal:

```nu
nu tests/mcp-tools.nu --suite qualify --output .workspaces/mcp-tools-reproduce/qualify
nu tests/mcp-tools.nu --suite scenarios --case all --arm all --repeats 2 --output .workspaces/mcp-tools-reproduce/scenarios
nu speed-bench/experiments/mcp-tools-20261025/exa-response.nu --output .workspaces/mcp-tools-reproduce/exa-response
```

The Python wire environment uses the existing SDK/WebSocket dependencies.
`scenarios/scenarios.json` retains all samples, actual call counts and raw handoff
text. Main reasoning remains native unless the user explicitly requests a
different mode. Do not compare a new model checkpoint, prompt or driver against
these numbers without recording that change.
