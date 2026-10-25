# Output-first recipe — 2026-10-26

Status: all **18 requests** completed by **04:15 UTC**, within the
**03:33–04:33 UTC** experiment window. The primary twelve-request screen ran
**03:42–04:01 UTC**.
Runtime is frozen C `aef775c2`, executable
`.workspaces/planning-abc-20261026/c-build/bin/ds4-server`.
Primary client experiment change `wnpkmouz`, snapshot `061d1f5d`.
The API-only follow-up uses the separately retained script and exact suffix
linked below; the runtime stays frozen for all eighteen requests.

## Matched setup

Compare the existing PTC description with the same description plus the exact
[recipe](recipe.txt). No production/runtime changes, automatic output spill or
inline-cap field. Reuse the [planning tasks](../planning-abc-20261026/README.md),
source evidence, schemas and external oracle. Native DeepSeek reasoning remains
enabled; temperature zero, eight tool calls and 4,096 output tokens are unchanged.

The guarded M5 Max/128 GiB server uses Flash-1017 IQ2_XXS/Q2_K imatrix,
Clef-Flash INT4, a 32 GB SSD expert cache, context 16,384, prefill/mixed quantum
2,048, MTP draft 1 and two batched sessions. Clef prefix caching, DS4 decoder
replay and server context suffix reuse are disabled. All twelve requests
report zero cached tokens. Large existing swap allocation is not treated as
growth; the guard checks pressure and swap change.

Each case runs control/recipe/recipe/control, one request per CLI invocation.
These two observations per arm support a small practical screen, not a general
accuracy or statistically established speed claim.

## Twelve-request screen

| Case | Description | Times, s (two requests) | Programs / fetches / helpers | Emitted bytes | Numeric / JSON / grounding |
|---|---|---|---|---|---|
| Ordinary | Control | 56.819 / 57.351 | 0 / 0 / 0 | 0 | 2/2 / 2/2 / grounded |
| Ordinary | Recipe | 80.226 / 81.871 | 0 / 0 / 0 | 0 | 2/2 / 2/2 / grounded |
| Favorable | Control | 110.011 / 108.898 | 2 / 2 / 1 | 15,264 | 2/2 / 2/2 / grounded |
| Favorable | Recipe | 128.701 / 127.938 (failed) | 4 / 3 / 1 attempted | 10,127 | 0/2 / 0/2 / no answer |
| Unfamiliar | Control | 76.819 / 79.281 | 1 / 1 / 0 | 9,642 | 2/2 / 2/2 / grounded |
| Unfamiliar | Recipe | 106.161 / 104.489 | 2 / 2 / 0 | 11,719 | 2/2 / 2/2 / grounded |

Counts, token values and byte values repeat within each row. Generated tokens
include reasoning tokens. Helper attempts include the failed HTTP400 call;
the favorable recipe has no successful helper inference. Emitted bytes sum
model-visible program result strings. HTTP bytes measure the raw JSON response
body, including client-visible audit data, excluding headers.

| Case | Description | Main input | Generated (reasoning) | Full HTTP response bytes |
|---|---|---:|---:|---:|
| Ordinary | Control | 1,334 | 542 (465) | 3,284 |
| Ordinary | Recipe | 1,433 | 889 (791) | 3,910 |
| Favorable | Control | 4,557 | 750 (473) | 66,381 |
| Favorable | Recipe, failed | 3,266 | 1,149 (657) | 85,249 |
| Unfamiliar | Control | 3,524 | 520 (364) | 25,822 |
| Unfamiliar | Recipe | 4,082 | 683 (321) | 42,095 |

## Inspection and decision

The first paired request diff contains exactly the appended description plus
the fixture's ephemeral localhost port. Ordinary recipes still answer directly:
both requests use 1,433 input / 889 generated / 791 reasoning tokens, versus
1,334 / 542 / 465 for both controls. Both variants provide the correct IDs,
count, maximum and grounded two-sentence handoffs. The added latency here is
extra direct-answer reasoning/generation, not an executed tool workflow.
The median is 57.085 versus 81.048 seconds, a 42.0% increase for the recipe.

Both favorable controls reproduce C's inefficient but correct plan: emit the whole
MCP result, fetch it again, construct properly targeted questions with
`lib.questionsFor`, then emit all questions and the full helper result as a
JSON string. Helper labels are TP3/TN5 in each; final handoffs are grounded.
Control median is 109.454 seconds. Recipe attempts take a median 128.319 seconds
and fail; their time cannot be counted as faster completed work.

Both favorable recipe runs generate identical programs and logical results.
The first program still emits the full MCP result as a JSON string (9,356 bytes).
The second fetches again and hand-builds choice questions using string choices
instead of the objects required by the native API and retained in the bound
template: the helper returns HTTP400
`ExpectedObject`. A third fetch inspects the correctly constructed library
question; the next program reaches the shared call limit before its retry
fetch. Neither request produces a final answer or successful helper labels.
They use 3,266 input / 1,149 generated / 657 reasoning tokens. Their reduced
emission is not a success: the recipe induces no stash/load or reduction before
the bulk dump, and repair/refetch consumes the task budget.

Both unfamiliar recipe requests correctly adapt to the nested output: they follow
`index.dataset_key` and preserves `narrative.current` in its projection. However,
its first inspection emits two complete long tickets (2,448 bytes), and its
second program fetches again and emits all tickets' notes (9,271 bytes). The
main model then classifies directly, without a helper. The final fields and
two-sentence handoff are correct and grounded, but the result uses more data
and an extra program/fetch versus the control's single full-result dump. No
reference is stashed or loaded. Median latency rises from 78.050 to 105.325
seconds (+34.9%); emitted bytes rise 21.5% and main input tokens rise 15.8%.

Do not adopt this output-first suffix as a general serving default. Controls
finish 6/6 with valid JSON and grounded two-sentence handoffs; the recipe finishes
4/6, retaining the favorable failures, and slows both successful case types.
No request stashes or loads a reference. These results establish neither a
reference-cache benefit nor efficient automatic helper composition. They also
do not establish that the helper model lost task quality: its control labels
are correct, while recipe requests fail native argument validation.

The visible manifest previews choice labels as strings, while the bound
template contains choice objects. That is a plausible source of confusion,
not an established cause. The follow-up tests a three-sentence API-only suffix
instead of the long recipe, preserving this primary negative result and frozen
runtime. It changes instruction content; gains cannot be assigned solely to
adding types or removing prose.

The guarded server stops cleanly with exit0, maximum pressure level1, zero swap
growth and no adverse samples. All measurements and report work finish within
the one-hour window. Raw artifacts remain ignored under
`.workspaces/output-first-20261026/`; concise results and scripts are tracked.

## API-only follow-up

After the frozen screen, one final candidate starts **04:03:29 UTC** on the same
server: replace the long suffix with [three API sentences](api-recipe.txt), not
the long recipe plus more instructions. The suffix says to preserve the bound
template with `lib.questionsFor`, use object choices rather than strings, and
read the native result's `answers` array. Two requests per case retain the same
tasks, settings and eight-call budget. A paired request diff again contains only
the description suffix and ephemeral fixture port. No further variant is tested.

| Case | API-only times, s | Median versus original controls | Programs / fetches / helpers | Numeric / JSON / grounding |
|---|---|---|---|---|
| Ordinary | 64.483 / 63.579 | 64.031 versus 57.085 (+12.2%) | 0 / 0 / 0 | 2/2 / 2/2 / grounded |
| Favorable | 98.574 / 98.451 | 98.513 versus 109.454 (−10.0%) | 2 / 2 / 1 | 2/2 / 2/2 / grounded |
| Unfamiliar | 152.039 / 150.440 | 151.239 versus 78.050 (+93.8%) | 2 / 2 / 1 | 2/2 / 2/2 / grounded |

| Case | Main input | Generated (reasoning) | Emitted bytes | Full HTTP response bytes |
|---|---:|---:|---:|---:|
| Ordinary | 1,383 | 619 (536) | 0 | 3,514 |
| Favorable | 3,488 | 563 (294) | 11,083 | 62,084 |
| Unfamiliar | 5,639 | 1,087 (658) | 17,564 | 71,864 |

Ordinary still answers directly; all facts and the cautious two-sentence
handoff are grounded. Its extra cost is not tool execution.

Favorable succeeds twice with native TP3/TN5. All eight original notes are
present in the helper input and every question targets its own ID. Both
program sequences first emit the full fetched result (9,224 bytes), refetch,
then use `lib.questionsFor` and emit the native helper object (1,859 bytes).
The controls' second emission instead contains the full questions array plus
the helper wrapper inside a JSON string (6,040 bytes). Removing that duplication
and double encoding reduces main input 23.5%, emissions 27.4% and wall time 10.0%
on this fixed task. It retains probabilities, two fetches and two programs;
no reference is stashed or loaded. Helper evidence here is the full MCP wrapper
rather than its structuredContent alone (2,643 helper tokens versus 2,621),
without changing the correct labels. Final handoffs are grounded/two sentences.

Unfamiliar now invokes the helper correctly, with native TP3/TN5 in each run,
but does substantially more work than direct classification after one dump.
Its inspection emits two full long tickets (2,493 bytes); a second fetch maps
the nested tickets to rows and calls `lib.questionsFor`. That program emits
criteria, all rows, all questions and all answers as a JSON string (15,071
bytes). All eight original notes and charges survive exactly. The helper input
omits `payload.criteria` although the question says to apply the policy in the
input; labels nevertheless match all eight oracle labels on this fixture.
Keep that contract omission distinct from a demonstrated label failure or
reference corruption. Finals retain correct fields and grounded two-sentence
handoffs. No references or further reduced handoff are used.

These are sequential follow-up requests compared with earlier bracketed
controls, not a fresh randomized comparison. Content differs from the long
recipe as well as its length. Do not infer that the observed change is solely
caused by adding API types, removing prose, or cache reuse. All six API-only
requests complete correctly, without the long recipe's malformed native
argument, but the performance result is mixed: favorable improves 10.0%,
ordinary regresses 12.2% and unfamiliar regresses 93.8%. This supports documenting
the actual helper types; it does not justify advertising the suffix as a
general performance preset or efficient automatic planning. Decision: reject
the long workflow recipe and retain the factual API-only hint for selective
known-layout use, without making it a global default. Preserve both suffixes,
reproduction scripts and results on the separate local `output-first` bookmark;
normal serving remains the qualified C build and main is unchanged. There is
no automatic output spill in this experiment and no natural reference-cache
win to report.

## Reproduction

Run one guarded server (the binary is the preserved qualified C executable),
then run the maintained sweep in another terminal:

```nu
let work = '.workspaces/output-first-20261026/reproduction'
mkdir $work
$env.DS4_SERVER_RESPONSE_STORE_DIR = ($env.PWD | path join $work responses)
$env.DS4_LOCK_FILE = ($env.PWD | path join $work server.lock)
$env.DS4_CLEF_PREFIX_CACHE = '0'
$env.DS4_V41_DECODER_REPLAY_ROWS = '0'
$env.DS4_SERVER_CONTEXT_SUFFIX_REUSE = '0'
python3 speed-bench/ideas/pressure_run.py ($work | path join guard) -- .workspaces/planning-abc-20261026/c-build/bin/ds4-server --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --decision-model gguf/clef-flash-int4 --ssd-streaming --ssd-streaming-cache-experts 32GB --host 127.0.0.1 --port 19333 --ctx 16384 --prefill-chunk 2048 --mixed-prefill-quantum 2048 --tokens 4096 --mtp-draft 1 --batched-session 2
```

```nu
nu speed-bench/experiments/output-first-20261026/sweep.nu --output .workspaces/output-first-20261026/reproduction-screen
# Same server, API-only candidate (six requests after the primary screen):
nu speed-bench/experiments/output-first-20261026/api-followup.nu --output .workspaces/output-first-20261026/reproduction-api
```
