# Server code mode and decisions — 2026-10-25

Server-side execution, bound evidence/templates and native decision calls work.
On this small fixed screen, the final generic API documentation lets Flint use
them without a user instruction naming a tool or supplying code. The favorable
24-record task finishes **22.8% sooner** than main-only, with correct facts, but
duplicates the final JSON. The four-record task is **2.20 times slower** because
the model delegates work it could answer directly.
With DeepSeek's native reasoning retained, the matched favorable task finishes
**32.5% sooner** through code mode: 86.366 seconds versus 127.862 direct. Both
arms complete with correct facts, clean JSON and grounded handoffs. An earlier
run silently disabled reasoning; its opposite result is retained as historical
evidence, not the result for ordinary DeepSeek use.

This is two repetitions of each fixed case, not a broad benchmark or an estimate
of general task accuracy. Source is jj change `olqomykynxyr`, parent `944863ed`.
Stages below are preserved separately under ignored
`.workspaces/ptc-decisions-20261025/`; no raw responses or logs are committed.

## Setup and comparison

M5 Max, 128 GiB; native Flint INT4 plus `gguf/clef-flash-int4` in one process.
Flint used the existing sibling bundle
`../flint/.local/share/rec/models/fdc85248ad77a0fa9fbf0abda8b1525bea8ee74c11ada2ec95eb66113353d677`.
The server used localhost:19333, context 16,384, prefill chunk 4,096, two resident
sessions and default output limit 4,096. Every request used temperature 0,
`max_output_tokens:4096` and `max_tool_calls:8`. Helper prefix caching was disabled
with `DS4_CLEF_PREFIX_CACHE=0`; `DS4_CLEF_TIMING=1` recorded helper stages.
One loaded model job ran under the existing pressure guard. Model loading is
outside the reported request wall time; profiling that synchronizes individual
kernels was not enabled.

[fixtures.nu](fixtures.nu) supplies the policy, records and separate gold oracle.
The ordinary case has four short records, two active account takeovers, and full
evidence visible in all arms. The favorable case has 24 longer records, six
positives and resolved/hypothetical/training/billing/software decoys. Main-only
reads all notes; hosted/code arms receive only the policy and ID/charge manifest,
while full notes remain bound to the server. Thus the favorable comparison
measures **data placement plus composition**, not voluntary delegation over equal
visible context or an isolated JavaScript-runtime effect.

All arms share the same natural task and generic tool-use policy. The final
candidate changes generic API documentation, not the user task, oracle or budget.
Its record/template example contains no ticket IDs or expected answer. Arm order
is reversed on the second repetition.

The first ordinary forward pass overlapped parallel CPU fixture compilation,
which ended at 14:39:17 UTC, and was excluded from final ordinary medians.
Its raw records remain in `wire/scenarios.json`; the clean
forward pass in `ordinary-clean/scenarios.json` replaces repeat0, while original
reverse repeat1 is retained. Numerical regrading treats ticket IDs as unordered;
the task never requested sorted IDs, and the original list-order check had
incorrectly marked the favorable main-only answer wrong.

## Final useful comparison

Wall times are medians of two observations. Input/output counts are main-model
tokens, including reasoning and internal tool-result continuation.

| Case and arm | Wall | Main input/output | Correct IDs/count/max | Clean JSON | Program/native calls per response |
|---|---:|---:|---:|---:|---:|
| Ordinary main-only | 5.480s | 388 / 1,225 | 2/2 | 2/2 | 0/0 |
| Ordinary final code + helper | 12.044s | 1,033 / 2,401 | 2/2 | 2/2 | 1/1 |
| Favorable main-only | 17.851s | 4,745 / 3,353 | 2/2 | 2/2 | 0/0 |
| Favorable final code + helper | 13.777s | 1,375 / 2,253 | 2/2 | 0/2 | 1/1 |

Favorable main input falls 71.0% and output 32.8%; native helper inference takes
2.579/2.528 seconds and consumes 7,413 helper input tokens. This still produces
more total tokens across both models; token counts across different models and
prefill/decode paths are not interchangeable performance units.

The generated program accesses `inputs.queue` and `schemas.triage`, spreads the
template, appends the target record ID to each question's instructions and makes
one native request for 24 questions. It emits all 24 ID/choice pairs, approximately
995 bytes, so the main model performs final filtering and the maximum calculation
using its manifest. It does not yet compute the whole final summary inside JS.
Full notes and probability vectors remain outside main-model context.

The HTTP audit remains large: 136,163 response bytes versus 15,806 for favorable
main-only, about 8.6 times as much. Saving generator work does not imply reducing
transport or durable audit size.

Ordinary delegation adds about 6.56 seconds. Its helper takes only 308/320 ms;
main generation grows from 1,225 to 2,401 tokens. Availability and better API usage
therefore do not imply that the model selects a worthwhile delegation boundary.

## Retained baseline and iteration

| Stage/case | Wall | Outcome and actual behavior |
|---|---:|---|
| Initial ordinary hosted | 7.093s | Correct fields; no tool call; duplicated final JSON |
| Initial ordinary code availability | 17.121s | Correct fields; no program/native call; duplicated final JSON |
| Initial favorable hosted | 22.559s | No final answer; 4,096-token limit; one native call with incorrectly targeted questions |
| Initial favorable code | 20.130s | No final answer; 4,096-token limit; two programs/two native attempts, neither native call succeeds |
| A: native errors become JS exceptions | 20.087s | No final answer; four programs/four visible native errors, no native inference succeeds |
| B: generic bound-library/API examples | 13.777s | One working program/native call; correct fields and handoff; duplicated final JSON |

Initial hosted questions used each ticket ID only as `name`, while instructions
were the same global predicate. OpenAI question names are output labels; this
implementation assigns internal IDs `q0`, `q1`, etc. A record selector must occur
in the instructions. That failure is question construction, not evidence that
the helper misclassified 24 properly targeted records.

Initial code rebuilt the visible manifest, invented a question shape and passed
an object where native input requires text/parts. It also used `return` instead
of `text()`. Returning a native error object without emitting it left an empty
successful program result. Stage A fixes that feedback: errors are visible and
the model successively corrects input serialization and question types, but it
still fails to use the bound objects before exhausting its generation budget.

Stage B documents actual bound-object access, template spreading,
`JSON.stringify` and `text()` with a generic example. The same natural task then
uses the library correctly. This is conditional on understandable tool
documentation; the initial failure remains part of the result.

## Transcript and quality review

Final favorable native labels are TP6, TN18, FP0, FN0 in each repeat; ordinary
labels are TP2, TN2, FP0, FN0. These are repeats of the same synthetic records,
not 48 independent classification examples. All final IDs, counts and maximum
charges match the external oracle.

Each completed answer's handoff field contains two appropriate sentences and
does not claim that locks, refunds or other external actions occurred. The final
favorable raw response repeats its JSON object across `</think>`, so it violates
the clean-JSON requirement even though the object fields and handoff are correct.
No output sanitizer was added. No bulk notes or probability dumps appear in the
main answer; those are retained in the audit instead. The initial hosted/code
favorable runs have no final user-facing handoff at all.

Final transport/replay qualification passes 56 checks, plus two real restart
checks. Those establish execution and lifecycle behavior separately from the
natural-task quality/performance observations above.

## Matched DeepSeek with native reasoning

The server binary is frozen at source `4cf6eacf`. The corrected client driver
omits `reasoning`, `reasoning_effort` and any `ds4` reasoning budget. Native
DeepSeek reasoning defaults to high, confirmed by the server trace and nonzero
reasoning-token usage. Both arms keep temperature zero and the same 4,096-token
output budget, which includes reasoning. No separate reasoning cap was added.
Future DeepSeek comparisons retain native reasoning unless the user explicitly
requests a different mode; the driver no longer silently forces none.

The fixture, generic tool documentation, binary and launch settings stay fixed
across sequential ABBA requests. A is main-only; B is code mode. Raw artifacts
are in `ds4-reasoning-on/` beneath the ignored work directory.

| Order | Arm | Wall seconds | Main input/output | Of output: reasoning | Cached tokens |
|---|---|---:|---:|---:|---:|
| A0 | Main-only | 127.380 | 4,767 / 1,078 | 948 | 0 |
| B0 | Code mode | 86.060 | 1,813 / 641 | 392 | 0 |
| B1 | Code mode | 86.673 | 1,813 / 641 | 392 | 0 |
| A1 | Main-only | 128.344 | 4,767 / 1,078 | 948 | 0 |

Median wall time falls from **127.862 to 86.366 seconds**, saving 41.496 seconds
or **32.5%**. Main input falls 62.0%, output 40.5% and reasoning tokens 58.6%.
Each B response uses one program and one successful native call; helper time is
3.189/3.395 seconds for 7,413 helper input tokens. No program error or retry
occurs. Both arms complete comfortably within the shared output budget; none
is marked incomplete or loses its final answer to reasoning-budget exhaustion.

The program batches 24 targeted questions using the bound queue and template,
then emits all 24 labels and confidences (1,565 bytes). It does not filter and
aggregate the result inside JavaScript; the main model calculates the count and
maximum using the visible charge manifest. Native labels are TP6/TN18/FP0/FN0
in each repeat. All four final answers have the correct six IDs, count six and
maximum 119,900 cents, clean JSON and exactly two grounded handoff sentences.
Unlike the earlier reasoning-disabled answers, they do not assert unsupported
refund or containment status. The B handoff says those actions have not been
confirmed here, without claiming they never occurred.

The full HTTP response remains larger: 127,972 bytes through code mode versus
1,652 direct, including bound-data/native-call audit. The favorable data-placement
inequality described above still applies. These are two repeats of one fixed
synthetic task, not a general DeepSeek or hardware performance claim. The guard
stayed at normal pressure with zero swap growth; model work ran sequentially.

## Earlier DeepSeek qualification

DeepSeek V4.1 Flash 1017 passes all eight explicit-code smoke checks. One natural
favorable task also completes with correct IDs/count/maximum and clean JSON:
76.385 seconds, 1,936 main input tokens, 458 main output tokens, two programs and
one successful native call. The helper consumes 7,413 tokens in 3.065 seconds
and again gives TP6/TN18/FP0/FN0. This earlier unmatched observation remains a
model-path qualification with reasoning disabled; it is excluded from both
matched comparisons.

Its first program classifies the queue and emits choices/confidence; its second
uses the returned IDs to calculate the maximum and construct the handoff.
The two-sentence handoff contains a weaker-grounded assertion: "No account has
yet been locked and no refund has been issued." In particular, the source does
not establish refund status for every ticket. Retain that semantic overstatement
separately from correct numeric facts and clean JSON.

## Historical DeepSeek run with reasoning disabled

Frozen source `4cf6eacf`, same M5 Max, fixture, binary and launch settings across
all four requests. A is main-only; B is code mode. Requests ran sequentially in
ABBA order with `reasoning:{effort:"none"}` inserted by the old client driver in
both arms. This was an unintended benchmark policy, corrected for the native-
reasoning comparison above. The effective mode was none, temperature was zero,
and the output budget remained 4,096. Raw artifacts remain in `ds4-matched/`.

| Order | Arm | Wall seconds | Main input/output | Main cached tokens |
|---|---|---:|---:|---:|
| A0 | Main-only | 53.277 | 4,742 / 100 | 0 |
| B0 | Code mode | 78.263 | 1,936 / 458 | 0 |
| B1 | Code mode | 77.923 | 1,936 / 458 | 0 |
| A1 | Main-only | 58.188 | 4,742 / 100 | 0 |

The medians are **55.732 seconds direct versus 78.093 seconds through code**:
22.361 seconds, or **40.1% more time**, despite 59.2% fewer main input tokens.
Each B response uses two programs and one successful native helper call, with
7,413 helper tokens and 3.079/3.476 seconds of helper time. The generated output
is 4.58 times the direct path. Additional code generation and continuations
outweigh the input savings on this fixture; exact attribution among those costs
was not profiled. This measures the reasoning-disabled setting only and does
not describe ordinary DeepSeek operation.

All four responses complete with clean JSON and correct ID sets, count six and
maximum 119,900 cents. Both B native calls give TP6/TN18/FP0/FN0. Every handoff
field has two sentences, but both direct and code answers assert that no refund
has been issued, which the fixture does not establish. This semantic caveat
therefore occurs in both arms. Code mode's first program emits labels/confidence;
its second calculates the maximum and constructs the handoff. The full response
is 157,327 bytes versus 1,586 direct, including bound-data/native-call audit.

No code or prompt was tuned between A0/B0/B1/A1. The guard stayed at normal
pressure with zero swap growth and exited cleanly. Four observations of one
fixed task do not establish a hardware-wide or model-wide performance rule.

## Reproduce and inspect

From the repository root, download the canonical aliases if needed, build, then
launch one guarded server with the measured settings. Choose a fresh work path
for each run because the pressure guard preserves its existing logs.

```nu
nu download-model.nu flint
nu download-model.nu clef-flash
nu tools/zig-build ds4-server -j3
let work = '.workspaces/ptc-decisions-reproduce'
mkdir $work ($work | path join responses)
$env.DS4_CLEF_PREFIX_CACHE = '0'
$env.DS4_CLEF_TIMING = '1'
$env.DS4_SERVER_RESPONSE_STORE_DIR = ($env.PWD | path join $work responses)
$env.DS4_LOCK_FILE = ($env.PWD | path join $work server.lock)
let server_args = [
    --model gguf/flint-int4 --decision-model gguf/clef-flash-int4
    --host 127.0.0.1 --port 19333 --ctx 16384 --tokens 4096
    --prefill-chunk 4096 --batched-session 2
]
^python3 speed-bench/ideas/pressure_run.py ($work | path join guard) -- ./zig-out/bin/ds4-server ...$server_args out+err> ($work | path join server.log)
```

In a second terminal, run the wire scenarios against that server. The drivers
do not start a model. [README.md](README.md) explains the fixture and comparisons.

```nu
nu tests/programmatic-decisions.nu --suite qualify
nu tests/programmatic-decisions.nu --suite scenarios --repeats 2
nu tests/programmatic-decisions.nu --suite scenarios --case ordinary --repeats 1 --output .workspaces/ptc-decisions/ordinary-clean
nu tests/programmatic-decisions.nu --suite scenarios --case favorable --arm ptc --repeats 2 --output .workspaces/ptc-decisions/affordance
nu tests/programmatic-decisions.nu --suite restart
```

The current source contains stage B. Reproducing rejected A/initial stages needs
their saved source/binary; changing only an output directory does not select an
older implementation. [summarize.py](summarize.py) produces separate numerical
summaries without modifying raw measurements; `--ordinary-clean` applies the
documented replacement. Add `--plot` only after final model timing has stopped.

For the matched DeepSeek launch, use a fresh directory and the same frozen
binary. This text-only run loads no vision encoder:

```nu
let work = '.workspaces/ptc-ds4-matched-reproduce'
mkdir $work ($work | path join responses)
$env.DS4_CLEF_PREFIX_CACHE = '0'
$env.DS4_V41_DECODER_REPLAY_ROWS = '0'
$env.DS4_SERVER_CONTEXT_SUFFIX_REUSE = '0'
$env.DS4_SERVER_RESPONSE_STORE_DIR = ($env.PWD | path join $work responses)
$env.DS4_LOCK_FILE = ($env.PWD | path join $work server.lock)
let server_args = [
    --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
    --decision-model gguf/clef-flash-int4 --ssd-streaming
    --ssd-streaming-cache-experts 32GB --host 127.0.0.1 --port 19333
    --ctx 16384 --tokens 4096 --prefill-chunk 2048
    --mixed-prefill-quantum 2048 --mtp-draft 1 --batched-session 2
    --trace ($work | path join trace.log)
]
^python3 speed-bench/ideas/pressure_run.py ($work | path join guard) -- ./zig-out/bin/ds4-server ...$server_args out+err> ($work | path join server.log)
```

Run the following in a second terminal. The maintained driver preserves native
DeepSeek reasoning by omitting overrides and keeps each sample separately.
The historical reasoning-disabled result requires its old driver/request files;
the commands below reproduce the native-reasoning comparison:

```nu
let root = '.workspaces/ptc-ds4-matched-reproduce'
let runs = [[name arm]; [a0-main main_only] [b0-ptc ptc]
    [b1-ptc ptc] [a1-main main_only]]
for run in $runs {
    nu tests/programmatic-decisions.nu --model deepseek-v4.1-flash --suite scenarios --case favorable --arm $run.arm --repeats 1 --output ($root | path join $run.name)
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
}
```
