# Native agent compaction planning

Item 71 separates the existing two-phase text compaction plan from the
worker's model/session transaction. The retained C Plan control passed its
ordinary agent and suffix fixtures. The selected native planner passes its ordinary-agent and suffix fixtures
in ReleaseFast and Debug, as does the explicit all-C fallback. All three real-vocabulary arms and both C/native actual summary/continuation
checks pass. Interactive start times differed between the generated-summary
checks, so they are independent functionality observations. No speed gain is claimed.

## Actual boundary

The [shared Plan](../../../ds4_agent_compaction.h) owns a private summary-request
vector, a replacement transcript vector and the summary/tail geometry. The
[native planner](../../../zig/agent_compaction.zig) selects thresholds, complete
turn/image boundaries and summary budgets, constructs the private request,
and rebuilds summary plus retained tail. It does not evaluate tokens or mutate
the live session. C retains the private summary generation, live publication,
suffix-cache/exact decision, cancellation and rollback transaction. Borrowed
images stay borrowed; token-vector ownership transfers explicitly or is freed
with the plan. The header holds the actual existing policy constants once.

The control is built after introducing the real C Plan actions while leaving
the native planner flag off. Its Context, Bash and engine owners were already
qualified natively, so `-Dnative=true` does not yet mean native compaction.
The [control commands/logs](evidence/control/) record a 2.439 s cached target
build, a passing 5.017 s ordinary agent fixture and a passing 0.040 s suffix
fixture. These are CPU development observations, not a latency ablation.
The retained binaries are under
`.workspaces/zig-agent-compaction-20261022/control/bin`.

Root will run the selected native planner against the existing production-linked
agent/suffix fixtures and real-vocabulary additions, then the prepared actual
compaction/continuation screen and integrated archival/repeated-suffix task.
Only bounded small commands, diagnostic failures and outcomes will be retained;
model weights, cache chunks and production context are excluded. Root owns
serial compilation and all guarded model/GPU work. No replacement harness or
additional parser is introduced by this report.


## Native preparation and first CPU results

The native Plan built on its first attempt in 4.034 s. Three unused C adapter
warnings were annotated without changing planner behavior, then the clean
build completed in 1.812 s. The actual ReleaseFast agent fixture passed in
4.975 s and the mock/full-compaction suffix fixture in 0.039 s. The native
Debug target build passed in 1.780 s. All
[completed commands and raw logs](evidence/native/) are retained; the later Debug
fixture and all-C fallback results are recorded below. These are cached
build/CPU-fixture observations, not model latency comparisons.

The [published vocabulary loop](qualify_vocabulary.py) reuses the same three
serial actual-fixture arms, now selecting the retained `control`, `native` and
`native-debug` binary directories under
`.workspaces/zig-agent-compaction-20261022`. It creates a fresh `vocab/` output
there and fails visibly if that output already exists. Root runs it under the
existing pressure guard; it does not evaluate model tokens.

The prepared [compaction smoke driver](compact_smoke.py) reaches the existing
interactive `/compact` command through a standard-library PTY. Piping slash
text through the noninteractive driver would instead submit a user turn, so
this is a concrete UI path requirement rather than a new production option.
The driver reuses the [committed agent helper](../zig-combined-20261022/run_agent.py),
asks the model to remember fixed facts before disposable audit prose, invokes
one real private summary generation and asks for the facts afterward. It is
a separate two-turn compaction/continuation screen, not the six-turn
external-archive/repeated-suffix workflow. The published driver differs from
its frozen prepared version only in resolving `ROOT` from its checked-in path.
The completed guarded outcomes are recorded below.


## Completed CPU fixture matrix

The native Debug agent and suffix fixtures also passed. The explicit
`-Dnative=false` reference built and passed both fixtures in ReleaseFast and
Debug, selecting the C planner and the original C application owners.

| Selection | Agent fixture | Suffix/full-compaction mock | Result |
| --- | ---: | ---: | --- |
| Native Debug | 5.297 s | 0.199 s | Both exit 0 |
| All-C ReleaseFast | 5.011 s | 0.071 s | Both exit 0 |
| All-C Debug | 5.128 s | 0.222 s | Both exit 0 |

The [fallback commands/logs](evidence/fallback/) retain its 1.467/0.444 s cached
ReleaseFast/Debug builds, and [native logs](evidence/native/) retain the actual
selected planner's results. Together with the retained C Plan control and
native ReleaseFast, ten existing-fixture invocations passed. This is a count
of executions, not ten independent model tasks or a claim of complete branch
coverage. No new compiler error or runtime policy repair was required; only
the three old C test adapters were marked unused to clean warnings. The
real-vocabulary screen passed as recorded below; the generated-summary screen
is recorded separately below.


## Actual-vocabulary Plan validation

All three [actual-vocabulary fixture arms](evidence/vocabulary/result.json)
passed: C Plan control in 3.028 s, native ReleaseFast in 3.067 s and native Debug
in 3.243 s. This reuses `test-agent-live-context`, now including a production
Plan scenario whose retained tail begins inside an unfinished assistant
thinking span. It checks summary-request prefix preservation, resumed assistant
prefix insertion, exact retained-tail tokens, system-prefix preservation and
explicit vector cleanup. The session stays at position zero: no private
summary or model token is evaluated. Existing context-edit/control/budget
checks also remain active.

Each arm's stdout/stderr and actual binary SHA-256 are retained. The printed
system/edited token counts were 2398/2427, 2399/2428 and 2402/2431 respectively;
each invocation passes the same internal invariants. This is not a claim that
all three prompt vectors or generated outputs were compared
byte for byte. The [guard](evidence/vocabulary/guard/result.json) completed
with child return code zero; all
[ten pressure samples](evidence/vocabulary/guard/pressure-summary.json) were
normal with zero swap growth. Peak wired/compressed memory was 4.553/8.824 GiB.
The [selected ten-file snapshot](evidence/native/native-source.json) was
[verified unchanged](evidence/native/source-verification.json) during report
assembly. The later generated-summary outcomes are
recorded separately below rather than inferred from token/text validation.


## Actual `/compact`: retained C Plan control

The [C Plan control](evidence/smoke-control/result.json) completed the real
interactive workflow in 238.856 s with exit zero. It accepted two ordinary user
turns, one actual manual `/compact` command and no external live-context edit.
The trace records 5,459 tokens reduced to 3,365 while retaining an 819-token
tail. Its generated private summary preserved `SABLE-7319` and count 37; the
retained tail did not contain the code, so the later correct JSON answer came
from the generated summary rather than a verbatim surviving fact. The private
compaction request was absent from the new mirror.

The command used the 1017 IQ2_XXS/Q2_K model, 32 GB SSD expert-cache target,
context 8,192, chunk 512, temperature zero, no thinking and a 256-token ordinary
response cap. The compact prefix cache was enabled; suffix reuse and pre-RoPE
reuse were disabled to isolate the existing summary/rebuild path. The standard
PTY reached the existing UI command. A narrow sandbox rule denied writes only
to the user's `.ds4_agent_history`; the actual HOME and normal CLI flow stayed
in place, and EOF did not save a named session.

The [execution record](evidence/smoke-control/execution.json), first/final mirror
snapshots and fixture are retained with [standard-gzip trace](evidence/smoke-control/agent.trace.gz)
and [PTY output](evidence/smoke-control/pty.log.gz). Raw-file
[hashes and original workspace paths](evidence/smoke-control/artifacts.json)
identify the unmodified originals; no cache files are checked in. This is one
actual C control, not evidence of native compaction success or an end-to-end
speed comparison. The corresponding native arm also passed as recorded below.


## Actual `/compact`: selected native Plan

The [native workflow](evidence/smoke-native/result.json) passed in 237.301 s,
with two ordinary user turns, one actual manual compaction and no external
context edit. It reduced 5,459 tokens to 3,314, retaining the same 819-token
tail geometry. The generated summary contained `SABLE-7319` and count 37;
the retained tail did not contain the code. The final answer correctly returned
both facts, and the private summary request was absent from the new mirror.
The [execution record](evidence/smoke-native/execution.json), all small synthetic
snapshots and [raw-artifact hashes](evidence/smoke-native/artifacts.json) are
retained with [gzip trace](evidence/smoke-native/agent.trace.gz) and
[PTY output](evidence/smoke-native/pty.log.gz). Unmodified originals remain in
the recorded ignored workspace paths; cache files are excluded.

| Actual compaction selection | Whole workflow | Tokens before / after | Retained tail | Later facts |
| --- | ---: | ---: | ---: | --- |
| C Plan control | 238.856 s | 5,459 / 3,365 | 819 | Correct, from summary |
| Native Plan | 237.301 s | 5,459 / 3,314 | 819 | Correct, from summary |

These are independent functional checks. The
[input diagnostic](evidence/smoke-inputs.json) records actual interactive
start times of `2026-10-22 11:39:12 EDT` and `2026-10-22 11:43:45 EDT`.
Interactive code intentionally ignores `DS4_AGENT_TEST_TIME`; that override
requires noninteractive mode. Visible input text matches only after removing
display IDs and those actual start-time strings. The model therefore did not
receive identical prompts. The generated summaries differ, and neither that
difference nor the observed times is assigned causally to the port. There is
no speed claim or identical-output requirement here.

The [native guard](evidence/smoke-native/guard/result.json) completed with child
return code zero. Its [229 normal samples](evidence/smoke-native/guard/pressure-summary.json)
had zero swap growth; peak wired/compressed memory was 42.902/10.365 GiB.
The C-control guard likewise completed zero with 230 normal samples and zero
swap growth. Including the vocabulary campaign, all 469 sampled observations
were normal with zero swap growth.

To repeat one interactive arm, retain its selected binary and use a fresh
output under the existing guard:

```sh
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-agent-compact-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-agent-compaction-20261022/compact_smoke.py \
    --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
    --agent .workspaces/zig-agent-compaction-20261022/native/bin/ds4-agent \
    --output .workspaces/zig-agent-compact-rerun
```

This qualifies native two-phase compaction planning, real vocabulary/control
handling and actual summary-based continuation with the retained C worker
transaction. It does not claim a wholly native agent worker or the later
six-turn external-archive/repeated-suffix workflow; that integration screen
follows final resource qualification.
