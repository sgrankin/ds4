# Native agent runtime owners

This report records the separate Context and Bash ownership slices. Native
Context ReleaseFast/Debug CPU fixtures and all three real-vocabulary arms pass.
Native Bash also passes the actual-process agent and suffix fixtures in both
build modes. The integrated six-turn task remains pending compaction
qualification; these checkpoints do not claim that later result.

The retained C Context control and ordinary agent fixture passed; their first
failures and repairs are preserved below.

## Retained C Context control

The control remains the C live-context implementation with the previously
qualified native engine/support modules. A prepared native Context object was
compiled while constructing these targets, but the C caller guard was not yet
selected; successful control runs are not native Context qualification.

The initial [build attempt](evidence/context-control/context-control-build.stderr)
found a missing generated `getentropy` declaration and an explicitly discarded
error set in the prepared native object. The header/import and error handling
were corrected before the control fixtures were run. Build attempts and raw
logs remain under [evidence/context-control/](evidence/context-control/).

| Control observation | Whole process | Result |
| --- | ---: | --- |
| Initial target build | 4.152 s | Compile errors retained |
| Corrected target build | 0.762 s | Exit 0 |
| Ordinary `ds4_agent_test` | 5.008 s | Exit 0 |
| Initial suffix mock | 0.034 s | SIGSEGV retained |
| Rebuild after mock repair | 1.039 s | Exit 0 |
| Fresh-directory fixed suffix mock | 0.040 s | Exit 0 |

The suffix mock originally lacked `ds4_session_is_decoder_replay`. Its fake
session therefore reached the actual production predicate, which interpreted
an incompatible fake record and faulted. The
[fresh-directory LLDB backtrace](evidence/context-control/context-control-suffix-lldb2.stdout)
identifies that exact call. Supplying the missing mock predicate, returning
false because this fake session has no replay state, fixed the fixture without
changing production replay policy. The
[passing fresh-directory run](evidence/context-control/context-control-suffix-test2.json)
and its [full output](evidence/context-control/context-control-suffix-test2.stdout)
are retained. It covers transaction/ownership, repeat suffix selection,
retained-tail compaction, stable opaque turn labels, literal controls and
cancellation/budget boundaries through the existing mock scenario.

An earlier [LLDB attempt](evidence/context-control/context-control-suffix-lldb.stdout)
reused a directory containing prior cache state and hit an earlier fixture
assertion. That failed diagnostic is also retained; it is not presented as the
cause of the fresh-directory segmentation fault or as a passing test. Future
runs use fresh output directories. These cached build and CPU-fixture times
are diagnostics, not a runtime performance comparison. No model/GPU job was
run by this report author.


## Item 68: native Context CPU qualification

The selected `DS4_NATIVE_AGENT_CONTEXT=1` path now owns live-context rendering,
body-label quoting, token-text/control validation and staged prepare/commit
transactions through the actual [Context records](../../../src/ds4_agent_context.h).
The [native owner](../../../src/agent_context.zig) contains ordinary Zig APIs;
its [C facade](../../../src/agent_context_entry.zig) is the boundary used by the
existing C worker. Only a successful caller-side graft/exact decision commits
staged transcript/turn owners. Graph/session policy and suffix selection stay
with the worker. Display labels do not enter model tokens.

The caller macro was selected only after the fixed C control passed. The
existing ordinary agent fixture and fresh-directory model-free suffix fixture
then passed with the actual native Context in ReleaseFast and Debug:

| Native Context observation | Whole process | Result |
| --- | ---: | --- |
| ReleaseFast agent/fixture build | 3.203 s | Exit 0 |
| Debug agent/fixture build | 1.791 s | Exit 0 |
| ReleaseFast `ds4_agent_test` | 4.998 s | Exit 0 |
| Debug `ds4_agent_test` | 5.327 s | Exit 0 |
| ReleaseFast suffix fixture | 0.046 s | Exit 0 |
| Debug suffix fixture | 0.201 s | Exit 0 |

The [six recorded commands](evidence/context-native/) and raw output retain
these actual executions. The ReleaseFast build's unused C
`agent_live_context_body_text` adapter warning is preserved; it is not a
fixture failure. The adapter was later marked unused during Bash preparation,
and the later builds are warning-clean. Bash remained the original C implementation in these Context-stage agent
runs. Timings are single CPU-fixture/cached-build observations, not a C/native
performance claim. The serial real-vocabulary C/native/Debug screen passed as recorded below.
The later integrated six-turn agent workflow will be recorded after
Bash/compaction activation; it is not claimed by this Context checkpoint.


## Item 68: actual-vocabulary qualification

The existing `test-agent-live-context` fixture passed against the actual 1017
IQ2_XXS/Q2_K model vocabulary in all three serial arms:

| Selected Context | Whole fixture | Result |
| --- | ---: | --- |
| Retained C control | 3.178 s | Exit 0 |
| Native ReleaseFast | 2.946 s | Exit 0 |
| Native Debug | 3.180 s | Exit 0 |

The [actual commands, binary hashes and results](evidence/context-vocabulary/result.json)
and each arm's raw output are retained. Every run checked unchanged/deleted
context, literal assistant/DSML prefix and suffix preservation, UTF-8 editing,
byte round-trip, separated-edit ambiguity rejection, protected prefix, NUL,
missing-file and budget behavior. The fixture printed `no token evaluation`:
this is real-vocabulary token/text validation, not inference or generated-answer
quality. Engine/Metal setup and static buffers were initialized, but no model
token was evaluated. The reported system/edited lengths were 2397/2426 for
C and native ReleaseFast, and 2400/2429 for Debug; each invocation independently
passed the same internal invariants. No cross-run generated-output identity or
speed claim follows from these three ordered runs.

The original [executed driver](evidence/context-vocabulary/qualify_vocab.py) is
retained unchanged. The small published [qualifier](qualify_vocabulary.py)
only resolves the repository from its own path and selects a fresh ignored
output folder; its fixture loop and arguments are unchanged. After retaining
these three binary prefixes, reproduce under the existing guard:

```sh
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-agent-context-vocab-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-agent-runtime-20261022/qualify_vocabulary.py
```

The fresh default output is `.workspaces/zig-agent-context-vocabulary-rerun`.
The [guard](evidence/context-vocabulary/guard/result.json) completed with child
return code zero. All [ten pressure samples](evidence/context-vocabulary/guard/pressure-summary.json)
were normal with zero swap growth; peak wired/compressed memory was
6.198/9.408 GiB. The selected [ten-file source snapshot](evidence/context-native/context-source.json)
was [verified unchanged](evidence/context-native/source-verification.json) at
the Context68 checkpoint, before later Bash integration changed shared files. Useful fixture workspaces and model files stay ignored;
only small commands, diagnostics and outcomes are retained here.


## Item 69: native Bash CPU qualification

The selected native [Bash owner](../../../src/agent_bash.zig) now owns the shell
job's monitor thread, process group, child reaping, descriptors, timeout checks
and bounded output observation. C retains list nodes/IDs and the UI callback.
The shared [immutable snapshot](../../../src/ds4_agent_bash.h) is copied under the
job mutex; disk reads happen after releasing that lock. This exposes coherent
observations without duplicating the job/worker implementation layout.

The original process-fixture commands and assertions are unchanged; tests now
observe whole snapshots through the actual API. The ordinary agent fixture
executes real child processes rather than a replacement subprocess mock. The
existing suffix fixture also passed with the combined native Context/Bash
selection.

| Native Bash observation | Whole process | Result |
| --- | ---: | --- |
| Initial ReleaseFast build | 2.958 s | Two compile errors, retained |
| Corrected ReleaseFast agent/fixture build | 0.641 s | Exit 0 |
| Debug agent/fixture build | 1.650 s | Exit 0 |
| ReleaseFast actual-process agent fixture | 4.984 s | Exit 0 |
| Debug actual-process agent fixture | 5.308 s | Exit 0 |
| ReleaseFast suffix fixture | 0.041 s | Exit 0 |
| Debug suffix fixture | 0.198 s | Exit 0 |

The first [compile log](evidence/bash-native/bash-build.stderr) reported a C
command-pointer versus empty-string type inference conflict and an inferred
`u16` length where `usize` was needed. Explicit local pointer/length types
fixed both. Removed `dupeZ` calls were changed to standard `dupeSentinel` before
that attempt. No process scheduling or shell policy was changed to make the
fixtures pass. The successful later builds are warning-clean.

All [commands and raw output](evidence/bash-native/), including the failed
attempt, are retained. The [eight-file snapshot](evidence/bash-native/bash-source.json)
was [verified unchanged](evidence/bash-native/source-verification.json) at
this Bash checkpoint. Earlier Context snapshots continue to describe that
stage, rather than asserting that later shared C/build files never change.
These are CPU ownership/behavior checks and cached development-build times,
not a model-task or performance result. The integrated six-turn archival and
repeated-suffix workflow remains pending the later compaction checkpoint.
