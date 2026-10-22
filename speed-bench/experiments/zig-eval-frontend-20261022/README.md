# Item113 — Complete handwritten Eval frontend

Qualified implementation checkpoint: `613c52ad` (main remains `ec5424de`).

The native `ds4-eval` now owns options, case selection, prompts, grading, trace
regrading, generation/retry, UI/input and main. Case literals and actual shared
records have one C definition; Engine/Session, sampling and transport are real
libraries. Small signal installation/default-reraise leaves remain C. The
independent original C frontend remains selectable. This qualifies the Eval
frontend, not every remaining backend owner or full evaluation-suite accuracy.

C ReleaseFast, native ReleaseFast and native Debug build and pass all 20 CPU
cases with identical return code/stdout/stderr (60 invocations). These include
the existing extractor self-tests, case validation/listing/filtering, help and
invalid/missing options. The initial native compile found three Zig name
collisions in Grading/UI; only those local identifiers changed before the clean
build. Build wall times are retained observations, not comparable clean-build
performance measurements.

Real Metal model qualification uses the 1017 IQ2_XXS/Q2_K imatrix target,
32GiB requested expert cache, SSD streaming, ctx8192/chunk512, temp0/seed1 and
Law case `6082513c8dba4ec68aa68f1bf5854d09`. Plain no-thinking runs repeat the
case with `--case-sequence 1,1`: C/native/Debug produce identical counted model
output bytes. Thinking C/native runs exercise the 16-token budget, soft8/hard4
forced thinking close and `--retry-incomplete`; both attempts (16 then32 tokens)
are byte-identical. All five fresh traces regrade identically through all three
binaries, including status/count/error output (15 regrade invocations).

These short Law runs are **INCOMPLETE and exit1**, not successful answers:
the no-thinking output uses all 32 tokens without reaching a final answer;
the thinking retries remain incomplete with `think_close: hard_forced`.
The evidence establishes matching generation/retry/grading behavior. It makes
no grade-success, task-quality or speedup claim. The first model driver treated
exit1 as a subprocess failure; that harness failure is retained, and the resumed
driver admits the frontend's documented graded/incomplete exit status.

The completed seven-arm real controlling-PTY replay passes: C/native/Debug
normal finish, native/Debug pause→resume→navigation→quit, and native/Debug
SIGINT. All enter and leave the alternate screen. Normal finish and key-driven
quit restore operational terminal attributes; the raw C/native/Debug delta is
the same Darwin `PENDIN` bit (`0x20000000`, pending input retype). Both SIGINT
arms return -2 and restore every recorded attribute. Raw deltas remain in
`pty-results.json`; no product change was made to suppress PENDIN. The PTY helper
keeps the controlling-session leader alive while recording terminal attributes.

Earlier no-controlling-terminal, revoked-terminal/EOF and strict-PENDIN harness
attempts are retained separately, with their failing logs/results/guards. They
do not replace the final seven-row replay or establish a product regression.
The final PTY pressure guard completes with child rc0, 46 samples at pressure
level1 and maximum swap0.009463GiB. Heavy model/PTY runs were
serial under the existing pressure guard; complete command, sample and outcome
records remain in evidence. Wall times are diagnostic observations only.

`evidence/cpu-results.json`, `model-results.json`, `pty-results.json` and
`qualification-summary.json` summarize completed observations. Per-command
records retain argv, exit status, elapsed time and raw outputs; counted model
blocks are in the traces. `final-source.json` hashes actual integrated files,
superseding the earlier source-candidate manifest; `binary-hashes.json` identifies
all three tested binaries. Logs larger than 1MB are gzip-compressed for publishing;
raw originals remain ignored. No model, binary, build cache or expert-cache files
are published. The retained drivers use repository/workspace paths and are
evidence of the actual campaign, rather than a standalone test framework.
