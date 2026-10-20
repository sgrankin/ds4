# Overthinking logit policy experiment

This experiment adapts the fixed hesitation-token penalty from [Quantized
Reasoning Models Think They Need to Think Longer, but They Do Not](https://arxiv.org/abs/2606.00206).
The paper evaluates dense 1.5B–32B reasoning models, AWQ/GPTQ 3/4-bit and
FlatQuant 4/8-bit, not DS4.1 Flash or this IQ2/Q2_K MoE recipe. Its gains do
not establish a DS4.1 result.

## Policy

```sh
DS4_OVERTHINK_PENALTY=0.5 ./ds4 -m MODEL.gguf --ssd-streaming \
  --think-level 75 --temp 0.6 --top-p 0.95 --min-p 0 -p 'YOUR PROMPT'
```

- The default is disabled. Zero is exactly disabled.
- A positive value subtracts that amount from raw logits for exact leading-space
  word tokens from the paper's 50-marker list, before temperature, greedy
  selection, top-k, top-p, and min-p. Split words are skipped to avoid suppressing
  an unrelated prefix. Startup reports the number mapped in the actual tokenizer.
- `DS4_OVERTHINK_SCOPE=think` is the default: the last `<think>`/`</think>` control
  token in the live checkpoint determines the gate. Final answers and tool
  payloads after `</think>` receive the ordinary distribution.
- `DS4_OVERTHINK_SCOPE=all` applies the policy at every step, closer to the paper.
  Think-only gating is an experimental adaptation.
- `DS4_OVERTHINK_AFTER=256` delays the policy until 256 newly generated thinking
  tokens in the current assistant turn. The default is 0, preserving the first
  experiment. Prefilled system/user/history/current-thought tokens and restored
  checkpoints do not spend the threshold. A new assistant marker resets the
  count; closing and reopening thinking in the same turn accumulates it. Control
  markers and final-answer tokens are excluded. This is another experimental
  adaptation, not the paper's fixed-at-every-step method. Think-only scope still
  prevents final-answer suppression. Explicit `scope=all` continues applying
  the penalty outside thinking after the current turn meets the threshold.
- `ds4`, `ds4-agent`, and `ds4-server` session sampling and session argmax APIs
  use the policy. A one-shot greedy CLI request uses the session implementation
  while the policy is enabled.
- The follow-up coverage fix routes direct `ds4_engine_generate_argmax` calls
  through ordinary session evaluation while enabled, including CPU and raw graph
  paths. `ds4_session_eval_argmax` also avoids GPU top-1-only shortcuts while the
  policy is active. The unchanged optimized shortcuts remain available by default.
- Speculative session APIs and batch speculation fall back to ordinary token
  evaluation while enabled. The pilot does not qualify a speculative sampler.
- The policy operates on a lazily allocated copied logit buffer. Forward passes,
  cached model state, `ds4_session_copy_logits`, token logprobs, and diagnostic
  top logprobs retain the raw model logits. Sampling has a small copy/arithmetic
  cost and no extra forward pass.
- `ds4_session_copy_sampling_logits` exposes adjusted raw logits before sampler
  temperature/filtering for policy NLL/KL. Raw-model NLL/KL must continue using
  `ds4_session_copy_logits`.
- The serial experiment API `ds4_engine_set_overthinking_penalty` changes the
  policy between cases without reloading the model. Do not change it concurrently
  with sampling on sessions of that engine.

The ordinary default sampler is T=1, p=1, min-p=.05. The paper uses T=.6,
p=.95; a raw penalty is stronger at the lower temperature. Before filtering,
a penalized token's odds relative to an unpenalized token are multiplied by
exp(-lambda/T).

## Validation

Built only `ds4`, `ds4-agent`, `ds4-server`, the CPU sampling test, and the pilot
with `make -j2`. The full CPU sampling suite passed. Added checks cover positive
penalty direction, greedy selection, application before temperature and candidate
filtering, agreement with the independent reference sampler and RNG over 4,608
cases, unchanged raw logits, zero-penalty behavior, and opening/closing/reopening
the reasoning gate. No model/GPU inference was used for these checks.
Small [build output](overthink-provenance/late-gate-build.log) and
[CPU sampling results](overthink-provenance/late-gate-cpu-tests.log) are tracked
for review; runtime traces, pressure logs, and prepared live manifests stay in
the ignored persistent artifact directory. Archived manifest copies in the
provenance directory identify whether a prepared run was unused or actually
executed; a manifest by itself is not evidence of execution.

## Generation pilot

```sh
make -j2 speed-bench/overthink_pilot
DS4_OVERTHINK_PENALTY=0.5 speed-bench/overthink_pilot \
  /ABS/PATH/DeepSeek-V4.1-Flash-Q2.gguf calibration-RUN_NAME 512 75 --cache-gb 64
```

The pilot loads one engine and reuses one session for four short verifiable
math/Python tasks, sweeping lambda=0/.5/1, with scope=think, T=.6, top-p=.95,
min-p=0, fixed seed 20261002, context 8192, and effort 75. Alternate cases reverse
arm order to reduce systematic cache favoritism. The token cap can be 512 or
1024 (up to 4096 is accepted). Every arm saves raw token text and one JSONL record
with final answer, expected answer, reasoning/total tokens, cap hit, reasoning
close, stop, prefill time, and decode time.

The expert cache is explicitly 64 GiB by default. `--cache-gb GIB` overrides
`DS4_PILOT_CACHE_GIB`; finite positive sizes, including 80 or 96 GiB, are accepted.
The runtime still reports its effective cache and full memory plan. Large cache
size alone is not a reason to reject a run; combined host pressure determines
whether the process can continue. Run names use letters, digits, underscores,
and hyphens. Output is always persistent under
`speed-bench/ideas/artifacts/RUN_NAME`, and existing output is never overwritten.
`--cases N` overrides `DS4_PILOT_CASE_LIMIT` and runs only the first N tasks of
the selected suite. Its default is the full suite. The limit is checked against
that suite before engine creation, and every result/manifest records the limit.

Use the shared model-job guard for every model command, including tokenizer-only
diagnostics. Prepare persistent binary/source/model-stat provenance before a run:

```sh
DS4_OVERTHINK_PENALTY=0.5 python3 speed-bench/prepare_overthink_run.py \
  hard-RUN_NAME /ABS/PATH/DeepSeek-V4.1-Flash-Q2.gguf --suite hard --candidate 1 \
  --cache-gb 64 --cases 2 --after 256 \
  --guard /Volumes/Models/ds4/speed-bench/ideas/pressure_run.py
```

The helper only writes a sibling manifest and prints the exact guarded command;
it never starts the model or reads the model tensor payload. Execute that command
from its recorded checkout after obtaining the serial model slot. The manifest
hashes the binary, source files, and guard, and records the revision, exact args,
cache, case count, late threshold, model path/size/mtime, and policy environment.
The printed command includes the environment assignments so its late threshold
survives copying the command. Guard pressure logs use a
separate persistent `RUN_NAME-pressure` directory.

After selecting a candidate from calibration, compare it with baseline on the
distinct eight-case held-out suite:

```sh
DS4_OVERTHINK_PENALTY=0.5 speed-bench/overthink_pilot \
  /ABS/PATH/DeepSeek-V4.1-Flash-Q2.gguf heldout-RUN_NAME 512 75 heldout 0.5 --cache-gb 64
```

The held-out tasks include correcting false multiplication/subtraction results,
rejecting an extraneous square-root solution, checking proposed integer factors,
Python's signed floor/remainder behavior, named `could`/`error` functions, and
exception handling that must print the exact phrase `could but error`. The final
phrase tests legitimate use of targeted language after the reasoning gate closes.
Omit the final lambda argument to sweep all three arms; that becomes another
exploratory sweep rather than a locked candidate evaluation.

The harder calibration suite contains six branching counting problems: circular
binary strings, constrained lattice paths with forbidden points, sorted integer
triples, digit arrangements with parity/divisibility constraints, primitive
integer triangles, and even permutations with separated fixed points. Its gold
answers are exhaustively enumerated by `speed-bench/overthink_gold.py`; the
circular, lattice, and permutation counts also agree with independent formulas.
Run the enumerator without a model before the generation pilot:

```sh
python3 speed-bench/overthink_gold.py
DS4_OVERTHINK_PENALTY=0.5 speed-bench/overthink_pilot \
  /ABS/PATH/DeepSeek-V4.1-Flash-Q2.gguf hard-RUN_NAME 1024 75 hard 1.0 --cache-gb 64
```

This compares the baseline with lambda=1 only. Calibration can inform the single
candidate locked before running the held-out suite. The runner accepts a 2048
cap if a repeated, separately recorded budget comparison is needed.

The strict scorer reads only text after a generated `</think>` and requires the
whole final answer to match the declared integer/fraction/list/printed-line
grammar. It accepts narrow enclosing Markdown/code/boxed wrappers, normalizes
integers and list whitespace, and preserves spaces in the literal text task.
It never accepts a matching substring. Every record distinguishes `CORRECT`,
`INCORRECT` (parsed, wrong value), and `INVALID` (missing/unparseable final), with
`parse_ok` and `parse_reason`. Any INVALID records cause exit status 3 after all
records are written. `--score-self-test` runs 17 parse/equality checks without loading a
model. Raw traces must be checked before calling unexpected-but-correct formatting
a reasoning failure.

A cap hit without a final answer is a failure under the stated budget, not
evidence of permanent inability to solve the task. This tiny pilot and its partly
warm expert cache cannot establish a quality-neutral speedup; promising settings
require more held-out cases and balanced timing repeats.

## 2026-10-02 calibration result

Native Metal run at effort 75, cap 512, context 8192 completed all 12 arms.
The actual tokenizer mapped 46/50 markers, skipping `Hmm`, `hmm`, `retry`, and
`recheck`. Every arm produced a parsed correct final answer with no cap hits.

| Lambda | Correct | Total tokens | Reasoning tokens | Cap hits |
| --- | --- | --- | --- | --- |
| 0 | 4/4 | 315 | 301 | 0 |
| .5 | 4/4 | 315 | 301 | 0 |
| 1 | 4/4 | 315 | 301 | 0 |

The full traces are byte-identical across all three lambdas for each problem.
There is no reasoning-length improvement on this calibration set. Aggregate
decode wall times were 15.905s, 12.823s, and 13.674s, respectively, but the expert
cache starts colder on a task's first arm. These timings do not establish a
policy speedup. The generated reasoning contains little hesitation; harder
reasoning cases are needed to test the paper's claimed failure mode.

Raw traces, JSONL records, full startup/run log, and SHA-256 trace hashes are
preserved in [the calibration evidence directory](overthink-calibration-20261002-1540/summary.json).
The sandboxed launch could not access Metal; the successful run used the
authorized native execution context.

## Interrupted hard run and recovery

The cap-1024 hard run used frozen source `ac39f789` and the previous automatic
cache setting: 79.51 GiB dynamic experts plus 7.12 GiB prefill headroom, with a
97.14 GiB full planned footprint. A watchdog kernel panic reboot interrupted it
at 16:02:59 UTC. Root's panic inspection reported an 80.263 GiB pilot RSS,
93.4 GiB total wired memory, 33.3 GiB compressed memory, and only 14 MiB free.
Two concurrently launched tokenizer diagnostics also appeared in that snapshot.
Their command path was intended to avoid tensor prefetch, but their observed
page-in counts were large. The exact contribution of those diagnostics is not
established; combined host pressure was exhausted. They should have been
serialized with the model process.

The completed easy calibration was committed and survived recovery. Hard raw
traces and partial JSONL were stored under `/tmp` and were lost at reboot. Before
interruption, the logged circular count was correct in both arms (415 baseline
tokens versus 389 at lambda=1). The lattice baseline was correct at 941 tokens
while lambda=1 capped at 1024 without a final answer. Primitive-triple, digit,
and triangle arms each capped in both settings. These are observed partial logs,
not a recoverable completed evaluation or evidence of a quality-neutral gain.
The recovered source now has explicit configurable cache sizing and persistent
artifact/provenance paths. No post-reboot inference was run for this recovery
change.

The separate late-gate child adds CPU tests for a 500-token history prefix,
prefilled thoughts, the 255/256 threshold, closed/final/reopened thinking, a new
assistant turn, and restored input. Its first planned generation comparison is
the first two hard tasks (circular and lattice), with the same baseline/candidate
seed and budget, to compare immediate and delayed intervention. This is fresh
evaluation; it cannot reconstruct the traces lost in the reboot.

## Fresh guarded two-case comparison

Both runs used frozen source `86bfd7a2`, the same executable/seed, effort 75,
cap 1024, and an explicit 64 GiB total SSD-cache budget (56.88 GiB dynamic
experts plus 7.12 GiB reserve; 74.51 GiB full planned footprint).

| Setting | Correct final answers | Total tokens | Cap hits |
| --- | --- | --- | --- |
| Baseline, either after setting | 2/2 | 1356 | 0 |
| Lambda 1, immediate | 1/2 | 1413 | 1 |
| Lambda 1, after 256 | 1/2 | 1439 | 1 |

Circular count: baseline 415 tokens, immediate 389, delayed 415, all correct.
The delayed circular trace is byte-identical to baseline. Lattice count:
baseline 941 correct tokens, both candidates capped at 1024 without closing
thinking or producing a final answer. The lambda-zero traces are byte-identical
across the immediate and delayed runs for both tasks, with no observed baseline
drift. Both tasks generated more than 256 thinking tokens, so the late gate was
applicable; the circular trace's equality does not mean the distribution policy
was inactive.

Both guards completed normally with child exit code 3 for the single scored
INVALID cap in each run. Peak wired memory stayed below 72.56 GiB, compression
below 6.66 GiB, swap unchanged at 0.038 GiB, and kernel pressure level at 1.
No pressure stop occurred. These arms do not show a quality-neutral reasoning
reduction and do not support enabling the policy by default. Startup/cache
order makes their wall times exploratory.

The [tracked compact summary](overthink-provenance/guarded-hard-two-summary.json)
includes exact configuration, trace hashes, active-gate applicability, strict
scorer outcomes, and pressure peaks. Full traces/results/logs remain under the
ignored persistent artifact directories named in that summary. The older
manifest names without `v2` were prepared against an earlier guard and were
never executed.

An optional immediate lambda=.5 comparison was prepared with the same frozen
binary/seed/cache/budget, but no serial model slot was granted for it. It has no
results or pressure directory and is explicitly
[PREPARED_UNEXECUTED](overthink-provenance/optional-half-status.json). Its
manifest is provenance for a proposed run, not measured evidence.
