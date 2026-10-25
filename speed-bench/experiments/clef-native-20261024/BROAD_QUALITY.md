# Broader native quality qualification

M5 Max, 128 GB, 2026-10-25 UTC. These are sampled task checks, not a full
Decision Index score or public vision index. Inputs and reproduction recipes
are in [STAGING.md](STAGING.md). Raw responses, distributions, source rows,
server logs and pressure records remain local under
`.workspaces/clef-quality-broad-20261025/`.

One server loads all three Clef models, context 16384, with one request at a
time. The unchanged pressure guard owns the server. CPU builds may overlap
quality requests, so reported wall times are observations rather than final
quiet latency comparisons. Model loading is outside each request's time;
initial lazy compilation and per-model cold effects can occur inside it.

## Frozen dense control

Server SHA256 `fa67e1d5f3a0ec19fc28b7da5110867b6dc5f6beb699a0d7ecef4ab0ac8b921b`,
at `.workspaces/clef-native-20261024/media-build/bin/ds4-server`:

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/clef-quality-broad-20261025/guard -- .workspaces/clef-native-20261024/media-build/bin/ds4-server --model gguf/clef-flash-int4 --decision-model gguf/clef-int4 --decision-model gguf/clef-omni-int4 --port 19317 --ctx 16384
```

The client was interrupted once to exclude Omni from the sequence after a real
kernel bug was identified. The upstream runner resumed the same output files;
the final full-model sample contains exactly 100 unique completed rows. Its
scorer finished before a coordinated clean server restart. The guard returned
child 0, pressure remained level 1 and swap decreased by about 0.09 GB.

The frozen Omni gathered-QMM kernel wraps its signed 16-bit routed row count
above 4095 text tokens. No Omni requests were run in this control. Its broader
qualification is deferred to a corrected frozen server; this is a correctness
fix, not an acceptable output-degradation experiment. Dense Flash/full do not
use that expert gather kernel.

## Full Clef: exact 100-request text sample

Edition 0.2.1, seed 20260919, 25 requests per task, complete upstream case groups
and exclusions. Existing runner and task scorers are unchanged. All 100 requests
are answered, with no unsupported results or errors.

| Task | Metric | Result |
| --- | --- | ---: |
| API-Bank | Accuracy | 100% |
| BANKING77 | Macro-F1 | 90.91% |
| CLINC150+OOS | Macro-F1 | 100% |
| When2Call | Accuracy | 64% |

BANKING77's sample accuracy is 96%. Across the 100 categorical decisions, mean
NLL is 0.35190 nats/decision, multiclass Brier score 0.16015, and ECE with ten
equal-width confidence bins 0.04983. ECE has high sampling uncertainty at this
size. NLL uses the probability assigned to the labelled answer; Brier sums
squared errors across every option. Neither is a token-level language-model
perplexity measurement.

Observed HTTP latency is median 4012 ms, p95 12890 ms, mean 5437 ms under the
quality-run conditions above. API-Bank median is 12238 ms, BANKING77 2923 ms,
CLINC150 5318 ms and When2Call 1138 ms. Different tasks have substantially
different token and option counts; these numbers should not be read as one
fixed-input inference speed.

## Corrected candidate and Omni text sample

The corrected HTTP candidate has SHA256
`470499725255f7ad3a36b14938f69b26397b9e323a62646ecf1cd5378ee18eeb`, at
`.workspaces/clef-native-20261024/fixed-build/bin/ds4-server`. Its served phase
explicitly sets `DS4_CLEF_OMNI_EXPERT_TILES=1`; the corrected old gather path is
retained for diagnostics. A model-free 32769-row regression and four paired
6472-token fixed-old/new forwards pass, with identical outputs. The earlier
large-context differences came from the real old-kernel overflow.

Reuse the already-completed Omni100 native CLI run rather than repeating that
model work through HTTP. Its exact inputs are the same 100-row sample, in the
same order; output was converted to the ordinary runner result shape and scored
with the existing scorer. Evidence is under
`.workspaces/clef-native-20261024/omni-fixed-proof/tiles-100/`. All 100 requests
are answered with no errors.

| Task | Metric | Omni result |
| --- | --- | ---: |
| API-Bank | Accuracy | 96% |
| BANKING77 | Macro-F1 | 100% |
| CLINC150+OOS | Macro-F1 | 88.89% |
| When2Call | Accuracy | 48% |

Omni mean NLL is 0.57196 nats/decision, Brier 0.24089 and ECE 0.19763. Its
observed native CLI median is 564 ms, p95 1987 ms and mean 835 ms. These timings
have a different boundary from full100's HTTP timings and should not be treated
as a matched speed comparison. One actual corrected-server Omni HTTP smoke also
returns 200 with the saved reference's billing choice.

The corrected server loads all three models for BLINK28 and Flash2000, under a
fresh unchanged pressure guard. The tile0 startup was stopped cleanly before any
request when the execution choice changed to tile1. Full100 remains attributed
to its original dense control.

## Real public images: BLINK28

Each model answers the same 28 public image questions, four from each of seven
subtasks. Twelve questions have one image and sixteen have two. All 84 requests
complete successfully, with no unsupported outputs or errors. The source's
961-row item and image digests match its pin; this selected subset is not the
full benchmark. Mean option-count chance is 35.71%.

| Model | Correct | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash | 18/28 | 64.29% | 0.75699 | 0.47963 | 0.17260 |
| Full | 22/28 | 78.57% | 0.53187 | 0.31246 | 0.15338 |
| Omni | 19/28 | 67.86% | 0.76765 | 0.46650 | 0.22540 |

NLL/Brier/ECE use the successful categorical probability distributions, with
the same definitions as the text checks. Twenty-eight cases give wide sampling
uncertainty, especially for four-case subtask breakdowns; these results do not
establish a general model ranking or native/reference parity.

| Model | Observed HTTP median | Mean | p95 |
| --- | ---: | ---: | ---: |
| Flash | 496 ms | 1301 ms | 3424 ms |
| Full | 1341 ms | 2943 ms | 7214 ms |
| Omni | 444 ms | 924 ms | 1942 ms |

These are one pass through varied inputs, including first-use compilation,
shape/cold effects and possible CPU builds. They are not matched warm latency
controls. The runner caps source images at 1,638,400 pixels; the server then
uses each model's native image budget. No same-weight MLX task reference was
run for this image sample yet. Per-subtask results and distributions are in
`vision-summary.json` and the three `*-vision-28/` directories locally.

## Flash: broader 2000-request text sample

All 2000 expected unique requests complete successfully: no errors,
unsupported results, abstentions or pending rows. Edition 0.2.1, seed 20260919,
500 requests per task, complete upstream case groups and exclusions.

| Task | Metric | Result | Observed HTTP median |
| --- | --- | ---: | ---: |
| API-Bank | Accuracy | 91.00% | 3261 ms |
| BANKING77 | Macro-F1 | 92.75% | 810 ms |
| CLINC150+OOS | Macro-F1 | 87.20% | 1495 ms |
| When2Call | Accuracy | 63.80% | 334 ms |

BANKING77 sample accuracy is 93.2%; CLINC150+OOS accuracy is 87.4%. Aggregate
mean NLL is 0.59825 nats/decision, multiclass Brier 0.25068 and ECE 0.09340.
Observed HTTP latency is median 1230 ms, p95 3511 ms and mean 1512 ms. It includes
varied request sizes and cold effects under the quality-run conditions described
above, and is not a matched native/reference latency comparison.

The first window completes 1660 rows (2540.7 seconds of request time). It stops
at the planned checkpoint for independent ownership/performance gates. A
second window resumes the remaining 340 rows (483.6 seconds), using an
independent byte copy of the same frozen binary and identical model/context/
tile1 settings. Total active request time is 3024.3 seconds; the intervening
pause is excluded. No completed result is dropped or rerun. The existing runner
and scorer handle the resume; final result IDs exactly match all 2000 inputs.
Both serving windows stop cleanly, with guard child 0, pressure level 1 only
and unchanged swap usage. The earlier dense control's swap decreases slightly.

The sample's first 100 selected cases are the earlier fixed sample, so their
new responses reuse the saved same-weight MLX reference without additional
inference: 100/100 top choices match, mean KL(reference||native) is 8.57e-5 and
delta-NLL is +0.00015805 nats/decision. The maximum individual option-probability
gap is 0.02219. This checks those 100 cases, not the entire broader sample or
media parity.

The completed task summary is `clef-flash-text-2000/benchmark-summary.json`;
`final-quality-summary.json` also records row-coverage checks, per-window request
time and all guard outcomes locally. The final serving copy is
`ds4-server-470499` under the same local result directory, SHA256 as above.
The maintained [qualify.nu](qualify.nu) reuses the existing text/vision runners;
the specific resume invokes [evaluate.nu](evaluate.nu) with the same output,
`sample-2000.jsonl.gz` and `sample-2000-suite` paths from [STAGING.md](STAGING.md).
