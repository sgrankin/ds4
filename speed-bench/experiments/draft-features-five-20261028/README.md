# DSpark history with the ordinary confidence policy

This follow-up measures the existing optional DSP1 checkpoint trailer under
Metal's ordinary confidence threshold, **0.6**, rather than the earlier eager
verification setting of zero. The sole changed setting is
`DS4_V41_CHECKPOINT_DRAFT_FEATURES=0/1`. The completed five-pair campaign
supports adopting DSP1 for DSpark-enabled persistent serving.

Five fresh processes per arm run in AB/BA/AB/BA/AB order. Each loads the same
1017 IQ2_XXS/Q2_K trunk and MXFP4 DSpark head, keeps high reasoning, and uses
context 8192, prefill 512, 32 GB expert cache, packed index/SIMD, full six-token
packet width and exact-sampling support. Requests use temperature 0, so their
verification is greedy. Suffix reuse and automatic semantic boundaries are off.
HOT0 forces stored-state materialization after switching among prompt contexts.

Each process retains three unmeasured seed requests followed by four measured
restored continuations: ordinary JSON, a strict tool call, a schema answer, and
a separate 32-token ordinary cap. The first three have semantic answer checks;
the capped probe checks status/token-budget behavior, not completed-answer
quality. Startup and seeds are excluded explicitly. All five measured samples
per side, order, variation and failures remain in the report; no outlier is
silently removed.

The [driver](run.nu) owns one sequential server at a time beneath the unchanged
pressure guard. The [analyzer](analyze.py) joins each of its seven client rows
to the actual traced request and native generation phase. It records completed
verification cycles, proposals/acceptances, output/reasoning/token changes and
cache credits. It checks actual DSP1 trailer headers/extents and cache file
bytes, rather than treating an enabled flag as evidence of restored drafting.

Run the already-frozen binary with adjacent `shaders/` through the driver:

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/convergence-20261028/dsp1-guard -- nu speed-bench/experiments/draft-features-five-20261028/run.nu FROZEN/bin/ds4-server .workspaces/convergence-20261028/dsp1-samples
python3 speed-bench/experiments/draft-features-five-20261028/analyze.py .workspaces/convergence-20261028/dsp1-samples --output .workspaces/convergence-20261028/dsp1-samples/summary.json
```

Raw requests/responses, command settings, stderr/trace and all samples stay
local under that output directory. Cache file bytes describe stored files,
not SSD traffic or guaranteed filesystem allocation. The last decode-progress
log before a verification cycle is only a coarse timing boundary, not an exact
first-draft token position.

See the [original DSP1 implementation and eager-policy qualification](../deepinfer-alignment-20261027/README.md#4-useful-dspark-history-across-canonical-and-ssd-restore)
for the 32-byte header, chronological BF16 target-feature arrays, old-payload
fallback and first-positive-temperature-cycle oracle. DSP1 stores exact raw
target features and rebuilds projected drafter KV; it does not copy
DeepInfer's projected-KV representation. The ordinary-policy results below
support enabling the existing option for DSpark checkpoint restore in the
DeepSeek fast named presets.

## Result: useful at ordinary confidence

Completed 2026-10-28 04:46 UTC on the M5 Max, with the unchanged pressure
guard exiting successfully. The frozen runtime is the `9446a21d` qualification
workspace plus the qualified DSML change recorded as `10e95da7`; no in-progress
main-FP4 code is included. Its executable is
`.workspaces/convergence-20261028/qualification/.workspaces/dsml-qualified/bin/ds4-server`,
with frozen adjacent shaders. Each arm retains its exact argv/environment in
`command.json`. Numerical tables below come only from the completed fresh
`dsp1-samples` cohort and its local `summary.json`.

The three complete restored tasks take **37.533 → 30.731 seconds on average:
18.12% less HTTP wall time**, with every pair improving. Paired improvements
range from **17.65% to 18.85%**. The ordinary JSON task improves 10.74%, the
strict tool task 23.30%, and the schema answer 23.64%. The separate 32-token
cap improves 25.40%; it measures a bounded continuation rather than a completed
answer. These are warm-file state-restore workflows: the OS file cache is not
flushed, and model startup plus the three seed requests are excluded.

All five samples per arm are retained below, in seconds. Pair order is
AB / BA / AB / BA / AB; A is `base`, B is `features`.

| Task | Arm | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 |
|---|---|---:|---:|---:|---:|---:|
| ordinary | base | 15.711004 | 15.349833 | 16.025634 | 15.758999 | 16.061026 |
| ordinary | features | 14.062387 | 13.700466 | 14.170662 | 14.169264 | 14.327056 |
| strict-tool | base | 10.272544 | 10.228527 | 10.499789 | 10.380733 | 10.588983 |
| strict-tool | features | 7.857535 | 7.959605 | 8.027846 | 7.957438 | 8.061014 |
| schema-answer | base | 11.273176 | 11.174243 | 11.541583 | 11.379859 | 11.420733 |
| schema-answer | features | 8.670362 | 8.604790 | 8.692741 | 8.694679 | 8.701437 |
| ordinary-short | base | 2.296537 | 2.362223 | 2.329152 | 2.330949 | 2.314276 |
| ordinary-short | features | 1.723506 | 1.770836 | 1.720814 | 1.739406 | 1.723355 |
| three complete tasks | base | 37.256724 | 36.752603 | 38.067006 | 37.519591 | 38.070742 |
| three complete tasks | features | 30.590284 | 30.264861 | 30.891249 | 30.821381 | 31.089507 |

| Task | Arm | Mean (s) | Median (s) | Range (s) | Sample SD (s) |
|---|---|---:|---:|---:|---:|
| ordinary | base | 15.781 | 15.759 | 15.350–16.061 | 0.287 |
| ordinary | features | 14.086 | 14.169 | 13.700–14.327 | 0.235 |
| strict-tool | base | 10.394 | 10.381 | 10.229–10.589 | 0.151 |
| strict-tool | features | 7.973 | 7.960 | 7.858–8.061 | 0.078 |
| schema-answer | base | 11.358 | 11.380 | 11.174–11.542 | 0.141 |
| schema-answer | features | 8.673 | 8.693 | 8.605–8.701 | 0.040 |
| ordinary-short | base | 2.327 | 2.329 | 2.297–2.362 | 0.024 |
| ordinary-short | features | 1.736 | 1.724 | 1.721–1.771 | 0.021 |
| three complete tasks | base | 37.533 | 37.520 | 36.753–38.071 | 0.561 |
| three complete tasks | features | 30.731 | 30.821 | 30.265–31.090 | 0.316 |

Time changes below are negative when DSP1 is faster. The mean-time ratio is
the change of arithmetic means; the range and SD describe paired percentage
changes. Both order groups improve in every task; there are only three AB
and two BA pairs, so these groups describe order variation rather than a
precise estimate of an order effect.

| Task | Mean-time change | Paired range | Paired SD (pp) | AB mean change | BA mean change |
|---|---:|---:|---:|---:|---:|
| ordinary | -10.74% | -11.58% to -10.09% | 0.54 | -10.95% | -10.42% |
| strict-tool | -23.30% | -23.87% to -22.18% | 0.65 | -23.64% | -22.76% |
| schema-answer | -23.64% | -24.68% to -22.99% | 0.68 | -23.86% | -23.30% |
| ordinary-short | -25.40% | -26.12% to -24.95% | 0.47 | -25.53% | -25.21% |
| three complete tasks | -18.12% | -18.85% to -17.65% | 0.48 | -18.36% | -17.75% |

## What changed during generation

All **60 complete-answer checks pass**: three seed and three restored tasks
in each of ten processes. All ten cap-budget checks also pass. All 20 matched
measured outputs preserve exactly the same text, tool argument strings,
reasoning text, completion status and output/reasoning token counts after
excluding generated identities. This API campaign does not capture raw token
IDs, teacher logits, delta-NLL or KL; it is concrete task/output evidence.

The native phase joins produce the following counts in **each of all five
pairs**. A verification cycle is a completed native DSpark verification, not
an accepted-token count. The unchanged high-reasoning traces remain present.

| Task | Output / reasoning tokens, both arms | Cycles base → DSP1 | Proposed base → DSP1 | Accepted base → DSP1 |
|---|---:|---:|---:|---:|
| ordinary | 260 / 202 | 42 → 84 | 108 → 222 | 89 → 172 |
| strict-tool | 169 / 82 | 7 → 42 | 35 → 136 | 35 → 124 |
| schema-answer | 183 / 151 | 11 → 54 | 45 → 144 | 45 → 124 |
| ordinary-short | 32 / 32 | 0 → 9 | 0 → 30 | 0 → 23 |

The restored drafter can immediately use the saved target-feature history.
Without DSP1, the complete-task traces do not show verification until after
the 100-token decode-progress log; with DSP1, verification begins before the
first progress log. This is a coarse log boundary, not a precise first-draft
token index. The cap32 control has no verification cycles, while DSP1 has nine.
Cached input credits match in both arms: 97 / 392 / 172 / 97 tokens for the
four measured requests. Fresh seed verification counts also match across arms.
The gain comes from preserving useful drafter history across actual stored-state
materialization, rather than reducing the generated reasoning or answer.

## Storage cost and decision

Every feature process publishes exactly three valid DSP1 endpoints; every
control publishes three base endpoints. The manifest/chunk inspection validates
version, saved position, min(position,128) row count, width 5120, layer IDs
37/38/39 and the full native trailer extent.

| Prompt frontier | Base native bytes | DSP1 native bytes | Trailer increase |
|---:|---:|---:|---:|
| 97 | 8,986,044 | 11,965,916 | 2,979,872 |
| 172 | 11,913,500 | 15,845,692 | 3,932,192 |
| 392 | 13,078,180 | 17,010,372 | 3,932,192 |

Total cache files are **33,988,308 → 44,833,332 bytes** in every process:
**+10,845,024 bytes (+10.343 MiB; +31.91%)** for these three short endpoints.
That comprises 10,844,256 trailer bytes plus 768 additional manifest bytes.
All five file-size samples per arm are identical: mean/median/min/max equal
the respective value, SD 0. Each trailer is bounded at 3,932,192 bytes; the
relative cost will depend on the underlying state size and cache reuse.
Logical file sizes are not measured SSD traffic or physical filesystem blocks.

**Decision: enable DSP1 in the DeepSeek fast named presets** (`context-fast`,
`resident-fast`, `persistent-fast`, and the `persistent-packed` alias).
`DS4_V41_CHECKPOINT_DRAFT_FEATURES=0` remains an explicit override. Ordinary
engine defaults remain unchanged; this belongs to the opt-in preset package
and only matters with DSpark. The ordinary-confidence restored-task gain is
repeatable, with matching observed behavior and a small bounded per-endpoint
cost. It does not establish a benefit for fresh-only requests, cold SSD reads, non-DSpark
serving or every possible reasoning task. The explicit six-token packet width
is part of this qualification; stock width 1 is not this configuration.
The previous confidence 0 cross-fixture repair remains a separate diagnostic;
these results use the ordinary confidence 0.6 throughout and do not justify
changing that policy. Main bookmark and installed executables are unchanged
by this qualification and preset decision.

## Retained unsuccessful attempts

Three attempts precede the completed fresh cohort, all local and excluded:

- `dsp1-attempt1-type-error`: unquoted Nushell `32GB` was passed as a filesize;
  launch failed before a model request. It produced no measured samples.
- `dsp1-attempt2-pressure-samples`: the unchanged pressure guard stopped the
  process during an excluded seed. It produced no measured samples. The cause
  of this pressure stop is unproven; the earlier 58 GiB PDF helper was already gone.
- `dsp1-attempt3-driver-error-samples`: a bare `ordinary` in a Nushell branch
  attempted to execute an external command before the cap32 request. Three
  base measurements existed, but the entire incomplete attempt is excluded.
  Pressure was normal. Quoting the literal fixes the driver; its four source
  selections were executed separately before restarting all five pairs fresh.

The successful cohort uses the unchanged pressure guard and contains no
stitched samples, filtered outliers or timing taken from these failures.
