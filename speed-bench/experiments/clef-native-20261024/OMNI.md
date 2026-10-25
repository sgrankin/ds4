# Omni expert tiles — 2026-10-25

The remaining large short-input gap was in sparse expert matrix work. On the
four ordinary records, an Omni-only gathered-matmul candidate cuts whole native
decision latency **40–46%**, with bit-identical probabilities. These are local
M5 Max 128GB timings with the pinned INT4 Omni checkpoint and full GPU head.
The per-expert tile path is now the default after the qualification below;
`DS4_CLEF_OMNI_EXPERT_TILES=0` retains the corrected old gather for diagnostics.

## One expert per tile

Our initial gather reused MLX 0.32.0's mixed-expert BM64 kernel from Flint.
At 161–262 text tokens, top8 over 128 experts averages only 10–16 routed rows
per expert. That kernel repeatedly computes its whole 64-row neighbourhood
with each encountered expert's weights, then stores only that expert's rows.

The actual reference runs MLX 0.32.3. Its
[gather dispatch](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/quantized.cpp#L1469)
and [NAX kernel](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/quantized_nax.h#L1452)
instead schedule each row tile for one expert, using BM32 below 64 routed rows
per expert and BM64 above. The native candidate follows that policy. It reuses
our existing router counts/offsets, builds a small map of `[start,count,expert]`,
then uses the existing affine dequantizer and NAX matrix leaves. Both FFN
projections reuse the map. Valid tiny packed-QMV cases retain their old path.
No model repacking, precision reduction or routing approximation was introduced.
The adapted Metal leaves carry the pinned MLX/Apple MIT credit.

## Mirrored screen

One frozen binary ran old/tiles/tiles/old under the existing pressure guard.
Four ordinary records have one warmup plus three timed repetitions each; two
larger, real Decision Index requests have one warmup plus two timed repetitions.
Totals include record preparation, backbone, lexical rows and head, excluding
model loading/HTTP. Builds and other model work were paused for timing.

| Tokens | Shape | Old backbone | Tiles backbone | Old total | Tiles total |
| ---: | --- | ---: | ---: | ---: | ---: |
| 166 | 1 question, 3 options | 110.13 ms | 58.13 ms | 114.17 ms | 62.21 ms |
| 262 | predicate + 3-level score | 128.07 ms | 74.30 ms | 132.68 ms | 78.96 ms |
| 161 | 1 predicate | 112.60 ms | 58.81 ms | 116.62 ms | 62.87 ms |
| 234 | predicate + 2 options | 125.19 ms | 69.03 ms | 129.59 ms | 73.49 ms |
| 3388 | 151 intent options | 756.53 ms | 664.46 ms | 783.39 ms | 690.48 ms |

Short mirrored controls are stable; the larger old controls vary 725.7–787.4ms
in backbone time, while tiles vary 662.4–666.5ms. Thus the 3388-token benefit
is about 9–15%, rather than a precise universal 12%. All listed distributions
are **bit-identical**, including all repetitions: delta-NLL and KL are zero.
For the four ordinary records, max probability gap against saved same-weight
MLX reference remains 0.0009394. Their native totals are now below the local
MLX engine's approximately 70–86ms; timing boundaries differ slightly because
native includes record preparation. This is a small corpus, not an across-task
latency ranking or a full Decision Index result.

Raw results remain ignored under
`.workspaces/clef-native-20261024/omni-tiles-abba/`; the private command is
`nu .workspaces/clef-native-20261024/omni-tiles-abba/run.nu`.
After building `clef-smoke`, set `DS4_CLEF_OMNI_EXPERT_TILES=1` for the candidate
and `0` for the old control. The first four requests come from
`tests/fixtures/clef-omni-record.json`; larger records are rows 30 and 0 of the
fixed existing 100-task sample, with their original state/questions unchanged.

## Long-input correctness failure found during comparison

A genuine 6472-token/53-option input initially showed an 8-point probability
shift despite the same top choice. This was not accepted as unexplained noise.
All shorter comparisons were bit-identical. Old long-input controls varied by
up to 0.02284 between runs; new controls were bit-stable.

The old shared gather narrows `M-row` to `short` before clamping to its 32-row
fragment. At 4096 Omni text tokens, top8 produces 32768 routed rows: the
conversion overflows, bounds become negative and output rows remain unwritten.
The new kernel clamps in integer width before narrowing, avoiding the defect.
The shared old kernel now has the same bounded correction, so its fallback can
serve as a valid control. A model-free `flint-smoke` regression uses 32769 rows,
constant exact affine weights and NaN output sentinels, checking every output
including the final ragged row.

The initial 6472-token latency/distribution comparison used that broken old
kernel and is therefore unsuitable as a quality/speed tradeoff. Its single gold
decision had delta-NLL -0.129 nats, but that is not evidence of general quality.
The model-free NAX regression passes on this M5 Max. Four 6472-token repetitions
then produce **bit-identical distributions** between the repaired old gather
and the new tiles: KL and delta-NLL are zero. This confirms that the original
large shift was a correctness defect, rather than an approximation tradeoff.

## Completed 100-task qualification

The exact existing 100-task sample completes without errors using the new
tiles and the full GPU head. Requests preserve the original state/questions
and order from `smoke-100.jsonl.gz`, changing only the model to `clef-omni`.
The ordinary upstream task scorers give:

| Task | Cases | Metric | Native Omni |
| --- | ---: | --- | ---: |
| API-Bank | 25 | Accuracy | 96.0% |
| BANKING77 | 25 | Macro-F1 | 100.0% |
| CLINC150+OOS | 25 | Macro-F1 | 88.89% |
| When2Call | 25 | Accuracy | 48.0% |

Mean NLL is 0.57196 nats/decision; multiclass Brier is 0.24089 and equal-width
10-bin ECE is 0.19763. This small sample does not establish full-suite quality,
calibration equivalence or an across-model ranking. We did not rerun all 100
against corrected-old or MLX: the measured parity claim covers the controlled
ordinary, 3388-token and 6472-token comparisons above. The broader run establishes
native task completion and actual sampled scores, without claiming 100-case
distribution parity.

All fixture/model guards exit zero and no model job remains from this slice.
Evidence, exact requests, raw answers, standard Decision Index result records
and task summary remain local under
`.workspaces/clef-native-20261024/omni-fixed-proof/`. Its `run.nu` first runs the
model-free regression, then fixed-old/new long controls, then the 100 tile
requests. These final task calls use the native CLI; their timing must not be
mixed with HTTP timing from the serving qualification.
