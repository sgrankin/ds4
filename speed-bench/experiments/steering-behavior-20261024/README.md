# Residual steering: held-out behavior

This tests actual generated answers from the residual vectors, separately from
the [runtime math and latency qualification](../steering-20261024/README.md).
Moderate additive steering gives useful observed compression on Flint, DS4.1
and Qwen, with key facts retained. Strong settings can instead cause repetition,
and some smaller negative settings merely run out of budget while still giving
coherent explanations. The small samples and extra factual errors are retained
below; an executable hook alone is not behavioral proof.

## Setup and checks

Baseline implementation is private steering change `sysvyppyvons` (`b7679480`).
Installed M5 Max/128GB binaries serve one model/condition at a time under the
existing pressure guard. Each fixed mode/strength starts an ordinary server;
each prompt is an independent stateless Responses request with `store:false`,
no conversation reference and temperature zero. Instructions are neutral:
`You are a helpful assistant.`

The raw target-minus-control residual banks are the previous experiment's
`residual-directions/{flint,ds41,qwen,glm}.f32`. Flint/DS41 use the first four
paired rows of the bundled `dir-steering/examples/succinct.txt` and `verbose.txt`;
Qwen/GLM use the first two. All trunk layers are selected. Project normalizes
rows at load; add retains magnitude and divides strength by the selected count.
These banks were not fitted to the held-out questions below.

| ID | Held-out prompt | Essential check |
| --- | --- | --- |
| 0 | Explain how a refrigerator keeps food cold. | Moves heat out; refrigerant/compressor cycle is a valid explanation |
| 1 | Why do we see lightning before hearing thunder? | Light reaches the observer much faster than sound |
| 2 | What is the difference between RAM and an SSD? | Volatile active workspace versus persistent storage |
| 3 | A recipe needs 300 grams of flour for four portions. How much flour is needed for ten portions? Explain your reasoning. | 750 g with valid scaling or unit-rate arithmetic |

The [request driver](request.nu) saves complete requests/responses and a small
`results.json`. Word counts use Nushell `str stats`; final answer and reasoning
are counted separately. We also read the answers for core facts, extra factual
mistakes, coherence and repetition. Length alone does not establish quality.
An incomplete, empty or repetitive response is never credited as succinctness.

Flint chooses whether to reason. Its final comparison uses a 2048-token output
allowance; the initial 1024 pilot capped two baseline answers and is excluded
from the final length comparison. Other full-prompt arms allow 512 output tokens
with reasoning disabled. No forced reasoning-close budget is used.

Projection's axis has no sign: `d` and `-d` give the same edit. Positive strength
removes the existing component and negative strength amplifies it. The effect
on style therefore must be measured, rather than inferred from the sign of
target-minus-control. Addition retains that difference's orientation.

## Flint: useful moderate points, overdrive failures

All four baseline answers complete at 2048 tokens. Each gets the core question
right, but the baseline is not factually perfect: lightning adds wrong distance
estimates and a false safety inference; RAM adds a misleading GB-versus-TB
capacity distinction. The quality notes below preserve those baseline defects.

| Mode / FFN strength | Completed | Final words by prompt 0/1/2/3 | Total change from zero | Reasoning words, total | Manual outcome |
| --- | ---: | --- | ---: | ---: | --- |
| Zero | 4/4 | 243 / 140 / 215 / 62 | reference: 660 | 2059 | Core facts correct; extra errors noted above |
| Project +0.1 | 4/4 | 237 / 119 / 157 / 41 | -16.1% | 1569 | All four shorter, coherent, core facts retained; lightning has an extra wrong distance calculation |
| Add +1 | 4/4 | 113 / 123 / 151 / 33 | -36.4% | 1581 | All four shorter, coherent, core facts retained; lightning retains a contradictory distance rule |
| Add -1 | 4/4 | 282 / 202 / 276 / 65 | +25.0% | 1875 | All four longer, coherent and core-correct; no concrete extra factual error flagged |
| Project +1 | 4/4 | 213 / 183 / 177 / 64 | -3.5% | 1443 | Two shorter, two longer; no consistent final-answer verbosity effect |
| Project -0.1 | 3/4 | capped / 157 / 221 / 63 | no full-set aggregate | no full-set aggregate | Refrigerator reasoning degenerates into repetition; flour has a stray `Combin? Wait` fragment despite correct arithmetic |

The clearer practical result is **bidirectional addition** on these four
held-out prompts: `+1` shortens all four, while `-1` lengthens all four. Project
`+0.1` also shortens all four here. This supports useful control for this bank
and model; four greedy answers do not establish a generally calibrated style
vector or monotonic response across strengths.

Concrete example: the baseline flour answer gives `300 / 4 = 75`, then
`75 * 10 = 750`, adds a second `300 * 2.5` method and repeats the conclusion
(62 words). Project `+0.1` keeps the correct unit-rate explanation and stops at
the conclusion (41 words). Add `+1` gives the same arithmetic in 33 words.
RAM similarly falls from 215 to 157/151 words while preserving volatility,
working-memory/persistent-storage roles and a useful analogy.

Important quality limits:

- Project `+0.1` lightning says five seconds means five miles, despite correctly
  stating five seconds per mile elsewhere. Add `+1` gives one mile in five
  seconds and then contradicts itself with three seconds per mile. These are
  extra factual errors, even though the requested light-versus-sound cause is
  correct. The baseline also gives incorrect distance numbers and an unsafe
  claim that a delay above 30 seconds means likely safety.
- Add `+1` simplifies SSD access latency to milliseconds; treat that as an
  overgeneralization, not a new accuracy result. Core RAM/SSD roles are correct.
- Project `+1` fixes the baseline lightning distance/safety additions, but its
  refrigerator explanation omits the expansion pressure-drop stage. It remains
  a coherent heat-pump answer, with less complete technical detail.
- Reported reasoning-word totals decrease about 24% at project `+0.1` and 23%
  at add `+1`. The reasoning traces have not been graded for reasoning quality;
  this is a count observation, not proof of better thinking or general quality.

The stronger screen identified unusable points rather than good short answers:

| Condition | Observed result |
| --- | --- |
| Project -1 | All four hit the 2048-token cap, mostly repeating `others` |
| Add -20 | All four hit the cap repeating `Combin` |
| Add +20 | All four hit the cap, dominated by repeated `shortened` |

Small reported word counts from unspaced repetition, or repeated occurrences
of `shortened`, are not style successes. These high-strength settings are
unusable for this all-layer bank. The lower-strength screen followed those failures,
instead of spending the experiment budget on more collapse runs.

## DS4.1: useful additive compression, unstable negative points

DS4.1's baseline completes all four prompts with core-correct, coherent answers.
It gives the complete refrigeration cycle, correct light/sound and distance
approximations, the RAM/SSD distinction, and 750 g. Its 327-word RAM answer
repeats the useful distinction through a list, table, analogy and conclusion.

| Mode / strength | Completed | Final words by prompt 0/1/2/3 | Manual outcome |
| --- | ---: | --- | --- |
| Zero | 4/4 | 257 / 112 / 327 / 91 | Core-correct and coherent; reference total 787 |
| Project +0.1 | 4/4 | 220 / 116 / 237 / 92 | Core facts and coherence retained; total 665 (-15.5%), two shorter and two slightly longer |
| Add +1 | 4/4 | 209 / 78 / 163 / 17 | All four shorter, coherent and core-correct; total 467 (-40.7%) |
| Project -0.1 | 3/4 | at least 380 / 252 / 237 / 98 | Refrigerator is coherent expanded detail until cutoff; other three complete and core-correct; mixed length direction |
| Add -1 | 1/4 | capped 379 / capped 368 / capped 346 / 145 | First three restart/repeat headings and paragraphs; flour completes with correct 750 g |
| Project -1 | 0/4 | capped repetitive bullet glyphs | No usable answer |

Project `+0.1` is a usable compression point on this tiny set, though the effect
is not uniform across prompts. Its refrigerator preserves all four major cycle
stages in 220 words; RAM preserves the main role/volatility distinction in 237.
Lightning and flour remain correct, at approximately their baseline lengths.

Add `+1` gives the clearest DS4.1 result: all four answers complete, retain the
key facts and become shorter. The flour answer falls from 91 words to 17:

> 300 grams for 4 portions means 75 grams per portion. For 10 portions:
> 75 × 10 = 750 grams.

The refrigerator keeps the main cycle and supporting details, lightning keeps
the correct speed explanation, and RAM/SSD keeps the volatility and role
distinction. There are minor formatting glitches (`circul-ates`, `each-other`)
in two answers, but no concrete extra factual error was flagged in this point.
This is useful observed control with acceptable small output defects, not a
claim that a four-pair bank is generally calibrated.

The project `-0.1` refrigerator is **not** a repetition collapse. It coherently
expands the same mechanism with insulation, thermodynamics and COP, then reaches
the cap mid-explanation. Its 380 words are a lower bound, not its natural final
length. This distinction matters: incomplete output may be coherent overlong
output rather than a broken answer, but it cannot earn completed-answer quality
or brevity credit.

Add `-1` has a different problem. The first three responses repeatedly restart
their headings/introductions mid-word, even though the initial core facts are
right. For example, the refrigerator switches from `refrigerant that circul`
directly into another `Understanding How a Refrigerator Works` introduction;
RAM restarts its comparison several times. These are observed repetition
failures, rather than an inference from the incomplete status alone. Its flour
answer is simply longer and correct.

## Qwen: useful addition within a fixed output budget

Qwen's initial 512-token zero baseline completes lightning and flour, but runs
out of budget during coherent refrigerator and RAM explanations. The partial answers
are not repetition failures: the refrigerator explains the cycle and is cut
near its end; RAM explains roles, volatility and analogy before extra material
is cut. Repeating only those two zero cases with a 1024-token allowance gives
naturally completed answers of 519 and 376 words. Both are exact text-prefix
extensions of the original capped answers; the higher limit adds continuation,
not a different answer before the old cutoff.

| Mode / strength | Completed | Final words by prompt 0/1/2/3 | Manual outcome |
| --- | ---: | --- | --- |
| Zero, initial 512 limit | 2/4 | at least 337 / 293 / at least 304 / 173 | Coherent and core-correct where covered; two unfinished tails |
| Zero, naturally completed | 4/4 | 519 / 293 / 376 / 173 | Cases 0/2 extended to a 1024 limit; total 1361 words, coherent and core-correct |
| Project +0.1 | 2/4 | at least 337 / 263 / at least 318 / 160 | Completed lightning/flour are correct and moderately shorter; other two remain coherent but capped |
| Project -0.1 | 1/4 | capped 371 / capped 332 / capped 270 / 191 | Elaborate coherent explanations, not repeated-token collapse |
| Add +1 | 4/4 | 209 / 128 / 154 / 91 | All complete/coherent/core-correct, including the two cases zero did not finish |
| Add -1 | 0/4 | capped 335 / capped 284 / capped 262 / capped 267 | Coherent, more elaborate exposition; unfinished, not repetition collapse |

Add `+1` preserves the complete refrigerator cycle in 209 words and the RAM/SSD
role/volatility distinction in 154. Lightning has correct light/sound speeds
and the approximate three-seconds/km or five-seconds/mile rule. Flour correctly
derives 750 g using the unit rate. No concrete extra factual error was flagged
in these four completed additive answers.

With those natural completions, add `+1` reduces the four-answer total from
**1361 to 582 words (-57.2%)**, with all four shorter and their key facts retained.
The two baseline retries had a higher ceiling; this is a comparison of natural
completed answers, not a claim that all runs had the same output allowance.

On the **original matched completed subset**, lightning plus flour fall from
466 to 219 words (**-53.0%**) at add `+1`; project `+0.1` gives 423 (**-9.2%**).
The two other additive answers are also shorter than the baseline's original
printed prefixes and finish within the same 512-token budget. This independently
shows useful bounded task behavior: four completions instead of two, without
pretending the original capped prefixes were natural full answers.

## GLM: one bounded arithmetic example

The first GLM baseline request used the lightning prompt and a 192-token cap.
It timed out at the client's 240-second deadline before returning a response.
This was actual decoding, not blocked startup: prefill finished in 11.3 seconds,
and logs reached 150 generated tokens at about 0.66 tokens/second. The pressure
guard stayed normal with unchanged swap; the owned server was stopped cleanly.

There is no returned answer from that first attempt to grade. This was an
evaluation-budget limit on the bounded SSD configuration, not evidence that a
GLM steering vector failed. It was replaced by **only the arithmetic case**,
at a shared 128-token allowance:

| Mode / strength | Result | Words printed | Output tokens | Request time |
| --- | --- | ---: | ---: | ---: |
| Zero | Core-correct, coherent, capped at an extra `Quick Check` heading | 77 | 128 | 193.2s |
| Project +0.1 | Core-correct, coherent, capped during an alternative derivation | 74 | 128 | 184.8s |
| Add +1 | Completed, coherent, correct 750 g with both arithmetic steps | 24 | 38 | 55.7s |

The additive answer explicitly gives `300 / 4 = 75` grams per portion, then
`75 * 10 = 750` grams and the conclusion. Zero and project also give the correct
answer before their unfinished tails; they are not repetition failures.
Addition therefore improves completion and avoids unnecessary output on this
one task. The timings include different output lengths and are not a kernel
speed comparison. There is no natural-final-length percentage, broader GLM
style claim or four-question accuracy claim from this one-case check.

## Steering fixtures now use Zig

The two new public-API steering fixtures were initially written in C to reuse
the existing fixture/build pattern. The C API did not require their test logic
to be C. They now live in:

- `tests/test_session_steering.zig`: session ownership, live changes, zero
  identity, batching and speculative/packet replay.
- `tests/test_metal_steering_hc.zig`: the shared GPU math/alias/BF16 oracle.
- `src/engine/steering.zig`: a pure unit check of bank normalization, additive
  magnitude preservation and nonzero-layer counting.

The two C predecessor files are removed. Zig still calls the public C ABI
through generated declarations; this preserves the actual interface coverage.
Older tests that directly include/access the retained private C engine stay
alongside that owner rather than acquiring another wrapper just for language
uniformity.

After conversion, the shared GPU oracle passes 342 cases plus 18 norm checks,
with exact zero results and maximum nonzero absolute difference 4.77e-7 against
its reference. Converted public-session tests pass Flint project/add, including
live settings, batches and exact full eight-token packet replay. The pure bank
test passes 1/1, and the complete host/server/agent/recovery/script suite passes
without build warnings. This turn re-ran the converted real-model fixture on
Flint; the prior C fixture's four-family results remain earlier evidence, not
a claim of fresh four-family execution after conversion.

These prompts test explanatory style and a few elementary facts. They do not
establish coding/tool-use quality, refusal ablation, long-context behavior or
the quality of hidden reasoning. Strengths and layer selections remain specific
to the model, bank and capture point.

## Reproduce and inspect

Use the [bank builder instructions](../../../dir-steering/README.md) to recreate
the raw-add bank from the paired source prompts. Start a direct server with
the desired mode and strength, a 4096-token context and no automatic KV/suffix
cache settings. For example, with Flint in one terminal:

```bash
python3 speed-bench/ideas/pressure_run.py .workspaces/style-check/guard -- \
  ./ds4-server --model /path/to/flint-int4 --port 19284 --ctx 4096 \
  --tokens 2048 --prefill-chunk 4096 \
  --dir-steering-file /path/to/flint-residual.f32 \
  --dir-steering-mode add --dir-steering-ffn 1 --dir-steering-attn 0
```

From another terminal, send independent cases with the maintained Nu driver:

```bash
nu speed-bench/experiments/steering-behavior-20261024/request.nu \
  http://127.0.0.1:19284 .workspaces/style-check/responses flint add 1
```

Stop that owned server before starting another condition. DS41 uses a 32GB
expert cache and chunk 512; GLM uses a 24GB expert cache with its native chunk
and static decode mapping disabled; Qwen retains its no-residency/no-warmup
settings. Raw commands, pressure logs, full requests, responses and manual
notes remain under ignored `.workspaces/steering-behavior-20261024/`.

Lightning extra-fact checks use [NWS thunder/distance guidance](https://www.weather.gov/safety/lightning-science-thunder)
and [NWS safety guidance](https://www.weather.gov/safety/lightning-tips). No
automatic readability score substitutes for the manual review.
