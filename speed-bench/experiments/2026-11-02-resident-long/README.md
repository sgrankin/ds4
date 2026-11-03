# Longer resident-routing quality screen

Allowing five substitutions produced more distribution drift than the existing
three-substitution preset, with a small average reference-token likelihood cost
on longer code, prose and a recorded coding session. The agent replay's mean
delta-NLL was +0.0059 nats/token against native routing, or 0.59% higher teacher
perplexity. Its final answer had a larger local change, so the whole-session
mean should not stand alone. These are matched-history diagnostics, not
independent task outcomes. All three policies also passed three narrow,
reasoning-enabled answer checks after an 8K seed and a 512-token warm append.

No new performance comparison was made. The earlier
[short screen](../2026-11-02-resident-six/README.md) measured the speed tradeoff
and recommended investigating cap five further.

## Setup

The one-hour experiment began at 21:46 Eastern on 2026-11-02. All arms used
engine revision `6da1b939`, an M5 Max with 128 GiB, and DeepSeek V4.1 Flash
1017 IQ2_XXS/Q2_K imatrix. The [driver](run.nu) clears inherited `DS4` settings
and specifies the policies and commands. Context allocation was 16,384, prefill
chunk 8,192, expert-cache target 48 GiB, with packed main and index KV and SIMD
index scoring. The cache and packing settings match the earlier study; context
allocation and prefill chunk are larger here. The selected tile was 64 and limit
1,024. Each candidate used gap .90, maximum rank 64, replacement and native
mixture weights. The existing preset allows three substitutions with per-slot
mass .25 and total mass .50; cap five allows five with both limits set to 1.
Native routing has substitution disabled.

For each policy, the corpus probe seeded 8,192 tokens from
`speed-bench/promessi_sposi.txt` or `src/server/generation.zig`, then appended
1,024 tokens. At each of the 8,192 and 9,216 frontiers it scored the same
1,024 teacher tokens with full-vocabulary logits. The first position is the
prefill frontier; the other 1,023 follow teacher-forced decode. The append is
exactly at the selected-cache boundary: **1,024 takes a native full sweep**;
selected prefill requires fewer than 1,024 tokens. All four frontier logits
match native bit for bit under both candidates. Thus these corpus measurements
isolate approximate decode after native prefill, not approximate warm prefill.
Logit capture makes the diagnostic timings unsuitable for performance claims.

## Matched-history results

Each row combines the two 1,024-position frontiers for one text. Delta-NLL is
candidate minus native in nats per teacher token; positive means the candidate
assigns lower likelihood to those reference tokens. Forward KL compares the
full native and candidate distributions on the same history. P95 and maximum
are across positions, not independent runs. Distribution metrics use raw logits
before sampling filters; top-1 agreement counts positions with the same most
probable next token.

| Text | Policy | Mean delta-NLL | Mean KL | P95 KL | Maximum KL | Top-1 agreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Prose | preset three | -0.0009 | 0.0313 | 0.1347 | 0.7609 | 93.90% |
| Prose | cap five | 0.0271 | 0.0619 | 0.2763 | 2.2489 | 91.41% |
| Code | preset three | 0.0017 | 0.0064 | 0.0309 | 0.4629 | 98.24% |
| Code | cap five | 0.0081 | 0.0140 | 0.0693 | 1.3285 | 98.19% |

Cap five's teacher perplexity is 2.75% higher for prose and 0.81% higher for
code. Those percentages do not measure task failure rates. A near-zero average
delta-NLL can hide changes in either direction, as the positive KL and tails
show. The [128-position KL plot](block-kl.svg) shows how drift varies along
the continuations rather than rising monotonically with position. These four
evaluation windows have disjoint target positions within each text, but share
context from just two texts, so 4,096 positions must not be
interpreted as thousands of independent tasks. The metrics describe forced
histories; sampled responses can diverge and require actual answer checks.

## Recorded coding session

The agent replay uses a retained historical passing Q2 coding fixture on the
current 1017 model. It preserves the original prompt/decode boundaries across
three user turns, 14 assistant rounds and 20 tool calls, scoring 3,371 teacher
decode tokens as context grows to 8,710. The fixture modifies a synthetic
payment-reconciliation project. Its recorded tool outputs are replayed rather
than executed again: this is not a new autonomous coding pass.

Both candidate runs used selected prefill for all 3,480 appended user/tool
tokens: 60 calls covering 139,200 layer-rows, plus one native fallback for the
initial 1,859-token system prefix. Cap three made 89,180 prefill and 90,996 decode
substitutions; cap five made 165,186 and 173,875. The cap is a maximum: actual
decode substitutions averaged 0.67 versus 1.29 per layer/token, not three or
five on every row. These counters establish activation, not physical SSD reads.

| Policy | Mean delta-NLL | Mean KL | P95 KL | Maximum KL | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preset three | -0.0025 | 0.0062 | 0.0372 | 0.3483 | 98.10% |
| Cap five | 0.0059 | 0.0168 | 0.0987 | 1.1003 | 96.68% |

Cap five's average loss increases over the three user turns, while KL does not
increase monotonically. The turns have different content, so this comparison
does not isolate the effect of context length:

| User turn | Teacher positions | Three: delta-NLL / KL | Five: delta-NLL / KL |
| --- | ---: | ---: | ---: |
| 1 | 1,565 | -0.0052 / 0.0064 | 0.0029 / 0.0187 |
| 2 | 1,302 | -0.0022 / 0.0055 | 0.0073 / 0.0131 |
| 3 | 504 | 0.0049 / 0.0074 | 0.0114 / 0.0207 |

The [256-position block plot](agent-block-kl.svg) and
[running mean delta-NLL](agent-running-mean-delta-nll.svg) retain the late tail.
For cap five, the last 256 positions average delta-NLL +0.0236 and KL 0.0213;
the final 147-token assistant round averages +0.0499 and 0.0310. The last plot
block has only 43 positions, with +0.1605 delta-NLL and KL 0.0862. That small
block is more affected than the broader ending and should not be mistaken for
another full 256-position block. Its worst token has delta-NLL +3.1488: the
reference token's probability drops from 15.68% to 0.673%. This is a local
distribution change, not an observed task failure.

That last block explains why deduplication must precede date filtering. At the
worst position, the recorded text says "filtering first could **drop** that
first occurrence". Native routing actually favors ` let` (33.15%) over the
recorded ` drop` (15.68%); cap five favors ` wrongly` (38.59%), ` let` (33.90%)
and ` incorrectly` (8.84%). A different clause opening need not change the
conclusion. This example shows why a large single-token loss is not itself a
semantic error; the forced replay cannot tell us how those alternatives would
have continued.

All 8,710 fixture token IDs decode with the current tokenizer; the three
original prompts reproduce byte for byte, and all 551 shared IDs used in the
comparison match bytes in the retained old Q2 trace. The original full
vocabulary hash is unavailable, so this does not establish full tokenizer
identity. Current 1017 native logits are the reference throughout; the old Q2
run supplies the fixed token history, not baseline logits. Teacher output is
retained locally under `replay/agent-*/logits/` in the workspace below.

## Generated-answer checks

Reasoning-enabled warm-tail answer checks used 8,704 prompt tokens: 8,192 native
seed plus a 512-token append, with a 512-token generation cap. Unlike the corpus
sweep, the candidate arms use selected prefill for the append. Both candidates
recorded 48 selected calls and six native seed sweeps; prefill substitutions
were 38,414 for three and 55,541 for five.

| Retrieval case | Native | Preset three | Cap five |
| --- | --- | --- | --- |
| Older fact in the middle of the seed | Pass | Pass | Pass |
| Recent fact in the selected tail | Pass | Pass | Pass |
| Tail correction supersedes an older fact | Pass | Pass | Pass |

All nine answers exactly matched their expected code and stopped naturally.
Generation used 23–103 tokens including reasoning; none reached the cap. The
helper scores a short reference answer first, then invalidates KV and prefills
the prompt again before independent greedy generation. Those teacher tokens do
not enter the answer's KV, but the expert cache retains its prior weight loads.
These are three variants of a synthetic retrieval task, not broad reasoning or
coding benchmarks. All corpus, replay and answer stages completed under the
pressure guard without interruption.

## Decision

The longer evidence keeps cap five as a reasonable speed/quality choice for
the measured 48 GiB cache configuration. Relative to native routing, its mean
teacher perplexity rises by 0.59% on the agent replay, 0.81% on code and 2.75%
on prose. Against the current three-expert policy, those increases are 0.84%,
0.64% and 2.84%. That is a small average cost alongside the earlier short
workload's 7.16% warm-time saving, with visibly larger local changes.

This does not establish an autonomous coding success rate or behavior at
100K-plus contexts. Fixed histories prevent sampled errors from changing later
inputs, and the traces are only three dependent workloads. The earlier speed
figure belongs to its 2K-seed workload; these logit captures do not measure
long-context speed. Presets remain at three because this request was a quality
comparison, not adoption. No new policy or subsequent experiment is started.

Raw logits, logs, and analysis remain local under
`.workspaces/resident-long-20261102/`; they are not committed.

## Reproduction and interpretation

Build the existing probes, then run each stage sequentially under the pressure
guard. Use a fresh output directory: the replay refuses to overwrite its teacher
logits. These commands reproduce captures, not the rendered report figures,
and are run from the repository root.

```nu
./tools/zig-build ds4-bench decoder-replay-bench agent-session-replay -j4
let work = '.workspaces/resident-long-recheck'
let driver = 'speed-bench/experiments/2026-11-02-resident-long/run.nu'
for stage in [quality replay answers] {
    python3 speed-bench/ideas/pressure_run.py $'($work)/($stage)-guard' -- nu $driver $'($work)/($stage)' $stage
}
```

The corpus dumps use the existing
[full-vocabulary comparator](../steering-20261024/compare_decode.py).
Replay captures a flat little-endian float32 matrix, one 129,280-element row
per entry in `rows.csv`. Compare only identical row metadata, normalize logits
with float64 log-sum-exp, and compute `log p[target] - log q[target]` and
`sum(p * (log p - log q))`, where `p` is native and `q` is the candidate.
Position and phase summaries use arithmetic means; perplexity ratios are
`exp(mean delta-NLL)`. Block plots average adjacent positions rather than
accumulating loss with sequence length.
The one-off aggregation and plotting script remains local at
`.workspaces/resident-long-20261102/analyze_quality.py`; it reuses the comparator's
normalization. It is not a maintained dependency of the probes.

The corpus inputs were the retained Italian prose fixture and the current Zig
server generation source. Their SHA-256 hashes were respectively
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`
and `8b66848b966f85bd670775b7b742caf7ca117516964d860a330646cee4e70c94`.
The recorded agent fixture `speed-bench/agent-session/session-v1.txt` had hash
`310159cecfb6688d65967cf30279e1626c15571dbedd02838549f0d1ff14f368`.
Each policy ran in a fresh process, but retained expert residency across its
own phases. The corpus probe restores its 8K KV snapshot between continuations;
that restoration does not reset the expert cache. The agent replay has no such
reset. Neither fixed-history probe measures freely diverging sampled tasks.
