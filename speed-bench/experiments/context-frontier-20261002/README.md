# Saved context frontiers, 2026-10-02

Base: `67696070`. Model: `DeepSeek-V4.1-Flash-Q2.gguf`, Metal SSD streaming,
Apple M5 Max, 128 GiB. No runtime default changes.

The current V4.1 session discards its graph when a prompt changes behind the
live end. This experiment restores a complete earlier prefix snapshot and
evaluates the edited suffix normally. It retains no stale suffix KV or latent
states, so it is distinct from approximate Suffix Cache Reuse in the CLM paper.

## Benchmark

Build `make -j2 speed-bench/context_frontier_bench`. Run from the repository root:

```
./speed-bench/context_frontier_bench \
  --model /Volumes/Models/ds4/gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --prefix 2048 --old-span 256 --new-span 256 --tail 512 \
  --ctx 8192 --prefill-chunk 2048 --repeat 1 --decode 4
```

The frozen input is `A + B + C`; the edited input is `A + B' + C`. A is the
saved prefix, B' comes from a different section of the same source, and C is
retained verbatim. Both control and candidate start from the same complete
old-context snapshot. Controls rebuild the entire edited input; candidates
restore A then evaluate B'+C. ABBA order balances warmed expert-cache effects
within one process. Restore time is included in the candidate's total; saving
A is reported separately so a one-edit result can also charge that cost.

Every candidate must match a fresh replay with the same A boundary in all
vocabulary logits, the complete serialized graph state, and four continuation
logit rows. The benchmark also reports differences from the ordinary cold
full rebuild. This distinction matters: changing prefill batch partitions can
change floating-point results even when prefix restoration itself is exact.
The aligned 2048-token case holds those partitions fixed.

The initial aligned ABBA passed every equality check. Rebuilds took 62.019 and
56.033 seconds; restored-prefix edits took 24.251 and 24.473 seconds. Aggregate
edit latency fell **58.73%** (2.423x speedup). Saving the 24,118,324-byte prefix
cost 12.396 ms once; restoration cost 0.861–4.596 ms. Charging that save to a
single edit still gives a 58.71% reduction. `aligned.csv`, `aligned.log`, and
`aligned-summary.json` retain the measured rows and exactness results. This is
one balanced same-process screen, not fresh-process or actual-agent validation.

`agent-prefix.tokens` contains the first 6000 tokens reconstructed from the
checked-in `speed-bench/agent-session/session-v1.txt` P/D trace. Its first 1859
tokens are that recorded agent's system prefix. `--tokens-file` accepts this
source directly, avoiding a tokenizer-dependent conversion of the fixture.

## Agent candidate

`DS4_AGENT_COMPACT_PREFIX_CACHE=1` makes `ds4-agent` restore the existing
`sysprompt.kv` before computing a compacted summary+tail. The cached rendered
system prompt, model, quantization and exact token prefix must all match. Cache
failure follows the ordinary rebuild path. The experiment skips image-bearing
contexts, tensor/distributed execution, directional steering, and other models.

The cache is loaded only when compaction occurs; no additional persistent
memory or snapshot-copy cost is added to ordinary turns. The compacted suffix
is recomputed in full. The switch is opt-in because a system-prefix boundary
need not match the cold compacted prompt's floating-point batch partition.

Build the focused native regression with
`make -j2 tests/test_agent_compact_prefix`, then run it with the Q2 model path.
It checks complete restored-state identity, disabled mode, mismatched system
text, and corrupt/missing cache fallback. `ds4_agent_test` covers existing
agent behavior without loading a model.

The focused native regression passed on the Q2 model: its real 1846-token
system prefix serialized to 22,824,716 bytes and restored in 5.728 ms. Complete
state identity and all fallback cases passed (`native-cache-test.log`). The
whole original image-bearing conversation is excluded, even if compaction
would discard every old image span.

## Scope

The aligned benchmark is a middle-edit engine experiment, not a measured
ordinary-agent speedup. Existing append-only agent turns already reuse their
live checkpoint and retain reasoning tokens. This mechanism helps edits and
compaction; it cannot improve turns that already append normally. The agent
candidate requires task-level validation before any default promotion.
