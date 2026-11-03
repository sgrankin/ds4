# Resident routing on longer histories

Five substitutions remains a reasonable candidate after the longer quality
screen. Its average likelihood cost is small across the measured texts and
recorded agent session, though individual positions change substantially.
The three-expert presets remain unchanged.

## Context and setup

The previous sweep recommended five for its warm-time saving, but its quality
probe covered only 64 teacher positions. The user asked for longer traces before
adoption. This one-hour slice fixed engine revision `6da1b939`, the M5 Max, the
1017 IQ2_XXS/Q2_K imatrix quant and a 48 GiB expert-cache target. It compared
native routing, the current three-expert policy and the proposed five-expert
policy; no runtime implementation changed.

Each policy scored 4,096 positions from code and prose after 8K/9K frontiers,
plus 3,371 assistant tokens from a recorded three-turn coding session with
20 tool calls. The exact 1,024-token corpus append takes a native full sweep,
so that arm isolates decode approximation. The recorded session and separate
512-token-tail answer checks exercise actual selected-prefill substitutions.
This was a quality experiment; diagnostic timings are not performance evidence.

## Findings and decision

Cap five increases average teacher perplexity by less than 3% against native
on each workload. Agent turn-average delta-NLL rises, but KL does not grow
monotonically; different turn content prevents isolating the effect of length.
The final recorded explanation has a larger local loss: inspecting the worst
token shows a preference for a different clause opening, which need not change
meaning. Fixed-history loss cannot decide how that alternative would continue.
All nine generated retrieval/correction answers pass with reasoning enabled.

The [result report](../../../speed-bench/experiments/2026-11-02-resident-long/README.md)
owns the settings, per-text and per-turn results, outliers, plots and reproduction
commands. The recorded coding task supplies forced history; this is not a new
autonomous coding pass or a long-context speed comparison. Keep the cap-five
recommendation as a measured tradeoff, without claiming a task-success rate or
extrapolating to much longer contexts. No preset promotion or further experiment
was part of this slice.
