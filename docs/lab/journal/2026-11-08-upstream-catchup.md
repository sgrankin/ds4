# Complete upstream adaptation through b3f50180

The catch-up integrates all surviving functionality from upstream `890898cf`
through `b3f50180`, while retaining the native owners and local optimizations.
It is a local adaptation merge; published main stays at `364ceeab`. The paused
performance campaign and separate Kimi branch remain outside this task. The
[result report](../../../speed-bench/experiments/2026-11-08-upstream-catchup/README.md)
owns settings, complete qualification, numerical comparisons and timing results.

## Context and setup

The user asked for a full catch-up, not a port narrowed to installed hardware.
We inventoried 59 changed upstream files, 96 non-merge commits and the final
choices in the intervening merges. Separate workers handled Qwen, DeepSeek and
GLM/shared code; one integration owner handled shared cache changes and all
heavy qualification. Only previously retired model/backend products were
excluded. Tests call native owners rather than substitute old C controllers.

The M5 Max has 128 GiB of RAM. Frozen baseline `df94063f` and the adapted runtime
ran with their own Metal sources, the same installed checkpoints and matching
settings. Each timing workload used five alternating pairs under the pressure
guard. Raw data and the completed temporary checklist remain ignored under
`.workspaces/upstream-b3f50180/`.

## Findings

Qwen gains SSD expert execution throughout prefill, decode and its predictor,
with memory admission, prepared predictor history and versioned state. A real
model test caught a startup mapping omission that synthetic kernel tests could
not: only token embeddings were mapped. The corrected native mapping now
includes all static trunk/predictor weights and has a focused host regression.
Resident and SSD execution then matched all nine complete vocabulary frontiers
in the ragged-prefill continuation check. Both Qwen and GLM passed the full
Responses conversation, tool, interruption and durable-restart checks.

Five-pair measurements establish a Qwen resident decode gain and a smaller GLM
prefill/total-inference gain in the measured configurations. DeepSeek timing is
inconclusive: append outliers occur on opposite sides at 32K and 6K, while
steady decode changes by less than 1%. We retained every sample and adopted the
port without claiming those noisy totals as a general gain or regression.

DeepSeek 32K, Qwen and GLM have exact fixed-history logit arrays. DeepSeek 6K has
small distribution drift with the resident-routing preset, and exact arrays
when substitutions are disabled. This isolates an interaction with approximate
routing; it is separate from the exact pruning-state checks. All most-likely
tokens agreed in the measured approximate comparison.

## Decision and limits

Keep the complete port, install the matching frontends, and fold its temporary
implementation commits into one adaptation merge with upstream ancestry.
The owning manuals explain the mechanisms; the result report records the
experiment. Raw benchmark data and worker workspaces do not enter commits.
Qwen v3 saves are readable only by the new runtime, though it accepts old v2
saves; durable transcripts remain the rollback/replay source.

Unavailable older GPUs, M6, physical two-Mac hardware and full GLM/V4 Pro weights
limit qualification only. Their retained code paths are included. No new
optimization experiment or publication is authorized by this catch-up's
completion.
