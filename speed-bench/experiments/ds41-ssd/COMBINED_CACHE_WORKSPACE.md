# Demand workspace plus aging256: native validation (2026-09-21)

The combination improves the recorded tool session and short interactive use,
but demand-workspace growth regresses substantially. Both settings remain
opt-in. SLRU runtime support was removed in commit `1f757780`; its historical
measurements and offline simulator remain available. Default eviction is still
frequency aging every 16 decode tokens with LRU ties.

## Setup and reproducibility

All runs use explicit `gguf/DeepSeek-V4.1-Flash-Q2.gguf`, SSD streaming, and the
same frozen executable built after removing SLRU. Shader sources are frozen
per experiment. GPU runs are serial, with no compilation during timing.
The main candidate sets both `DS4_METAL_V41_DEMAND_WORKSPACE=1` and
`DS4_METAL_STREAM_CACHE_DECAY_TOKENS=256`; controls unset both. The interactive
harness now accepts repeated `--candidate-env` arguments and removes every
candidate key from its control environment.

The full replay teacher-forces a tool-using session. It measures inference,
excluding actual tool execution and sampling. Primary turn time is append
prefill plus decode; the initial system prefix is reported separately. Native
expert pread counters count requested bytes, not measured physical SSD traffic.
All replay comparisons check exact full-vocabulary phase logits and full final
continuation state. JSON manifests, summaries, comparisons, and per-run logs
are retained in the evidence directories linked below. Binaries and large
state dumps are not checked in.

Example main replay command (use a new output directory):

```sh
python3 speed-bench/agent-session/replay_abba.py /tmp/combined-full \
  --model gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --candidate-env DS4_METAL_V41_DEMAND_WORKSPACE=1 \
  --candidate-env DS4_METAL_STREAM_CACHE_DECAY_TOKENS=256 --order ABBA
```

## Full tool-session ABBA

| Metric | Default | Combination | Change |
|---|---:|---:|---:|
| Append prefill | 93.441 s | 85.761 s | -8.22% |
| Decode | 186.118 s | 173.064 s | -7.01% |
| Turn time, excluding initial system prefix | 279.558 s | 258.825 s | -7.42% |
| Initial system prefix | 15.927 s | 16.046 s | +0.75% |
| Native expert pread, including initial prefix | 671.37 GiB | 617.40 GiB | -8.04% |
| Native cache misses | 80,226 | 74,844 | -6.71% |

Turn-time samples in ABBA order: 279.934, 258.994, 258.657, 279.182 s.
Both candidates beat both controls. Counter values repeat exactly. Phase logits
and final states match. Including the initial prefix, total inference falls
from 295.485 to 274.872 s (6.98%).

The demand workspace starts at 2,048 instead of 8,192 rows, allowing the dynamic
expert cache to increase from 7,822 to 8,253 slots (72.51 to 76.50 GiB).
No workspace growth occurs in this workload: each append fits the smaller
workspace, even as accumulated context grows. Do not add gains from earlier
independent experiments to these combined measurements.

Evidence: [combined-full](combined-full/).

## Actual-agent ABBA

Prompts are hello / 2+2 / hello, with a 256-token generation limit. All runs
complete identical 59/15/28-token answers; input/output token sequences match
across fresh and restored sessions, and restored sessions hit the KV cache.

| Sum of three response times | Default | Combination | Change |
|---|---:|---:|---:|
| Fresh | 8.903 s | 8.443 s | -5.17% |
| Restored | 13.064 s | 12.737 s | -2.51% |

Both candidates beat both controls in both modes. The eight saved command
records confirm both candidate flags are set and both control flags absent.
This is a small interactive workload, not broad prompt coverage.

Evidence: [combined-agent](combined-agent/).

## Workspace growth: reproducible regression

`workspace-transition.txt` exercises a short initial exchange, then 8,192 and
16,384 appended tokens, then a short append. Demand workspace grows from 2,048
to 4,096 to 8,192 rows, trimming the expert cache as required. The growth tests
use two opposite-order pairs, with other measurements between them; they are
not a contiguous balanced ABBA block.

| Growth comparison | Default turn time | Candidate turn time | Change |
|---|---:|---:|---:|
| Combination, BA | 64.080 s | 104.517 s | +63.10% |
| Combination, reverse AB | 65.619 s | 82.175 s | +25.23% |
| Workspace alone, AB (aging16 in both) | 67.308 s | 87.533 s | +30.05% |

Both combination samples are slower than both controls. The magnitude varies
substantially; 63% is not a stable estimate of the effect. Workspace alone also
regresses, so aging256 is not necessary to trigger the slowdown. Most of that
workspace-only regression is append prefill: 61.576 to 81.880 s; decode is
5.732 versus 5.654 s.

All growth runs have identical phase logits and final states, and no mlock
failures. Both expansions complete. Final runtime tensor sizes match controls
(8,263 MiB); this does not establish a leak. Native expert pread is 30.45 versus
30.23 GiB in the first pair, insufficient to explain the large latency gap.
The cause remains unresolved. Next investigation should time synchronization,
cache trimming, workspace release/allocation, and subsequent GPU/I/O phases.
A partially retained slab and transient allocation pressure are hypotheses,
not measured causes.

Evidence: [combination BA](combined-growth-ba/),
[combination AB](combined-growth-ab/), [workspace alone](workspace-growth/).
This observed regression blocks unconditional default promotion.

## Incremental aging comparison at the larger cache

Full-session AB with demand workspace enabled in both variants and aging256
only in B isolates aging's incremental effect at 8,253 slots:

| Metric | Workspace alone | Workspace + aging256 | Change |
|---|---:|---:|---:|
| Append prefill | 101.378 s | 95.877 s | -5.43% |
| Decode | 186.092 s | 178.541 s | -4.06% |
| Turn time | 287.471 s | 274.418 s | -4.54% |

Logits and final state match exactly, including the primary full-session
reference. This supports an incremental benefit, but a single AB pair is
exploratory rather than balanced validation. Absolute times differ from the
earlier main ABBA block, so compare within each experiment, not across blocks.
Evidence: [combined-incremental](combined-incremental/).

All GPU jobs completed. Both binaries were rebuilt after SLRU removal; all 33
Python harness tests pass. The actual-agent combination exercises the new
repeated-candidate-env support, with all eight command environments verified.

## Correction: what the smaller-cache evidence actually showed

The earlier native smaller-cache experiment tested **aging64**, not aging256,
on a shorter 343-token excerpt at 4,086 slots. Relative to aging16, it increased
misses 21.38% and native expert pread 30.41%. Mean turn time increased 2.77%,
with overlapping timing samples. That undermines the proposed capacity rule
of aging64 for small caches and aging256 for large caches. It does **not** prove
that native aging256 regresses at small capacity, nor establish that no
universal setting can work.

Offline aging256 did regress at 4,096 slots, while helping larger simulated
caches. But those recordings omit prefill and seeding, so simulated misses do
not establish native latency. Native aging256 at that smaller capacity remains
untested. See [the prior native study](CACHE_POLICY_RUNTIME.md) and
[offline study](CACHE_POLICY_STUDY.md).
