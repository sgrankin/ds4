# DS4.1 SSD optimization handoff

Third round: 2026-09-13, approximately 17:18-18:18 UTC, M5 Max /128 GiB.
Use jj, one commit per experiment. Vision reproduction and session saving remain
deferred until the user is at keyboard. No vision experiment in this round.

## Accepted defaults

Warm text appends use exact selected-expert tiles up to128 rows below1024
remaining tokens when the dynamic cache fits at least half the model's experts.
Smaller cache budgets retain tile8/limit256. Tiles clamp to actual prefill_cap;
sweep classification uses the same cap. Scalar causal attention order and BF16
boundaries remain intact. Selected tiles avoid whole-layer expert sweeps.
Existing early scalar selected-weight loads still overlap shared GPU work.
Image, quality, TP, format and backend capability guards remain unchanged.
No memory allocation default changed. At context100000:8.01GiB scratch,
72.51GiB dynamic experts,7.12GiB prefill reserve,97.61GiB total estimate.
Explicit --prefill-chunk remains supported independently of context.

New rollback: DS4_METAL_DISABLE_V41_WIDER_SELECTED_TILES=1 restores tile8/256.
Older DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS=1 also disables the previous
round's selected short batching and early scalar loading. Older full-layer
exact tail fallback retains DS4_METAL_DISABLE_V41_EXACT_SHORT_PREFILL.
Diagnostic DS4_METAL_V41_SELECTED_TILE and DS4_METAL_V41_SELECTED_LIMIT
allow controlled tuning without changing defaults.

## Controlled results from this hour

- Whole5000-token benchmark (4096 prefix +904 tail):92-93 to114 tokens/sec,
  about10.25s saved. Tail alone24.2-24.8 to35.1-35.4tps. Full logits identical.
- Final warmed fixed-agent ABBA: hello prefill fresh2.861->2.319s (~19% lower),
  restored4.421->3.697s (~16% lower). Through first output2.930->2.399s fresh,
  4.517->3.920s restored. Six-token turns essentially flat. Earlier fixed ABBA
  had a fresh-control outlier; prefer this final repeat for headline figures.
- Complete answers (three turns,41/15/28 generated tokens): mean sum of response
  times fresh8.968->8.617s, restored12.657->12.085s. Roughly4-5% improvement;
  do not generalize the larger prefill gain to full responses.
- Restored first decode remains about0.22-0.25s versus about0.09s on the old
  policy. Pipeline creation totals only6ms across the restored first turn,
  so it does not explain the gap. Needs finer I/O/GPU attribution.

Evidence: README.md experiments25-50; tail904-selected-abba.json,
agent-final-warmed-abba.json, agent-fixed-tile128.json, agent-fixed-responses.json,
pipeline-create-profile.json.
Agent and benchmark binaries rebuilt with the accepted defaults.

## Important benchmark correction

The first hello appends38 tokens,33 from injected datetime context. Earlier
agent experiments varied live timestamp tokens and sampling seed; treat those
as exploratory. Fixed-token model benchmarks are unaffected. Current harness
pins DS4_AGENT_TEST_TIME=1789319891, TZ=America/New_York, --seed1234, freezes
executable and Metal sources, and checks actual input/output token hashes across
ABBA. Test-time override is noninteractive only. Trace/session clocks stay live.
Startup and submit-to-ready wall times are recorded separately. Harness uses
isolated DS4_AGENT_CACHE_DIR, never the user's session files. System KV restore
does not restore expert weights, explaining some fresh/restored difference.

## Correctness and reproduction

Full serialized continuation snapshots match scalar controls after indexed2048
prefixes, tails2..767 across boundary cases, and8 decoded tokens;4096-prefix
768/1023 tails also match. Selected128 under16GiB cache target (8.88GiB dynamic)
passes eviction stress.4GiB target leaves one expert and passes scalar fallback.
Final defaults pass tails2/3/4/5/6/7/8/9/17/40, tiny-prefix32-row-cap tails
2/3/17/128, and cap1 fallback. See round3-*-state.txt, final-default-indexed.txt,
final-small-prefix-cap32.txt, final-cap1-fallback.txt, and tile128-state.txt.

    python3 speed-bench/experiments/ds41-ssd/agent_abba.py /tmp/NEW-DIR --candidate-env DS4_METAL_DISABLE_V41_WIDER_SELECTED_TILES=1
    python3 speed-bench/experiments/ds41-ssd/agent_abba.py /tmp/NEW-RESPONSES --candidate-env DS4_METAL_DISABLE_V41_WIDER_SELECTED_TILES=1 --gen-tokens 256

Never run concurrent GPU benchmarks. Metal requires sandbox escalation, not
sudo. jj mutations need backing Git access. Fixed benchmark input was extracted
from bd66c402 ds4.c into /tmp/ds41-prefill-input.c, SHA256
1776dbfed177ea14f3ce6cac1d8d0b1c1b44dfff2c2663769a9a5634aeec34e7.

## Experiments retained only as opt-ins

- DS4_METAL_V41_FUSED_SHARED_BF16: fused Q8 shared gate/up, intermediate BF16,
  SwiGLU, final BF16.32 synthetic shapes exact;32-row micro ~24% faster,
  actual agent flat. Full logits and continuation snapshot tests pass.
- DS4_METAL_V41_BATCH_SHARED_OVERLAP: launch shared GPU work while loading
  routed weights after router completion. Exact tests; exploratory gains
  unconfirmed with fixed timestamp harness. Not a default.
- DS4_ENGRAM_SMALL_BATCH_PARALLEL, DS4_METAL_V41_SMALL_ENGRAM_PREFETCH,
  DS4_ENGRAM_FILE_CACHE: correctness passed; no convincing first-turn benefit.
- Prior fused norm, Q8/BF16, Engram lookup and async expert worker experiments
  remain optional. See README for measurements and switches.
- Smaller scratch frees memory but auto cache consumes it; no demonstrated
  short-agent win. No cache budget change adopted.

## Best next work

1. Attribute restored first-decode delay: expert misses/read time, command-buffer
   waits and first scalar work after batched prefill. Pipeline creation ruled
   out as the main explanation in the measured helper paths.
2. Fixed-input confirmation of batch shared overlap; sweep pread concurrency
   and bounded prefetch with full-response checks and real eviction workloads.
3. Longer conversations and cache budgets: the128/1024 policy is conditional,
   not evidence that128 is universally optimal.
4. RoPE + quantization + direct KV writes, preserving block quantization rules.
   HC fusion must consume previous sublayer mixer, not the newly computed mixer.
5. Demand-allocated scratch with explicit memory headroom; avoid handing every
   saved byte directly to expert cache. Account for startup and I/O costs if
   adding saved-system-KV expert hints or background warmup.
6. Reproduce vision-enabled text slowdown and session-save limitation with user.
