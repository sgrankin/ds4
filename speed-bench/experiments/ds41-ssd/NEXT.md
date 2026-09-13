# DS4.1 SSD optimization handoff

Objective: good agentic tool-use performance balanced with interactive latency.
Machine: Apple M5 Max, 128 GiB; DS4.1 Flash Q2; SSD streaming required.
Run GPU benchmarks serially. Metal and jj mutations need sandbox escalation;
profiling our own process with `sample` worked without sudo. Vision reproduction
and the vision session-save limitation remain deferred until the user is present.

## Active routing-prediction experiment

User authorized predicted expert prefetch. Exact-route oracle implementation
committed d47db637; first results b471a8ce. Short ABBA completed at
/tmp/ds41-oracle-preattention-short: 18.405 -> 16.984 s decode (-7.7%), combined
33.773 -> 32.813 s (-2.8%, one candidate prefill outlier). Exact routes, logits,
full snapshot, cache misses and bytes. Evidence oracle-preattention-short.json.
Capture /tmp/ds41-oracle-routes-short.bin is 343 tokens * 40 layers, 493936 bytes,
SHA256 80bfffb6a8481e3762d531ff6590f4d7e5eb1cbdfcd297e9c806c375c8a7174a.

Second uncommitted diagnostic adds DS4_V41_ORACLE_DEFER_CHECK to pre-attention:
GPU-copy actual routes into a 960-byte trace, validate after each token, avoiding
per-layer CPU route waits while retaining native weights/arithmetic. It is an
oracle architectural bound, not usable with a fallible predictor without an
exact GPU check/fallback mechanism. Existing cache in-flight protections apply.
Short ABBA currently running /tmp/ds41-oracle-deferred-short (session48987),
A = pre-attention oracle, B = pre-attention plus deferred check. First A16.590s
and B13.316s decode, exact state/logits/routes. Wait for balanced completion.
Binaries already rebuilt; avoid builds/CPU-heavy analysis during timed runs.
Next: full route capture and full direct-default-vs-combined ABBA, and a separate
profiled comparison using the new pending-read CPU wait counter.

Backend diagnostic pending wait_ms measures only CPU join wait. Existing
pread_ms includes the whole begin-to-consumption lifetime, including overlap;
do not use it as exposed-stall time. Counters reset with cache stats.

Offline route_stats.py screens previous-token, hot-frequency and previous-layer
transition predictors. It is only a within-session chronological split and uses
all-route recall, not cache misses/deadlines. Short recall at six candidates:
previous token22.5%, frequency28.7%, transition45.9%; at12 transition63.0%.
Do not confuse this with independent-task predictor validation. Eight extractor
and route-parser tests pass. Need full-session stats before judging baselines.

Research anchors:
- https://arxiv.org/html/2410.22134v3 (ProMoE learned prediction and scheduling)
- https://arxiv.org/abs/2511.10676 (pre-attention expert prediction)
- https://arxiv.org/abs/2607.24787 (predictions only for transfers, frozen routing)

Score useful misses ready before deadline, wasted bytes and eviction, not just
all-expert prediction accuracy. For training, hold out entire sessions/tasks.
Try earlier-activation gate and transition-statistics baselines before an MLP.
Preserve exact execution routing; wrong predictions may cost time, not quality.

## Working-session benchmark now available

Registered in `speed-bench/README.md`; instructions and source are in
`speed-bench/agent-session/`. `live.py` runs three tasks in a disposable Python
ledger project and independently checks task results after every user turn.
The passing baseline uses 20 tool calls, 14 generation rounds, 3371 generated
tokens, and ends at context 8710. Turn wall time is 291.714 s, with 17.420 s
startup reported separately; model prefill/decode are 94.280/196.339 s.

`session-v1.txt` records exact prefill/decode boundaries and tokens from that
successful run. SHA256:
310159cecfb6688d65967cf30279e1626c15571dbedd02838549f0d1ff14f368.
The replay retains KV and expert-cache evolution, checks full phase logits and
final continuation snapshots, and excludes tool execution/sampling/rendering.
Full control replay closely reproduces live model time. Live task success and
short fresh/restored interactive responses remain required complementary checks.

    python3 speed-bench/agent-session/live.py /tmp/NEW-LIVE
    python3 speed-bench/agent-session/replay_abba.py /tmp/NEW-ABBA --candidate-env NAME=VALUE

The default replay is the complete session. `--order AB` is exploratory; pair
with BA using identical binary/shader hashes before treating it as balanced.
Repeat `--candidate-env` for combinations. Buffer and cache studies are supported:
`DS4_REPLAY_PREFILL_CHUNK=2048`, or `--cache-gb N` (total target including reserve).

## Latest round completed: no new runtime defaults

Evidence and numerical hashes are in README experiments 51 onward and adjacent
JSON files. Runtime files were restored to accepted revision 8e8d7794 in commit
4442d372. The following new runtime experiments survive in jj history only:

- Wider 512-row layer sweeps with <=128-row MoE subtiles (92e05db7): full ABBA
  turn model time 291.389 -> 295.204 s, about 1.3% slower. Prefill misses rose
  31818 -> 33952; decode misses were almost unchanged. Exact state/logits.
- Router-event overlap (0fb424f3): short-session ABBA decode about 0.75% faster,
  but combined append/decode about 0.8% slower. Exact state/logits.
- Broad current-layer cache protection (25689657): slightly fewer reads, about
  1% slower overall in short ABBA. Exact state/logits.
- Protection scoped only to future MoE subtiles (ec701a4a): about 0.8% slower
  overall in short ABBA. Exact state/logits. Do not retain either pin by default.

A targeted three-second CPU sample finds the largest main-thread wait at router
Metal completion, then selected SSD reads, then F_RDADVISE calls. Metal waits
include actual GPU work, so this is not a measure of removable overhead.
The sample perturbed decode; use `decode-profile.txt` only for attribution.

Existing flag `DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1` gave a promising
short fixed-session screen: decode 18.388 -> 17.981 s (2.2% faster), combined
turn model time 33.570 -> 33.246 s (1% faster), exact state/logits. This is only
the first seven complete replay phases (343 decoded tokens, context 2665),
not the complete working session. Completed interactive ABBA rejects the
unconditional ablation: fresh response time 8.222 -> 8.157 s (-0.8%), restored
12.261 -> 12.623 s (+3.0%), with identical input/output token hashes. Restored
decode is 5.3% slower. Read-ahead stays enabled; skip full-session promotion for
this variant. Evidence: `noadvice-interactive.json`, README experiment 59.

No GPU runs remain active. Agent, benchmark and replay binaries are rebuilt.
The successful live task, exact replay comparisons and four extractor tests
passed. No sudo or user input was needed. All negative experiments have separate
jj commits; do not reintroduce their runtime branches without new evidence.

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

## Accepted results from the previous round

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

1. Test `--prefill-chunk 2048` on the complete session. It frees about 4.25 GiB
   of scratch (8.01 -> 3.76 GiB), which automatic sizing gives to expert cache.
   Earlier short questions were flat; this recording has 3371 decode tokens and
   meaningful eviction. Measure full ABBA plus interactive responses before
   changing defaults. Keep memory savings distinct from cache growth.
2. Recheck `DS4_METAL_V41_BATCH_SHARED_OVERLAP=1` with the fixed session; sweep
   `DS4_METAL_STREAMING_EXPERT_PREAD_THREADS=18` versus default 9 separately.
   Prior shared-overlap agent results preceded fixed timestamp inputs.
3. Attribute restored first-decode latency with per-phase I/O and GPU timings.
   Investigate the extra drain before the second Engram layer only after proving
   every consumer of the reused buffer has completed in the eligible path.
4. RoPE + quantization + direct KV writes, preserving block quantization rules.
   HC fusion must consume the previous sublayer mixer, not the newly computed
   mixer. Require actual-session improvement as well as exact kernel results.
5. Demand-allocated scratch with explicit memory headroom. A possible extension
   lends part of the 7.12 GiB prefill reserve to decode/selected appends, reclaiming
   it before full-layer sweeps. Prove slab release, in-flight protection and all
   admission paths first; prefer the existing prefill-chunk knob for now.
6. Longer conversations, smaller cache budgets and compaction coverage. The
   current synthetic session is useful but does not establish long-context
   behavior or general coding-task quality.
7. Reproduce vision-enabled text slowdown and session-save limitation with user.
