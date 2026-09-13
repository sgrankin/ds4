# DS4.1 SSD optimization handoff

Objective: good agentic tool-use performance balanced with interactive latency.
Apple M5 Max, 128 GiB, DeepSeek V4.1 Flash Q2; SSD streaming required. Use jj,
commit experiments separately, and run GPU workloads serially. Metal/jj require
sandbox escalation; no sudo needed. No subagents. Vision reproduction and the
vision session-save issue remain deferred until the user is present.

## Current routing investigation

User authorized investigating predicted expert prefetch. No new production
scheduling default has been enabled. Diagnostic flags are opt-in and require
one scalar session per process; native routing is always retained or verified.

The checked working-session recording has 20 tool calls, 14 generation rounds,
3371 decoded tokens and final context8710. Its live fixture passed all three
independent task checks. Live turn time291.714s excludes17.420s startup;
model prefill/decode94.280/196.339s. The replay preserves token boundaries,
KV/expert-cache evolution and checks full phase logits/final continuation state.
See speed-bench/agent-session/README.md and the registry speed-bench/README.md.

Full-session routing results (README experiments60-64):

- Exact pre-attention prefetch oracle: decode198.199->172.594s (-12.9%),
  append93.918->93.997s, combined292.117->266.592s (-8.7%,25.5s saved).
  Per-layer CPU route validation stays in place. Balanced ABBA uses the last A
  of the combined study followed immediately by BBA; complete_abba.py checks
  executable/model/shader/token/route/environment and numerical identity.
  Shared control is one observation, not an extra independent repeat.
  Evidence oracle-preattention-full*.json; commit0fe6c413.
- Combined pre-attention + deferred validation oracle: decode197.511->136.838s
  (-30.7%), append94.140->97.418s (+3.5%), combined291.651->234.256s (-19.7%).
  Every actual route is GPU-copied into960 bytes per token and checked after
  the token; CPU address binding uses exact recorded future IDs. A real
  predictor cannot safely do this without GPU validation and fallback.
  Evidence oracle-combined-full.json; commits50ce2817/b1c9fdb5.
- Native routes, phase logits and full final snapshots match in all trials.
  Cache budget stays7822 experts/72.51GiB. Combined oracle reads670.63GiB
  versus670.61GiB control, with two additional misses; it saves time through
  scheduling rather than reducing expert bytes. These are free-future-information
  bounds for the tested schedules, not achievable-generation speed claims.
- Instrumented census:134840 layer events,75.3% all-resident,33302 mixed and
  3 all-missing. CPU pending-read joins total19.858s in197.939s decode.
  This excludes I/O preparation/advice and is not a ceiling on all I/O-related
  savings. Old pread_ms includes begin-to-consumption overlap, not just stalls.
  Evidence oracle-full-capture.json; commitd99dbefd.
- Cheap within-session chronological predictor baselines: at6 candidates,
  previous-token recall24.5%, frequency16.0%, layer-transition34.6%; transition
  at12 candidates48.6%. These are all-route metrics, not miss/deadline metrics
  or independent-task validation. Evidence route-stats-full.json.

Completed gate probe: existing pre-attention gate predicts 50.2973% of native
selected experts; all-six coverage 1.44245%, per-layer recall 24.4–67.4%.
Full native routes, logits and final state match baseline. Evidence gate-probe-full.json.
Intrusive probe timings are not deployment performance. Real bounded asynchronous gate prefetch is now implemented behind
DS4_METAL_V41_GATE_PREFETCH=1 and REJECTED for default enablement: short ABBA
has exact logits/state but decode +11.71%, combined turn +6.06%. Evidence
gate-prefetch-short.json and comparison. One-slot loader waits for wrong reads;
no demand priority/cancellation. Intrusive AB profile completed: expert reads87.53->127.24GiB (+45.4%),
pending CPU wait1.392->2.373s, joins2302->5467. Exact logits/state throughout.
Evidence gate-prefetch-profile.json. No full trial warranted for this regression.
Prioritize GPU cache-hit scheduling, confidence/cache-aware admission and demand
priority before another unconditional predictor trial. All new modes default off.
Missing-route schedule guard was tested on a one-token input and exits nonzero.
Wrong-route injection in deferred mode also exits nonzero. Twelve Python tests
pass, covering extraction, route parsing and balanced-continuation identity.

## Current continuation (experiments67-70)

Experiments67-69 implemented sparse cache execution before CPU routing readback,
then GPU all-hit guards, batched resource declarations and slab deduplication.
Native logits and continuation state were exact. Experiment67 and preliminary68
missed address-table maintenance and hit-path token aging; their timings are
superseded. Corrected68 preserves identical cache traffic and is +4.29% decode,
+2.55% short-session inference. Deduplicated69 remains +4.19% decode,+3.04% turn.
All are rejected. Code archived in jj commits458ee395,99121f52,468d504c;
restored out in4ba2a3a3. Runtime sources are back to experiment66 baseline.
This schedule still waits once per layer and is not the oracle's wait-free path.

Experiment70 completed: chunk2048 full ABBA exact; decode -2.95%, append
-2.07%, combined -2.665% (293.128->285.316s), reads670.61->582.57GiB.
Budget7822->8253 experts /72.51->76.50GiB. Results chunk2048-full-*.json.
No default changed: explicit --prefill-chunk2048 is the tested option.

Experiment71 ACTIVE: /tmp/ds41-chunk2048-large8192, unified session17089.
run_abba.py --candidate-prefill-chunk2048 --tokens8192 --ctx100000
--gen-tokens8, normal restored binary and frozen shaders. Finish, check frontier
logits and throughput, save evidence and decide whether this is a user-selectable
tradeoff rather than a default. GPU runs serial; no other active trials.

## Reproduction and artifacts

    make ds4-agent ds4-bench speed-bench/agent-session/replay
    python3 speed-bench/agent-session/replay_abba.py /tmp/NEW-PREFETCH --routes --candidate-env DS4_V41_ORACLE_PREATTENTION=1
    python3 speed-bench/agent-session/replay_abba.py /tmp/NEW-COMBINED --routes --candidate-env DS4_V41_ORACLE_PREATTENTION=1 --candidate-env DS4_V41_ORACLE_DEFER_CHECK=1

--routes without a path uses checked session-v1.routes.bin.gz (1.5MB,
commit80a11919; a command-local jj size-limit override was used, no repo config
change). Provenance is session-v1.routes.json. The native-endian uint32 payload
is4,854,256 bytes, SHA256:
3f3a99820e1e4e26f452092301fec1e67c5ca3c42a3acda7cc918a46e2943b43.
Token recording SHA256:
310159cecfb6688d65967cf30279e1626c15571dbedd02838549f0d1ff14f368.
Full replay logits SHA256:
dbb7917fd94cef6332878b8e74bfa328ecab7c28001de6eb1b02c212c1d55be0.
Full snapshot SHA256:
4038547fda7c3050dbb562e488a12b33f89527ea7c7927d142d978b2c36319fe.
Raw full comparison directories are /tmp/ds41-oracle-combined-full,
/tmp/ds41-oracle-preattention-full-tail and the merged
/tmp/ds41-oracle-preattention-full. No timed processes remain active.
Route capture and probe commands are in the agent-session README.

## Next substantial work

1. Evaluate the existing-gate probe, then collect pre-attention activations,
   actual selections and demand cache-miss masks across independent tasks.
   Current route-only recording lacks activation/miss labels. Score useful
   misses ready before deadline, wasted bytes and eviction; hot experts that
   are already cached can flatter ordinary recall.
2. Completed experiment66: asynchronous exact-fallback prefetch regresses.
   For a subsequent version, improve admission and demand priority. Predict into a separate,
   immutable GPU ID buffer, signal an event, and let the service worker read it
   and start loads while the main thread encodes attention. Join before native
   demand/cache mutation. Don't add a blocking CPU readback before attention or
   overwrite prediction IDs before the worker reads them. Demand reads need
   priority over wrong predictions; current one-slot loader can block on them.
3. GPU cache-hit scheduling: validate sparse addresses and compute the all-hit
   MoE in the routing submission. On a miss, preserve norm/shared/attention state
   and fall back before consuming the routed output. Guard every pointer/kernel
   and protect cache resources in flight. Stop within the layer initially;
   avoid speculative later KV updates and rollback. Existing
   ds4_gpu_stream_expert_cache_validate_selected still immediately waits on CPU,
   and its caller requires IQ2 gate/up. Header verification confirms this model
   uses IQ2_XXS gate/up and Q2_K down; the early selected override bypasses
   that validator, and its immediate CPU wait remains the relevant limitation.
4. If needed, train a small predictor: shared5120->64 stem plus40 separate
   64->384 heads is about1.31M weights (~2.6MB FP16), excluding biases. This is
   a candidate, not a trained model or performance claim. Hold out whole tasks,
   preserve the native router, and charge prediction/readback/contended I/O time.
5. Deeper lookahead needs bounded multi-request I/O and cache admission; the
   current oracle measures one pending load before same-layer attention only.

Research anchors:
- https://arxiv.org/html/2410.22134v3 (ProMoE prediction and scheduling)
- https://arxiv.org/abs/2511.10676 (pre-attention expert prediction)
- https://arxiv.org/abs/2607.24787 (transfer prediction with frozen routing)

Other queued ideas: prefill-chunk2048 versus larger expert cache on the full
session; fixed-input batch shared overlap; pread threads18 versus9; restored
first-decode attribution; RoPE/quantize/direct-KV fusion; demand-allocated
scratch with explicit headroom. HC fusion must use the previous sublayer mixer.
Preserve slab/in-flight/admission proofs before lending prefill reserve to decode.
Earlier wider-layer/router-event/cache-pin attempts were exact but slower and
were restored out in4442d372; their historical implementations and evidence are
in README52-58. Read-ahead removal loses3% on restored interactive responses.

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
