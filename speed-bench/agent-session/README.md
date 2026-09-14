# Agentic session benchmark

This is a synthetic coding session through the real `ds4-agent` tool loop,
paired with a fixed token replay for controlled SSD performance experiments.
It complements short interactive prompts; it does not replace them.

The fixture is a standard-library Python payment reconciliation project. Three
user turns require inspecting code, observing failing tests, fixing money,
refund and retry handling, adding a date-filter API with tests, and producing
checked reports from 275 CSV records. Independent held-out checks run outside
the editable project after every turn. Existing test definitions and data must remain
unchanged; additional tests may be added. Task success, tool calls, context growth, turn wall time, prefill and
decode times are recorded. Model reasoning is enabled, matching agent defaults.

## Live task

From the repository root:

    python3 speed-bench/agent-session/live.py /tmp/agent-session-RUN

The runner copies the fixture, executable and Metal sources, isolates agent KV
files, fixes datetime/timezone and seed, and uses greedy sampling. Tools run in
the copied project. This runs actual model-selected commands; use a disposable
workspace. It never points tools at the ds4 checkout. `--env NAME=VALUE` enables
an experimental path; `--cache-gb N` sets the total expert-cache target.
`--prefill-chunk N` controls the agent prefill allocation; the replay equivalent
is `--candidate-env DS4_REPLAY_PREFILL_CHUNK=N`. The timeout is 1200 seconds
by default. Validation time is excluded from turn
wall time, but startup is reported separately. Tool and host work are included
in turn wall time; model timings alone are not time to task completion.

The live run must pass all three task stages and execute at least six tool calls
before emitting summary.json and replay.txt. A failure is a failed benchmark,
not a faster result. Live variants may take different valid tool paths: compare
success and elapsed time, and do not claim identical-workload speedups unless
the recorded token streams match.

## Fixed session replay

    make speed-bench/agent-session/replay
    python3 speed-bench/agent-session/replay_abba.py /tmp/session-ABBA \
      --replay /tmp/agent-session-RUN/replay.txt \
      --candidate-env DS4_METAL_V41_BATCH_SHARED_OVERLAP=1

`--order AB` is an exploratory screen; pair a promising result with `--order BA`
using the same binary/shader hashes before treating it as balanced evidence.
The default remains complete ABBA.

Repeat `--candidate-env NAME=VALUE` to test a combination against a control
with all those flags unset. Record single-factor studies before combinations.

Each replay retains the recorded system prefix, every user/tool-result prefill
boundary, and every generated token. Generated tokens are teacher-forced one
at a time through the ordinary decode API. It preserves evolving KV and expert
cache state within the session. Separate-process ABBA freezes executable,
shaders and tokens; it aborts if full-vocabulary logits at any phase boundary
or the complete final continuation snapshot differ. Large binary comparison
artifacts are hashed and deleted after checking; hashes and timings persist.

The replay excludes tools, sampling, tokenization and terminal rendering; use
it to attribute model runtime, and confirm useful gains with live tasks and
interactive probes. It does not simulate pauses, saved-session restoration,
images, compaction or speculative decoding. Trace extraction rejects compaction
and incomplete token blocks rather than silently fabricating a workload.
The token recording is specific to the DS4.1 tokenizer/model configuration.

The checked recording is `session-v1.txt`, SHA256
310159cecfb6688d65967cf30279e1626c15571dbedd02838549f0d1ff14f368.
It was captured from the passing live baseline on an M5 Max /128 GiB with
DeepSeek-V4.1-Flash-Q2, SSD streaming, context100000 and automatic cache sizing.
It includes20 tool calls (7 reads,6 shell commands,3 edits,3 directory lists,
and1 write),14 generation rounds, an1859-token system prefix,
3371 generated tokens and final context8710. Tool/user suffixes range62-826
tokens. Baseline user-turn wall times158.705/93.843/39.166s, total291.714s;
startup17.420s is separate. Model prefill94.280s and decode196.339s account for
nearly all turn time; the remainder includes tools and host overhead.
See baseline-live-v1.json and baseline-checks-v1.json for independent success
checks. This is a small synthetic working session, not coverage of long-context
compaction, every tool, or arbitrary coding-task quality.
`test_trace_session.py` checks extraction boundaries and rejection of
incomplete/unsupported traces.

## Exact-route prefetch oracle (diagnostic)

This deliberately uses future information. It measures scheduling opportunity,
not achievable generation throughput. Native routing and weights still execute;
every scalar route is checked and mismatches abort. Batched prefill is unchanged.
The first version starts selected SSD loading before the same layer's attention,
using the existing pending-load slot, workers and cache budget. It does not yet
model a learned predictor, wrong predictions, or multi-layer speculative queues.

    DS4_V41_ROUTE_RECORD=/tmp/routes.bin python3 speed-bench/agent-session/replay_abba.py /tmp/route-capture --candidate-env DS4_V41_ORACLE_PREATTENTION=1 --order A
    python3 speed-bench/agent-session/replay_abba.py /tmp/route-abba --routes /tmp/routes.bin --candidate-env DS4_V41_ORACLE_PREATTENTION=1

The complete recording is also checked in as `session-v1.routes.bin.gz` (about
1.5 MB), with provenance in `session-v1.routes.json`. For the default session,
`--routes` without a filename uses this recording, so no capture is required:

    python3 speed-bench/agent-session/replay_abba.py /tmp/oracle-ABBA --routes --candidate-env DS4_V41_ORACLE_PREATTENTION=1

Both comparison variants load and validate the frozen route file. Only B starts
loads before attention. The route file must come from the same token recording,
model and scalar schedule. Its header checks model dimensions; position, input
token, layer and ordered expert IDs are checked during execution, and unused
trailing records cause failure. Files are native-endian uint32 data, version 1,
for local experiments; the harness records their SHA256. Recording refuses to
overwrite an existing file. Use one session per process, ordinary nonvision,
single-GPU SSD mode. Do not enable oracle flags for real agent use.

A second bound adds `DS4_V41_ORACLE_DEFER_CHECK=1` alongside pre-attention mode.
It copies each native GPU route into a tiny per-token trace and checks all routes
at token completion, allowing host address binding from the oracle without the
per-layer router wait. Native routing weights remain in use. This is deliberately
stronger future knowledge than a fallible prefetch predictor can safely exploit:
a real implementation would need GPU-side validation and a miss/fallback path.
Use it to distinguish synchronization opportunity from I/O overlap, never as a
production optimization. Any route mismatch still fails the entire run.

For diagnostic attribution, `DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1` plus
`DS4_REPLAY_PROFILE_MEMORY=1` reports cumulative `streaming pending ... wait_ms`:
time actually spent in pending-read joins, excluding overlap and installation.
Collect these separately from uninstrumented headline timings.

`python3 speed-bench/agent-session/route_stats.py /tmp/routes.bin` reports cheap
previous-token, frequency and previous-layer transition prediction baselines.
It trains on the first 60% and tests on the last 40% of one recording. This is
an exploratory within-session split, not independent-task validation. Recall is
across all experts, not just misses; candidate budgets and all-selected coverage
are reported explicitly. The previous-token baseline always offers only its
original selected set, even in the larger-budget table.

To reuse an immediately preceding matching control, run `--order BBA` and then
`python3 speed-bench/agent-session/complete_abba.py PREVIOUS FOLLOWING OUTPUT`.
Use the previous frozen executable and run consecutively. The helper checks
binary, model, shaders, tokens, oracle data, environment and numerical identity.
The shared control is one observation, not an additional independent repeat.

An offline activation probe uses `DS4_V41_ROUTE_PROBE=1` together with route
recording mode. It applies each existing gate to the same layer's pre-attention
normalized activation, then measures overlap with the native post-attention
route. It never substitutes its predictions or prefetches them. This adds GPU
work/readback and its timings are not performance results; native logits/state
must match the baseline. Per-layer recall counters are printed at process exit.

The opt-in real predictor experiment can be screened without any oracle file:

    python3 speed-bench/agent-session/replay_abba.py /tmp/gate-prefetch-abba \
      --candidate-env DS4_METAL_V41_GATE_PREFETCH=1

It retains native exact routing and only speculates on cache loads. It is
currently slower on the short tool-session screen and is disabled by default;
see experiment66 in the SSD experiment notebook. Do not combine it with oracle
modes, image/quality execution, or the older asynchronous expert-load experiment.

For the measured small-input tool session, `--prefill-chunk 2048` on ds4-agent
freed about4GiB for cached experts and reduced full replay inference time2.7%.
The replay equivalent is `--candidate-env DS4_REPLAY_PREFILL_CHUNK=2048`.
Short interactive latency was near-neutral; large16384-token input showed a
possible throughput penalty, so the general default remains unchanged. See
SSD experiments70-73 for samples and exact-output checks.

### Learned routing predictor study

Capture pre-attention BF16 activations, early-gate predictions and native labels:

    DS4_V41_ROUTE_RECORD=/tmp/new-routes.bin DS4_V41_ROUTE_FEATURES=/tmp/new-features.bin \
      python3 speed-bench/agent-session/replay_abba.py /tmp/new-capture \
      --candidate-env DS4_V41_ORACLE_PREATTENTION=1 --order A

Only A runs, so the candidate flag is not enabled. Both output files must be
new. Capture adds GPU readback and gate work; do not use its times as performance
results. Match its native route, logit and state hashes against the fixed replay.
The version1 feature header is eight native uint32 values: magic0x44534631,
version1, layers40, hidden5120, experts384, selected6, bytes-per-feature2,
reserved0. Each row contains position/token/layer uint32,5120 BF16 values, and
six int32 early predicted IDs. Native labels are in the paired route recording.
Full session capture is about1.39GB; keep raw data outside the repository.

    python3 speed-bench/agent-session/predictor_holdout.py /tmp/new-ttl-holdout
    python3 speed-bench/agent-session/train_route_predictor.py \
      /tmp/new-features.bin /tmp/new-routes.bin /tmp/new-training \
      --holdout-features /tmp/new-ttl-holdout/features.bin \
      --holdout-routes /tmp/new-ttl-holdout/routes.bin

Training requires NumPy and MLX and uses the GPU. Run capture, training and
performance benchmarks serially. Chronological60/20/20 splits keep all layers
of each token together; validation selects the best epoch. Test and the separate
TTL-cache task are evaluation-only. These measurements lack demand cache-miss
and deadline labels and do not establish useful prefetch or runtime speed.

Cache-aware offline screening pairs `DS4_V41_ROUTE_CACHE=/tmp/new-cache.bin`
with native route recording. The16-byte uint32 header is magic0x44534331,
version1,layers40,experts384. Each row has position/token/layer then384 residency
bytes, taken before native scalar demand loading without cache aging changes.

    python3 speed-bench/agent-session/score_route_prefetch.py \
      /tmp/new-features.bin /tmp/new-routes.bin /tmp/new-cache.bin \
      /tmp/new-training/best.safetensors /tmp/new-miss-score.json

Default scores the last20% of scalar tokens. `--split all` is appropriate for
an independent evaluation task. It filters predicted residents and reports cold
miss coverage, false reads and reads on native all-hit layers. Confidence uses
uncalibrated6*softmax scores. This holds native cache evolution fixed: it omits
speculative eviction, contention, predictor overhead and readiness deadlines.

`train_miss_predictor.py FEATURES ROUTES CACHE OUTPUT` trains an alternative
binary cold-demand objective. Supply `--baseline-weights CHECKPOINT`,
`--holdout-features FILE`, `--holdout-routes FILE`, and `--holdout-cache FILE`.
It selects checkpoint and one global score threshold using validation only,
maximizing covered misses with unnecessary reads <=10% of native demand. That
budget is not guaranteed on test or other tasks. `--warm-start` initializes
from the route checkpoint and fine-tunes at a lower learning rate. Both models
rank six cold candidates before thresholding and can abstain entirely. This
still assumes fixed native cache residency, not a deployed prefetch simulation.
