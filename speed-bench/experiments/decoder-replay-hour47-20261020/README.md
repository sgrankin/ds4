# Item47: bounded decoder replay

2026-10-20, source base `61bee1df`. **Opt-in qualified on three direct fixtures and one unbatched HTTP tool
workflow; defaults remain unchanged.** The primary server result is **7.03%
whole-workflow saving**, including nine fewer search-output tokens. The separate
exact 8,192-token direct result is 38.17%. The opt-in
`DS4_V41_DECODER_REPLAY_ROWS` accepts 0, 128, 256; default 0 preserves the native
exact decoder dependency suffix. The same binary runs the bracket 0,128,256,0.

## Fixed experiment

Apple M5 Max/128 GiB, verified 1017 IQ2_XXS/Q2_K weights, direct Metal sessions.
Context 16,384, requested prefill chunk 8,192, 64 GiB expert-cache target, native
routing, MTP off. Each arm runs three exactly 8,192-token prompts: a current fact
near the middle; a favorable fresh fact within the final 100 tokens; and an old
fact at one-eighth followed by a correction 512 tokens before the query tail.
Exact positions and shared token vectors are recorded by the probe.

Each prompt first receives a shared fixed-teacher answer plus EOS for paired
ΔNLL/KL. A separate fresh prefill generates the independent first greedy answer
with a 48-token cap. The timed practical scope is `greedy_sync_ms + greedy_ms`;
engine open, teacher work and analysis are separate. Retain first failures and
caps. Compare with both native brackets, report drift and plot the measured
speed/quality points. The teacher-perplexity 20% line is a reference, not a gate.

The [runner](run.py) clears inherited DS4/PT settings and records commands,
binary hash and source base. It runs under the existing pressure guard.
[Probe source](../../decoder_replay_bench.c), [paired analyzer](analyze.py),
[portable algorithm and bounds](../../../recipes/12-bounded-decoder-replay.md).

## State/publication contract

Encoder rows and compressed/index state remain complete. Decoder raw attention
is clipped to the replay start; missing keys are not zero-attended. The live
session is approximate before writes, sticky across continuations, and blocked
from canonical payload/layer/snapshot export and suffix capture, even with
original pre-RoPE tape. Invalidate/recompute or canonical load to obtain exact
state again. Runtime activation requires an existing wide sweep of at least
8,192 rows. The batched preset's ordinary 2,048-row HTTP path does not exercise it. Default
unbatched sync can; its separate qualification is recorded below.

## Phase 1 result: no practical speed gain

Completed 22:37 UTC with the same binary (SHA256
`711bf327c34885bf3d15c9a7994756e9e14890013bf78320380b65d076c597df`)
for 0,128,256,0. Source base `61bee1df`; the exact runtime patch and build log are
retained. Aggregate practical native bracket time was 69.586→69.042 s
(−0.781% drift), mean 69.314 s. Replay 128 took 70.697 s (**2.00% slower**);
replay 256 took 70.391 s (**1.55% slower**). These are sums of the three separate
direct prefill+first-answer timings, not a single agent task.

| Case | Native bracket mean | Replay 128 | Saving 128 | Replay 256 | Saving 256 | Native drift |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary |20.737 s|21.588 s|−4.10%|22.260 s|−7.35%|+2.80%|
| Favorable |23.728 s|24.854 s|−4.75%|24.093 s|−1.54%|−1.23%|
| Changed |24.850 s|24.255 s|+2.39%|24.037 s|+3.27%|−3.25%|

All 12 first greedy answers stopped and exactly matched their expected code;
there were no caps. Both native arms' 22 teacher rows and logits were byte
identical, including EOS. Candidate teacher argmaxes never flipped, but their
22 logit rows were all byte-different from native.

| Replay rows | Mean ΔNLL, nats/token | Mean KL | Maximum ΔNLL | Maximum KL | Teacher perplexity ratio |
|---|---:|---:|---:|---:|---:|
|128|+0.0000226651|0.00000263555|+0.000206096|0.0000229597|1.000022665|
|256|−0.000000231732|0.000000371817|+0.0000296724|0.00000319604|0.999999768|

These are tiny distribution changes on only 22 fixed teacher rows; correct
three-fact answers do not establish broad model quality. Timing drift is
material relative to the small candidate differences. The approximation flag
was true and payload/suffix-cache size APIs returned 0 on every candidate case;
native sizes remained available. This checks the intended publication boundary,
not a disk-cache performance gain. The guard recorded 537 normal samples and
zero swap growth; all four processes exited 0.

**Phase 1 alone did not earn promotion:** quality changes were small here, but reducing decoder rows did not
reduce practical completion time. No I/O bottleneck or cause is claimed from
this timing curve alone.

[Paired summary](analysis/phase1/summary.json),
[per-row quality](analysis/phase1/02-rows128-paired.csv),
[256 quality](analysis/phase1/03-rows256-paired.csv),
[first timings/answers/tokens/executions](evidence/phase1/),
[pressure summary](evidence/phase1/guard/pressure-summary.json),
[build](evidence/phase1/build.log), [source patch](evidence/phase1/source.patch).
Raw 11.38 MB logits per arm remain in the ignored workspace; their SHA256 values
are retained in the paired summary. No raw logits or binaries are published.

## Diagnostic profile and phase 2 method

A separate ordinary-only profile compared 0 and 128 using the phase 1 binary and
both `DS4_METAL_GRAPH_PREFILL_PROFILE=1` and
`DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE=1`. Each arm includes two
prefills (teacher and independent greedy). Its instrumented wall times are
**excluded from the practical curve**; this is a work/dispatch diagnostic.

| Across two sweeps | Native 0 | Replay 128 |
|---|---:|---:|
| Total layer records |126|120|
| Decoder layer records |46|40|
| Decoder output rows |48,300|5,120|
| Full decoder-layer pread records |40|40|
| Profile decoder map+seed time |8.362 s|9.385 s|

The decoder did much less row work but still read/mapped/seeded whole expert
layers. Rounded per-record sizes imply about **149 GiB of logical full-layer
pread bytes per arm**, from 3.73 GiB×20 layers×two sweeps. These are logical
read requests inferred from rounded logs, **not measured physical SSD traffic**:
OS file cache can serve them. No hardware I/O counter was collected. Profile
map+seed times do not establish that storage was the latency bottleneck or
predict the uninstrumented follow-up's result. The corrected summary parses
signed floating-point fields and retains the actual 126/120 and 46/40 counts.
[Profile summary](evidence/profile/profile-summary.json),
[commands and raw logs](evidence/profile/),
[profile guard](evidence/profile/guard/pressure-summary.json).

Phase 2 adds `DS4_V41_DECODER_REPLAY_SELECTED=1` when bounded replay is active.
Eligible Apple streaming decoder layers use the EXISTING selected-expert batch
path for 128/256 rows, skipping the full-layer read/map/seed and force-resident
path. Native router choices and mixture weights are retained; this is not
resident-expert substitution. Unsupported layers keep the ordinary full-layer
fallback. Encoder publication, replay-floor masking, approximate flags and
export/suffix-capture exclusions remain the phase 1 contract. Encoder lookahead
is kept below the decoder boundary and skips selected-layer targets; static
mapping is restored when transitioning from a full-layer path to selected work.

The phase 2 curve again brackets 0,128,256,0 with one changed binary; it must be
assessed separately from phase 1. The measured result follows; profile latency
is excluded from that curve.
[Phase 2 source patch](evidence/phase2/source-phase2.patch),
[build](evidence/phase2/build-2.log), [server checks](evidence/phase2/server-tests.log).

## Phase 2 result: selected experts make replay useful on these fixtures

Completed 22:48:21 UTC. The changed binary SHA256 is
`0b0981251d5ecad7e3da9ccfa705cfd40d8e6462f284cd37a77b81ae5406104d`.
`DS4_V41_DECODER_REPLAY_SELECTED=1` was fixed for the whole 0,128,256,0 bracket;
rows 0 does not activate selected replay. Native practical sums were
68.258/68.688 s (closing drift +0.631%), mean 68.473 s. Replay 128 took
42.335 s (**38.17% saving**), and 256 took 44.970 s (**34.32% saving**).
Against the faster native bracket, savings were 37.98% and 34.12% respectively.
These remain sums of direct prefill plus independent first answers, not
whole-agent time.

| Case | Native mean | Replay 128 | Saving 128 | Replay 256 | Saving 256 | Native drift |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary |20.557 s|12.713 s|38.16%|13.914 s|32.32%|+7.64%|
| Favorable |23.774 s|14.633 s|38.45%|15.498 s|34.81%|−1.47%|
| Changed |24.142 s|14.989 s|37.91%|15.559 s|35.55%|−2.99%|

All 12 first answers stopped and matched their expected codes. The 22 native
teacher rows and logits were byte-identical before/after. Both candidate arms
had zero teacher argmax flips; all 22 candidate logit rows differed bytewise.

| Replay rows | Mean ΔNLL, nats/token | Mean KL | Maximum ΔNLL | Maximum KL | Teacher perplexity ratio |
|---|---:|---:|---:|---:|---:|
|128|+0.0000325041|0.00000446931|+0.000299239|0.0000446463|1.000032505|
|256|+0.0000128427|0.00000116427|+0.000162426|0.0000142981|1.000012843|

The ordinary-case native timing drift is larger than aggregate drift and is
retained above. The result qualifies an **opt-in candidate on these fixed direct
fixtures**; three facts and 22 teacher rows are not general reasoning/coding
quality evidence. Replay 128 is the faster measured point; 256 changes the
teacher distribution less here. No universal winning setting is inferred.
All 440 pressure samples were normal, with zero swap growth. Candidate
approximation flags and zero payload/suffix-cache size checks remained correct.

[Paired summary](analysis/phase2/summary.json),
[128 row comparisons](analysis/phase2/02-rows128-paired.csv),
[256 comparisons](analysis/phase2/03-rows256-paired.csv),
[measured curve](analysis/phase2/tradeoff.svg),
[first outputs/settings/timings](evidence/phase2/),
[guard](evidence/phase2/guard/pressure-summary.json).

### Selected-path diagnostic confirmation

A separate instrumented ordinary 128 run reported `selected_layers=20 full_layers=0`
on both teacher and independent greedy sweeps. Across them it retained 40 encoder
full-layer pread records and issued **zero decoder full-layer pread records**,
versus 40 decoder records in the earlier profile. This confirms the intended
whole-layer-path avoidance; it does not measure physical SSD traffic or imply
that selected-expert reads disappeared. Its 30 pressure samples were normal,
with zero swap growth. All instrumented profile latency remains excluded from
the uninstrumented curve.
[Summary](evidence/selected-profile/profile-summary.json),
[raw logs and first answer](evidence/selected-profile/01-rows128/),
[guard](evidence/selected-profile/guard/pressure-summary.json).

## HTTP qualification: measured within a native bracket

The default unbatched server calls full session sync before entering any
quantum loop, so a sufficiently wide prefill can already activate replay with
no new server API. Omit `--batched-session`; an explicit 0 is invalid. The
batched preset's ordinary 2,048-row chunks do not activate this wide path.
The first 340-audit-line fixture rendered 8,162 input tokens, below the 8,192
activation gate. Root intentionally stopped its owned guard; that attempt is
**ineligible**, with no timing or task-quality claim. Its first wire outputs,
requests, server logs and interruption reason are retained in
[ineligible attempt evidence](evidence/http-ineligible340/qualification-status.json).
The corrected 360-line fixture adds explicit actual-usage and replay-log
eligibility checks. Its first complete bracket and the subsequent policy fix are
recorded below. This corrects fixture geometry rather than selecting a better
generated answer. Server quantum behavior is unchanged.

### Eligible HTTP360 result exposed a live-continuation policy bug

The eligible 360-line bracket completed 22:55:12 UTC; first request input was
8,623 tokens and candidate runtime logs proved 128-row/20-layer selected replay.
All three first workflows passed: actual note reads, function/path grades and
final archived-code/receipt/current-key answers were correct, with clean exits.
Workflow times were **55.80993 / 113.32834 / 54.75790 s** for 0/128/0. The native
mean was 55.28392 s: the candidate was **104.99% slower**. This is a valid
negative HTTP result retained in full, not a successful feature latency claim.

Native R1/R2/R3 cached-token counts were `[0,8680,8901]`; candidate counts were
`[0,0,0]`. The old approximate-state live policy required a nonempty suffix
profile. With replay enabled and suffix reuse off, the profile stayed empty,
so R2 and R3 discarded valid matching live state and recomputed their complete
prompts. This differs from blocking approximate state from canonical storage:
that exclusion remains necessary while matched live continuations remain useful.
[Unfixed summary](analysis/http-unfixed/summary.json),
[all first wire/requests/results](evidence/http-unfixed/),
[guard](evidence/http-unfixed/guard/pressure-summary.json).

A narrow server policy fix now gives replay-only live state its own profile,
scoped by model/forward profile, replay rows, selected-path setting and optional
client hint. Existing token/parent checks and canonical export/capture guards
remain in force. The suffix-reuse-enabled profile contract is unchanged.
[Exact patch](evidence/http-profile-fix/http-profile-fix.patch),
[build](evidence/http-profile-fix/build-3.log) and
[server-suite pass](evidence/http-profile-fix/server-tests-3.log).
A fresh 0/128/0 bracket uses the same fixture definition and corrected policy,
but its new owned path changes BPE tokenization: all fixed arms render 8,625
input tokens, versus 8,623 in this unfixed bracket. Compare timing within each
native bracket; a raw unfixed-to-fixed duration difference is not a token-identical
binary-only comparison. Source/logs establish the policy mechanism.
**The fixed-policy result follows below.**
The direct selected-curve result above is unchanged; no whole-agent result is
inferred from either HTTP version.

### Fixed HTTP360 result: bounded whole-workflow qualification

Completed 22:59:17 UTC with one fixed-policy binary in the 0/128/0 bracket,
SHA256 `4a8c959bc6668f5a801656a0812271ff5318ee23865e24d4d369ee81a1e29cd2`.
All three first search/read/final workflows passed, each with an actual local
file read, valid function/path arguments and the expected final JSON. Every
first request had 8,625 input tokens. Native workflow times were 52.305368 and
52.445182 s; mean 52.375275 s, closing drift +0.2673%. Replay 128 completed in
**48.691598 s: 7.033% saving**, or **6.909%** versus the faster native control.
Startup is excluded; the three requests and actual file read are included.

| Arm | R1 / R2 / R3 output tokens | R1 / R2 / R3 cached tokens |
|---|---|---|
| Both native controls |57 /71 /26|0 /8682 /8905|
| Replay 128 selected |48 /71 /26|0 /8673 /8896|

The candidate's two follow-up requests visibly matched live prefixes with
`approximate=1`; valid private continuation survived while canonical export,
SSD prefix storage and suffix capture remained excluded. The difference of 9
cached tokens tracks the 9 fewer search-output tokens. The **whole-workflow
package gain includes that shorter initial search output**. It is not a pure
prefill speedup, and no HTTP KL/ΔNLL is assigned to these diverging histories.
All 153 guard samples were normal with zero swap growth; all servers exited 0.

The exact 8,192-token direct probes saved 38.17% with 128 rows. This HTTP prompt
also processes a 433-token tail normally and performs real tool round trips;
its measured 7.03% belongs to that different scope. The profile fix, native
brackets and token geometry are retained; no token-identical unfixed/fixed
binary-only comparison or whole-agent/coding result is claimed.

**Decision: retain opt-in qualification for the measured direct fixtures and
this unbatched HTTP tool workflow.** Defaults/presets remain unchanged. Batched
2,048-row dispatch does not activate replay; this feature still requires a
wide eligible sweep and both replay-row/selected-path controls. Private replay
state remains unavailable as canonical SSD KV or relocatable suffix state.
[HTTP summary](analysis/http-fixed/summary.json),
[all first outputs and executions](evidence/http-fixed/),
[visible continuation logs](evidence/http-fixed/02-rows128/stderr.log),
[guard](evidence/http-fixed/guard/pressure-summary.json),
[fixed-policy checks](evidence/http-profile-fix/server-tests-3.log).

The linked agent still builds with default flags and its model-free checks pass:
[agent build](evidence/final-checks/agent-build.log),
[agent checks](evidence/final-checks/agent-tests.log). These checks do not measure
agent replay speed or task quality.

To reproduce the bounded HTTP bracket with a fresh owned path:

```sh
make -j2 ds4-server speed-bench/decoder_replay_bench
python3 speed-bench/ideas/pressure_run.py .workspaces/item47-http-rerun-guard -- \
  python3 speed-bench/experiments/decoder-replay-hour47-20261020/serve.py \
  --model "$PWD/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf" \
  --output .workspaces/item47-http-rerun
```

The driver fixes incremental schemas on, native routing, suffix reuse off,
application SSD KV cache off and unbatched mode; it checks actual prompt length
and candidate runtime activation. A different path can change BPE tokenization,
so retain each run's actual usage and compare within its own native bracket.

Credit [DeepInfer's decoder SWA replay](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md#L179-L191)
at `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`. Its cluster speed claims are
separate from this bounded Metal/Q2 experiment.
