# Serving project journal

Last updated: 2026-10-25. This is the compact handoff for resumed work and context
compactions. Keep the current state accurate and append short decision/progress
entries at meaningful checkpoints. Raw experiment output stays local.

## Clef family — implemented and qualified, 2026-10-25

On M5 Max128GB, native Zig/Metal now serves Clef-Flash, full Clef and Clef-Omni.
All three pinned INT4 bundles are downloaded/verified; their scoped artifact
identities and22 downloader checks prevent metadata-name collisions. Production
has no Python/MLX dependency. Flash/full share a dense hybrid executor; Omni
has a sparse thinker, DeepStack vision and audio encoder. Record/tokenizer,
learned GPU joint head, CPU media and actual serving are connected. See
[setup/APIs/limits](../docs/CLEF.md) and [component sources](models/clef/README.md).

Both `/v1/systemone` and `/v1/decisions` share one native operation. Decision
models are stateless/non-streaming; the generator retains Responses HTTP/WS
and multiple conversations. `--decision-model DIR` adds independent helpers;
`--model CLEF_DIR` is decisions-only. Named `ds4-local --server` profiles select
all three bundles. Models discovery reports actual loaded capabilities/media.

The complete GPU head is the ordinary default:11–14% lower short Flash request
time than its CPU head, about67/108/68/90ms on four records. CPU matrices are not
expanded for this path, and head pipelines prepare before readiness. Omni's
one-expert tile map is also default,40–46% lower ordinary whole-decision time,
bit-identical to the corrected old gather on controlled short/3388/6472-token
cases. Audio16-row GEMM saves44%, reaching22.4ms vs local MLX21.5ms on the short
component; longer34.1ms vs28.2ms remains a measured gap. Diagnostic fallbacks
remain, not a requirement to enable the ordinary fast paths.
[Head](../speed-bench/experiments/clef-native-20261024/HEAD.md),
[Omni](../speed-bench/experiments/clef-native-20261024/OMNI.md),
[media](../speed-bench/experiments/clef-native-20261024/MEDIA.md).

Numerical qualification exposed/fixed actual implementation defects: already
folded RMSNorm1+w, packed QMV's invalid K768 tail, and cast-before-clamp overflow
above32767 routed rows. The latter left unwritten outputs beyond4095 Omni text
tokens; the model-free regression and corrected old/new long comparison pass.
These are distinct from acceptable quantization/approximation tradeoffs.
Thirty tokenizer/record cases and eight media-placement cases match exact IDs,
spans and positions; tower cosine comparisons are recorded separately.

Flash completes2,000/2,000 fixed Decision Index0.2.1 requests (500 each): API-Bank
91.0%accuracy, BANKING77 92.75%macro-F1, CLINC150+OOS87.20%macro-F1, When2Call
63.8%accuracy. NLL0.59825 nats/decision, Brier0.25068, ECE0.09340. Its first100
preserve all same-weight MLX top choices (KL8.57e-5, delta-NLL+0.000158).
Full and Omni each complete the same100-request smoke. Public BLINK samples
complete84/84: Flash18/28, full22/28, Omni19/28. These are four-task text and
seven-subtask vision samples, never full-index scores. Vision task parity against
MLX remains unmeasured; component math and actual task quality are distinct.
[Broader quality](../speed-bench/experiments/clef-native-20261024/BROAD_QUALITY.md).

Controlled native HTTP media tests pass5 requests each for Flash/full and11 for
Omni: image contrast, question-specific OpenAI images, explicit/encoded video;
Omni also follows spoken three/seven and explicit/encoded soundtracks while
answering the video's final color. Combined three-model/tower RSS76.9GiB is an
observation, not minimum RAM. Pressure stays normal with no swap growth.
[Media tasks](../speed-bench/experiments/clef-native-20261024/MEDIA-TASKS.md).

Shared runtime owns Metal device/queue and owner-aware submitted groups. DS41
Core owns actual Host/Metal state, including caches, pipelines, pools and
profiling; workers capture their owners explicitly. Two actual DS41 engines
preserve nine full-vocabulary logit comparisons exactly through interleaving,
closing A and continuing/fresh-graph B. They share a referencecounted process
file lock, so one Core's close cannot drop the surviving Core's protection.
Other GGUF/TP/CPU paths retain previous admission. Final DS41 controls/candidate
are788.23/787.68/783.09ms; all28 mixed protocol/SDK/cancellation checks pass.
[Ownership](../speed-bench/experiments/clef-native-20261024/OWNERSHIP.md).

All four existing native model families match frozen control on32 full-vocabulary
FP32 rows byte for byte (KL/delta-NLL0). Engine-aware thinking/memory callers
replace ambient shape assumptions; ten metadata-only CLI token streams match.
Full host/server/agent/recovery/script checks pass, with no production build
warnings. Flint's larger reported buffer estimate is an accounting correction
for its existing65536-token allocation blocks, not newly allocated memory.
[Existing-engine regression](../speed-bench/experiments/clef-native-20261024/ENGINE-REGRESSION.md).

The final local stack is two coherent changes: model/download implementation
`oxmmmnktsvnl` and runtime/serving integration `wzvyrkmkpkyn`. Earlier plan/download
checkpoints were consolidated. Main stays at published steering`ce3afb7999b7`;
new Clef work is local. Raw logs/weights/dumps/plots stay ignored, while concise
reports, standard numerical recipes and necessary compact golden fixtures are
tracked. No model/test server remains running after qualification.

Remaining proposals: aggregate automatic model-memory reservations, a server
registry for multiple generators, reusable decision features, broader task
coverage and additional hardware. Current GGUF automatic admission budgets one
engine; qualified co-hosting commands use explicit cache budgets (32GiB main+
Flash,8GiB each in the two-Core proof). Helper-first loading alone does not alter
that planner. These proposals do not become claims of implemented support.

## Cooperative helper scheduling — measured opt-in, 2026-10-25

Source inspection found a practical remaining mixed-workload limit: Clef encodes
its entire text backbone into one GPU command, so a long full-model request can
hold the device for seconds. A bounded experiment adds `DS4_CLEF_LAYER_QUANTUM`
(default0, unchanged one-command behavior; test1/4/8), splitting only at complete
layers. The same weights, buffers and arithmetic remain; completed groups provide
other models/cancellation a scheduling point. Runtime submission admission now
queues new owner groups behind already waiting owners, while allowing dependent
same-owner submissions to avoid resident/streaming deadlock. The actual CPU
lease fixture passes that invariant. The actual full-Clef2881-token helper screen
is complete: q0/1/4/8/0 with two repeats per point. Main completion4.14→0.885s at
q1, helper3.99→4.20s (+5%); observable main gap3.92→.323s. All25 helper probability
vectors are bit-identical (KL/delta-NLL0), main answer remains57 tokens. Closing
control matches opening. q4 main2.17s/helper4.16s; q8 main3.98s/helper4.00s.
Keep default0: short-helper overhead and other contention patterns are outside
this screen. [Measured tradeoff/reproduction](../speed-bench/experiments/clef-native-20261024/SCHEDULING.md).

Reuse `tests/mixed-serving.nu --mixed-timing-only --helper-repeat 250` with Flint
and full Clef for a realistic long-helper pause. Existing WebSocket event logs
already record delta timestamps; no new timing harness is needed. Compare whole
main/helper latency, longest delta gap, ordinary Flash/full/Omni overhead and
unchanged decisions. Plot the measured group-size tradeoff and choose a default
only if the responsiveness gain justifies solo cost. Source/private builds are
under `.workspaces/clef-native-20261024/scheduling-build/`. The standalone Pareto
PNG/SVG and JSON summary are in `.workspaces/clef-scheduling-20261025/`; only the
concise report and plotting recipe are tracked. The slice completes within its
one-hour checkpoint. GPU was explicitly returned to broad quality at03:37UTC
with340 Flash rows remaining.

Final ownership hardware gates are also complete: actual two initialized Metal
owners survive A cleanup; two actual nativeDS41 Core engines preserve nine frozen
full-vocabulary logit comparisons exactly, including A close/B continuation and
fresh B graph. Host pools pass concurrent5120/4096 shapes. NativeDS41 engines
share a referencecounted process file lock, preserving accidental second-process
admission protection; other GGUF/TP/CPU paths retain their previous guard. Final
DS41 control/candidate/closing medians788.23/787.68/783.09ms, all28 mixed checks
pass, pressure normal/no swap growth. [Ownership report](../speed-bench/experiments/clef-native-20261024/OWNERSHIP.md).
All five frontends build; engine-aware thinking/memory calls replace old ambient
shape assumptions, and ten metadata-only CLI token streams match frozen controls.
Full host/server/agent/recovery/script suite passes. The final four-family frozen
logit regression now passes all32 full-vocabulary rows byte for byte.

Automatic cache admission remains a per-GGUF-engine planner, not an aggregate
allocator. Merely loading helpers first would not reserve them: it uses device
recommended capacity rather than current other-owner allocations. This slice
documents explicit co-hosting cache budgets (qualified32GiB main+Flash,8GiB each
in the two-Core proof); an explicit other-owner reservation is future work.

## Clef family and decision serving — plan, 2026-10-24

At the user's request, published the qualified residual steering and behavioral/
Zig-test commits through `ce3afb7999b7e97aa697dc3ec3520650c165a5f0` to `main`
and `origin`. Remote main matched the starting revision before publication and
the push was verified. This is the historical approved plan; the execution
checkpoint above records what is now implemented and actually qualified.

Implement native Zig/Metal **Clef-Flash, Clef and Clef-Omni**, both decision APIs,
and separate engine ownership for multiple loaded models in one process.
Keep the existing policy: no Python/MLX production inference; use reference
implementations, kernels, offline conversion and numerical oracles. Cleffa and
Decision Index are cloned locally under ignored `.workspaces/clef-references/`.
Use ordinary explicit engine records and a small loaded-model table; no plugin
framework or all-engine rewrite before obtaining the first Flash result.

### Execution and API boundaries

- Start with Flash text: exact record/token/span preparation, affine INT4
  backbone prefill and the learned joint schema head. Preserve all final hidden
  rows, selected output-embedding rows and question/option pooling spans; this
  is not last-token sampling, JSON generation or a full vocabulary projection.
- Flash is Qwen3.5-9B (32 layers, hidden4096); full is card-described Qwen3.8-27B
  with `qwen3_5` config (64 layers, hidden5120). Both have three GatedDeltaNet
  layers per full-attention layer. Reuse appropriate Qwen/Flint operations after
  checking their actual contracts; existing HC/PLE/indexed-attention graphs are
  not drop-in dense Clef graphs. Full should reuse the dense executor.
- Keep the joint head and media towers at their reference precision initially;
  use the existing affine quantization layout where possible. Profile CPU/
  Accelerate versus GPU head placement on real shapes. Repacking, mixed
  precision and kernel changes are allowed when measured gains justify them.
- One native decision request/result operation feeds `/v1/systemone` and
  `/v1/decisions`. Keep question order and typed option values separately from
  display names. Match each wire format's probabilities/score/confidence rules;
  Clef's max-probability wrapper differs from current TypeSafe confidence, and
  OpenAI does not publish a formula guaranteeing Luna-equivalent calibration.
  State the local confidence policy. These decision endpoints are stateless,
  non-streaming operations; existing Responses HTTP/WebSocket remains the
  generative protocol. Do not invent conversation references for decisions.
- Clef's supported primitives are noul/predicate, choice and score. Unsupported
  box questions/media forms must fail visibly. Text first is an intermediate
  milestone, not completed family support. Preserve each API's image placement
  semantics when adding vision. Omni audio/video needs a documented local media
  extension because neither decision contract covers its complete native input.
- Runtime owns shared Metal device/queue and process services; engine owns
  weights, pipelines, mappings/cache and execution state; request/session owns
  bounded continuation/features. Start with serialized GPU leases and bounded
  chunks/packets. CPU preparation/head work need not hold the GPU lease.
  Closing one engine must not shut down another's services. DS4's model mappings,
  expert residency/I/O and scheduling globals need real migration, not removal
  of the single-instance lock. Pin generative ancestry and durable state to model.
- Omni is a distinct Qwen3-Omni thinker: 48 layers, hidden2048,128 experts/top8,
  plus DeepStack vision and a 16kHz audio/log-mel frontend. Implement text MoE,
  images, audio, then synchronized video+sound separately. Skip unused talker/
  code2wav weights. M5 Max is the performance target; do not claim M1 validation
  from a fallback that only builds.

### Evidence and qualification

Use the [Decision Index kit](https://github.com/apolinario/decision-index/tree/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d)
directly through its HTTP/SystemOne engine. First selectively rebuild API-Bank,
BANKING77, CLINC150+OOS, CRUXEval and When2Call (plus BFCL when useful), run a
fixed small smoke subset preserving case groups, then the built-in stratified
~2000-request sample. Pin edition **0.2.1** for the published Clef card comparison;
report current public0.3 separately. The live0.4 headline contains private tests
and cannot be reproduced by this local run. A subset is never a full-suite score.
Reuse its public vision suite after image support; audio/video also need matched
reference fixtures and task checks. Keep downloaded evaluation data local.

Measure end-to-end p50/p95 latency and decisions/s, including tokenization,
preprocessing and head; separately record backbone/head/GPU time, queue delay,
input tokens, number of questions/options, peak memory, pressure and swap.
Compare cold input, repeated state with changed questions, long input, and mixed
main-model/helper workloads. Avoid holding a blanket GPU lease across long CPU
work. Qualify A→B→A continuation/probabilities, closing either engine, failed
startup, cancellation and wrong-model ancestry rejection.

Initial performance target: match or beat the fastest applicable local reference
on matched weights/shapes/settings; investigate repeatable >5% deficits. Existing
single-model latency should remain within ordinary measured variation (use5% as
an investigation threshold, not permission to stack regressions). Report task
accuracy/F1, coverage/errors, decision NLL/KL, Brier/calibration and top-choice
agreement. First separate runtime drift from quantization loss using the same
weights. Evaluate quantization tradeoffs rather than rejecting every changed
answer. Plot quality/latency/memory Pareto points for tunable precision, caching,
batching and packing; optimize the measured bottleneck only.

External baselines, **not our results**: [Cleffa](https://github.com/zknpr/cleffa/tree/24fc91e28d154ff8dadfa58462a30b437413c3f4)
reports M5 Max128GB warm engine time including its CPU head, excluding encoding/
HTTP: Flash122ms/full439ms at346tokens;476ms/1580ms at1382tokens. Use matching
local runs rather than treating hosted GPU/network times as an M5 speed target.
The [Flash quant card](https://huggingface.co/mlx-community/clef-flash-4bit)
reports a sampled0.2.1 index54.65 versus55.63 BF16,96.4% top-answer agreement;
the [full quant card](https://huggingface.co/mlx-community/clef-4bit) reports
57.02 versus57.96 INT8,98.1% agreement. Their sampled and full-card scores differ.

Pinned references/download candidates (sizes are decimal on-disk, not RAM):

| Model | Official revision | INT4 candidate and revision | Size |
| --- | --- | --- | ---: |
| Flash | `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f` | `mlx-community/clef-flash-4bit` at`b5adbe2c3a95200d4bf04a0727c8ec457157d5e1` |6.194GB |
| Full | `a30162b99d68a481d8bd75d38df069a0cf83df40` | `mlx-community/clef-4bit` at`b973b6f405d02f8e9b83208c7e82ddf43a742eb6` |16.311GB |
| Omni | `df2d7c9505e62ba8a527b147513a4fa66a77dc5d` | `mlx-community/clef-omni-4bit` at`040a6e49d2e5d2f4f3ba68e70a6a0654833e23e1` |19.773GB |

Official source/configs: [Flash](https://huggingface.co/Cloudflare/clef-flash),
[full](https://huggingface.co/Cloudflare/clef), [Omni](https://huggingface.co/Cloudflare/clef-omni).
Protocol sources: [TypeSafe OpenAPI](https://api.typesafe.ai/openapi.json),
[TypeSafe confidence](https://docs.typesafe.ai/confidence),
[OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions).
Use CPU/high-precision reference outputs for Omni; the reference implementations
document PyTorch MPS numerical problems. Preserve credit/licenses when adapting
kernels. The executable sequence and hour budgets are at the top of
[next experiments](../speed-bench/ideas/next-experiments.md).

## Completed follow-up — behavioral testing and Zig fixtures, 2026-10-24

The user asked for actual behavioral evidence and consistent Zig tests after
pointing out that changing logits did not prove useful steering. This bounded
follow-up generates and manually inspects answers to four held-out neutral
questions: refrigerator, lightning, RAM/SSD and proportional recipe arithmetic.
Each condition uses independent stateless Responses requests and greedy decoding.
Runtime is the existing private b7679480 build; no inference math changed here.

With the existing small residual banks, additive strength +1 gives:

| Model | Final answer words, zero → add+1 | Observed effect |
| --- | ---: | --- |
| Flint | 660 → 420 (-36.4%) | All four shorter, complete, coherent and core-correct |
| DS41 | 787 → 467 (-40.7%) | All four shorter, complete, coherent and core-correct |
| Qwen | 1361 → 582 (-57.2%) | All four shorter, complete, coherent and core-correct |

Flint additive -1 makes all four longer (+25%), showing bidirectional style
control. Projection +0.1 also compresses Flint (-16.1%) and DS41 (-15.5%) totals,
but its per-prompt direction is not uniformly consistent on DS41. Projection
is sign-invariant in its axis and changes the existing component; do not infer
the desired behavior from target-minus-control orientation as with addition.

These are four-question demonstrations, not broad quality or refusal-ablation
validation. Flint retains incorrect extra numerical claims in lightning answers.
Strong settings cause repetition/caps; coherent verbose output can also hit a
cap and must not be confused with collapse or credited as succinctness.
Flint's baseline needed 2048 tokens; Qwen's two capped baseline cases were
extended 512→1024 and exactly preserve their original text prefixes. Comparisons
use natural completed answers, with changed ceilings explicitly documented.

GLM's larger baseline timed out at 240seconds (~0.66t/s). A bounded arithmetic-only
check at 128 tokens gives correct but capped zero/project +0.1 answers, while
add +1 completes in 24 words/38 tokens with correct 75×10=750g. This is one-case
completion evidence, not a general GLM style/length percentage.

The two new standalone steering fixtures now use handwritten Zig; C predecessors
are removed. Public ABI coverage remains through generated C declarations.
Bank preparation checks moved from the private C session fixture to a Zig unit
test beside the pure helper. The converted GPU oracle passes 342 cases plus 18
norm checks (max nonzero gap 4.77e-7, zero bit-exact); converted real-model session
checks pass Flint in both modes with independent live state, batches and exact
full eight-token packet replay. Pure bank unit 1/1 and full host/server/agent/recovery/
script suites pass, without build warnings. Prior four-family C fixture results
remain earlier evidence; this conversion re-ran its real-model checks on Flint.
Older tests that directly exercise retained private C stay with that owner.

[Full behavioral results and reproduction](../speed-bench/experiments/steering-behavior-20261024/README.md).
Raw requests/answers/logs and useful workspaces remain ignored under
`.workspaces/steering-behavior-20261024/`. Main is unchanged; this is a local
validation/test-language follow-up, with the production executables unchanged.

## Runtime steering checkpoint — 2026-10-24

Private jj change `sysvyppyvons` adds selectable **project** (default) and **add**
through `--dir-steering-mode` in CLI/agent/server/bench and engine options. The
user accepts a few percent overhead, superseding the original zero-slowdown
requirement. Old FFN-output vector compatibility is explicitly unnecessary.
The default hook is the residual after FFN/full-block publication; the attention
flag optionally edits its intermediate residual too. Each HC stream uses the
same hidden-width direction independently. DS41/Flint preserve ordinary BF16
publication, edit it, then round the result back to BF16.

Project normalizes each nonzero row and uses full per-layer user strength.
Add preserves learned magnitudes and divides strength over nonzero selected
rows, matching our local oMLX convention (`omlx/patches/steering.py`). Zero rows
mask layers. No runtime stacking: additive banks can combine offline with layer
budgets accounted for; one summed vector generally cannot reproduce sequential
projections. Mode/bank are immutable at startup; live strengths remain in user
units, affect subsequent computation, and retain earlier KV influence.

All four installed native models pass both modes: DS4.1 Flash 1017, Flint,
Qwen3.8 Flash Next and GLM 5.3 Flash. Checks cover exact zero scalar logits,
independent live settings, future-session defaults and mixed/homogeneous batches.
DS41/Flint batch gaps are zero; Qwen/GLM MTP replay gaps stay below 0.00021 logit.
Flint's full eight-token packets match scalar replay exactly in three cycles
per mode. The shared GPU oracle passes 342 cases plus 18 weighted-norm checks
with zero drift; three Flint hardware schedules pass math checks on M5 Max.
Host/script/recovery checks pass, including a small builder contract test in
`tests/verify.nu`. Localhost recovery required running outside the sandbox.
Retained C Pro/full-GLM paths and legacy batch/mode dispatch are adapted and
syntax-checked; their weights and physical Pro-device performance are unqualified.

Fresh banks use four succinct/verbose prompt pairs for DS41/Flint and two for
Qwen/GLM, captured from post-block residuals (HC mean for extraction). They
qualify extraction and execution, not a useful refusal/style direction. Builder
supports raw-add/unit-project output and layer masks; the strength sweep is Nu.

Short mirrored screens, two samples per state, positive means more elapsed time:

| Native model / mode | Prefill | Fixed decode |
| --- | ---: | ---: |
| Flint project, strength 0.1 | +0.41% | +2.43% |
| Flint add, strength 1 | +7.42% (about 26ms) | +1.52% |
| DS41 project, strength 0.1 | +4.73% | +5.12% |
| DS41 add, strength 1 | -2.13% | -0.41% |

Flint uses 256+256 tokens; DS41 uses 256+64, chunk 2048 and a 32GB expert cache.
Controls vary by a few percent, so these are coarse screens, not universal
speed promises; DS41's negative add point is not an established speedup. One
bounded Flint additive condition-hoist is retained with preserved BF16 math;
its earlier 5.47% decode screen is historical, not a precise causal comparison.
Failed derived-row machinery is removed from the runtime, with source/evidence
preserved locally. No further optimization loop is scheduled for this slice.

Automatic KV SSD/suffix reuse stays excluded when a bank is loaded, even at zero;
expert-weight SSD streaming remains available. New-hook long-append/context and
Qwen/GLM latency, calibrated behavioral/quality curves and steering-aware cache
identity/restoration remain separate future experiments. No automatic quality
rejection based on NLL/KL is implied.

All five root executables are installed byte-identically from
`.workspaces/steering-20261024/residual-final/bin`; source/results/raw banks stay
local below `.workspaces/steering-20261024/`. Main/origin remain at published
`2e1502a322ec`; no steering push or main advancement.
[Reproduction and results](../speed-bench/experiments/steering-20261024/README.md),
[bank conventions and recipes](../dir-steering/README.md).

## Published checkpoint — model retirement, 2026-10-24

The user ended the ideas discussion and authorized retiring GLM 5.2 and original
DeepSeek V4 Flash, then advancing and pushing main with the qualified native
runtime stack. Keep V4 Pro in C and preserve shared full-GLM-5.3 DSA support.
New constellation/voice implementations remain proposals. Source cleanup and
qualification are complete. Main publication advances from
`635d5d534e77f723ac3677e3840b783fddd24af4` through the qualified native runtime
`06128389a665`, the separate design journal update and the cleanup commit
containing this entry. Origin was fetched again with no divergence. Downloaded
checkpoints and historical evidence are preserved; retired active support and
advertising are removed. No model server was running at the start, and none
is left running by qualification.

Cleanup removes the old Flash profile, Flash-only scheduling/controllers and
hotlist, GLM 5.2's builtin hotlist, retired download targets, request aliases and
obsolete Flash-only launchers/fixtures. Shared DSA code is now explicitly the
full GLM 5.3 profile: admission requires its GGUF `general.version="5.3"`,
verified against the maintained converter and a 4 MiB header range from the
catalog checkpoint. No full GLM 5.3 weights were downloaded or run. Pro's
profile, hotlist and previous arithmetic fallback paths remain; neither Pro nor
full GLM 5.3 has been hardware-qualified in this cleanup. Native DS4.1, Qwen,
GLM 5.3 Flash and Flint arithmetic is unchanged.

The placement fixture previously assumed 43 Flash layers; it now tests Pro's
61 layers and is included in `tests/verify.nu` (132 checks pass). Host fixtures,
server/agent tests, localhost recovery and download/launcher/driver checks pass.
Raw logs stay under `.workspaces/model-retirement-20261024/`; a concise final
report records scope and reproduction commands in
[model retirement](../speed-bench/experiments/model-retirement-20261024/README.md).
GPU state/kernels, Qwen/GLM state recovery, four repeated F16 suffix relocations
and all 103 four-model Responses/WebSocket protocol checks pass. Qwen/GLM
answer probes pass 14/14 each and Flint 16/16. DS4.1 remains 11/14, with complete
check/answer records matching the frozen control's three existing disconnect
answer failures. The trained DSpark admission/tensor oracle passes, including
cross-family rejection, 49 finite tensors and exact repeated forward logits.
All ten pressure-guarded children exit zero. `make all` is warning-free and all
five root executables are byte-identical to the qualified installed binaries.
Pro/full GLM 5.3 remain C paths; their native ports and new constellation/voice
engines remain future work. GLM bounded-SSD performance remains backburnered.

## Native model constellation — direction clarified 2026-10-24

The user wants a constellation of generative, decision and extraction models:
Clef, Clef-Flash, Clef-Omni and small specialists alongside the existing engines.
DS4.1 + Flint was an example, not a limit on model choice. This records the
desired scope and implementation constraints; the new specialist engines are
not implemented or locally qualified. The cleanup/publication task above is
separate from implementing those future capabilities.

### Implementation policy

- No production Python inference, Python sidecars or Python implementation
  embedded behind a wrapper. Follow the Flint approach: study reference code,
  learn its algorithms, packing, precision boundaries and optimized kernels,
  then implement native model execution under our serving runtime. Python
  remains useful for offline conversion, numerical oracles and measurements.
- Reusing a C/C++ library is an option, not the boundary of what we can support.
  Port or adapt useful kernels and execution schedules when that better serves
  our Apple hardware. An existing implementation's memory/precision constraints
  are not requirements for our implementation. In particular, cleffa's BF16
  weights and large resident footprint do not prescribe our Clef layout.
- Metal, CPU/Accelerate and Core ML/Neural Engine are all valid execution paths.
  Choose by measured task latency, memory, quality and interference with other
  loaded models. A CPU/ANE helper might overlap main-model GPU work; shared
  memory bandwidth and actual accelerator placement still need measurement.
- Keep native Zig ownership and model-specific execution components, with narrow
  platform bridges where useful. Joint heads, encoders, tokenization and media
  preprocessing are part of support: loading a compatible backbone alone does
  not implement a decision model.
- Expose the operations a model actually supports: generation, typed decisions
  and/or extraction. Do not force encoder/head workloads through token sampling
  or allocate conversational KV for a model that does not need it. Cache shared
  input representations where the model permits; cross-model KV is not reusable.
- Keep separate engine ownership, model-pinned conversational state, shared
  scheduling and one combined memory/pressure budget. GPU transaction exclusion
  must not unnecessarily serialize an independent CPU/ANE helper. Use actual
  implementations to shape the common interface, not a speculative framework.
- Reformatting, repacking, quantization and mixed precision remain first-class
  choices for every model, including specialist heads and media encoders. Keep
  sensitive tensors at useful precision when measurements justify it. Compare
  end-to-end latency, working memory and quality, including conversion/load cost
  and the cost imposed on the main model; record Pareto curves for tunable choices.

### Candidate models and references

These are researched candidates, not a promise that all checkpoints, modalities
or reference-runtime features already work in our server. Smaller specialist
selection remains subject to practical task measurements.

| Candidate | Intended role and native input modalities | Implementation references |
| --- | --- | --- |
| Clef 27B | General typed decisions; text, images, video frames | [Official model](https://huggingface.co/Cloudflare/clef), [MLX backbone + joint head](https://huggingface.co/mlx-community/clef-4bit) |
| Clef-Flash 9B | Lower-memory decisions; text, images, video frames | [Official model](https://huggingface.co/Cloudflare/clef-flash), [MLX implementation](https://huggingface.co/mlx-community/clef-flash-4bit) |
| Clef-Omni 30B-A3B | Sparse decisions; text, images, audio, video with sound | [Official model](https://huggingface.co/Cloudflare/clef-omni), [complete quantized MLX implementation](https://huggingface.co/mlx-community/clef-omni-4bit) |
| Laya family | Small text-only decisions; English/multilingual/task-tuned checkpoints | [Models](https://huggingface.co/convaiinnovations/laya), [MLX encoder and decision heads](https://github.com/mizorewww/laya-mlx) |
| GLiNER2 base / GLiNER2.5-Decide | Text-only schema extraction / dynamic-label classification | [Base MLX port](https://github.com/Andrew-Chen-Wang/gliner2-mlx), [Decide MLX port](https://github.com/souuzaa/gliner2-mlx), [base Swift/MLX port](https://github.com/MacPaw/Gliner2Swift) |
| GLiClass Edge Core ML | Very small text-only router/classifier, including CPU/ANE evaluation | [Application-tuned model and runtime](https://huggingface.co/FluidInference/gliclass-edge-apps-coreml), [measurements and limits](https://huggingface.co/FluidInference/gliclass-edge-apps-coreml/blob/main/RESULTS.md) |

[Cleffa](https://github.com/zknpr/cleffa) is a particularly relevant C/Objective-C/
Metal reference, explicitly derived from ds4's approach. It implements dense
Clef variants and Omni with media processing; the head mostly uses CPU Accelerate,
with large memory-side projections on GPU. Its BF16-only footprint and unfinished
Omni long-context performance are opportunities to improve, not constraints to
inherit. The MLX ports provide alternative quantized implementations and Apple
measurements; their Python host code remains reference material only.

Do not carry published runtime timings or numerical parity into our native port
as achieved results. Measure ordinary and favorable cases, input processing and
cache reuse; small-encoder short-input timings are not long-conversation timings.
Evaluate actual decision/extraction accuracy, calibration and escalation behavior,
not just output shape or agreement with another implementation.

The emerging decision convention is TypeSafe-compatible `/v1/systemone`, already
used by Clef and [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api).
OpenAI's [Decisions API](https://developers.openai.com/api/docs/guides/decisions)
uses similar predicate/choice/score concepts with different wire shapes.
The user now wants both `/v1/systemone` and OpenAI-compatible `/v1/decisions`,
sharing native decision execution and adapting request/result shapes at the
protocol boundary. Preserve each contract's probability, confidence, refusal
and supported-modality semantics; do not claim full equivalence from a common
route name. This is the selected API direction, not implemented endpoints.
[Clairic](https://developers.cloudflare.com/workers-ai/models/clairic/) has a
separate typed-field extraction contract and currently hosted text/image input.
OpenResponses does not currently standardize these operations. Keep internal
decision/extraction semantics separate from compatibility adapters or hosted
Responses tools. The extraction API and voice compatibility surface remain
under discussion; multiple external APIs can use the same underlying runtime.

## Voice frontends and backend task lifetime — discussion 2026-10-24

The user is exploring voice support, particularly a continuous conversational
frontend that delegates heavier work without canceling that work whenever the
user interrupts speech. No voice engine, transport or model port is implemented
by this discussion. The native-only production policy above also applies here.

- Separate microphone/playback transport, speech processing, live conversation
  state and backend task state. Duplex means simultaneous listening/speaking;
  tool calling and continued conversation during backend work are additional
  capabilities, not consequences of duplex alone.
- OpenAI [Realtime](https://developers.openai.com/api/docs/guides/realtime)
  (`/v1/realtime`) and [GPT-Live](https://developers.openai.com/api/docs/guides/live)
  (`/v1/live/sessions`) are distinct APIs. Realtime exposes a voice/reasoning/tool
  session; Live explicitly separates continuous spoken interaction from delegated
  backend work. WebRTC carries audio tracks and a JSON data channel; a primary
  WebSocket carries audio and events together. A sideband is another control
  connection to the same voice session, not the reasoning-model connection.
- Live client delegation can call our own agent/model/service. The application
  owns transcript history and task state: a delegation-created event gives an
  ID and timing metadata, not a ready-made user request. Build the backend request
  from that state and return verified results/progress. Managed Responses
  delegation instead selects a supported OpenAI backend; it is not an arbitrary
  local Responses base-URL setting. [Delegation contract](https://developers.openai.com/api/docs/guides/live-delegation).
- Treat interruption of playback, cancellation of speech generation, and
  cancellation/revision of a backend job separately. Continuing useful backend
  work is desirable; an explicit correction may update or supersede it. Keep
  completed actions and stale-result handling in application-owned task state,
  so an interrupted or replaced voice session does not lose work or repeat it.
- CPU/Accelerate and Core ML/ANE recognition/synthesis are attractive on the
  shared computer, but GPU voice models remain candidates. Measure combined
  frame deadlines, backlog, audible continuity and main-model throughput, not
  only isolated audio real-time factor. Bounded GPU submissions and audio-aware
  scheduling are candidates, not a guarantee that all models coexist smoothly.
- NemotronLabs VoiceChat 11B has a real native Apple
  [implementation](https://github.com/soniqo/speech-swift/blob/main/docs/inference/voicechat.md),
  using MLX throughout its current duplex path. Isolated INT5 M5 Pro runs report
  about 73–75 ms compute for an 80 ms audio frame; this is not concurrent DS4
  qualification or user-perceived first-audio latency. Its shared language KV
  currently prevents free-form speech during function-channel work; microphone
  perception/transcription continues and captured speech is replayed afterward.
  Duplex support therefore does not establish the desired Live-style behavior.
  Retain it as a model/kernel reference without inheriting this scheduling
  limitation as a requirement. [Model boundaries](https://github.com/soniqo/speech-swift/blob/main/docs/models/voicechat.md).

### Voice model and implementation references

The reference shortlist is broader than one model or one accelerator. Python
implementations are oracles/algorithm sources, and Swift/C++ implementations are
also references rather than required production languages or linkage boundaries.

| Workload | Candidate and reason to examine it | Existing Apple/native references and limits |
| --- | --- | --- |
| Fast transcription | Parakeet Ultra 0.6B, 25 European languages | [FluidAudio Core ML/ANE](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/ASR/ParakeetUltra.md); file/window throughput is not live first-partial latency |
| Streaming recognition | Nemotron streaming ASR | [Native cached streaming/latency tiers](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md); language/accuracy varies by checkpoint and chunk size |
| Multilingual recognition | Qwen3-ASR 0.6B/1.7B | [Models](https://github.com/QwenLM/Qwen3-ASR), [MLX encoder/decoder/alignment](https://github.com/moona3k/mlx-qwen3-asr); examined streaming path uses chunking and revisable text |
| Mature recognition baseline | Whisper | [whisper.cpp Metal/Core ML](https://github.com/ggml-org/whisper.cpp), [WhisperKit](https://github.com/argmaxinc/argmax-oss-swift); repeated-window transcription is not a cached transducer |
| Lightweight speech synthesis | PocketTTS 100M and October 20 tiny English 18M/7M/2M releases | [Official models](https://github.com/kyutai-labs/pocket-tts), [FluidAudio CPU/Core ML/ANE implementation](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/TTS/PocketTTS.md); accelerator split differs by variant, tiny-model cloning is not wired in that port |
| Expressive synthesis / cloning | Qwen3-TTS 0.6B/1.7B | [Models](https://github.com/QwenLM/Qwen3-TTS), [native MLX model/codec/streaming](https://github.com/Blaizzy/mlx-audio-swift/tree/main/Sources/MLXAudioTTS/Models/Qwen3TTS); Base, CustomVoice and VoiceDesign have different capabilities |
| Long-form and streaming synthesis | VibeVoice | [Microsoft releases](https://github.com/microsoft/VibeVoice), [community fork](https://github.com/vibevoice-community/VibeVoice), [Apple implementation](https://github.com/soniqo/speech-swift/blob/main/docs/models/vibevoice.md); original long-form code was withdrawn September 2025, separate official Realtime-0.5B followed December 2025; examined native port still returns whole waveforms from its stream API |
| Expressive synthesis comparison | Fish Audio S2 Pro | [Weights and architecture](https://huggingface.co/fishaudio/s2-pro); published headline latency is H200 evidence, not Apple performance |
| Duplex conversation | Moshi / PersonaPlex | [Official MLX and Rust/Metal](https://github.com/kyutai-labs/moshi), [native PersonaPlex work](https://github.com/soniqo/speech-swift/blob/main/docs/models/personaplex.md); native duplex does not itself provide backend delegation |
| Duplex with backend delegation | Realtime-Venus | [Project and models](https://realtime-venus.github.io/); architecture matches the frontend/backend split, Apple execution not verified |

[0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) is another useful native
C++/ggml/Metal model and codec reference, including Qwen/Pocket/Vibe. Its remote
release and published CUDA speed table were not equated to the installed
`../audio.cpp` checkout or to Apple model qualification.

For protocol research, [Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech#realtime-api)
demonstrates a tested core OpenAI Realtime subset over WebSocket/WebRTC on a
cascaded ASR → language model → TTS pipeline. Original Moshi/PersonaPlex protocols
are different; a `/v1/realtime` path alone does not establish client compatibility.
Our existing Responses WebSocket and a future voice session adapter are distinct
protocols over shared application state. Keep heard speech/playback state apart
from generated-but-unplayed output and durable backend progress.

Voice qualification should measure first useful audible response, interruption
response, recognition errors on names/numbers and relevant languages, speech
naturalness, frame p95/p99/backlog and coexistence with main-model decode/prefill.
Real-time factor alone is insufficient. All candidate timings remain authors'
measurements, and none of these voice models has been run by this discussion.

## Multi-model composition and Flint lessons — discussion 2026-10-24

The user wants these ideas retained before resuming execution. This entry records
direction and source-based candidates, not implemented multi-model support or
measured DS4.1 speedups. Runtime work in this design remains proposed; the later
cleanup/publication authorization at the top supersedes the earlier main hold.

### Multiple loaded models in one process

- Initial example: DS4.1 remains the main agent model while a small, fast Flint
  handles bounded subtasks in its own context. Both can retain weights and
  sessions in memory; switching requests need not unload/reload either model.
  A future server-hosted agent tool could use this arrangement, but that tool
  and its agent loop are separate proposed features. The constellation above
  broadens this example to other generative and non-generative models.
- Prefer one Responses HTTP/WebSocket frontend with explicit model selection,
  separate engine instances and a scheduler. Pin conversations, inference state
  and `previous_response_id` ancestry to the appropriate model; resuming a
  response must not silently send another model's KV/state to the new engine.
  Same-process composition is the desired follow-up, superseding the earlier
  initial preference for separate processes. Current boot still opens one engine.
- No cross-model megakernel is required. Staying loaded, alternating GPU work
  and executing simultaneously are separate capabilities. Same-model batching
  can reuse weights; batching unrelated models does not create that reuse.
  Megakernels are useful for launch/synchronization overhead and activation
  locality within an execution pipeline, not a prerequisite for multi-model use.
- Start with serialized GPU transactions: encode, submit and finish one engine's
  work before another uses shared resources. Bound prefill chunks and decode
  packets so other conversations can run and cancellation remains responsive.
  Parallel host preparation/I/O may still overlap. Simultaneous GPU execution is
  a later measured option; overlapping Flint's cooperative resident kernels can
  violate their whole-grid progress assumptions.
- Ownership should follow actual resources: process runtime owns the Metal
  device/queue, shared services and total budget; an engine owns model identity,
  weights/mappings, expert residency and model-specific execution state; a
  session owns its continuation/KV/frontier. Share scratch through a scoped
  lease when transactions serialize. Keep real records single-source, without
  mirrored Engine/Session state or per-session copies of every scratch buffer.
- Flint already owns most model buffers, pipelines and session state while
  borrowing the shared Metal device/queue. That boundary was our native port's
  implementation choice around upstream's kernel contract, not proof that two
  engines currently compose safely. DS4.1 still has process-global Metal model
  mappings/cache/scheduling state and shutdown paths. In particular,
  [Core cleanup](engine/core.zig) shuts down shared threads/GPU services when one
  model retires. Give shared services a runtime lifetime; do not merely bypass
  the instance lock or assume strong Objective-C references make cleanup safe.
  Derive shared interfaces from actual model components, including decision and
  extraction engines, rather than a speculative plugin layer.
- Admit the combined weights, expert caches, all sessions and shared scratch
  under one pressure-aware budget. Earlier short-context estimates suggested
  roughly 65–70 GiB for this pair, not a qualified simultaneous measurement;
  Flint's two configured 262144-token sessions can materially increase that.
  Preserve the pressure guard and account for shared allocations once.
- Qualification should exercise A → B → A continuation/logits, two independent
  conversations, model-pinned response references, cancellation, failed startup
  and closing one engine while the survivor continues. Measure alternating task
  latency and memory pressure before considering concurrent GPU kernels.

### Execution lessons from Flint

Source review: upstream `recursive-oss/rec` revision
`e53663b262af4cb1ca498e596ca8b1a2d79e0800`, available in `../flint`.
Our [native decoder notes](metal/flint/README.md) and
[prefill notes](metal/flint/prefill/README.md) describe the preserved contracts.

- Separate load-time packing, batched prefill and persistent decode. Flint's
  M5 Max decoder uses a fixed 76 × 256 resident pool with phase-specific roles;
  its small residual state stays in registers and normalized inputs are reused
  from threadgroup memory. Other GPUs need their own qualified launch envelope.
- Expert-local teams synchronize only their own gate/up → down dependency.
  Six routed experts use eight groups each and the shared expert sixteen.
  Teams redundantly find their assigned routing rank to avoid a global top-k
  publication barrier. DS4.1 could borrow the scheduling idea while preserving
  its own router transforms, correction biases and mixture normalization.
- Compact layer descriptors and aligned packed weight banks reduce hot-path
  decisions. Projection kernels reuse activation loads across output rows and
  paired gate/up dots. DS4.1 already has packed GGUF dispatch, scratch layouts
  and fused expert operations; measure the remaining traffic and barriers.
- Greedy Flint generates up to eight tokens inside one resident launch;
  filtered packets chain forward, GPU sampling and token handoff in one command
  buffer. DS4.1's ordinary path still reads full logits to the host in
  [Executor.logits](engine/deepseek41/executor.zig), then selects through
  [Selection](engine/session_sampling.zig). The runtime has a GPU argmax primitive,
  but that does not establish its use on this DS4.1 path. Device-side selection
  is a concrete candidate before attempting longer GPU-only decode packets.
- Tune dispatch to actual shapes. Flint uses packed matrix-vector kernels for
  small row counts and tiled matrix multiplication for larger prefill batches.
  DS4.1 already distinguishes scalar, batch and exact-small-row paths; examine
  resumed prompts and rows per expert rather than copying Flint's thresholds.
- DS4.1 SSD expert demand is a real host boundary. Its selected-load path already
  overlaps some I/O with shared work. Larger persistent regions must respect
  expert misses and yield/cancel boundaries; a fully resident Flint decoder
  cannot simply replace a streamed DS4.1 pass. Prefer measured resident work
  regions over an assumed whole-model megakernel win.

The user explicitly wants model reformatting/repacking kept on the table for
DeepSeek and other models. GGUF is not a required hot execution layout: a simple
reproducible conversion or load-time packing step may enable better aligned
expert reads, compact scales/metadata, activation reuse and shape-specific
kernels. Compare total memory/cache capacity, SSD bytes, load/conversion cost,
prefill, decode and end-to-end latency; a faster inner kernel alone is insufficient.
For tunable layouts or precision, plot useful speed/memory/quality tradeoffs so
we can choose measured Pareto points rather than assume one universal format.
Distinguish lossless layout/packing changes from requantization or changed math.
Preserve the source checkpoint, check lossless candidates against the baseline,
and measure delta-NLL/KL plus practical output quality when arithmetic changes.
Approximation is acceptable when the measured tradeoff is useful. Flint's INT4
format, fixed shapes and BF16 rounding are examples to study, not DS4.1 defaults.

Suggested bounded experiments, pending execution authorization: refresh DS4.1
handoff/head timing; try device-side output selection; examine resident expert
teams/activation reuse; tune small-prefill dispatch. Repacking can be a separate
one-change experiment when profiling identifies a format/layout limitation.
Use the existing one-hour baseline → change → measure → inspect → decide loop,
retaining negative results; reconcile with the experiment queue before starting.

GLM's low bounded-SSD qualification rate remains backburnered at the user's
request. The persistent all-layer decode mapping is implemented and ported;
`DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP=1` selects the bounded fallback
used for qualification. This is a configuration choice, not missing native code.

## Model retention direction — 2026-10-24

The user revised the cleanup direction after discussing V4.1 Pro:

- Keep DeepSeek V4 Pro's existing C path for now, including helpers shared
  with it. Do not port it speculatively or remove it with old Flash support.
- Retire V4 Flash and GLM 5.2-specific support where it can be separated from
  retained models. The cleanup above implements this decision.
- Full GLM 5.3 is a future native port target. Its legacy `glm-dsa` profile
  shares the fixed 79-layer DSA layout; the cleanup preserves that C path and
  verifies checkpoint identity/header admission, without claiming live-model
  qualification.
- Treat V4.1 Pro as a future model, with no guessed dimensions or promised
  compatibility. The native V4.1 family is the likely starting point; old
  Pro remains a useful reference for larger-model handling.

DeepSeek's [V4.1 announcement](https://api-docs.deepseek.com/news/news260910/)
names a future V4.1-Pro. The later [API changelog](https://api-docs.deepseek.com/updates/)
reverses V4 Pro's announced API retirement and keeps its service/billing.
The user supplied [this release-status discussion](https://www.yottalabs.ai/post/deepseek-v4-1-pro-release-date-what-is-known-how-to-prepare-2026).
These references explain the retention decisions; they do not establish support
for an unreleased architecture. See the current task above for publication state.

## Shared runtime migration — completed 2026-10-24

The user authorized this sequence with “Ok. Make it so.” The plan is twelve
one-hour ceilings, with earlier completion encouraged: Qwen prompt mapping;
GLM prompt mapping; Responses publication integration; generic payload I/O;
snapshot/staged-file envelopes; selected-expert workers; page-in/model-view
services; common session control; session resources; engine startup/admission;
engine Metal lifecycle; assembled-runtime qualification and removal of old
owners. [Detailed sequence and checks](../speed-bench/ideas/next-experiments.md).

The target is one Zig owner per active single-Metal component, keeping model
arithmetic, formats and cache/threading policy stable during the migration.
Use the qualified `e37e53d9` model port as baseline and leave main unchanged.
Keep actual records single-source and cold/GPU objects separate. Source prep
can overlap; integration and guarded model runs remain sequential. Prefer
existing fixtures and stable live-session timing cases. Metal ARC/MSL,
third-party libraries and separate legacy/reference/distributed C are explicit
boundaries, rather than targets for line-count conversion.

Slice 1 completed within its first hour (start about 10:33 UTC): Qwen full/live
prompt, schema/tool-call and checkpoint-suffix formatting now has one C-free
model owner. Exact existing server fixtures, six pure assembly cases, three
focused CPU tests and all 25 real Responses/WebSocket checks pass. C retains
small compatibility/test adapters. [Progress and commands](../speed-bench/experiments/native-runtime-20261024/README.md).

Slice 2 is complete too: GLM full/live templates and tool/schema formatting
select its pure model owner. Six pure GLM tests, seven native preparation cases,
exact server fixtures and all 25 real GLM Responses checks pass by 11:09 UTC.
Slice 3 is complete: native Responses publication and prompt parsing pass the
existing server/store fixtures and all 25 real Qwen Responses/WebSocket checks.
Slice 4 is complete: bounded native FILE/tensor/F16 transfers pass the full
host suite and Qwen/GLM frozen → native → frozen payload/fresh-logit exchange.
Slice 5 is complete: one native envelope handles public and suffix snapshots,
with CPU lifetime/error fixtures, full host checks, Qwen/GLM exchange and all
eight DS41 F32/F16 cross-binary/conversion campaigns passing.
Slice 6 is complete: the native pthread mailbox passes failure/busy/reuse
checks, and actual DS41 async loading matches frozen bytes/fresh logits at
six frontiers with its flag enabled. Slice 7 is complete: native real-file/thread tests, full host suite, DS41/GLM
state/fresh-logit comparisons and actual profiled 256-token layer preads pass.
A clearly named C span-selection leaf remains until slice 10, where its native
replacement is prepared. Slice 8 is complete: shared native routing, control/frontier and family payload
selection pass host checks, Qwen public eval_argmax, GLM partial-cancel recovery,
three-family frozen payload/logit comparisons and all 28 Flint Responses checks.
Generic common batch policy is an identified bounded follow-up after pools exist;
model batch arithmetic was already native. Slice 9 is complete: native resource lifetime passes all-family checks, Qwen
full recovery/batch/arena growth, DS41 exact charge/decharge, all 25 GLM Responses
checks and fixed-build GLM cancellation recovery. Independent review found and
fixed a stale pre-fit reserve capture; a focused construction-order probe checks
fresh admission and exact decharge. Slice 8b is complete: common batch validation/publication and mixed sequencing
pass Qwen full ordinary/MTP/failure recovery, host tests and all 28 Flint
Responses checks. A final audit found single-session continuation/rewind policy
still in C for Qwen/GLM/Flint; a bounded SessionView follow-up will close it.
Slice 10 is complete: native admission/cache policy, shared spans and direct
page-in selection pass host/CPU checks, frozen GLM/DS41 state+fresh logits and
all 40/45-layer real pread runs. Slice 11 is complete: all five executables, full host suite, three GGUF inspect
paths, Qwen unowned-error isolation, frozen DS41/GLM state+fresh logits and Flint
ordinary inference/teardown pass. The actual Core configuration is authoritative;
legacy globals are compatibility reads, with old DS41 TP ownership distinguished.
Common continuation and all-family sampling are now complete: isolated Qwen
state/MTP/recovery and all25 Responses checks pass, all28 Flint continuation
checks pass, plus host and47 real tokenizer parity cases. Multimodal is complete: four CPU transaction cases, host+GPU state fixtures,
seven exact Qwen/GLM image outputs and exact frozen/public DS41 image comparison
pass. Small rewrite/steering policy passes Fast/Debug CPU probes and host checks;
Qwen steering getter now reads its actual graph. Stateless generation is complete:
host/Fast/Debug probes and isolated Qwen/GLM ABBA runs pass with exact output.
Warm decode is effectively level; first cold arms are retained, not speedup
claims. GLM storage closure is complete: CPU decision tests, host checks,
frozen payload/fresh logits,16 actual MTP snapshot cycles (seven double-token),
context reuse and all45 profiled256-token layer preads pass. Dead-helper cleanup
is integrated:133 production C lines and77 private Objective-C definitions
removed; no warning suppression or kernel change. All-five final build is
warning-free and the complete final host suite passes. The immutable prefix is
`.workspaces/runtime-migrations-20261024/final/bin`. Combined state/kernel and
Qwen/GLM recovery checks pass; four repeated F16 suffix relocations pass, and
all103 protocol checks pass across DS41/Qwen/GLM/Flint. Qwen/GLM/Flint answer
probes pass; DS41 repeats three known wrong disconnect JSON answers, now being
matched exactly by the frozen baseline. Runtime screens preserve exact outputs;
Qwen live append changes0.13%, GLM decode0%. A measured Flint terminal argmax
was removed; counter probes, host and28 serving checks pass. Longer256-token
Flint ABBA is level (243.57/243.88 frozen vs245.41/242.99 native t/s).
Fresh builds initially grew14.24→19.22s. Compatible hot exports and cold
math/spans now share separate compile units; standard Zig CPU parallelism
replaces the wrapper's12-job cap. Full build is11.87s (baseline11.41s with
equal parallelism; old capped wrapper14.24s), cached0.97s. Final grouped state,
fresh-logit, GPU and Flint serving checks pass. Grouped Qwen append differs
0.08%, decode0.70%; longer Flint decode1.07%, with exact output/frontiers.
These are small regression screens with visible point-to-point variation.
All requested owners are complete. The five installed root executables match
the qualified binary hashes. The local native-runtime stack is consolidated
into five reviewable commits; original checkpoints are retained on
archive/native-runtime-steps-20261024. Main stays unchanged; nothing is pushed.

Final routing audit identified additional actual ordinary controllers beyond the
initial inventory: Qwen/GLM sampling gates/history/logprobs, common multimodal
assembly/encode orchestration, small continuation/rewind/canonical-rewrite and
live-steering policy, and stateless greedy generation. These get separate
bounded one-hour components after engine composition, before combined slice 12;
model arithmetic and capabilities stay fixed. This is additional migration work,
not completed ownership or a reason to count inactive C fallbacks as runtime.
 Independent payload-I/O, snapshot-envelope,
selected-worker and session-control/resource source preparation overlaps; only
root integrates shared files and schedules guarded model jobs. Frozen baseline
binaries and original shared sources are under
`.workspaces/runtime-migrations-20261024/baseline/`. The work bookmark is
`native-runtime`; the qualified model port stays on `qwen-glm-zig`, main unchanged.

## Native Qwen 3.8 and GLM Flash ports — completed 2026-10-24

The user authorized handwritten Zig model ports, explicitly reusing downloaded
weights. Both ordinary single-Metal model paths are now native: configuration,
weight binding/validation, graph resources, attention/experts, prefill/decode,
recurrence, MTP, snapshots/payloads, batching, session control and vision.
Qwen owns disk-only n-grams/PLE and pooled GDN; GLM owns compact/indexed MLA,
KDA/mHC and chunk/cache/page-in ordering. Actual records are defined once in
shared headers; generic selected-load/page-in services live in `stream_io.h`.
Cold model/n-gram/memory exports are separate from hot GPU objects so CPU-only
fixtures never pull a second C core. Existing Metal/MSL primitives remain.

The ordinary algorithms do not forward to C model traversal. At that checkpoint
C retained common engine/session lifecycle and platform/storage services, GLM52, independent CPU
oracles and explicit profiling/distributed/legacy paths. GLM expanded full-KV
allocation remains unreachable as in the baseline; disabling indexed prefill
retains the existing rejection. This is a Flash single-Metal completion boundary.

Branch/bookmark: `qwen-glm-zig`, based on `030a040e` (preserved by
`before-qwen-glm-ports`). Main remains unchanged. Installed checkpoints are
`gguf/Qwen3.8-Flash-Next-Q4.gguf` and `gguf/GLM-5.3-Flash-Q2.gguf`, with their
matching vision sidecars. No new weights were downloaded. Reports/module maps
and commands: [native family qualification](../speed-bench/experiments/qwen-glm-zig-20261024/README.md)
and [canonical payload exchange](../speed-bench/experiments/model-family-ports-20261023/PAYLOAD.md).
Raw binaries, commands, logs, state and timings remain ignored under
`.workspaces/model-family-ports-20261023/`.

Completed qualification on the 128 GiB M5 Max, one guarded model job at a time:

- Both models' 2047/2048/2049 frontier matrices match frozen C bit for bit:
  1,209,600 floats per paired matrix. Qwen also matches the 2047→4095 extension.
- Qwen's five state fixtures, depth-three packets (38 three-row cycles with
  12 second-draft rejections), mixed ordinary/MTP batches, failed n-gram reads
  and exact recovery, context limits and shared-arena growth all pass.
- GLM snapshot, continued prefill, 84 pure geometry cases and actual bounded
  MTP pass. Continued-vs-cold chunk ordering retains the same inherited numeric
  difference (top token 39058, six matching top-eight ranks, max delta 0.298615).
- Seven vision cases/repeated encodes match all 12,574,720 floats. Complete
  Qwen image/follow-up turns pass ordinary/MTP; GLM image identity, replacement,
  replay/restoration and fresh image-conditioned decode pass.
- Each family passes all 25 real Responses/WebSocket checks, including tools,
  concurrent conversation overlap and durable response/conversation recovery.
- Frozen → native → frozen payloads at 1/3/4/5/129 match bytes and freshly
  computed continuation logits. Invalid-header preservation, truncated-state
  invalidation/recovery and the candidate-only early-logit cut all pass.
- Both GPU kernel fixtures, full host/script suite, formatting and all five
  executable builds pass. Useful reports/scripts are tracked; raw data is not.

Three inherited GLM53 correctness fixes were included: stage token/nextn/head
mappings for bounded SSD MTP (seven double-token snapshot cycles plus one reuse
acceptance, no map failures); invalidate nonfinal prefill before mutation
(canceled 17/19 frontier rejects sampling/persistence and rebuilds 154,880 logits
exactly; cancellation before work preserves old state); invalidate payloads at
first live-logit write before later validation can fail. No wire-format change,
extra intermediate output-head work or new runtime configuration was added.
Independent review found no additional resource/ABI/default-dispatch blocker.

Runtime ABBA screens show effectively level performance: Qwen 2048-token live
append 1242.27 vs 1245.23 t/s (0.24% longer), ordinary decode 0.03% longer and warm
repeat 0.07%; GLM decode 0.73% longer and warm prefill 354.25 vs 355.10 t/s. Qwen first
prefill varied 222–948 t/s in both binaries even after process warm-up; keep that
uncertainty instead of claiming a startup speedup. The within-session append
is stable at 1241–1247 t/s. Fresh-project-cache builds of all five executables took
14.88 s native vs 15.50 s frozen; cached builds 1.08 s vs 1.05 s. These are small
regression screens, not optimization claims.

The initial GLM global-map attempt was pressure-stopped and a 4096-context
allocation refused 111.42 GiB > 110 GiB. The existing
`DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP=1` fallback plus a 24 GB expert cache
then completed both frozen/native 2K campaigns, state, persistence and serving
with normal pressure/no swap growth; guards were never relaxed. Qwen uses
`DS4_METAL_NO_RESIDENCY=1 DS4_METAL_NO_MODEL_WARMUP=1`, no SSD-expert flag.

Automatic server SSD/suffix caching still admits native DS4.1 only. Qwen/GLM
Responses durability replays stored items after restart; public canonical model
payloads are a separate qualified interface. Resident GLM shared-weight batch
math is ported but has not been exercised with this SSD-only host setup.

## Upstream adaptation and Metal follow-up — 2026-10-23

Completed `jj git fetch --all-remotes`: upstream main moved from `5379c1c9ac31`
to `ee7e8405232c`, adding exactly two commits. The user authorized catch-up using
an adaptation merge. Main remains unchanged.

- `a1012463383a` adds verified, resumable V4.1 Flash **1017 MXFP4** downloads.
  This is another quantization of the existing checkpoint. Our native Metal
  runtime already dispatches MXFP4, and our Q2 checksum already matches the
  upstream correction. Adapt the target/identities into the Nu catalog and
  downloader, reusing the upstream GGUF shard assembler and small fixture tests.
  Standard GGUF shards require tensor-directory reconstruction and 16 KiB
  alignment; the existing raw Q4 part concatenator cannot assemble them.
- `ee7e8405232c` sorts/coalesces CUDA selected-expert prefill reads, makes CUDA
  whole-layer prefetch opt-in, and handles publication after upload completion.
  Its sole shared-engine hunk excludes Apple. Native Metal already publishes
  after joined reads, but selected-prefill I/O tasks remain in router order.
  CUDA coalescing does not transfer directly to our per-expert gate/up/down
  slabs, and Metal's existing whole-layer preparation serves a different path.

We use an **adaptation merge** instead of rebasing the rewrite. Its first parent
is our current tip, its second parent is the reviewed upstream tip, and its tree is our
implementation plus the selected ports. This preserves published history and
records the upstream synchronization boundary. The second parent means every
upstream change through that point has been considered; record significant
omissions/deferred work because future incremental reviews will not offer them
again. Preserve both parents when tidying integration history. Ordinary local
feature changes can still be squashed/rebased. Review sync changes against their
first parent; continue deliberate ports rather than relying on automatic merges
to infer C-to-Zig correspondence.

Created the sync on `upstream-sync-ee7e840`, with first parent `9944e786f715`
(also preserved by `before-upstream-ee7e840`) and second parent `ee7e8405232c`.
Restored its initial tree from our first parent before deliberately applying
ports; the first-parent diff was empty at that point. This resolved automatic
rename/delete conflicts without resurrecting root C files or shell launchers.

Phase 1 completed in about six minutes of its one-hour budget. Added
`./download-model.nu ds41f-mxfp4`, ten published size/hash identities, native Nu
space accounting and verified/resumable assembly. The focused Python binary
assembler remains byte-identical to upstream at its original `gguf-tools/`
path. Existing Q4 and Flint paths are retained. All five executables build
without diagnostics; the full model-free host suite passes, including 22
offline downloader tests, 11 Nu launcher cases and the driver contracts.
Independent comparison confirmed every new artifact identity against upstream.
Raw checks remain under `.workspaces/upstream-sync-20261023/`.

The upstream CUDA implementation and its CUDA-only documentation are
intentionally out of scope for this Metal build. Its transferable idea was
phase 2: file-offset sorting only in the temporary selected-prefill I/O task
array, preserving routing, allocations, decode and native lookahead. The
installed Q2 has uniform IQ2_XXS gate/up and Q2_K down across all 40 layers,
satisfying the selected-prefill type guard.

Phase 2 completed within its one-hour budget: the implementation was correct
but its performance benefit was inconclusive, so the runtime knob was removed.
The automatic-cache diagnostic reached the intended path but the pressure
guard stopped it cleanly on elevated macOS pressure. Matched 64 GiB-budget
ordinary appends and 32 GiB-budget forced-128-row appends completed normally.
Their ABBA mean times favored sorting by 2.13% and 1.72%, respectively, but the
ordinary candidates matched the first control within 0.1% and the wide-batch
candidates matched the final control within 0.2%; an unchanged 2K-prefill
control varied by 2.89%. All compared full 129280-logit frontier rows matched
exactly. The default larger-cache/128-row regime remains unqualified after the
pressure stop. This does not reject CUDA's combined sorting/coalescing policy.

Runtime source is restored byte-for-byte to the qualified pre-experiment
version. Keep the concise [experiment and reproduction patch](../speed-bench/experiments/upstream-selected-read-20261023/README.md)
for future storage/layout comparisons; raw timings/logits/pressure logs remain
ignored under `.workspaces/upstream-selected-read-20261023/`. No staging,
coalescing, slab repacking or new runtime configuration was retained.

Full MXFP4 qualification is separate: the merged file is 467.16 GiB and retained
shards plus output/reserve require about 935.3 GiB, excluding HF cache extras.
The audit found about 458 GiB free on Models, so acquiring it requires more
storage first. Downloader fixtures can be qualified without those weights.

## C frontend warning cleanup — 2026-10-23

Baseline `ec1dd845` compiled obsolete C-to-Zig migration facades and fixture-only
adapters into the production support translation units: 94 unused-function
warnings in `ds4_server.c` and 74 in `ds4_agent.c`. Removed 118 helpers with no
runtime or fixture callers. The remaining fixture adapters compile only for
their existing test macros, including agent fixtures that define only
`DS4_AGENT_TEST_NO_MAIN`. Removed the old unused attributes; production warning
flags remain enabled. Runtime bridges, Qwen/GLM rendering and shared prompt
bytes are retained.

Both support files compile without diagnostics under the production C flags
plus `-Werror=unused-function`. Before/after object exports and the `__text`,
`__cstring` and `__const` sections match exactly; this is source/build cleanup,
not an inference optimization. All five executables build cleanly with
`nu tools/zig-build ds4 ds4-server ds4-agent ds4-bench ds4-eval`.
`nu tests/verify.nu` passes, as does the model-free suffix transaction fixture;
the real-vocabulary live-context fixture also builds. No model job was needed.
Raw baseline copies, warning logs and checks remain ignored under
`.workspaces/c-warning-cleanup-20261023/`. Main is unchanged.

## Flint reasoning visibility fix — 2026-10-23

The user's dsh `openai-responses` profile received no thinking because our server
required `reasoning.summary` even though OpenResponses does not require that
hidden-by-default policy. Flint Responses now emits its generated reasoning
through the existing summary events/items by default, on HTTP and WebSocket.
An explicit `reasoning.summary: null` suppresses that text. Effort/thinking hints
are accepted and ignored, matching Flint's model-controlled reasoning policy;
the response envelope reports `reasoning.effort: null` instead of a fictitious
effort. DeepSeek visibility defaults are unchanged.

No regression checks were added or run, at the user's request. The obsolete
fixture assertion that Flint must reject effort was removed.
`nu tools/zig-build ds4-server` succeeded, and the root executable was replaced
atomically for `ds4-local` without interrupting the running server.
The already-running server needs restarting to pick up the new binary.

## Intended product

The user wants a focused local inference server for their M5 Max, with possible
M1 Max support. We follow upstream implementations and selectively copy or
reimplement relevant model changes, kernels and fixes, retaining attribution.
We do not need every upstream backend, frontend or deployment mode.

- Responses is the conversational protocol, including WebSocket transport and
  incremental conversations. HTTP and WebSocket share request handling and
  output events. Add both System One and OpenAI-compatible Decisions adapters
  over shared decision execution; voice protocols are being explored separately.
- Many independent conversations per engine. Durable history outlives sockets,
  active inference slots and cache eviction. Concurrent conversations need fair
  scheduling, cancellation and bounded prefill work.
- Multiple loaded engines in one process are now the desired follow-up, with
  model routing, separate ownership and a shared scheduler/budget as discussed
  above. Current serving still opens one engine; separate processes remain a
  usable deployment option. Configure combined physical RAM and SSD bandwidth.
- Assemble a serving executable from protocol/conversation handling, scheduling,
  model implementations, platform compute and RAM/SSD storage. Choose Metal,
  CPU/Accelerate or Core ML/ANE per workload; production inference stays native,
  with Python implementations used as references. Start with normal linked
  modules and explicit composition; derive shared interfaces from actual
  implementations rather than speculative plugin requirements.
- Model implementations own tokenizer/prompt rendering, tool/reasoning and
  multimodal translation, execution, and saved-state semantics. Original DS4.1
  and 1017 share family code with explicit variant configuration. Native Metal
  backends now include DeepSeek, Flint, Qwen 3.8 and GLM-5.3 Flash.
- Learn from ds4-agent's context editing, archival, compaction and reuse behavior.
  Expose useful serving controls through explicit API extensions as needed so
  other agent harnesses can use them. Agents can execute their own tools.

## State and storage boundaries

Keep three concepts explicit:

1. Durable conversation items and response ancestry.
2. The selected/edited/compacted working context presented to a model.
3. Model inference state: KV, compression/recurrent state, pre-RoPE values and
   any other state needed for continuation.

Persistent-agent facts can leave working context while remaining in checkable
storage. Qualification must exercise actual archival and tool retrieval.
Approximate activations can preserve influence from an earlier prefix; rebuilding
from text may change that influence. Preserve checkpoint lineage and distinguish
checkpoint restoration from rebuilding a context.

Share mechanisms such as immutable byte storage, I/O, pinning, eviction and
accounting where useful. Each model owns capture/restore/relocation and validity
rules. Current DS4.1 checkpoint formats are not a universal per-token KV format.
Expert weights and conversation checkpoints have different residency policies.

USRBIO refers to 3FS's user-space ring I/O interface, not DeepInfer's IPC or GPU
communication rings. Its reusable lessons are batching, reusable buffers,
completion handling and overlapping I/O with computation. A single engine
process can start with an ordinary bounded queue and I/O workers; cross-process
shared-memory machinery is unnecessary for that arrangement. Measure submission,
copying and SSD wait costs before choosing a more elaborate queue.

## Candidate serving features — discussion 2026-10-23

Record these as a design palette: the user may want some, all, or a remix.
This discussion does not authorize implementation or select a scripting runtime,
memory database, or universal tool framework. Steering remains separately
deferred with its zero-speed-regression acceptance requirement.
Connectome-like context aging and CLM are reference ideas; no particular memory
system or agent harness is selected.

| Piece | Intended role and execution ownership |
| --- | --- |
| Server-side programmatic tool calling / code mode | Model-generated programs compose server-exposed capabilities within a response. OpenAI's current Responses API provides a precedent; our pinned OpenResponses core does not standardize it, but permits hosted extensions. Tool execution location remains explicit: a server-run program may still need results from client-owned tools. |
| Client-side composition, like `functions.exec` | The external agent composes its own tools in its own runtime. This exists in our current client tooling; it is not an implemented DS4 server capability. Keep it compatible with, rather than dependent on, server-side composition. |
| Direct tools and program bindings | Expose capabilities directly to the model and through code bindings when useful. Prefer the same names, schemas and result semantics in both surfaces; avoid divergent duplicate tool definitions. |
| Server-side MCP | Let the serving process connect to MCP servers, discover their tools and execute calls. Those tools could be available directly and from server-side programs, enabling server plugins without moving the agent frontend into the model process. Not implemented yet. |
| Explicit and automatic compaction | Support deliberate `/responses/compact` calls and separately consider request-configured thresholds and automatic/default policy. OpenResponses 2026-04-24 includes the standalone endpoint; OpenAI's current automatic `context_management` configuration goes beyond that pinned core. Our server implements neither compaction mode yet. |
| Specific CLM/context edits | Operate on the server-owned working view using stable turn/item IDs, preserving the full journal and archived evidence. Offer inspect/archive/replace/remove primitives; programs may compose them, but an individual edit can also be structured arguments without code. Apply changes at a safe inference boundary and return a receipt. |
| Cross-session memory | Keep useful facts, decisions and evidence in durable, checkable storage, available to later sessions through retrieval tools. This is distinct from response ancestry, prompt/KV caching and a compacted working context. No storage/index choice is selected. Qualification should retrieve an archived fact absent from the active prompt. |

### Candidate memory and thread tools

The user specifically suggested explicit memory, thread listing, archiving and
search. Web/browser capabilities can come through server-side MCP; concentrate
our own tools on state the serving process owns. Names below are sketches, not
an implemented API or a commitment to separate tool schemas.

| Capability | Useful contract |
| --- | --- |
| `memory.put/get/update/retract` | Curated cross-session notes: facts, decisions, constraints and evidence references, scoped to a project or user. Preserve source thread/item IDs and distinguish superseded notes from current ones. Start with explicit text records; indexing and automatic memory promotion remain undecided. |
| `threads.list/open` | Discover threads by title, project, recency or status; inspect a bounded outline or selected messages instead of dumping full history into the active prompt. |
| `recall.search/read` | Search notes, archived material and thread history with source filters; return small snippets and stable references, then fetch exact selected ranges. Keep search/read behavior consistent across these sources. |
| `threads.archive/unarchive` | Remove completed work from the active thread list while retaining its history and searchability. This differs from deleting data or removing turns from working context. |
| `context.inspect` | Report token budget, a bounded turn index and large context contributors so the model can decide what to retain or archive without rereading its entire context. |
| `context.archive/replace/remove` | Archive selected older turns durably, then reduce the working view to a summary and references. Return an edit receipt and resulting token usage; suffix selection/recomputation stays runtime policy. |
| `threads.fork` (later) | Explore an alternative from a selected turn without overwriting the original. Fork logical history and reuse KV opportunistically; do not imply that an exact approximate-state snapshot exists. |
| `threads.ask` (later) | Run a bounded question against another thread and return an answer with source references, reducing how much unrelated history enters the current model context. This would spend additional inference; it is not just a search alias. |
| `context.restore` (later) | Undo a bad working-context edit using a retained logical revision. Keep logical restoration distinct from exact restoration of approximate inference state. |

Concrete qualification workflow: finish an investigation, archive its detailed
turns, save the measured conclusion with source references, and leave a small
pointer in working context. In a fresh session, find why a kernel variant was
rejected and retrieve its actual measurement and command. Archive completion
precedes removal of the only copy. For concurrent sessions, target edits at a
known thread head/revision so stale changes fail visibly. Direct calls and code
bindings should share these operations rather than grow parallel implementations.

### Candidate server-side agent tasks

The user added server-side agent tasks to the palette: delegate a bounded goal
to a model/tool loop on the server; detached jobs may outlive the client or
one response stream. Examples include investigating an archived thread,
running a bounded experiment, and preparing an evidence-backed memory update.
These are execution jobs, distinct from a todo-list record or a cron schedule.

Clarified user intent: the central capability is **an agent as a server-hosted
tool**, not only a separately submitted background job. A model (directly or
through a server-side program) invokes something like `agent.run`; our server
runs that child agent's model/tool loop and returns its result to the caller.
The parent need not orchestrate the child's individual tool calls, and the
external client need not execute that inner loop. The child has its own working
context, explicit goal/tool scope and budget; return a concise result, artifacts
and evidence references, with its detailed trace retained separately.

The same runner can support two call policies: await a bounded child result and
continue the parent within its response, or detach a longer job with a durable
handle for later retrieval. These are invocation/lifetime choices around one
implementation. An MCP adapter would expose the same runner to external agents;
it is not a second executor. This remains a design clarification, not an
instruction to start implementation.

- Detached invocation sketch: submit a task with a source thread/context, goal,
  allowed tools and budget; return a stable task ID promptly. Inspect progress/results, supply
  requested input or cancel later. Task listing/discovery is also useful as our
  own capability, regardless of whether a wire protocol has a list method.
- A task may own its own conversation, working context and memory references.
  Multiple logical jobs share the existing engine/scheduler; this does not imply
  parallel GPU execution or a speedup. Keep explicit inference/tool budgets.
- Detached task lifetime must be independent of the invoking socket; an awaited
  child may instead follow parent cancellation. Current Responses
  WebSocket disconnect cancellation does not implement this behavior. Persist
  sufficient task/context/progress state for the recovery promise we choose;
  returning an ID alone does not make execution restart-safe. Record completed
  actions so recovery does not blindly repeat external side effects.
- The MCP Tasks extension is a possible interoperability layer, not the agent
  runtime itself. An MCP tool could start an agent job and return a task handle;
  the host polls in protocol code instead of spending model turns on polling.
  An optional MCP endpoint for our own agent tasks is distinct from the already
  proposed outbound MCP client used to plug third-party tools into the server.
- Reuse one job owner behind any hosted Responses tool and MCP adapter; do not
  build two schedulers. Server-side MCP composition must also handle tasks
  returned by downstream tools, with input requests and cancellation propagated.

Protocol research on 2026-10-23: MCP introduced experimental Tasks in
`2025-11-25`. The `2026-07-28` revision moved/redesigned them into the official
`io.modelcontextprotocol/tasks` extension: polling/final results through
`tasks/get`, input through `tasks/update`, cancellation through `tasks/cancel`.
Do not copy the old `tasks/result` or `tasks/list` wire flow into a new adapter.
The current [Claude Code documentation](https://code.claude.com/docs/en/mcp#mcp-tasks-that-run-on-the-server)
describes public-beta support requiring v2.1.304+; that confirms support, not
the exact day its rollout became visible. Protocol sources:
[original experimental Tasks](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks),
[2026-07-28 changelog](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/changelog.mdx),
and the [current Tasks extension](https://modelcontextprotocol.github.io/ext-tasks/specification/2026-07-28/tasks.html).

### Opportunistic background compaction — proposal

The user suggested preparing compaction or Connectome-like memory consolidation
while the server is otherwise idle. Treat computation and publication separately:
an idle job can prepare a candidate without changing a completed response or the
current working context. This is not implemented or scheduled for implementation.

Proposed minimal flow:

1. Response R completes. Freeze a useful older prefix and prepare one private
   compaction candidate for that thread, retaining complete conversational/tool
   boundaries. Summarization must not overwrite the live session or R's history.
2. New input U arrives. Foreground work takes priority. Use a finished candidate
   only when its source prefix and existing model/prompt/configuration checks
   still match and the current request permits compaction. Otherwise use the
   ordinary context; do not wait for speculative compaction when avoidable.
3. At the start of response S, form the actual working context from the compacted
   prefix, exact retained tail and all fresh user/tool input. Publish a replayable
   compaction item covering that complete input frontier before S's assistant
   output, and generate from that same view. Never append new output to completed R.
4. Continue normal Responses chaining. Stateless clients retain/replay the new
   checkpoint; stateful reconstruction recognizes it as replacing covered context,
   not as another copy of the old history. Keep the previous recovery path until
   the new checkpoint is durable or completely delivered as required by the mode.

The checkpoint cannot merely contain the earlier background summary: if it is
emitted after U and the client prunes everything before it, U and retained tail
must still be recoverable. Each branch gets a checkpoint for its actual adopted
context even if it shares an immutable prepared prefix with another branch.

This is a proposed scheduling policy behind compaction, not a standardized idle
maintenance command. OpenAI documents compaction output during an active response;
our pinned OpenResponses core does not define unsolicited idle compaction output.
No extra idle event is needed for the flow above. A future task/inspection surface
could expose preparation status when useful. The Responses `background` field
means asynchronous execution of a requested response, not general idle maintenance.

Foreground priority needs bounded GPU work and yield/cancel checks; host-thread
priority alone is insufficient. Preserve the interactive KV where practical and
include session-switch/restoration cost in observed foreground latency. Run only
when pressure/headroom and configured budgets permit; storage behavior must honor
the session's persistence policy. Memory extraction/linking can be a separate
idle task without rewriting the active context.

Start with one candidate per thread and existing fallback/eligibility machinery.
Measure foreground delay, candidate adoption, saved compaction/prefill time,
wasted background work and practical retrieval/answer quality. Do not assume that
idle compute is free or that a smaller prompt necessarily lowers whole-task time.
Sources: [OpenAI compaction](https://developers.openai.com/api/docs/guides/compaction),
[background responses](https://developers.openai.com/api/docs/guides/background),
and the local [OpenResponses stream/extension contract](../.workspaces/openresponses-reference/src/specifications/2026-04-24.mdx).

### Compaction and recovery contract

Server-side compaction need not make the server the sole owner of recoverable
state. The client need not receive a readable summary: automatic compaction can
return an opaque `compaction` item in ordinary response output, and standalone
compaction returns a replacement input window. Clients can retain and replay
these items without interpreting their contents.

- A client-held, self-contained compaction payload plus the retained window can
  recover without durable per-session server history. The compatible decoder
  and any decryption keys must survive; an opaque reference into server RAM is
  not a self-contained checkpoint, nor does this imply cross-provider portability.
- If the client keeps only response IDs/deltas and the server never returns
  sufficient replay state, reliable restart recovery requires durable server
  storage of the effective context and its referenced data.
- Replaying the full original history is another fallback, but recompacting it
  can produce a different working context. Keep that distinct from restoring
  the previous compacted state.
- Durable logical history/working-context state and optional inference snapshots
  have different jobs. Ordinary KV can be rebuilt from effective tokens. An
  approximate suffix graft may carry influence absent from the edited text;
  restoring that particular state needs its snapshot or an explicit rebuild.
- Account for disconnection before checkpoint delivery: retain enough old history
  or a durable committed checkpoint to recover. Do not silently replace the only
  recoverable state with an undelivered ephemeral compaction.

Existing durable Responses storage is a starting mechanism, not proof that
compaction recovery is already implemented. A possible arrangement is durable
logical context plus replayable client checkpoints, with GPU/SSD inference caches
managed independently. Keep the recovery choice explicit when later selecting
features from this palette.

Sources: [OpenAI programmatic tool calling](https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling),
[OpenAI compaction](https://developers.openai.com/api/docs/guides/compaction),
and the local [OpenResponses hosted-tool and compaction specification](../.workspaces/openresponses-reference/src/specifications/2026-04-24.mdx).

## Current implementation

- `main`, `origin/main` and `flint-metal` identify the qualified Flint stack at
  `635d5d534e77`, published at the user's request on 2026-10-23 and verified
  directly on origin. `zig-rewrite` and `zig-engine` retain the earlier qualified
  Zig runtime at `37498248b4cb`. The stack starts
  at condensed base `60813d03`. The original condensed commit before artifact cleanup was
  `da7940e2f2d8`; use bookmarks for its rewritten successor.
- Detailed implementation history is preserved at
  `zig-rewrite-history-20261022`; the mechanical translation is archived at
  `zig-mechanical-archive`.
- `qwen-glm-zig` contains the completed native Qwen/GLM Flash port, based on
  the upstream adaptation at `030a040e`; it is local and has not advanced main.
- Supported native scope is DeepSeek V4.1 Flash, Flint, Qwen 3.8 Flash Next and
  GLM-5.3 Flash on Metal plus the
  shared server/CLI stack. DeepSeek also supports the native Agent/benchmark/
  Eval/Web stack. Flint's internal ds4-agent tool mapping is not implemented;
  use its Responses server with an external agent. Legacy GLM52, independent
  CPU oracles and explicit diagnostic/distributed model paths retain C.
- Actual Metal objects/ARC operations and pipeline catalogues remain Objective-C;
  MSL kernels and third-party C dependencies remain in their existing languages.
- Build using `tools/zig-build`; selected binaries are in `zig-out/bin`.
  Make delegates to the same build and supplies root executable names. Presets
  can select a separate installation with `./ds4-local --bin-dir ./zig-out/bin ...`.
- The server already has one engine, multiple active session slots, a decode /
  bounded-prefill coordinator, Responses continuation and durable conversations.
  Stateful Responses WebSocket serving is implemented and qualified. See
  [usage and supported scope](serving/RESPONSES.md).

Request preparation now has a C-free canonical input/model boundary. Engine
detection and C conversions stay in the runtime adapter; DS4.1 formatting and
reasoning policy live in native model modules. Execution/scheduling still import
shared C server/session records, and state storage still understands DS4.1
checkpoint kinds. Further component independence is unfinished.

## Qualification and costs

See [the final qualification report](../speed-bench/experiments/zig-qualified-20261022/README.md)
and [the implementation guide](README.md) for exact scope, commands and limits.

- Recorded campaign: 98 host/CPU fixture invocations, 45 GPU invocations and six
  diagnostic invocations; complete ordinary prefill vectors match across arms.
- Practical checks include repeated agent context edits / SSD suffix reuse with
  actual archive retrieval, server restart and concurrent conversations, trained
  DSpark comparisons, and vision tasks.
- Measured ordinary/task runtime is approximately neutral, with a slower ordered
  vision point. This establishes neither a speedup nor universal non-regression.
- Full rebuild: C 9.221 s, native 11.938 s. Measured controller edit: C 3.319 s,
  native 1.092 s. Native no-op build overhead is about one second.
- Passing behavior checks does not establish good source architecture. The
  original mechanical translation failed readability expectations. Handwritten
  owners improve readability, while C interop remains a source of complexity. The adoption cleanup removes
  the duplicate C implementations rather than maintaining two products.

## Current architectural slice and next steps

Completed serving goal (authorized 2026-10-23): stateful Responses WebSocket
serving using OpenResponses, reusing the native request/model path and qualified
for incremental conversations, concurrency and reconnect/restart under the
pressure guard. Main remains unchanged and raw results remain local.

OpenResponses `2026-04-24` requires sequential turns on each WebSocket; concurrent
conversations use separate sockets. Newer OpenAI documentation's `stream_id`
multiplexing is a separate extension and is not the initial target. Reference
checkout: `.workspaces/openresponses-reference`, commit
`8907bc3b7c4300ff5c21fd647cfde9929243afa1` (ignored).

Implemented: RFC6455 framing/upgrade, direct semantic-event sink
shared with HTTP, latest connection-local parent for `store=false`, durable
fallback for stored responses, and complete OpenResponses response envelopes.
ReleaseFast and Debug each pass the framing/native-boundary fixtures and all
26 integration checks, including the official SDK, real tool retrieval and clean
shutdown/restart. Ten selected unchanged official OpenResponses tests pass when
run serially. A stable tool loop reuses 292/317 prompt tokens and correctly reads
stored facts. The observed answer error in three optional cancellation probes
also reproduces on the independent C baseline. See the
[qualification report](../speed-bench/experiments/responses-websocket-20261023/README.md).
The installed `zig-out/bin/ds4-server` is byte-identical to the qualified binary.

On 2026-10-23 the user authorized the canonical request/model architectural
change. The completed slice separates logical request/history handling from
DS4.1 prompt, tool and reasoning materialization.

- New native records live in `serving/types.zig`, with model policy in
  `models/ds4.zig` and the DS4.1 renderer in `models/deepseek41/prompt.zig`.
- Production request materialization invokes those modules through
  `server/canonical.zig` and `server/model.zig`; required C interop remains in
  those compatibility adapters. The old whole DS4.1 renderer was removed.
- Existing native server fixtures and five standalone boundary cases pass in
  ReleaseFast and Debug. The standalone fixture requires no server/engine
  objects or Metal. Guarded HTTP restart/concurrency campaigns pass 22/22 checks
  and 5/5 graded answers each, for C -> refactored and refactored -> refactored.
  See [the focused report](../speed-bench/experiments/zig-serving-boundary-20261023/README.md).

Next proposed work: qualify server-owned context editing/suffix relocation and
idle-state eviction through an ordinary WebSocket agent workflow. Keep archival
and real retrieval in that task; existing HTTP/Agent features are not new
WebSocket qualification of every approximation setting.
Execution/session records and persisted payload semantics remain DS4-specific;
further backend independence is unfinished. Let a second actual model reveal
additional shared interfaces. Defer a universal kernel framework, runtime plugin
ABI and custom ring-I/O implementation.

Keep one heavy model job at a time under the pressure guard. Use bounded
experiments and existing tools; do not expand the harness after a useful
measurement is possible. Large RAM usage is expected; watch pressure and swap.

## Progress log

### 2026-10-23

- Recorded candidate server/client code composition, direct tool visibility,
  server-side MCP, manual/automatic compaction, specific context edits and
  cross-session memory in the design palette above. These are discussion notes,
  not new implemented capabilities or authorization to start them. Recovery
  requires sufficient state somewhere: client-carried replayable compaction,
  durable server context, or an explicit full-history reconstruction fallback.

- Planned, explicitly deferred: add native Metal V4.1/1017 activation steering,
  starting with FFN outputs. **Do not implement yet. Zero speed degradation is
  a hard acceptance requirement with steering disabled/zero and enabled.** Do
  not accept a small slowdown in exchange for behavioral control; compare fixed
  token workloads so shorter answers cannot hide slower inference. Prefer
  projection fused into existing kernels, retaining cache exclusions initially.
  The restriction already existed in upstream's first V4.1 implementation; it
  was not a feature removed by the Zig rewrite. The initial future experiment
  has a one-hour ceiling; see the [deferred plan](../speed-bench/ideas/next-experiments.md#deferred-v411017-activation-steering--planned-2026-10-23).

- Completed the M4 Pro and M5 Pro ports and cross-profile qualification on the
  available M5 Max. Auto detection selects Recursive's 36/32 resident schedule
  on Pro devices; `--flint-profile` permits a compatible smaller profile for
  qualification. M4 uses portable SIMD-group matrix/attention prefill with a
  2048-token chunk cap; M5 retains the existing NAX path and 4096-token cap.
  Profiles have distinct shader entry names; oversized resident grids and M5
  NAX on M4 are rejected before loading weights. The original M5 Max decoder
  source is unchanged.
- All three profiles ran two prompts with greedy/default sampling. Current
  M5 Max and M5 Pro reproduce the frozen native Max's 128-token continuations
  and eight teacher distributions exactly. M4 differs numerically: 8/8 teacher
  top tokens agree, mean KL 0.001505 and delta-NLL -0.014220 nats on the small
  eight-point probe; its free-running continuations eventually differ.
  Twelve Pro scalar/packet continuation comparisons match. Model-free GPU
  fixtures cover portable matrices, expert tails, attention/window/sinks,
  actual KV bank crossing/local-ring wrap and sampler handoff.
- Both Pro profiles pass 29 WebSocket/protocol checks and 17 answer probes,
  including tools, concurrent conversations, SDK, cancellation, durable restart,
  and 4855-token prefill followed by cached continuation. The host/server/
  storage/agent/script regression suite passes. On this M5 Max, M4's 798-token
  prefill is 478 ms versus 217 ms for NAX, with generation around 219 versus
  232 tokens/s. These are cross-profile measurements, not predictions for Pro
  hardware. Physical M4/M5 Pro residency, compiler behavior and performance
  remain unqualified. See the [profile report](../speed-bench/experiments/flint-profiles-20261023/README.md);
  raw evidence stays in `.workspaces/flint-profiles-20261023/`. `make all`
  installed the qualified executables; installed CLI/server bytes match the
  copies used for qualification. Changes remain on the local Flint stack.

- Added `download-model.nu flint` / `flint-int4` for the qualified native
  safetensors directory. Uses Recursive's pinned Hugging Face revision
  `3b93f9053b530488405ae9fca51d4900d6ce21f0`, twelve file identities from the
  original packaging manifest, and the existing resumable HF/SHA-256 flow.
  Downloads into `gguf/flint-int4` without changing the default GGUF symlink.
  All 15 offline download tests pass. A live run downloaded all seven metadata
  files from the pinned revision and verified the five existing weight shards
  through workspace symlinks, avoiding another 22.6 GB transfer. Nu parse and
  format checks pass; no inference/kernels changed.

- Completed Recursive Flint's native Zig/Metal backend and existing Responses
  HTTP/WebSocket integration. User explicitly chose the existing Metal runtime
  only: no MLX library or embedded Python in production. Reused licensed shader
  source and the original M5 Max persistent decode schedule. Source reference
  is `/Volumes/Models/flint` at `e53663b262af`; installed affine INT4 weights
  are under its `.local/share/rec/models/fdc85248...` directory (22.57 GB).
  [Usage and reproduction](../docs/FLINT.md),
  [qualification report](../speed-bench/experiments/flint-native-20261023/README.md).
- Reference baseline completed under the pressure guard after the user stopped
  their DS4 agent. Original Python Engine, M5 Max, context 262144, 128 generated
  tokens, three repetitions; medians of repetitions 1 and 2: short 21-token
  prompt prefill 43.5 ms / greedy 246.1 tok/s / default grammar 224.94 tok/s;
  798-token prompt prefill 227.3 ms / greedy 239.75 tok/s / default grammar
  212.61 tok/s. Raw results and exact driver remain ignored under
  `.workspaces/flint-baseline-20261023/`. This is the reference, not a native
  performance result. Default sampling is T=1, top-p=.95, top-k=20.
- Native tokenizer/prompt, raw safetensors bank packing, batched prefill,
  resident decode and GPU filtered sampling are implemented. Tokenizer matches
  47 HF cases in Release/Debug; seven packaged templates match. Native short
  prefill improved 202 -> 61 -> 37 ms by testing one small-batch QMV change at
  a time. Selected packed QMV retains NAX for larger batches. Default-sampling
  decode is 247.1 / 225.9 tok/s (short / 798-token), reference 224.9 / 212.6;
  greedy longer-task total is neutral (756.8 vs 757.7 ms), with slightly slower
  decode. These are narrow warm measurements, not a universal speed guarantee.
- Eight teacher-forced distributions inside reasoning: mean delta-NLL +0.013817
  nats, mean KL 0.0010015, all eight top tokens equal; perplexity ratio 1.0139.
  This is a small two-prompt proxy. Native RNG differs from MLX; native serving
  has no grammar masks. Do not claim exact output equivalence.
- Final real-model WS qualification passes 29 checks and 17 quality grades:
  independent concurrent conversations, real tool retrieval, disconnect,
  reconnect/restart and official SDK. A 4,855-token prompt plus post-decode
  continuation passes; input 5,011 correctly credits 4,855 cached prefill rows.
  Generated continuation is recomputed through prefill, so it is not counted as
  reused KV. Actual GPU fixtures pass token-65,536 bank crossing and wrapped
  local-ring publication; full 262k-token inference is not yet qualified.
- Per-conversation BF16 KV and bounded local-ring/logit snapshots are native.
  An engine mutex serializes prefill/resident packets: overlapping two resident
  kernels risks GPU barrier deadlock. Flint packets are distinct from MTP.
  Durable Responses JSON survives restart; Flint GPU-state SSD checkpoints,
  suffix relocation, vision and internal ds4-agent tools remain unsupported.
  The server-only `flint` preset selects model defaults and clears incompatible
  DeepSeek cache settings. Main is unchanged; new work stays local.
- Full model-free host/script verification passes, as do Debug server/session
  fixtures and all 11 launcher cases. A guarded existing DeepSeek control and
  final runtime produce identical selected IDs and all reported top-logprobs.
  Raw artifacts remain ignored in `.workspaces/flint-baseline-20261023/` and
  `.workspaces/flint-tests/`; commit source, concise reports and reproduction
  helpers only.

- Followed up on the user's script preference: use plain `#!/usr/bin/env nu`
  and `nu script.nu` invocations, removing blanket `--no-config-file` flags
  from entry points, callers and test children.
- Qualified the Nushell script migration using Nu 0.116.1. `ds4-local` and
  `tools/zig-build` keep their executable names, arguments and process-replacing
  behavior. Downloader/test/benchmark/service drivers use `.nu` names; the
  downloader is now `download-model.nu` with a standard JSON target catalogue.
- Keep Python for model/numerical work and the small split-model binary-I/O
  helper (flock, streaming copy/hash and fsync). All 30 download targets and eight
  pinned identities are preserved. Historical experiment snapshots retain their
  recorded scripts; the C-era build helper is a Nu wrapper for archived objects.
- Validation: the full Nu host verifier passes; `make test-scripts` passes ten
  launcher cases, thirteen genuinely hashed offline download cases and driver
  contracts. One-time controls match 66 GPU CLI invocations, DSpark cases, GLM
  prompts and sweep status/arguments. JPEG fixture bytes reproduce exactly.
  Nu parse/format checks and Linux-runner helper checks pass; Linux service
  actions and model/GPU runs were not part of this script migration.
- `make test` now delegates to `tests/verify.nu`; `make test-scripts` selects its
  fast script-only subcommand. Build flags are forwarded as arguments. Runtime
  binaries remain unchanged while the user's agent is running. Raw evidence is
  local under `.workspaces/nu-launchers/`, `.workspaces/nu-downloader-20261023/`,
  `.workspaces/nu-drivers-20261023/` and `.workspaces/nu-scripts-20261023/`.
  Keep these script changes separate from the pending research journal notes.

- Promoted the qualified runtime through `37498248b4cb` to `main` and pushed to
  `origin` at the user's explicit request. Confirmed remote main had not moved,
  ancestry was linear and no conflicts/raw benchmark dumps were introduced.
  Publication reused existing qualification. Pending local research notes were
  preserved outside the published commits.

- Completed a source review of Strata, local Flint and Luciano Martins's eight
  October 6–8 vLLM optimization PRs against Zig runtime `37498248b4cb`.
  This is research and proposed work; no runtime change or model benchmark ran.
  Flint remains at `e53663b262af`; Strata snapshot is `7ddc17a38fe7`.
- Strata's [cache-conditioned router](https://github.com/Niko1221/Strata/blob/7ddc17a38fe7f31637e895ecfba58c2cd43d65ec/src/kernels/cuda/route_prior.cu)
  biases selection toward resident experts while retaining unbiased mixture
  weights. DS4 already has global decayed hotness, bounded resident substitutions,
  selected-cache overlap and exact decoder dependency suffixes. Proposed follow-up:
  compare a simple residency bonus with the existing policy, retaining DS4's
  transformed-affinity/bias semantics and measuring task quality and SSD bytes.
- The first architectural target is exposed host/GPU handoff cost on cache hits.
  Existing [SSD experiments 62–66](../speed-bench/experiments/ds41-ssd/README.md#62-full-session-route-and-exposed-io-census)
  found 75.3% all-resident layer events, a 30.7% decode-time reduction with free
  future routes plus deferred checks, and a slower real prefetch predictor.
  These are historical workload results and oracle bounds, not current Zig speed
  measurements. Refresh the baseline before changing the selected-load schedule.
- A small independent candidate remains head-only Q4_K. Header-only inspection
  of the local 1017 checkpoint confirms `output.weight` is Q8_0, 5120 × 129280,
  670.703 MiB; Q4_K would occupy 355.078 MiB. Existing native dense dispatch
  accepts Q4_K. Profile head time, then measure speed, continuation delta-NLL/KL
  and practical outputs; the 47.1% head-byte reduction is not an end-to-end gain.
- vLLM sources: [attention tuning](https://github.com/vllm-project/vllm/pull/60317),
  [KV-sharing metadata](https://github.com/vllm-project/vllm/pull/60445),
  [vision/audio](https://github.com/vllm-project/vllm/pull/60478),
  [MoE shapes](https://github.com/vllm-project/vllm/pull/60504),
  [vision batching](https://github.com/vllm-project/vllm/pull/60643),
  [diffusion batching](https://github.com/vllm-project/vllm/pull/60663),
  [text fusion](https://github.com/vllm-project/vllm/pull/60674) and
  [MTP/head work](https://github.com/vllm-project/vllm/pull/60689).
  Only 60643 is merged at this review; seven remain open. Transferable proposals
  are actual-shape M5 tuning, fewer intermediate passes and host dependencies,
  and speculation selected by accepted tokens per wall time and unique expert
  bytes. Preserve the existing pressure guard and one-change/hour workflow.

- Completed source layout after the user clarified that C/H files and backend
  trees belong under `src/` too: 225 additional tracked files preserved. Common
  C/H/include tables are in `src/`; Metal driver/shaders join `src/metal/`;
  retained CUDA/ROCm sources are in `src/cuda/` and `src/rocm/`. No root C/H or
  backend source directories remain. Algorithms and supported-backend scope
  are unchanged; CUDA/ROCm are retained source, not newly hardware-qualified.
- `make all` / `make test`, native suffix transactions, four packet/ownership
  fixtures and the official scorer build pass. Default shader discovery works
  in a guarded DeepSeek run; tokens and all reported top-20 logits/logprobs
  match the saved pre-layout baseline exactly. The preset launcher also works.
- Updated active helpers and documentation links, leaving historical experiment
  source lists intact. JPEG tests now load the existing production image library
  instead of compiling the obsolete C decoder alone; all three pass. Existing
  benchmark helper checks pass 8 + 5 cases, and the calibration helper discovers
  the repository and extracts 184 Eval records without rewriting datasets.
- Independent review confirms file preservation and path-only backend changes.
  Reproduce the host checks with `make all test`; run extra fixtures through
  `tools/zig-build agent-context-suffix-test native-activation-test
  native-norm-test native-hc-projection-test native-f16-publication-test
  quality-score`. The layout smoke uses the same model command as the adoption
  report. Raw logs/manifests remain in
  `.workspaces/source-layout-completion-20261023/`; JPEG/helper logs are under
  `.workspaces/src-layout-20261023/`. Main is unchanged.

- Adoption cleanup completed: all five production mains are Zig. Removed the
  migration build option, native-selection switches and duplicate C owners;
  retain Qwen/GLM C engines behind the shared frontends and actual Objective-C
  Metal driver boundary. Server signal flags have one Zig owner shared with C
  signal/I/O helpers. Make now delegates to Zig, preserving root launch paths;
  GGUF scoring links the production engine. Main is unchanged.
- Qualified frozen-parent vs cleanup inference on DeepSeek, GLM and Qwen:
  short generated continuations and every reported top-20 logit/logprob value
  match exactly. GLM uses SSD streaming; Qwen uses its normal Metal path with
  residency/warmup disabled (n-grams remain disk-only, no Qwen SSD support).
  22 ReleaseFast host fixtures, seven Debug fixtures, Qwen/GLM GPU suites and
  all 26 required live WebSocket checks pass. Both live servers exit cleanly.
  Installed the same executables through Make at the usual root paths; both
  `make all` and `make test` pass end to end.
- The default DeepSeek Metal fixture has a pre-existing FP4 rounding-midpoint
  failure reproduced identically by the frozen parent; no kernel fix is bundled
  with adoption. The earlier three WebSocket answer-quality failures also remain
  unchanged. A resident GLM baseline was pressure-stopped before switching to
  the working SSD path. Full details and commands are in the
  [cleanup report](../speed-bench/experiments/zig-adoption-20261023/README.md);
  raw results stay under `.workspaces/zig-adoption-cleanup-20261023/`.
- Completed the authorized `zig/` to `src/` directory rename as a separate
  change after adoption. Component structure and runtime behavior are unchanged;
  build paths, C includes, test imports and current documentation links follow
  the new location. Historical experiment source lists retain their old paths
  for the revisions they describe. Build metadata stays at root; the resume
  journal is now `src/JOURNAL.md`. `make all` and `make test` pass, as do
  the model-free suffix transaction fixture and four Metal packet/ownership
  fixtures. All 376 files are preserved; implementation content matches after
  path substitution, and all 31 updated Markdown links resolve. No new model
  or latency campaign was needed for this directory-only change. Raw build/test
  logs stay under `.workspaces/src-rename-20261023/`.

- User authorized the request/model architectural change. Added native logical
  records and a C-free DS4.1 preparation implementation, replacing the previous
  whole renderer. Model aliases, reasoning policy and runtime format selection
  now belong behind the model adapter; durable history remains server-owned.
- ReleaseFast and Debug build and CPU checks pass. A build caught a signed C
  enum conversion mismatch; review caught the disabled fallback for unknown C
  reasoning values. Both were corrected before final CPU verification.
- Reusing the existing guarded HTTP two-session/restart/cancellation campaign;
  both cross-version and refactored-only campaigns completed normally. They
  cover fresh requests, concurrent decode, cancellation, surviving requests,
  streaming and durable Responses/conversation restart. No numerical kernel or
  checkpoint format changed. Results are local under
  `.workspaces/zig-serving-boundary-20261023/`; concise results and commands are
  [tracked](../speed-bench/experiments/zig-serving-boundary-20261023/README.md).
- The current slice is complete. Responses WebSocket transport remains the next
  feature in this sequence; the subsequent user-authorized goal completed it.
  A generic execution backend and second native model remain later work.
- Completed stateful WebSocket serving using OpenResponses `2026-04-24`, with
  sequential sockets and cross-socket concurrency. Added direct event sinks,
  ephemeral parent replay, durable recovery, framing and shared response schemas.
- ReleaseFast and Debug integration each pass 26 required checks; the real
  function tool reads a stored record whose facts are absent from its prompt.
  Official Python SDK 3.37.0 works. All test servers exit cleanly.
- The stock OpenResponses runner overfills the two-slot server with ten parallel
  tests and a 30 s timeout (3/10 pass). The unchanged ten official cases pass
  serially with 180 s timeouts. Do not present the first timeout run as a schema
  failure or claim that all optional protocol features are supported.
- Quality records remain separate: each full run passes 12/15 answer probes;
  the other three repeat one wrong JSON answer before/during/after cancellation.
  An independent C-server control produces the same answer and token counts.
- Stable instructions/tools yield 292/317 cached prompt tokens in a real file
  lookup loop; changed instructions in the broader integration yield no prefix
  hit, as expected. This is reuse evidence, not a whole-agent speedup claim.
- Installed the qualified server in `zig-out/bin`. Results, first failures and
  reference sources stay local under `.workspaces/`; only concise reports,
  implementation and necessary tests are tracked. Serving goal is complete.

### 2026-10-22

- Completed and qualified the handwritten rewrite, condensed it into one branch
  commit, and retained detailed history without advancing main.
- Reviewed source quality: handwritten Zig owners are generally preferable for
  engine/agent work; small C routines can remain more direct. Organization is a
  large part of the benefit. Keep the real Objective-C Metal boundary.
- User clarified the serving product and USRBIO reference. Adopt the direction
  above, with simple in-process I/O as the starting point.
- User requested readable commits: keep raw benchmark logs, generated snapshots
  and run dumps local/ignored. Keep concise reports, reproduction scripts and
  necessary fixture inputs tracked. Removed 6,749 new generated files (55.5 MB)
  from tracking, preserving every local file byte-for-byte; 234 reports, scripts
  and fixture inputs remain tracked. Older main history is outside this cleanup.
  Verification inventory is local under
  `.workspaces/benchmark-tracking-cleanup-20261022/`. Runtime source is unchanged.
- Added this journal and its resume/update instructions in `AGENTS.md`.

## Source references

- [DeepInfer storage deployment notes](../../DeepInfer/doc/inference_deployment_qa.md)
- [3FS USRBIO](https://github.com/deepseek-ai/3FS/blob/main/src/lib/api/UsrbIo.md)
- [Responses WebSocket semantics](https://developers.openai.com/api/docs/guides/websocket-mode)
- [Responses conversation state](https://developers.openai.com/api/docs/guides/conversation-state)

## Qualification constraint — resolved 2026-10-24

A terminal-launched user Flint server on 127.0.0.1:8001 started at 08:21 Eastern
(PID38036); it is not one of this task's qualification binaries. Qwen candidate
and frozen control were both pressure-stopped with unchanged guards. Actual
Flint continuation passes all28 checks; common continuation CPU/host probes pass.
The user stopped that server; isolated Qwen continuation then passed with normal
pressure and unchanged guards. The two stopped runs remain recorded failures.
Combined qualification and root installation are complete. The user's Flint
server is restored on127.0.0.1:8001 with its original model, two sessions,
262144 context and4096 prefill chunk; GET /v1/models confirms Flint/context.
Account for this live server before future latency/model campaigns. Main stays
unchanged; the private review stack contains five commits.
