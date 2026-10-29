# Handwritten Zig runtime

Start with [project direction and current status](../docs/project.md).
The [lab journal](../docs/lab/README.md) records research decisions and evidence.
The [serving request/model boundary](serving/README.md) describes native logical
input and the DS4.1 preparation path, including its remaining compatibility edges.
[Stateful Responses](../docs/serving/responses.md) documents HTTP/WebSocket usage and
the supported OpenResponses contract.

`src/` contains handwritten Zig model implementations for DeepSeek V4.1 Flash,
Flint, Qwen 3.8 Flash Next, GLM-5.3 Flash and the Clef decision family on Metal,
plus the Agent, server,
CLI, benchmark and evaluation frontends. The optional Web client is a separate
Zig library. All five program entry points are Zig. Shared engine/session,
startup/admission, continuation/batch, payload and SSD map/page-in policy have
native owners too. The [runtime migration report](../speed-bench/experiments/native-runtime-20261024/README.md)
records their qualification. Third-party dependencies and the Metal driver
retain their original languages.

## Scope and organization

Project C/H and common include tables live here alongside the Zig owners.
`metal/` contains the Zig host controllers, Objective-C driver and MSL shaders;
retained CUDA and ROCm sources live in `cuda/` and `rocm/`. The supported build
remains macOS Metal. Tests, tools, third-party dependencies and model weights
have their own top-level directories.

| Area | Native ownership |
| --- | --- |
| [Engine](engine/) | Engine factory/admission, model configuration and weights, input/vision, graph/workspace lifetime and execution, sessions/batching/continuation, sampling, snapshots/payloads, suffix reuse, trained DSpark/speculation, selected-expert workers and SSD view/page-in policy. |
| [Metal host controllers](metal/) | GPU startup/retirement policy, model-map/residency/warmup transactions, dense and routed matmul, expert cache/tables/load scheduling, attention/output projection, compressor updates, embedding/gather, RoPE, normalization, HC, router selection, quantization/copy primitives, resource and command scheduling, and selected-route diagnostic files. MSL kernels remain unchanged. |
| [Agent](agent/) | Options/startup, worker sessions, turns and tools, terminal/streaming output, history and live context, archive retrieval, editing and Bash, compaction/private summary, checkpoints, cancellation and shutdown. |
| [Server](server/) | HTTP/input/JSON, request materialization, OpenAI/Anthropic/Responses output, conversations and tool memory, image resources, batching, cache reuse/publication, generation, routing and serving lifecycle. |
| Frontends | Complete `ds4`, `ds4-bench` and `ds4-eval` controllers; the optional Web/CDP transport and recovery client. |
| Shared libraries | GGUF metadata/directory parsing, vocabulary/BPE/pretokenization/chat, numeric sampling, Engram, images, state cache/KV store, SSD configuration and response-file storage. Shared tokenizer/chat utilities also serve GLM and Qwen callers. |
| [Clef](models/clef/) | Dense hybrid and Omni MoE backbones, native tokenizer/record, learned GPU joint head, image/audio towers and model-specific media placement. [Usage and qualification](../docs/models/clef/usage.md). |
| [Shared runtime](runtime/) | Process Metal device/queue lifetime and owner-aware in-flight submission groups. Clef/Flint resources and native DS4.1 Host/Metal contexts have independent owners; retained legacy and TP admission limits differ. [Ownership qualification](../speed-bench/experiments/clef-native-20261024/OWNERSHIP.md). |

Algorithms live in ordinary Zig owners with typed borrowed inputs and explicit
allocation/resource lifetimes. Small entry modules expose each existing C ABI
once. Mutable process state has one owner, in a sole entry or the actual shared
driver record. Pure helpers compose without another export or duplicate state.

Project records and shader packets are declared once in actual shared headers.
`build.zig` generates bindings from those headers and shares binding modules
where type identity matters. The implementation is handwritten Zig, rather than
translated C bodies or checked-in SDK declarations. Ownership stays compatible
with the public API: libc allocations remain libc allocations, borrowed model
and tensor views remain borrowed, and scoped retained handles release their
independent references after the transaction's safe boundary.

## Build and run

The [build and test manual](../docs/development/build-and-test.md) owns native
prerequisites, Zig targets, cache locations, script conventions and diagnostics.
Programs install in `zig-out/bin`; Make uses the same build for root compatibility
names. Use [the agent guide](../docs/serving/agent.md) and
[server guide](../docs/serving/overview.md) for presets and startup.

## C, Objective-C and shader boundaries

Objective-C implements the actual Metal driver boundary: device/queue/library
creation, literal PSO/function-constant catalogues, real `MTLBuffer` ownership,
encoder bindings, submission/wait and strong-reference retirement. These
catalogues include shared other-family resources and remain explicit driver
source. Native Bootstrap owns feature/environment decisions and phase order;
ModelLifecycle owns map replacement, residency and warmup policy. Native route
diagnostics own binary replay/recording and hotlist file transactions over libc.
The final combined qualification covers all three closing owners.

Native controllers decide packet contents, offsets, geometry and transaction
order. ARC retains/releases real Apple objects, with lexical autorelease pools
and explicit borrowed/retained contracts. Resource caches and existing memory/
timing report I/O remain named C/Objective-C libraries. TP process/service-thread
transport remains C; local gate fixtures do not establish two-host behavior.

MSL remains the GPU math implementation. Third-party C libraries, including
image codecs, line editing and radix-tree support, remain dependencies. Public
C compatibility shells, platform/signal installation, selected diagnostic I/O
and TP/distributed transport also remain explicit boundaries.

Ordinary calls still use C platform/library services for allocation admission,
token-vector storage, process/thread/lock operations, memory measurements,
page warming and diagnostic/report I/O. Compatibility entry points allocate
private Engine/Session shells and borrow actual shared component addresses;
Zig does not duplicate their layouts. These services are distinct from the
separate C model implementations below.

DeepSeek V4 Pro, full GLM 5.3 DSA, independent CPU references and explicit legacy
or distributed diagnostic paths retain C implementations. Original V4 Flash and
GLM 5.2 are retired. Ordinary DS4.1, Flint,
Qwen and GLM53 Metal execution and prompt mapping have native owners.
This build supports macOS Metal; other backend code still present for
upstream correspondence is not a maintained alternate product. Migration
switches and duplicate C bodies for the adopted Zig implementation are removed.

## Qualification and evidence

Fixture targets build the actual selected production owners. Run their binaries
separately; targets do not automatically start models or GPU work.

```sh
./tools/zig-build server-test agent-test sampling-test session-state-test
./tools/zig-build -Doptimize=Debug server-test agent-test
./tools/zig-build graph-test metal-test dense-mpp-test ssd-experts-test
```

Model/GPU campaigns run serially under
[`pressure_run.py`](../speed-bench/ideas/pressure_run.py), with commands, source
and binary identities, raw observations and failures retained. Debug checks and
C/native comparisons protect ownership, continuation and protocol behavior;
passing a CPU fixture is not a GPU or real-task qualification. Known shared
baseline failures and required math modes belong in the corresponding report.

Grouped records provide the detailed scope and reproducible commands:

- Engine and sessions: [input/lifetime](../speed-bench/experiments/zig-engine-input-20261022/), [execution](../speed-bench/experiments/zig-engine-execution-20261022/), [session controllers](../speed-bench/experiments/zig-session-controllers-20261022/), and [vision](../speed-bench/experiments/zig-engine-vision-20261022/).
- State compatibility: [canonical payloads](../speed-bench/experiments/zig-engine-payload-20261022/), [repeated suffix reuse](../speed-bench/experiments/zig-engine-suffix-20261022/), [trained DSpark](../speed-bench/experiments/zig-engine-dspark-20261022/), and [speculative cycle](../speed-bench/experiments/zig-engine-cycle-20261022/).
- Agent: [compaction](../speed-bench/experiments/zig-agent-compaction-20261022/), [turn execution](../speed-bench/experiments/zig-agent-turn-20261022/), and [complete frontend/actual task](../speed-bench/experiments/zig-agent-frontend-20261022/).
- Server: [wire protocols](../speed-bench/experiments/zig-server-protocols-http-20261022/) and [complete serving/restart integration](../speed-bench/experiments/zig-server-full-20261022/).
- Components and earlier Metal checks: [tokenizer](../speed-bench/experiments/zig-tokenizer-hour8-20261022/), [CLI](../speed-bench/experiments/zig-cli-frontend-hour13-20261022/), [benchmark](../speed-bench/experiments/zig-bench-frontend-hour11-20261022/), [Metal resource lifetime](../speed-bench/experiments/zig-metal-resources-20261022/), and [exact cache](../speed-bench/experiments/zig-metal-exact-cache-20261022/).

These component reports describe their recorded revisions. The final combined
[qualification](../speed-bench/experiments/zig-qualified-20261022/README.md)
covers the selected source at `5e5bb690`, plus the documented include-path-only
Make fix. It records 98 host/CPU,45 GPU and six diagnostic invocations, exact
ordinary frontier vectors, repeated Agent/SSD reuse, durable server restart,
trained speculative cycles and six actual vision tasks in each implementation.

## Historical rewrite measurements — 2026-10-22

| Measurement | C reference | Native selection |
| --- | ---: | ---: |
| Full rebuild | 9.221 s | 11.938 s |
| No-op build | 0.014 s | 1.025 s |
| F32 controller edit | 3.319 s | 1.092 s |
| Ordinary process, bracket mean | 20.700 s | 21.061 s |
| Six-turn persistent Agent | 229.193 s | 230.237 s |

The full rebuild costs 29.5% more; the measured controller edit costs 67.1% less.
No-op overhead remains about one second. Ordinary steady decode differs by
under 1%; the control prefill rate drifts about 7% within the bracket. Task timings
are near neutral here, with an ordered vision point 4.73% longer. No blanket
build or runtime speedup is claimed. Commands, individual points and limits
belong to the final report. These are October 22 qualification results; the old
branch and publication state are historical. See [current direction](../docs/project.md)
for the present stack.
