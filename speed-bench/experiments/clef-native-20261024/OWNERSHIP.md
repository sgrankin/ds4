# Native engine ownership

The Clef integration requires a loaded helper to survive a generator opening,
closing or failing to open. It also exposed the next concrete restriction:
DeepSeek's former model globals prevented two ordinary native DS4.1 engines in
one process. This change gives those resources actual engine owners. It does
not combine two architectures into one kernel.

## Actual storage and entry boundaries

| Lifetime | Owner and contents |
| --- | --- |
| Process | Shared Metal device/queue, submitted-GPU-group arbitration, selected-expert worker mailbox, immutable environment policy and the GGUF process admission lock |
| DS4 Metal engine | `DS4MetalContext`: bootstrap, model views, streaming cache and read pool, command schedule, pipelines, dispatch scratch, exact-view cache, live tensor registry and optional route diagnostics |
| DS4 host engine | `ds4_host_context`: shape, compression/Qwen parameters, streaming policy, requested CPU parallelism and pool, expert profile, allocation guard, native route/oracle diagnostics |
| Session | Existing graph/KV resources, history, checkpoints and steering state |
| Flint/Clef engine | Existing independent weight/buffer/pipeline owners, borrowing the shared process device/queue |

The native Core holds the actual Host and Metal handles. Compatibility `g_*`
names now select fields in those records; they are not copies of a process-wide
model snapshot. Immutable lookup data remains shared. The optional process GPU
timeline logger remains a process logger; its per-encoding batch is model-owned.

Public C engine/session/chat/tokenization entries and native Zig entries select
their Core's owners, then restore the previous selections. Threads do not inherit
thread-local selection. Selected-expert jobs, persistent expert readers, CPU
workers and Metal keepalive/TP threads explicitly capture their relevant owner,
pool, descriptor or model view. The existing finish/join boundaries end those
borrows before retirement. The shared selected-expert mailbox still has one
start/finish slot; a competing submission uses the existing synchronous fallback.

CPU callbacks use their engine's CPU pool and selected Host shape. Different
engines can prepare CPU work independently. One engine's submitted GPU group
may contain overlapping command buffers, preserving resident/expert dependencies;
another engine waits until that group drains. CPU preparation and the CPU head
hold no GPU lease.

## Retirement and admission

Core retirement stops its CPU pool and diagnostic owners, drains GPU submissions
and expert reads, retires GPU views/caches, then closes the underlying model
maps and descriptors. It restores the prior selected owners before destroying
its own records. This also repairs the inherited cleanup order that closed maps
before the GPU/load cleanup which could still borrow them.

Only actual allocation/initialization creates a tensor-registry or exact-cache
record. Stats on an empty owner return zero; repeated empty cleanup does nothing.
Disabled expert profiling does not allocate its large profile table merely to
test whether profiling is active.

Ordinary local native DS4.1 Cores share a process-owned referencecounted file
lock. The first Core acquires the existing lock, each later Core in the same
process borrows it, and the last close releases it. Each Core records its own
borrow. Closing A cannot drop the guard protecting B against an accidental
second process. Retained CPU/legacy paths, TP and other GGUF families keep their
existing single-Core admission pending separate qualification. Flint/Clef do
not acquire the GGUF admission lock.

This source boundary enables multiple native DS4.1 engines through the public
engine API. The server's current generator-plus-decision-helper registry is a
separate routing boundary; this change does not claim arbitrary simultaneous
loading/routing of all supported generative families.

## Qualification checkpoints

The first shared-runtime/selected-owner slice passed the 28-check real
Flint+Flash and DS4.1+Flash serving suites. That checkpoint's matched short
DS4.1 screen was 785.32 ms versus a 773.67 ms frozen control, with ordinary
decode around 25–26 tokens/s. It preceded the complete Host/PSO ownership move,
so those measurements are historical evidence, not the final owner performance
bound. See [mixed serving](mixed-serving.md).

The complete ownership source builds `ds4-server`, `runtime-host-test`,
`runtime-lifecycle-test` and `runtime-two-engine-test` in
`.workspaces/clef-owner-final-build`. The model-free Host fixture passes actual
independent 2/3-thread pools, concurrent callbacks with different 5120/4096
shapes, explicit worker selection and B surviving A's destruction.

On integration change `wzvyrkmkpkyn`, the complete host/server/agent/recovery
verification also passes. Ten model/token metadata cases match the frozen
control, including standalone metadata/token dumps which now select a temporary
inspection Host rather than mutating a live engine's shape.

The final single-model old/new numerical screen also passes for DS4.1, Qwen,
GLM and Flint: all 32 full-vocabulary FP32 rows are byte-identical, with zero
ΔNLL/KL and all eight guards completing normally. These are bounded fixed-input
checks, not performance measurements. See [engine regression](ENGINE-REGRESSION.md).

The final actual-Metal lifecycle passes: two initialized DS4 contexts run
A→B→A GPU primitives, then B's pipelines and tracked tensor handles survive A
cleanup. It also checks async completion, shared queue identity, failed third
shader startup, empty queries/cleanup leaving no orphan owner record, and zero
final runtime borrows. The pressure guard completes with child 0, normal level 1
throughout and no swap growth. Evidence:
`.workspaces/clef-owner-final-20261024/lifecycle/`.

The public-engine gate also passes. It loads the installed
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` twice, with context 512 and
separate 8 GiB SSD caches. Every frozen full-vocabulary FP32 logit comparison is
bit-identical, with unchanged argmax and committed histories during A→B→A.
After A closes, B continues, tokenizes another prompt and allocates a fresh
graph successfully. An independent lock descriptor remains blocked while B
survives and acquires successfully after both engines close. Runtime borrows
return to zero and public entries restore the ambient selections. The guard
completes with child 0 and normal pressure. Evidence:
`.workspaces/clef-native-20261024/two-engine-actual/`.

## Final matched serving screen

The final owner candidate passes all **28** HTTP/WebSocket/SDK, ancestry,
distinct-helper-state and cancellation checks in the existing mixed-serving
suite with DS4.1 plus Clef-Flash loaded in the same process. All servers close
cleanly. The lifecycle guard and all three serving guards finish with child 0,
normal pressure level 1 throughout and no swap growth.

The main uses the installed 1017 IQ2 checkpoint, context 8192, prefill 512,
two resident sessions, mixed-prefill quantum 512, SSD streaming and the unchanged
automatic cache budget. Both frozen controls and the candidate load the Flash
helper and set `DS4_CLEF_FULL_HEAD_GPU=1`. Each arm performs the same warmup and
five main-only WebSocket requests, generating the same five-token JSON answer.
Other model jobs and CPU builds are paused during measurement.

| Arm | Main median |
| --- | ---: |
| Frozen pre-owner server, opening control | 788.23 ms |
| Final owner candidate, helper resident and idle | 787.68 ms |
| Frozen pre-owner server, closing control | 783.09 ms |

The candidate is 0.07% below the opening control and 0.59% above the closing
control. This short screen finds no noticeable repeatable regression. It does
not establish a sub-percent overhead bound or long-context performance.

The full mixed suite's three-repetition medians are:

| Request | Isolated | Overlapping |
| --- | ---: | ---: |
| DS4.1 generator | 789.94 ms | 849.62 ms |
| Clef-Flash helper | 68.24 ms | 70.68 ms |

This is measured GPU contention for one small request pair. It is not a
throughput-scaling result. The subsequent bounded layer-group scheduling
experiment is separate from this ownership qualification.

Candidate SHA-256:
`44b3930bcbca121ab95df2f87df643fd47d69cdb32a6f2a06e33ce3d64e7d9f1`.
Frozen control SHA-256:
`fa67e1d5f3a0ec19fc28b7da5110867b6dc5f6beb699a0d7ecef4ab0ac8b921b`.
Complete commands, socket events, pressure samples and summary stay ignored in
`.workspaces/clef-owner-final-20261024/`.

```nu
nu tools/zig-build ds4-server runtime-host-test runtime-lifecycle-test runtime-two-engine-test --prefix .workspaces/clef-owner-final-build
.workspaces/clef-owner-final-build/bin/test-runtime-host
python3 speed-bench/ideas/pressure_run.py .workspaces/owner-check/lifecycle/guard -- .workspaces/clef-owner-final-build/bin/test-runtime-lifecycle
python3 speed-bench/ideas/pressure_run.py .workspaces/owner-check/two-engines/guard -- .workspaces/clef-owner-final-build/bin/test-runtime-two-engines gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
```

For a serving arm, launch under the existing guard in one terminal:

```nu
$env.DS4_CLEF_FULL_HEAD_GPU = '1'
python3 speed-bench/ideas/pressure_run.py .workspaces/owner-check/serving/guard -- .workspaces/clef-owner-final-build/bin/ds4-server --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --decision-model gguf/clef-flash-int4 --host 127.0.0.1 --port 19309 --ctx 8192 --prefill-chunk 512 --ssd-streaming --tokens 128 --batched-session 2 --mtp-draft 1 --mixed-prefill-quantum 512
```

Run the existing client from another terminal, with a new output directory:

```nu
nu tests/mixed-serving.nu --url http://127.0.0.1:19309 --main-model deepseek-v4.1-flash --output .workspaces/owner-check/serving/client --repeats 3
```

Use `--timing-only --repeats 5` for the matched idle-helper main loop. Stop the
owned server cleanly and wait for the guard to complete before the next arm.

## Source map

- [Metal owner](../../../src/metal/context.inc.m),
  [public handle](../../../src/metal/context.h)
- [Host owner](../../../src/engine/host_context.h), implementation in
  [the retained C support library](../../../src/ds4.c)
- [Core retirement](../../../src/engine/core.zig),
  [factory/admission](../../../src/engine/engine_lifecycle.zig),
  [native entry scope](../../../src/engine/metal_scope.zig)
- [selected-expert job capture](../../../src/engine/stream_selected.zig),
  [expert read pools](../../../src/metal/expert_io.zig)
- [Host fixture](../../../src/runtime/host_test.zig),
  [Metal lifecycle fixture](../../../src/runtime/lifecycle_test.zig),
  [public engine fixture](../../../src/runtime/two_engine_test.zig)
