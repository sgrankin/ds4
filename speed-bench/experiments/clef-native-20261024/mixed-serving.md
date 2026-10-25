# Mixed native engine qualification

This report preserves the staged measurements. The subsequent complete DS4
ownership move and its qualification boundary are recorded in
[native engine ownership](OWNERSHIP.md).

Working-tree screen on 2026-10-25, based on plan revision `6262ae0ee7ea`.
Hardware: M5 Max, 128GB. Published main stayed at `ce3afb7999b7`; no root binary
was overwritten. One pressure-guarded model process at a time, with the other
model jobs paused during timings. Every owned server shut down cleanly, all four
guards returned child 0, pressure stayed normal and swap did not grow.

## Actual capabilities

One process loads native Flint plus Clef-Flash. **28 checks pass** through the
[Nushell driver](../../../tests/mixed-serving.nu) and its focused
[wire leaf](../../../tests/mixed-serving-wire.py), reusing the existing WebSocket
client and the official OpenAI SDK. The leaf is testing code, not a production
Python dependency.

The checks cover both loaded models in discovery, rejection of an unloaded known
model, cross-endpoint model rejection, distinct concurrent helper states, correct
main/helper answers under overlapping requests, incremental WebSocket ancestry
around a helper call, rejection of a helper model on a generator continuation,
main disconnect while the helper is active, helper disconnect followed by both
engines recovering, and a real official-SDK Responses WebSocket request.

This found and led to fixes for two concrete server gaps: known model names were
previously accepted even when that family was not loaded, and durable parent
responses were not checked against the active model family. Discovery and
request routing now use the loaded model's alias policy. `previous_response_id`
checks the stored parent model; conversation input remains reusable logical
history. The mixed socket test exercises the live continuation path. A restart
under a different generative family remains a separate ancestry qualification.

Shared device/queue lifetime also passes the model-free actual-Metal fixture:
A→B→A, asynchronous completion, one context closing while another survives,
DS4 bootstrap/cleanup preserving the survivor and queue identity, failed shader
startup, and zero final runtime borrows. Same-owner command buffers may overlap;
other engines wait for that owner's submitted work to drain. CPU preparation and
the CPU head do not hold this GPU exclusion. This is not simultaneous execution
of both models in a combined kernel.

DS4's model/cache/scheduler globals still permit only one GGUF engine owner.
Independent Flint/Clef engines coexist with that owner; this slice does not
claim arbitrary multiple GGUF engines or a fairness/priority scheduler.

## Matched small latency screen

Flint used its existing INT4 checkpoint, M5 Max profile, context 8192, two resident
sessions, prefill chunk 4096 and output allowance 1024. The requested answer was
`{"ready":true}`. All arms generate the same 57 tokens, including 50 reasoning
tokens, from 28 input tokens. Repeated cached-prefix requests determine these
medians; the first unused resident session can still incur a cold request.
These are HTTP/WebSocket wall times, not kernel-only decode measurements.

| Arm | Repetitions | Main latency median |
| --- | ---: | ---: |
| Previous executable, opening control | 5 | 229.65 ms |
| New runtime, main alone | 5 | 234.37 ms |
| New runtime, helper resident but idle | 3 | 231.01 ms |
| Previous executable, closing control | 5 | 233.08 ms |

The new-alone arm is 2.1% above the opening control and 0.55% above the closing
control. This small screen finds no noticeable repeatable runtime regression;
it cannot establish a sub-percent overhead bound or long-context performance.
The closing control was repeated with the same Python 3.12 wire environment as
the opening and new-alone arms after the SDK-inclusive fixture had switched to
the existing Python 3.15 SDK environment. Both closing client variants agree
within about 1%.

With both requests submitted together, same-process GPU contention is visible:

| Request | Isolated median | Overlapping median |
| --- | ---: | ---: |
| Flint generator | 231.01 ms | 290.16 ms |
| Clef-Flash decision | 69.31 ms | 130.23 ms |

All answers remain correct. This measures one tiny fixed pair, not throughput
scaling or a claim that concurrent GPU execution is faster. The decision engine
used `DS4_CLEF_FULL_HEAD_GPU=1`; that separate kernel experiment retains its own
numerical and task qualification. GPU work is serialized between engine owners,
while CPU preparation can proceed independently.

The first mixed run passed all 27 functional checks then failed because its
reference-only Python environment lacked the OpenAI SDK. The driver now defaults
to the already installed `.workspaces/responses-ws-venv/bin/python`; the complete
repeat passes all 28 checks including the SDK call. This was a client dependency
failure, not a server error. No package was installed for the fix.

## Reproduction and evidence

The existing model is at
`../flint/.local/share/rec/models/fdc85248ad77a0fa9fbf0abda8b1525bea8ee74c11ada2ec95eb66113353d677`.
The helper is `gguf/clef-flash-int4`. Candidate SHA-256:
`87966019284388f0ff1ad2f96fb6d833377d35d8590c98601913670e1bdbf95c`.
Previous executable SHA-256:
`68c6e8d0bf574d303fcc14775121581a25797cedae43f2b48a4fa9c775cd1f4a`.

```nu
$env.DS4_CLEF_FULL_HEAD_GPU = '1'
python3 speed-bench/ideas/pressure_run.py .workspaces/mixed-check/guard -- .workspaces/clef-native-20261024/build/bin/ds4-server --model ../flint/.local/share/rec/models/fdc85248ad77a0fa9fbf0abda8b1525bea8ee74c11ada2ec95eb66113353d677 --decision-model gguf/clef-flash-int4 --host 127.0.0.1 --port 19309 --ctx 8192 --prefill-chunk 4096 --tokens 1024 --batched-session 2 --mtp-draft 1
```

From another terminal, while no other model job is running:

```nu
nu tests/mixed-serving.nu --url http://127.0.0.1:19309 --main-model flint --output .workspaces/mixed-check/client --repeats 3
```

`--timing-only` runs just the generator loop and can target the previous
executable without helper support. `--python PATH` selects an already installed
client environment. Use a new output directory for every run.

Raw commands, complete socket events, requests/replies, server logs, pressure
samples and summary stay ignored in `.workspaces/clef-mixed-20261024/`.
The authoritative complete run is `new-helper/qualified-client/result.json`;
`summary.json` records the four arms and guard outcomes. Root also performed
separate image grounding checks against this resident Flash helper after the
final timings; those results are recorded separately from the contention screen.

## Remaining DS4 owner migration: inspected, not implemented

The shared runtime changes actual device/queue lifetime. They do not make DS4's
entire backend per-engine. Inspection of the current source finds roughly 411
static `g_*` declaration lines in `src/metal/ds4_metal.m`, including many PSOs.
The important mutable owners are command scheduling, model views/file descriptor,
expert cache and asynchronous expert-load pool, exact/pipeline caches, transient
and attention/compressor/Qwen scratch, model/streaming/quality flags, and
TP/keepalive state. Some are already typed records, but their concrete instances
remain global. C also publishes shape/compression/Qwen configuration globally;
`g_ds4_shape` has 118 references in `src/ds4.c`.

`Core.freeRecord` stops the DS4 CPU `g_pool`, closes its maps and performs the DS4
Metal cleanup before releasing its instance lock. Clef's Accelerate head and
Flint do not use this pool. Moving CPU-pool shutdown to the last shared Metal
borrow would therefore give it the wrong lifetime. Two GGUF owners would also
race the pool's single callback/context slot; a reference count alone cannot
make concurrent submissions safe.

Making just the small bootstrap scalar record per-owner would leave the real
maps, scheduler and workers global, so it is not a meaningful first migration.
Do not remove the single-GGUF instance guard based on that cosmetic change. A
concrete next sequence is:

1. Give the existing typed model-view, expert-cache and expert-load records one
   actual DS4 owner. Pass that owner to asynchronous worker callbacks and retire
   its work before its mapped weights. Retain the single-GGUF guard while this
   slice is compared against existing decode/prefill results.
2. Move command scheduling, model-dependent PSO caches and scratch into the same
   owner; pass its identity at the encoding boundary. A TLS alias or global
   active-owner pointer does not by itself isolate concurrent CPU encoding.
3. Remove global configuration reads from the live GGUF path and give CPU work a
   concrete pool/submission owner. Only then qualify alternating and concurrent
   two-GGUF engines and consider relaxing the guard.

These are proposed bounded experiments, not completed capabilities. The current
one-GGUF plus independent native-helper use case should be validated directly
before expanding this scope.

## First actual DS4 owner slice, implemented and hardware checked

After the inventory above, native DS4.1 Core now owns an opaque ARC Metal context.
The context contains the actual device/queue/library references, bootstrap,
command schedule, model views, expert cache, expert-load/pread pool, current model
FD and mode flags. The existing leaves use field-access macros over this owner;
there is no second shadow copy of those records. Public C shells and hot native
entries select/restore the concrete engine explicitly. Selected-expert jobs and
Metal keepalive/TP threads capture their owner; pread workers receive the cache
and FD for each submitted read group rather than looking them up from another
thread. CPU page-in and Engram readers already carry their model/table inputs.

The actual-Metal lifecycle fixture passes again after this activation. It checks
that two contexts have different real cache/schedule/load records, then initializes
one DS4 owner, cleans it up and confirms the independent Flint context still
works with the same runtime queue. This does not initialize two DS4 PSO owners.

The installed DS4.1 IQ2_XXS/Q2_K checkpoint plus Clef-Flash then pass the same
**28 real HTTP/WebSocket/SDK/concurrency/cancellation checks**. Context 8192,
prefill 512, two resident sessions, mixed-prefill 512, SSD streaming, automatic
cache budget, `DS4_CLEF_FULL_HEAD_GPU=1`. Automatic admission chooses 97.14 GiB
planned for DS4, including 79.51 GiB dynamic expert cache; large RAM use is expected.
The unchanged guard stays normal with zero swap growth and child 0 on both runs.

| DS4.1 + Flash screen | Main median | Helper median |
| --- | ---: | ---: |
| New owned context, isolated requests | 785.32 ms | 68.26 ms |
| New owned context, overlapping requests | 838.03 ms | 71.28 ms |
| Frozen pre-owner server, isolated main control | 773.67 ms | — |

The same short main answer generates 5 tokens; logs show about 25–26 decode tokens/s.
The owner arm is 1.5% slower than this one subsequent control. This tiny unmirrored
screen shows no noticeable regression, but is not a precise overhead bound.
No new quality approximation or model arithmetic was introduced.

Evidence: `.workspaces/clef-owner-20261024/{lifecycle,ds41-helper,ds41-control}/`.
Both servers shut down cleanly before releasing the GPU. Remaining PSO/scratch
and C configuration/CPU-pool globals still require separate actual owner slices;
the single-GGUF guard remains in force throughout.
