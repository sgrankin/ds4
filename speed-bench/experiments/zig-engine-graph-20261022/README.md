# Native V4.1 graph ownership and layout

Phase 2 passed the existing graph planning/accounting fixtures in ReleaseFast
and Debug, plus six bounded model runs. All four default-mode frontier vectors
matched the frozen input-stage control exactly; the F16 control/native pair also
matched exactly. This qualifies the mechanical ownership/layout change. It does
not establish an inference speed gain. Work ran from 12:41 to approximately
12:51 UTC on 2026-10-22, within the one-hour budget.

## Scope

The active path moves V4.1 graph layout, memory estimation, allocation, workspace
growth, reset and destruction into handwritten Zig:

- [layout.zig](../../../zig/engine/deepseek41/layout.zig) uses the validated
  per-model configuration to describe scratch, carry and alias storage.
- [resources.zig](../../../zig/engine/deepseek41/resources.zig) owns tensor/view
  handles, host hash buffers and disk-read Engram descriptors. Scoped cleanup
  releases partial construction; `take` transfers the actual graph record.
- [graph.zig](../../../zig/engine/graph.zig) connects that owner to existing C
  sessions through the shared [graph record](../../../ds4_deepseek41_graph.h).

C execution-row copies borrow the record and must not destroy its resources.
Model/weight mappings outlive the graph. Aliases use retained view handles and
are freed as views. Allocation accounting remains the allocation's recorded
size even if logical capacities or environment settings later change.

Attention, routing, matrix arithmetic, executor and session coordination remain
C in this checkpoint. Concurrent `ops.zig`, scalar/batch attention and MoE drafts
were **not linked or qualified here** and are excluded from recorded source
hashes. `-Dnative=false` retains the C graph owner. This phase preserves existing
layouts and optional F16 storage rather than introducing a new memory saving.

## Ownership fixtures and build results

The [driver](graph-qualify.py) ran the existing modes serially, first C then
native. Both the [ReleaseFast campaign](evidence/releasefast/results.json) and
[Debug campaign](evidence/debug/results.json) passed all four fixture invocations
and reported identical success output. Each session-accounting invocation
created and freed three sessions, including two simultaneously live sessions,
for twelve successful session allocation/free cycles across both campaigns.

| Mode | Invariant and cost |
| --- | --- |
| `--memory-plan` | Context-dependent prefill capacities, whole-model/TP/SSD admission and disk-only Engram bounds. Maps descriptors but allocates no GPU graph or weight pages. Simulated 128/256/512 GiB host budgets are arithmetic. |
| `--session-accounting` | Allocates 16k-context sessions, frees them after changing layout controls, and checks exact live graph-byte accounting. Fixed 16 GiB expert-cache target; no inference. Recorded static graph allocations were 2,726.05 and 6,826.77 MiB under the two existing layouts, for both implementations. |
| `--prefill-alias PROMPT_FILE` | **Not run.** Optional existing stress check requiring more than 26,624 encoded tokens, context 34,816 and a fixed 64 GiB expert cache. This is not a short-prompt test and is excluded from the default driver. |

The fixture's two direct `ds41_graph_alloc` calls now pass an explicit zero
prefill chunk, as does its direct `ds41_graph_bytes` call. Four capacity
assertions use `ds41_graph_memory(ctx, 0).prefill_cap`, available through either
implementation. No fixture assertions were weakened.

The [build records](evidence/build/) preserve the first failures and fixes:

- Initial native build failed on five field-name loops needing explicit
  compile-time evaluation. The same attempt exposed duplicate forward typedef
  warnings after record extraction. Both were corrected; the final native
  production/fixture build completed in 9.097 s with empty stderr.
- The first Debug fixture link exposed thirteen unresolved GPU symbols from
  unreachable other-backend code in the engine-including C translation unit.
  The C fixture now uses the existing strict optimized flags (`-O3`,
  `-fno-fast-math`) in both modes; the selected Zig graph owner remains Debug.
  No GPU stubs were added. The final Debug build passed in 2.150 s.
- The C production/fixture build passed in 9.483 s with empty stderr.

These are development builds with different cache/target histories, not a clean
build performance comparison.

## Bounded model comparison

The [model driver](model-qualify.py) reused the committed
[hour-26 raw prompt](../zig-metal-f32-hour26-20261022/prompt.txt) and existing
`ds4-bench`: 512 prefill tokens, 128 teacher-forced decode tokens, context 1,024,
chunk 512, Metal, 48 GB SSD expert-cache target, default math and no quality
mode. The control is the frozen first-input native binary, isolating this graph
owner change from earlier native modules. All six processes completed the
expected 512/128 token counts. These are fixed teacher tokens, not independent
answer-quality tasks.

| Default bracket, in order | Whole process | Prefill tok/s | Decode tok/s | Steady decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Control | 26.054 s | 32.90 | 18.99 | 21.76 |
| Native | 25.949 s | 33.39 | 19.10 | 21.90 |
| Native | 23.373 s | 33.45 | 19.19 | 21.65 |
| Control | 26.095 s | 34.07 | 19.52 | 21.89 |

The native whole-process mean was 24.661 s versus 26.075 s control (5.42% lower),
with control whole-process drift +0.157%. However, the native process times
varied by 2.58 s, and the actual prefill/decode means were respectively 0.19%
and 0.57% lower throughput than control; steady decode differed by -0.23%.
The observation does not support an inference speed improvement. The four
saved full-vocabulary frontier vectors were identical.

The ordered F16 compatibility pair took 23.069 s control and 25.460 s native
(+10.37% whole-process cost). Prefill was 33.89/34.31 tok/s and decode
19.57/19.66 tok/s. This unbracketed pair cannot isolate a storage-mode timing
cost or demonstrate a gain. Its two full frontier vectors were identical.
Only the prefill frontier is dumped; completing 128 teacher steps is not a
claim that logits were compared at every decode step.

[Model results](evidence/model/results.json) retain exact commands, environment,
binary hashes, CSV rows and timings. Every full frontier JSON is retained as
standard gzip beside its raw logs/CSV. [Summary calculations](summary.json)
preserve the bracket drift and pair cost. The existing CSV `kvcache_bytes`
field was zero; that field is not a measurement of this graph's allocations.
Actual allocation sizes appear in stderr and are checked by the ownership
fixture.

All three [pressure guards](evidence/pressure-summary.json) completed with child
status zero: 155 normal-pressure samples in total, no swap growth, model-bracket
peak wired memory 58.289 GiB and compressor 9.825 GiB. Recorded active source
hashes stayed unchanged throughout each campaign and match the
[final capture](evidence/active-source-sha256.json).

## Reproduction

Reconstructing inference time as `512 / prefill_tps + 128 / gen_tps` from
the rounded CSV columns gives 21.944 s control versus 22.006 s native means
(about +0.28%). The two control inference points drift by 3.22%. This agrees
with treating the ownership change as timing-neutral in this small screen.

Run from the repository root with the 1017 IQ2_XXS/Q2_K model under `gguf/`.
The existing Make target is `tests/test_deepseek41_graph`; selected-object Zig
target `graph-test` produces `bin/test-deepseek41-graph`:

```sh
tools/zig-build -Dnative=false --prefix .workspaces/zig-engine-graph-c install graph-test
tools/zig-build --prefix .workspaces/zig-engine-graph-native install graph-test
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-engine-graph-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-engine-graph-20261022/graph-qualify.py \
    --c-bin-dir .workspaces/zig-engine-graph-c/bin \
    --native-bin-dir .workspaces/zig-engine-graph-native/bin
```

The fixture driver defaults to fresh `.workspaces/zig-engine-graph-rerun` output.
Use separate prefixes/output directories and `-Doptimize=Debug` to repeat the
Debug owner check. Existing output directories fail visibly.

For the exact model ablation, retain the first-input native control binary
before building this checkpoint, then run the six-arm driver under the guard:

```sh
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-engine-graph-model-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-engine-graph-20261022/model-qualify.py \
    --control-bin-dir .workspaces/zig-engine-20261022/native/bin \
    --native-bin-dir .workspaces/zig-engine-20261022/graph-native/bin
```

The model driver defaults to fresh `.workspaces/zig-engine-graph-model-rerun`.
Use `--out` for another run. It runs control/native/native/control in default
storage, then one control/native pair with `DS4_V41_COMPRESSED_KV_F16=1`.
Vectors are compared within each mode. Root ran all model/GPU jobs under the
existing pressure guard; the report author ran no builds or model jobs.
