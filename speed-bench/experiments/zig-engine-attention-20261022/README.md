# Native V4.1 scalar and batch attention

Phase 3 ran from 12:52 to approximately 13:09 UTC on 2026-10-22, within its
one-hour budget ending at 13:52 UTC. The ReleaseFast/Debug builds, six frontier benchmark runs and six ReleaseFast
scalar-decode runs passed. The actual Debug decode and the existing repeated pre-RoPE agent workflow
also passed. This is a bounded compatibility result; no inference speed gain
was established. The control is the frozen [graph-owner checkpoint](../zig-engine-graph-20261022/README.md)
with C attention; the candidate selects native scalar and batch attention.

## Scope

The active native entry points call handwritten
[scalar attention](../../../zig/engine/deepseek41/attention.zig),
[batch attention](../../../zig/engine/deepseek41/batch_attention.zig) and shared
[support operations](../../../zig/engine/deepseek41/ops.zig). These controllers
own publication/selection order, raw and compressed cache geometry, the existing
F32/F16 storage choices, projection and quantization boundaries, attention
kernel selection, inverse RoPE and output projection. They call the existing
public GPU operations; shaders and matrix arithmetic remain unchanged.

Graph/model owners from the prior phases remain active. C retains other
executor, projection and prefill helpers and process/session coordination.
Native MoE, Engram-prefetch and prefill-policy drafts are not active in this
qualification and are excluded from source hashes. The shared graph and model
records are real C declarations consumed through generated bindings, including
`ds4_gpu_tp.h` for TP gate calls.

A read-only review of scalar attention and support operations against actual C
found no concrete Metal-path difference in publication/tape order, cache
capacities, candidate selection, replay floors, RoPE, BF16 boundaries, grouped
output offsets or TP sums. Source review is not execution qualification.

The initial candidate build exposed deprecated `std.meta.fields` use under the
installed Zig 0.17 compiler. It was changed to compile-time field-name iteration.
The first failure and native-unused C-helper warnings are retained in the
[build evidence](evidence/build/). The corrected ReleaseFast install/graph
fixture build passed in 9.219 s; Debug install/graph fixture passed in 2.662 s.
Both final stderr files were empty. These are cached development builds, not
matched clean-build performance measurements.

## Two complementary model screens

[model-qualify.py](model-qualify.py) reuses the phase-2 `ds4-bench` workload:
512 raw prompt tokens, 128 fixed teacher-forced decode tokens, context 1,024,
chunk 512 and 48 GB SSD expert-cache target. It runs control/native/native/control
in default storage, followed by one control/native pair with compressed KV F16.
Full prefill frontier vectors are compared exactly within each mode. CSV
prefill/decode timing and actual token counts are retained. It does not dump
all decode-step vectors or grade independent answers.

[decode-qualify.py](decode-qualify.py) complements that with actual scalar greedy
decode through the existing CLI `--dump-logprobs` path. It runs three ordered
control/native pairs: default storage, F16 storage, and F16 with
`DS4_METAL_V41_PACKED_F16_GATHER=0`. Each uses a fixed raw prompt, context 2,048,
chunk 512, expert-cache target 48 GB, MTP width 1, seed 73, temperature zero,
a 32-step cap and top-20 logprob dumps.

The [fixed decode prompt](decode-prompt.txt) takes the first 300 whitespace words
of the committed [hour-26 prompt](../zig-metal-f32-hour26-20261022/prompt.txt),
then asks for continued audit entries. The actual prompt contained **711 tokens**, beyond the 512-token sparse
selection frontier and with room for 32 steps. It is recorded by the CLI rather
than guessed by a custom tokenizer. The generated continuation is a cache-path
exercise, not a coding-quality evaluation.

The decoder driver retains every complete JSON step and selected token ID. It
reports full JSON identity, the first token disagreement, and maximum absolute
logit/logprob drift over common top-20 entries while the two runs share the same
history. Float or token differences are reported, rather than automatically
rejected. Execution and finite results must succeed. Early EOS is explicit in
`all_complete_32_steps`; it is not retried or padded. Top-20 drift is **not**
full-vocabulary KL or a delta-NLL measurement.

## Reproduction

Root builds selected binaries and runs one heavy job at a time under the
existing pressure guard. Retain the phase-2 native binaries as the control;
both scripts accept `--control-bin-dir`, `--native-bin-dir`, optional `--model`
and a fresh `--out` directory:

```sh
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-attention-model-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-engine-attention-20261022/model-qualify.py \
    --control-bin-dir .workspaces/zig-engine-20261022/graph-native/bin \
    --native-bin-dir .workspaces/zig-engine-20261022/attention-native/bin
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-attention-decode-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-engine-attention-20261022/decode-qualify.py \
    --control-bin-dir .workspaces/zig-engine-20261022/graph-native/bin \
    --native-bin-dir .workspaces/zig-engine-20261022/attention-native/bin
```

Default output directories are `.workspaces/zig-engine-attention-model-rerun`
and `.workspaces/zig-engine-attention-decode-rerun`. Existing outputs fail
visibly. Both scripts preserve command lines, environment overrides, binary and
active-source hashes, raw logs and results. Unqualified drafts are not hashed.

## Results

[Frontier benchmark results](evidence/model/results.json) show all four default
vectors identical, and the F16 control/native vectors identical. Every run
completed 512 prefill and 128 teacher-forced decode tokens. Full vectors are
retained as standard gzip beside the CSV/raw logs. Only the prefill frontier is
dumped by this benchmark.

| Default bracket, in order | Whole process | Prefill tok/s | Decode tok/s | Steady decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Control | 24.080 s | 31.52 | 19.65 | 22.03 |
| Native | 25.490 s | 34.25 | 19.73 | 22.04 |
| Native | 22.889 s | 34.14 | 19.67 | 21.86 |
| Control | 25.796 s | 34.28 | 19.61 | 21.89 |

Native whole-process mean was 24.190 s versus 24.938 s control (3.00% lower),
but control wall time drifted +7.13% and native wall time varied by 2.60 s.
Prefill throughput means were 34.195/32.900 tok/s (+3.94% native), while control
prefill itself rose 8.76% across the bracket. Decode means differed by +0.36%;
steady decode by -0.046%. These observations qualify compatibility, not an
inference speed improvement.

The F16 pair took 22.678 s control and 25.409 s native (+12.05% whole-process
cost). Prefill was 34.54/34.37 tok/s and decode 19.67/19.79 tok/s. One ordered
pair cannot isolate storage-mode performance. The existing CSV KV-byte field
was zero; it does not measure this graph's allocations.

[ReleaseFast scalar decode results](evidence/decode/results.json) completed all
192 capped steps: three modes × two implementations × 32 steps. All three
pairs had identical complete JSON, selected IDs and top-20 values. The measured
maximum absolute logit and logprob drift was zero over each pair's 640 common
scores at shared histories (1,920 paired scores total). These are top-20
observations at dump precision, not full-vocabulary KL or a coding evaluation.

[Summary calculations](summary.json) retain the bracket drift and pair cost.
[Actual Debug scalar decode](evidence/debug-decode/result.json) also passed
all 32 steps from the same 711-token prompt, with complete JSON equal to the
default-mode control. Its guard completed with child status zero.

The [existing agent workflow](evidence/agent/result.json) passed six turns using
context 8,192, chunk 512 and a 32 GB expert-cache target. All three archival
queries actually read the archived file afresh and correctly returned current
inventory total `416` and archived code `RIVER-73`. Active tool output and the
latest user request remained fresh. Its three observed SSD pre-RoPE grafts
reused 2,061, 1,538 and 1,113 tokens. Two occurred after the receiving state was
already approximate; the last also used an approximate source. The report does
not claim every move came from a single immutable original checkpoint. The
workflow completed in 226.354 s; this single functionality run had no timing
control.

The [agent trace](evidence/agent/agent.trace.gz), query traces, actual commands
and fixture are retained without copying KV cache chunks or whole edited
contexts. Reproduction uses the existing
[agent driver](../zig-combined-20261022/run_agent.py); its exact command and
environment are in [the guard record](evidence/agent/guard-command.json) and
[execution record](evidence/agent/execution.json).

All four [pressure guards](evidence/pressure-summary.json) completed with child
status zero: 546 normal-pressure samples, no swap growth, peak wired memory
58.933 GiB and compressor 10.008 GiB. The active source hashes stayed unchanged
throughout the measured model/decode campaigns. Afterwards, standard `zig fmt`
changed only `ops.zig` formatting; the exact [measured source](evidence/ops-before-format.zig)
and [formatting patch](evidence/format.patch) are retained, with measured and
post-format hashes [listed separately](evidence/active-source-sha256.json).
The qualified binaries were not rebuilt after formatting. Unqualified future
engine drafts remain excluded. These checks cover this model and bounded
workloads; they do not establish broad coding quality or long-context
performance.
