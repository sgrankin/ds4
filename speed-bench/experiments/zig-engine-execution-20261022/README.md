# Native V4.1 execution and prefill pipeline

Item 58 starts at 13:10 UTC on 2026-10-22 with a one-hour budget ending at
14:10 UTC. The selected ReleaseFast/Debug builds, eight frontier runs, three scalar
decode pairs, actual Debug decode, short vision check and six-turn persistent-agent
workflow passed. The completed qualification ended at about 13:30 UTC. This phase
compares the frozen [native-attention checkpoint](../zig-engine-attention-20261022/README.md)
with native V4.1 execution, MoE orchestration, streaming and prefill.

## Active boundary

The pipeline passes validated model/configuration, weights and graph owners
explicitly between the existing native support/attention modules and new
execution modules. These own layer/token sequencing, shared/routed expert
scheduling, selected-expert I/O and temporary views, mapped weight spans,
Engram prefetch, prefill policy/carry buffers and small
session-batch execution. Matrix/attention shaders remain unchanged.

C keeps existing private diagnostic and feature owners through complete service
actions: selected-load worker start/finish, oracle capture/check/cache output,
resident-route adjustment, DSpark capture and imatrix observation. These are
transitional actions on actual records, not getters for copied layouts. Public
engine/session state, payload serialization and DSpark internals remain C in
this phase. Future `state_payload.zig`, `dspark.zig` and `encoder_residency.zig` drafts are excluded from
qualification/source hashes. The source list is explicit and contains the actual integrated modules; the
unused encoder-residency draft is absent from recorded run hashes.

## Reused qualification

The [model driver](model-qualify.py) reuses the same committed
[hour-26 prompt](../zig-metal-f32-hour26-20261022/prompt.txt) and existing
`ds4-bench` interface:

| Case | Order | Prefill / teacher steps | Context / chunk | Storage |
| --- | --- | --- | --- | --- |
| Ordinary | Control/native/native/control | 512 / 128 | 1,024 / 512 | Default |
| F16 compatibility | Control/native | 512 / 128 | 1,024 / 512 | Existing compressed-KV F16 |
| Long compatibility | Control/native | 8,192 / 8 | 9,216 / 512 | Default |

Every case uses Metal and a 48 GB SSD expert-cache target. Each saves a complete
prefill frontier vector and CSV metrics, requires actual prefill/decode counts
to match the requested geometry, and compares frontier vectors within the case.
The long case reaches the carry/default decoder-suffix policy with bounded
512-row scratch; it does not invoke the old 26k-token/64 GiB alias stress
fixture. The benchmark itself rejects a prompt shorter than the required
8,200 tokens. No custom token-count parser or repeated fixture adjustment is
included.

[decode-qualify.py](decode-qualify.py) reuses the attention-stage scalar screen:
three control/native pairs in default, F16-packed and F16-expanded-gather modes.
The identical [raw audit prompt](decode-prompt.txt) contained 711 tokens in the
prior checkpoint. Each run requests a 32-step greedy cap and top-20 logprob
JSON, context 2,048, chunk 512, 48 GB cache, MTP width 1 and temperature zero.
Actual prompt/step counts are recorded. Full step identity, selected IDs and
common-top20 logit/logprob drift while histories match are reported; numeric or
token differences are not automatically rejected. Finite successful execution
is required. Top-20 drift is not full-vocabulary KL or delta-NLL.

Ordinary ABBA timing is retained alongside control drift. F16 and long ordered
pairs are compatibility screens, not evidence of a speed gain. The benchmark
frontier dump does not compare every teacher step; the CLI screen adds actual
scalar output observations. The additional checks use one actual Debug scalar run, the existing six-turn
archival/repeated-suffix agent workflow and an image probe.

## Reproduction after source freeze

Retain the attention-stage native binaries before building the candidate.
Both drivers accept explicit binary directories and a fresh `--out`; they run
serially under the existing pressure guard:

```sh
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-execution-model-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-engine-execution-20261022/model-qualify.py \
    --control-bin-dir .workspaces/zig-engine-20261022/attention-native/bin \
    --native-bin-dir .workspaces/zig-engine-20261022/execution-native/bin
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-execution-decode-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-engine-execution-20261022/decode-qualify.py \
    --control-bin-dir .workspaces/zig-engine-20261022/attention-native/bin \
    --native-bin-dir .workspaces/zig-engine-20261022/execution-native/bin
```

Default output directories are `.workspaces/zig-engine-execution-model-rerun`
and `.workspaces/zig-engine-execution-decode-rerun`. Existing output directories
fail visibly. Commands, overrides, binary/source hashes, logs, CSV and vectors
are retained by the same small loops used in prior stages. Root owns builds,
GPU/model execution and the integration freeze. The report author has run no
build or model job.

## Results

The first [native build attempts](evidence/build/) exposed five shadowed
`layer` parameters and a missing translated `getpagesize` declaration, then a
compile-time row-field loop exceeding the compiler's default evaluation quota.
These were fixed without changing the numerical policy. Native-unused C-helper
warnings were also cleaned up. The third ReleaseFast install/graph build passed
in 1.635 s and the Debug install/graph build passed in 2.177 s, both with empty
stderr. These cached development builds are not clean-build performance
measurements. The [existing batch-admission fixture](evidence/batch-admission.json)
passed in both modes.

All eight [frontier benchmark arms](evidence/model/results.json) completed their
actual token counts, with full vectors identical within each case. Raw metrics
and standard-gzip vectors are retained beside the commands/logs.

| Ordinary bracket, in order | Whole process | Prefill tok/s | Decode tok/s | Steady decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Control | 26.064 s | 33.90 | 18.92 | 22.45 |
| Native | 23.905 s | 32.97 | 19.24 | 21.92 |
| Native | 23.337 s | 33.48 | 19.40 | 21.91 |
| Control | 23.112 s | 33.99 | 19.47 | 21.80 |

Native whole-process mean was 23.621 s versus 24.588 s control (3.93% lower),
but the control changed by -11.33% across the bracket. Native prefill throughput
mean was 2.12% lower, decode 0.65% higher and steady decode 0.95% lower. This
supports compatibility with essentially similar inference throughput, not a
speed breakthrough.

The F16 compatibility pair took 23.040/23.219 s control/native; prefill was
33.96/33.83 tok/s and decode 19.67/19.26 tok/s. The long pair completed actual
8,192-token prefill and eight teacher steps in 23.645/23.686 s, with
392.30/390.26 prefill tok/s. Long full frontier vectors matched exactly. These
ordered pairs are not matched storage-mode or long-context speed curves. The
existing CSV KV-byte field was zero; it is not this graph's allocation measure.

All six [ReleaseFast scalar runs](evidence/decode/results.json) completed the
711-token prompt and 32 steps. Each of the three control/native pairs had
identical complete step JSON and selected IDs, with zero reported common-top20
logit/logprob drift over 640 paired scores per mode. This is 192 observed steps,
not full-vocabulary decode KL. The [actual Debug decode](evidence/postchecks/results.json)
also completed 32 steps in 27.181 s with complete JSON equal to the default
control.

The [short vision check](evidence/postchecks/results.json) reused the previously
qualified image/REPL input and real BF16 vision model. It encoded the 112×84
image and returned `RED` for both requested answers in 20.920 s. This one
functionality check is not a broad vision-quality or latency comparison.

The [persistent-agent workflow](evidence/agent/result.json) completed six turns
in 226.348 s with three context edits, three correct archived-fact answers and
three detected fresh archive reads. All three grafts used pre-RoPE suffix reuse
from SSD; two happened after a prior approximate graft. The final graft's source
was itself approximate. This demonstrates the repeated workflow, not a claim that
all reused activations came from an immutable original or a timing ablation. The
[actual command and overrides](evidence/agent/execution.json), first outputs and
compressed traces are retained; edited contexts and cache chunks are omitted.

All four [pressure campaigns](evidence/pressure-summary.json) completed with child
status zero. Their samples remained at normal pressure; measured swap growth and
peak wired/compressed memory are retained per campaign. The
[explicit active-source hashes](evidence/active-source-sha256.json) match the
measured sources before and after both drivers and at publication. Unused draft
modules are excluded. [Summary calculations](summary.json) retain the timing drift
and scope of every successful check.
