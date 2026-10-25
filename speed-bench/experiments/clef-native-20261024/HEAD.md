# Clef learned head on Metal — 2026-10-25

The complete GPU joint head removes **11–14% of whole-request time** from four
short Flash decisions. Its warm head costs 4.2–4.7 ms, versus 12–20 ms on the native
FP32 CPU path. It remains slower than the local MLX head's 2–3 ms; this is a useful
measured improvement, not a claim that every native workload matches MLX.

This slice used the working tree following plan `6262ae0ee7ea`, recorded at
`34af0bd5cb6c`; main stayed `ce3afb7999b7`. The frozen qualification executable
was `.workspaces/clef-native-20261024/full-head-build/bin/clef-smoke`, SHA-256
`392cbfa0653a9414af0b14b7b23c70981275e51c58d7a62428804e74a5a1e513`.
Hardware is M5 Max 128GB, the pinned INT4 checkpoints from [README.md](README.md),
unchanged BF16 learned head weights and FP32 head activations. No production
Python/MLX dependency was added.

## Change and matched timings

The first GPU step moved hidden normalization, memory projection and six memory
K/V projections onto Metal. Mirrored CPU/GPU/GPU/CPU controls improved total
time 6–7%, with maximum probability difference 7.45e-9. The remaining learned
graph now also runs on Metal: input/span projections, two evidence layers,
question summaries, four decoder layers, option scoring and cosine/prior terms.
Only the final small softmax and caller-order mapping stay on CPU.

| Flash tokens | CPU total | Partial GPU total | Full GPU total | Full GPU head | Saving from CPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 169 | 75.95 ms | 70.70 ms | 67.22 ms | 4.28 ms | 11.5% |
| 266 | 123.73 ms | 115.99 ms | 107.53 ms | 4.66 ms | 13.1% |
| 157 | 76.16 ms | 71.59 ms | 68.05 ms | 4.16 ms | 10.7% |
| 238 | 105.27 ms | 98.24 ms | 90.11 ms | 4.53 ms | 14.4% |

Each case has one untimed and three warm repetitions. Totals include native
record preparation, backbone, lexical rows and head, but exclude model loading
and HTTP. CPU/partial controls were mirrored; full-GPU numbers are the next
isolated warm screen, independently confirmed by later short controls. Builds
and other model work were paused during latency measurement. All guards exited
normally; raw stage logs remain local under `head-abba/` and `full-head-first/` in
`.workspaces/clef-native-20261024/`.

For reproduction, use four repetitions of the first four tracked fixture requests:

```nu
mkdir .workspaces/clef-head
(open tests/fixtures/clef-record.json | first 4 | each {|case|
    1..4 | each {|_| $case.request | to json --raw }
} | flatten | str join (char nl)) | save -f .workspaces/clef-head/requests.jsonl
nu tools/zig-build clef-smoke -j2 --prefix .workspaces/clef-head/build
$env.DS4_CLEF_FULL_HEAD_GPU = '1'
$env.DS4_CLEF_TIMING = '1'
python3 speed-bench/ideas/pressure_run.py .workspaces/clef-head/guard -- .workspaces/clef-head/build/bin/clef-smoke gguf/clef-flash-int4 .workspaces/clef-head/requests.jsonl
```

Set `DS4_CLEF_FULL_HEAD_GPU=0` for the CPU control, or also set
`DS4_CLEF_HEAD_GPU=1` for the partial-GPU control. Later serving integration makes
the qualified full GPU head the default, retains these diagnostic controls,
and prepares its weights/pipelines before readiness. The first lazy call in
the original experiment paid about 170 ms for head loading/JIT; it was excluded
from warm results. CPU matrices need not be expanded when using the full GPU head.

## Quality and attribution

The four short Flash records preserve all six decisions with maximum probability
gap 1.30e-8 against the paired native CPU result. The exact existing 100-request
Decision Index 0.2.1 sample also completes 100/100 requests and preserves all 100
top choices against both saved CPU and same-weight MLX results. This means the
sampled task scores in [README.md](README.md) are unchanged, not 100% accuracy.

Against the saved MLX run, mean KL is 1.01e-4 and mean delta-NLL is +0.00153 nats
per decision, with maximum absolute probability gap 0.01895. Against the older
native CPU run, mean KL is 3.90e-5, delta-NLL +0.00137 and maximum gap 0.01210.
Those larger gaps cannot be attributed to the GPU head: rerunning CPU and GPU
heads with the **same current binary** on the four worst rows agrees within
1.79e-7, and the fresh CPU result reproduces the old-control discrepancy. Keep
that intervening source/control drift distinct from the measured head change.

Full and Omni each pass four native requests/six reference decisions with the
GPU head. Maximum probability gaps against their saved MLX oracles are 0.0001224
and 0.0009394 respectively. These are spot checks; broader full/Omni task quality
is still separate work. Raw evidence remains local in `full-head-100/`,
`full-head-debug/` and `full-head-families/`.

## Tried, not adopted: the reference GDN threshold

Cleffa uses serial GatedDeltaNet except full-model 48-head inputs of at least 4096
tokens. Our candidate tried that policy against the current chunked path at
T≥256, with mirrored controls. At 266 tokens serial saved only 1.1 ms total
(107.71→106.60 ms), preserving all 24 repeated top decisions; maximum probability
gap was 0.000203. At 2256 tokens serial increased backbone time from ≈712.2 to 714.7 ms.
Other short controls stayed stable. These small, mixed gains do not justify
another policy, so the existing T≥256 chunk dispatch remains. Evidence is in
`gdn-serial-abba/` and `gdn-long-screen/`; this was not silently accepted as a win.

The head graph follows the published Clef implementation; wide MPP operations
adapt credited [Cleffa 24fc91e](https://github.com/zknpr/cleffa/tree/24fc91e28d154ff8dadfa58462a30b437413c3f4)
leaves, with its MIT notice in `src/metal/clef/LICENSE.cleffa`. Selected-row affine
weights and the existing Metal bridge remain owned by the individual engine.
