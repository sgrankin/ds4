# Clef kernel performance pass — 2026-10-25

Long dense inputs now expand each INT4 projection into one reusable BF16 scratch
matrix, then run MPP GEMM. Short inputs retain the existing quantized kernel.
Direct BF16-input SwiGLU also removes an intermediate FP32 matrix and dispatch.
Both are ordinary defaults: automatic dispatch uses expansion from2048 input
tokens; below that it uses the old64×64 kernel. Omni is unchanged by these
dense-model changes. [Prefix caching](PREFIX-CACHE.md) is measured separately.

## Measured result

M5 Max128GB, pinned `clef-int4`/`clef-flash-int4`; one guarded model job at a time,
with no concurrent builds. Every case has four calls; discard its first and
take the remaining three's median. Times include preparation/backbone/head,
exclude model loading/HTTP. These are small warm screens, not confidence bounds.

| Model | Actual input tokens | Control | Expanded + BF16 SwiGLU | Time saved |
| --- | ---: | ---: | ---: | ---: |
|Full |2881 |3633ms |3144ms |13.5% |
|Full |6731 |10368ms |8730ms |15.8% |
|Flash |2881 |947ms |870ms |8.1% |
|Flash |6731 |2800ms |2410ms |13.9% |

Full's table uses the closing control; opening control was3617/10321ms at the
same long lengths. Flash uses its own control. The plotting recipe retains the
opening reference and closing drift explicitly instead of silently averaging
the controls. Short direct-SwiGLU calls save approximately2–3% (e.g. full266
tokens349→341ms, Flash266 tokens108→106ms). Forcing expansion at266 tokens
instead regresses about26%; the automatic threshold prevents that choice.

An additional forced-expansion full-model length screen gives1121 tokens1177→1176ms,
2111 tokens2480→2276ms (8.2% saved), and4421 tokens6080→5618ms (7.6%). This
supports keeping the old path at1121 and selecting expansion above2048 on the
measured hardware; it is not an M1/M4
qualification or a claim that every possible length has its optimal tile.

All32 compared probability vectors per full/Flash expanded screen, including
all repeated decisions, are bit-identical to control: KL, probability difference
and delta-NLL0. The extra length screen's12 vectors are also identical. The
new100-request frozen-control/candidate task-quality comparison also preserves
all100 complete responses and probability distributions exactly (KL/delta-NLL0).
Both runs completed normally under the pressure guard. The earlier broad
[Flash2000 results](BROAD_QUALITY.md) belong to the preceding implementation.

## Implementation and memory

The quantized loader normally repeats dequantization for token tiles. The new
path invokes the same MLX affine helper once per matrix, preserving its BF16
rounding, and multiplies BF16 operands with explicit FP32 MPP accumulation.
It stores one rounded BF16 projection, as before. No full FP32 projection or
resident BF16 copy of the whole model is introduced. GEMM reduction order can
differ, so measured probability parity remains a qualification result rather
than an assumption.

One lazily allocated scratch buffer is reused across projections/layers:
340MiB for full Clef,192MiB for Flash. It is owned by the engine and released
with its ordinary buffers. The checkpoint and resident weight matrices remain
INT4. Direct BF16 SwiGLU converts each loaded gate/up value to float for the
existing activation math, avoiding a full widened gate/up matrix.

The baseline profile shows most GPU time in projections. The separate local
MLX diagnostic also finds affine work dominant, but it synchronizes module
boundaries and includes Python/submission overhead; native diagnostics use
Metal command GPU timestamps. Neither has ordinary submission/fusion behavior.
They do **not** prove that the earlier15% whole-request gap was all GEMM cost.
For example, attention buckets also differ materially. Headline results above
come from unprofiled whole requests. The MLX diagnostic preserves hidden rows
exactly relative to its own ordinary pass.

## Tried, not adopted as the default

- 64×128 tile: slower at266/2881 tokens, about5% faster at6731.
- 128×64 tile: about5% faster at2881,14% slower at266, slower at6731.
- Rows-first traversal: about5% faster at2881 and13% at6731, but expanded GEMM
 wins both long cases; its intermediate-length results are mixed.
- Groups of eight token tiles: about5%/8% faster on the two long full cases,
 also behind expansion. These remain diagnostic comparisons, not active policy.

`DS4_CLEF_QMM_TILE=auto` and `DS4_CLEF_FFN_BF16=1` are now defaults. Overrides
`64x64`, `64x128`, `128x64`, `rows-first`, `grouped`, `expanded`, or
`DS4_CLEF_FFN_BF16=0` exist to reproduce controlled comparisons. Expansion and
alternative tiles are diagnostic options; users need no flags for the selected
fast path. `DS4_CLEF_PROFILE=1` is diagnostic synchronization, never a speed run.

## Frozen evidence and reproduction

The pass starts from `4bead13005cc5fa51beaaf3d1923d4c75d556dee`. The frozen
control adds disabled diagnostic/dispatch scaffolding and uses64×64/old SwiGLU:
`.workspaces/clef-performance-20261025/tiles-build/bin/clef-smoke`, SHA256
`ccf2b89d1dd33cc79a4ae626146dec5a6e8891f1eae41ba232013b4ead33f0e1`.
The successful full/Flash expansion binary is
`expanded-fixed-build/bin/clef-smoke` under that workspace, SHA256
`693194e765b5c1bd7f3eba2e5b2e8e92741c0a4678c237244d22ffcf4c9fc726`.
Guard command records retain the exact binary/model/JSONL arguments. All
successful screens finish normally with normal pressure and no swap growth.

Use the first four tracked `requests.jsonl` records plus the scheduling billing
request with250 and600 repeated state lines (2881/6731 actual tokens); repeat
each input four times. [SPEED.md](SPEED.md#reproduction) describes the same
SystemOne question/state. `prefix_requests.nu OUTPUT --model clef` also writes
the exact250-repeat chart request as its first JSONL row. Change only its state
repeat count for the600-repeat case. The length screen uses90/180/390 repeats.

```nu
nu tools/zig-build clef-smoke -j3 --prefix .workspaces/clef-performance/build
with-env {DS4_CLEF_TIMING: '1', DS4_CLEF_QMM_TILE: 'auto', DS4_CLEF_FFN_BF16: '1', DS4_CLEF_PROFILE: '0', DS4_CLEF_LAYER_QUANTUM: '0', DS4_CLEF_PREFIX_CACHE: '0'} {
    python3 speed-bench/ideas/pressure_run.py .workspaces/clef-performance/guard -- .workspaces/clef-performance/build/bin/clef-smoke gguf/clef-int4 REQUESTS.jsonl
}
```

Repeat with tile`64x64`/FFN`0` for control, and with the Flash bundle. Preserve
stdout as`answers.jsonl` and stderr as`run.log`. The existing local`run.nu`
records all actual invocations. `plot_performance.py WORKSPACE` summarizes the
named screens and emits ignored JSON/PNG/SVG, including complete probability
comparison; it uses the existing logs rather than another timing harness.
An initial MPP compiler failure was fixed before measurements; its failed log
remains excluded. One overlapping early launch was rejected by the unchanged
pressure-guard lock and made no model request. Raw evidence stays local under
`.workspaces/clef-performance-20261025/`.

The affine helper is credited MLX/Apple MIT code; MPP tensor setup follows
[Cleffa24fc91e](https://github.com/zknpr/cleffa/tree/24fc91e28d154ff8dadfa58462a30b437413c3f4),
with its MIT notice retained beside the adapted kernels.
