# Cooperative long-helper scheduling

A long full-Clef request can hold the GPU for seconds when its whole backbone
is one command. On this fixed co-serving workload, one-layer groups reduce the
other model's completion time **4.14→0.885 seconds** (79%), at about **5%** higher
helper latency. All25 compared helper probability vectors are bit-identical:
delta-NLL, KL and maximum probability change are zero.

This remains **opt-in**, `DS4_CLEF_LAYER_QUANTUM=1`. Default0 preserves the single
prefill command. Short-helper cost and other main/helper contention patterns are
not qualified by this screen; choosing a universal automatic policy would exceed
its evidence. Full GPU head and Omni expert tiles remain ordinary fast defaults.

## Fixed baseline, one change, measurement

One frozen server binary, SHA256
`2ba3a395a86bf80e1e29dd02361ab52f163b63d0c1faa9326b2b10bb60752238`,
loads existing Flint plus pinned full Clef INT4 on M5 Max128GB. Context16384,
two generator slots, Flint prefill4096; the helper repeats one billing state250
times, producing2881 actual input tokens with one two-option question. Every
main request produces the same57-token answer, including50 reasoning tokens.

Run0/1/4/8/0, with a clean process per setting, one warmup per model and two
isolated/mixed repetitions. Builds and other model work are paused. Each trial
uses the unchanged pressure guard and exits0, with normal pressure and no swap
growth. Raw records are under `.workspaces/clef-scheduling-20261025/`.

| Layers per group | Mixed main | Mixed helper | Largest observable main gap | Isolated helper |
| ---: | ---: | ---: | ---: | ---: |
|0, opening control |4.140s |3.990s |3.921s |3.425s |
|1 |0.885s |4.197s |0.323s |3.523s |
|4 |2.174s |4.163s |0.447s |3.595s |
|8 |3.979s |4.001s |0.610s |3.590s |
|0, closing control |4.139s |3.992s |3.922s |3.451s |

These are medians of two observations, not confidence bounds or an SLA.
Observable gap includes request→first meaningful delta as well as gaps between
later deltas, using the existing WebSocket event timestamps. Solo main time
varies0.41–0.57s. Do not attribute that variation to a helper-only setting.
The one-layer helper range4.087–4.307s overlaps the four-layer range4.135–4.190s;
its tiny difference from four layers is not a precise throughput ranking.

The runtime admits waiting owners before a fresh GPU group, while continuing
to admit dependent submissions from the current owner until its group drains.
The actual CPU lease fixture protects both fairness and resident/streamed work
from deadlock. Clef yields only after complete layers; buffers/arithmetic remain
unchanged and cancellation gains another check. An engine's scratch mutex stays
held throughout, so another request cannot overwrite its incomplete state.

## Reproduction and Pareto plot

Launch the ordinary server with Flint as `--model`, full Clef as
`--decision-model gguf/clef-int4`, and settings above. Set the quantum per process
using `with-env {DS4_CLEF_LAYER_QUANTUM:'1'} { ... }`. Wait for `/health`, then:

```nu
nu tests/mixed-serving.nu --url http://127.0.0.1:19321 --main-model flint --helper-model clef --mixed-timing-only --helper-repeat 250 --repeats 2 --output .workspaces/clef-scheduling-20261025/q1/client
```

Use directoriesq0/q1/q4/q8/q0-closing for the sequential settings. Raw output is
ignored. No new timer/client framework is used; the maintained wire test reuses
the established Responses WebSocket client. One initial client ran before
readiness, made no request, and is excluded; the successful q1 client is stored
as `client-ready`. The plotter selects successful result records.

```nu
.workspaces/clef-artifacts-20261024/venv/bin/python speed-bench/experiments/clef-native-20261024/plot_scheduling.py .workspaces/clef-scheduling-20261025
```

This produces `summary.json` plus standalone `pareto.png`/`pareto.svg` in that
ignored workspace. The plotted axes are main/helper completion during overlap;
error bars show the observed ranges. The table above retains every measured
point without committing generated dumps or plot binaries.
