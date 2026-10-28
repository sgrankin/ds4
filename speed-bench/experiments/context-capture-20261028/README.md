# Checkpoint selection and capture during prefill

2026-10-28 follow-up to the [semantic checkpoint measurements](../context-boundaries-20261028/RESULTS.md).
This experiment separates two changes: selecting useful anchors and collecting
multiple endpoints within an existing layer sweep. The [selection comparison](SELECTION.md)
completes five alternating pairs with60/60 correct replies and7.8% less combined
initial-plus-two-edit time. The separate [collector comparison](COLLECTION.md)
passes60/60 replies and reduces that horizon20.6% against segmented anchors.
The [live collector fixture](QUALIFICATION.md) passes. Server `persistent-fast`
enables both improvements with explicit overrides; ordinary runtime defaults
and other presets remain unchanged.

The first comparison holds capture implementation fixed and changes
`DS4_SERVER_CONTEXT_BOUNDARY_POLICY` from `recent` to `anchors`. The latter keeps
the earliest and latest members of the already bounded recent selection. It
does not predict future edits or hardcode this workload's token positions.
The second comparison holds selection fixed and changes only the in-sweep
capture setting; both arms enable the same layer/PREAD profiling.

## Fixed workload and reproduction

`run-five.nu` reuses the original [HTTP workflow](../context-boundaries-20261028/run.nu)
and [analyzer](../context-boundaries-20261028/analyze.py). It starts one server at
a time beneath the existing pressure guard. Both arms enable semantic
checkpoints and ongoing-user suffix reuse. Default settings match the previous
campaign: ordinary/favorable cases, rows24, initial plus two edits, high
reasoning, temperature0, min_p0.05, main/index FP4, 32GB experts, HOT512,
SSD4096, context32768, configured chunk8192, two batched sessions, no MTP.
The ordinary coordinator quantum remains2048. No cache directory is shared
between processes. Each arm receives its own JSON record of environment
overrides, retained in the manifest and every process command.

Five processes per arm run in `AB, BA, AB, BA, AB` order. Initial cost, every
edit, startup, answers, capture errors, prefix/suffix geometry and cache page
counters are retained. The two cases within a process are separate contexts;
the second shares warm model weights. Raw results stay ignored under
`.workspaces/`; concise conclusions and reproduction scripts are tracked.

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/checkpoint-capture-20261028/selection-guard -- nu speed-bench/experiments/context-capture-20261028/run-five.nu .workspaces/checkpoint-capture-20261028/selection/bin/ds4-server .workspaces/checkpoint-capture-20261028/selection-five --control-env speed-bench/experiments/context-capture-20261028/selection-control.json --candidate-env speed-bench/experiments/context-capture-20261028/selection-candidate.json

.workspaces/bench-plots-venv/bin/python speed-bench/experiments/context-capture-20261028/report.py .workspaces/checkpoint-capture-20261028/selection-five --control-label Recent --candidate-label Anchors --plot .workspaces/checkpoint-capture-20261028/selection-five/pareto.svg

python3 speed-bench/ideas/pressure_run.py .workspaces/checkpoint-capture-20261028/collection-guard -- nu speed-bench/experiments/context-capture-20261028/run-five.nu .workspaces/checkpoint-capture-20261028/collection/bin/ds4-server .workspaces/checkpoint-capture-20261028/collection-five --control-env speed-bench/experiments/context-capture-20261028/collection-control.json --candidate-env speed-bench/experiments/context-capture-20261028/collection-candidate.json
```

The frozen binary prefix must contain `bin/ds4-server`, `shaders/*.metal` and
`source-revision.txt`. The manifest records those paths and revision.
`--control-binary` supports a separately frozen control when needed; using one
binary with explicit feature settings isolates the change more simply.
The existing `bench-plots-venv` supplies Matplotlib for `--plot`; ordinary
`python3 report.py OUTPUT` produces the numerical report without that option.

For an orchestration smoke, use a fresh output directory and add
`--pairs 1 --dry-run`. This writes both arms' fixtures and exact command records
without loading weights or opening a server. Smaller runs are not comparative
performance evidence. The driver clears inherited `DS4_*` settings.

## Correctness qualification for in-sweep capture

Compare completed snapshots against stop-and-save endpoints at positions that
exercise a partially filled compressor, a rolled128-row window and both packed
main/index history. Restore each snapshot, score a continuation and perform
repeated suffix relocation from its original pre-RoPE values. Keep MTP/drafter
state covered when that head is loaded, or explicitly retain the existing
fallback for that mode. A completed prompt must retain its normal final state
and logits after collecting earlier endpoints. Cancellation or a failed
collection must never publish a partially assembled endpoint.

Use the existing changed/deleted-fact and user-control workflows for behavioral
checks. Functional endpoint/restore tests are distinct from five-pair HTTP
timing and from teacher delta-NLL/KL. Per-segment staging/read measurements
should explain any remaining slowdown; cache file sizes do not measure SSD
traffic.

The in-sweep comparison can enable
`DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE=1` and
`DS4_METAL_GRAPH_PREFILL_PROFILE=1` in both arms. `report.py` then aggregates
profiled layer-0 sweeps, logical PREAD GiB, join waits and layer-stage times for
each process and each workflow's initial/edited requests. Multiple chunks
inside one layer count as one sweep. `layers_seen` retains coverage; unprofiled
scalar work is outside that count. Request groups use the existing prompt-start
and generation-finish records, including suffix-graft work between them.

PREAD bytes are logical API reads, rounded to two decimals per GiB log record,
and can hit the operating system cache. They do not measure physical SSD
traffic. Worker durations overlap GPU work; join wait can overlap the layer map
category, so those totals must not be added to reconstruct wall time. A
`capture_ms=-1` record means its capture work is included in forward time. The
report excludes that sentinel from isolated capture totals and counts such
captures separately.
