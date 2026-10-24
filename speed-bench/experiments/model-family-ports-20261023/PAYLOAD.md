# Cross-version model payload exchange

Both families pass frozen → native → frozen exchange. The frozen helper builds
from `030a040e`.
This adapts the existing
[DS4 canonical exchange fixture](../zig-engine-payload-20261022/README.md)
without changing that historical experiment.

Build `payload_exchange.c` against the frozen `030a040e` and final production
cores using target `model-payload-test` (artifact `model-payload-exchange`).
The helper uses only `ds4.h`; it has no private graph or session layout.
Root owns the build targets and guarded model schedule.

```sh
python3 -B speed-bench/ideas/pressure_run.py \
  .workspaces/model-family-ports-20261023/qwen-payload-guard -- \
  nu speed-bench/experiments/model-family-ports-20261023/payload_exchange.nu \
    qwen .workspaces/model-family-ports-20261023/frozen-payload/bin/model-payload-exchange \
    .workspaces/model-family-ports-20261023/candidate-build/bin/model-payload-exchange \
    .workspaces/model-family-ports-20261023/qwen-payload-exchange
```

Use `glm` and fresh output/guard directories for GLM. An optional `--model PATH`
overrides the installed family checkpoint. The helper selects one session at
context256: Qwen chunk128, disk-only n-grams and no expert SSD streaming; GLM
SSD24GB, default prefill chunk and the bounded per-layer mapping already used
for qualification. No MTP proposal is enabled in this canonical-state test.

Each family runs frozen -> candidate -> frozen at token prefixes1,3,4,5,129.
Three/four/five exercise GLM indexer pooling and short Qwen convolution history;
129 also covers the Qwen two-chunk extension. Every arm saves full canonical
bytes and full-vocabulary logits after fresh teacher token987. Imports must
resave identically and produce the same freshly computed continuation. Loaded
saved logits alone do not count as continuation evidence.

The two existing failure checks retained here are bad-header preservation and
an actual truncated cache read followed by good-load/fresh-compute recovery.
The truncated file retains GLM's row-count arrays or Qwen's MTP count so the
failure occurs after destination invalidation. GLM's candidate additionally
checks a cut immediately after changed live logits, before row-count arrays.
The inherited C bug leaves that frontier valid, so this assertion runs only
with the candidate's `--check-early-invalidation` option. All candidate checks
pass, including invalidation, rejection of sampling/persistence and exact
fresh-compute recovery. Wire bytes and continuation logits match at every
frontier in both directions.

This is wire compatibility and functional qualification; it makes no latency
or task-quality claim. Raw state
and logits remain ignored under the chosen `.workspaces/` directory.
