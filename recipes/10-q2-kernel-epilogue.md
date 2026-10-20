# Selected-cache IQ2 epilogue experiment

**Not adopted:** a register-fed fused SwiGLU epilogue produced no useful decoder
gain on the 1017 IQ2_XXS/Q2_K checkpoint. Baseline averaged **19.525 tokens/s**;
candidate **19.530 tokens/s**, +0.026%, within the four-run spread. Main runtime
and the default shader remain unchanged.

The portable idea is small: a selected-cache paired gate/up kernel already has
four reduced rows in registers. Activate those rows across four lanes instead
of rereading device outputs and activating them serially in one lane. Preserve
packed-weight interpretation, reductions, clamp/activation arithmetic, route
weights, output buffers and wrapper offsets. This candidate changed three
selected-cache decoder wrappers; it left contiguous resident and prefill
kernels alone. The complete [patch](../speed-bench/experiments/1017-upgrade-20261020/kernel/q2-register-epilogue.patch)
and frozen baseline/candidate shaders are published with the
[experiment report](../speed-bench/experiments/1017-upgrade-20261020/kernel/README.md).

One fixed-teacher ABBA used native routing, the same 2,048-token prompt, 128
continuation tokens, chunk 512 and 64 GiB requested expert cache on Apple M5 Max.
Generation throughput was **19.49, 19.70, 19.36, 19.56 tokens/s** in baseline,
candidate, candidate, baseline order. Each arm started a fresh process. This is
a negative result for that candidate and workload, not proof that every kernel
epilogue is already optimal. All sampled host pressure remained normal without
swap growth.

A separate short candidate profile logged **120 gate/up profile records
(480 stage records total)** for `iq2_slots6_pair_swiglu`, with IQ2_XXS gate and
Q2_K down projections, confirming production-path coverage. Each of gate/up,
activation/weight, down and sum contributed 120 records; these observations are
not a GPU-dispatch count. Profiling forces drains, so its timing is excluded
from the ABBA comparison.

The dumped frontier vectors were byte-identical, KL=0, but were captured
**after prefill and before decoder execution**. They do not qualify the changed
decoder's logits. Baseline and candidate both passed the existing SSD-address
mixed-batch numerical cases. Both passed the initial exact-output portion of
the SSD-experts test, then failed its retained-RSS seed/release check. The full
test therefore fails; root cause remains unresolved. A begin-only autorelease
pool trial failed the same baseline check and was reverted. No fresh stock
upstream binary was run, so this is not a demonstrated upstream leak diagnosis.

Reproduction uses the existing benchmark and guard, with one source override
per fresh process:

```sh
make -j2 ds4-bench
python3 speed-bench/ideas/pressure_run.py .workspaces/kernel-recheck-guard -- \
  python3 speed-bench/experiments/1017-upgrade-20261020/kernel/run_abba.py \
  --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  --output .workspaces/kernel-recheck
```

Use fresh output paths. The driver strips inherited experimental variables and
records commands and shader hashes. Preserve failed numerical logs beside the
timing evidence; do not promote a neutral optimization to justify its extra
kernel code. [Metrics and first outputs](../speed-bench/experiments/1017-upgrade-20261020/kernel/metrics.json)
remain available for another implementation to revisit the tradeoff.
