# Selected-cache IQ2 register epilogue: no useful gain

2026-10-20, Apple M5 Max / 128 GiB, checksum-verified 1017 IQ2_XXS/Q2_K
weights. Keep the baseline. The candidate averaged **19.530 tokens/s** versus
**19.525 tokens/s** for baseline, a **+0.026%** difference within the observed
run spread. No runtime or default shader change was retained.

The candidate keeps the IQ2 packed loads, dot products, route coefficients,
clamping and gate/up outputs. It changes the selected-cache fused SwiGLU
epilogue: reduce four rows into registers, then let four lanes activate one row
each, replacing a device-output reread and lane-zero serial activation. It
affects slots6, address-table and masked-address decoder wrappers. Prefill and
the contiguous resident kernels are unchanged. See the
[source patch](q2-register-epilogue.patch) and the two frozen shader sources.

| Arm | Shader | Generation tokens/s | Steady tokens/s | Prefill tokens/s |
|---|---|---:|---:|---:|
| A1 | Baseline | 19.49 | 19.86 | 29.41 |
| B1 | Candidate | 19.70 | 20.02 | 29.65 |
| B2 | Candidate | 19.36 | 19.75 | 29.52 |
| A2 | Baseline | 19.56 | 19.85 | 29.61 |

Each fresh process used the same 2,048-token prompt frontier, 128 fixed teacher
continuation tokens, context allocation 4,096, chunk 512, native routing and
64 GiB requested SSD expert cache. Means use the rounded CSV throughput values.
Four arms are a bounded screening result, not a statistical equivalence claim.
The ABBA completed before the later Homebrew activity; the interrupted scheduler
run is a separate experiment. All pressure samples stayed at normal level 1,
with no swap growth. Commands, shader hashes, logs and CSVs are in [abba/](abba/);
[metrics.json](metrics.json) summarizes the result.

The four dumped full-vocabulary logit vectors were byte-identical, KL=0. These
are **post-prefill frontier logits, before the changed decoder runs**. They do
not establish decoder output equivalence or decoder ΔNLL/KL. The original four
approximately 1.5 MiB JSON vectors remain in
`.workspaces/upgrade-20261020/q2-epilogue/abba/`; their common SHA256 is recorded
in metrics. They are omitted from this compact publication.

A separate candidate coverage run completed successfully and logged **120
gate/up profile records (480 stage records total)** for
`gate=iq2_xxs down=q2_k path=iq2_slots6_pair_swiglu`. The four stages each
contributed 120 records: gate/up, activation/weight, down and sum. These records
confirm that the modified production decoder path executes on these weights;
they are not a count of GPU dispatches. Its
profiling flag forces GPU drains: [coverage/](coverage/) is dispatch evidence,
not a latency comparison or decoder quality measurement.

Numerical checks have a concrete limitation:

- Baseline and candidate `test_metal_moe_prefill --ssd-address` pass all five
  mixed IQ2/Q2 batch-versus-split cases, rows 169, 170, 171, 257 and 512.
- Both `test_metal_ssd_experts` runs pass the initial exact batch-output portion,
  then fail the seed/release retained-memory check. Footprint rises from about
  252 to 392 MiB across six iterations. **The full test fails on both shaders.**
  The failure is reproduced against our unchanged shader baseline; its root
  cause is unresolved. No fresh stock-upstream binary was tested.
- A begin-only autorelease-pool trial failed the same baseline check and was
  reverted before ABBA. The trial wrapped only
  `g_batch_cb = ds4_gpu_new_command_buffer();` in `@autoreleasepool { ... }` inside
  `ds4_gpu_begin_commands`. This description comes from the experiment operator;
  no separate patch was saved. The two build logs and first failure logs are
  retained, including [numerical-after-pool/](numerical-after-pool/).

To reproduce, build the existing targets, then run the frozen driver once under
the existing pressure guard from the repository root. Choose fresh output paths
to preserve prior attempts:

```sh
make -j2 ds4-bench tests/test_metal_ssd_experts tests/test_metal_moe_prefill
python3 speed-bench/ideas/pressure_run.py .workspaces/kernel-recheck-guard -- \
  python3 speed-bench/experiments/1017-upgrade-20261020/kernel/run_abba.py \
  --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  --output .workspaces/kernel-recheck
```

The published driver differs from the original only in resolving repository and
shader paths, accepting model/output arguments and explicitly setting the child
working directory. It clears inherited DS4/PT settings and applies one shader
override per fresh process. Historical execution records retain their original
workspace paths. For either numerical test, use the recorded command with
`DS4_METAL_MOE_SOURCE` pointing to the chosen frozen shader; run sequentially
under the same guard. Do not interpret the retained numerical failures as a
successful full test suite.
