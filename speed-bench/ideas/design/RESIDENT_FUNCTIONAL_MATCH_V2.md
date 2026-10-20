# Resident expert matching by observed function

Design only, 2026-10-03. No implementation, fingerprint scan, model access,
build, quality result, speed result or default promotion is implied.

The present replacement planner uses router affinity and bias to identify a
resident alternative. Those scores express routing preference; they do not
measure the difference between the two expert outputs. Native coefficients
can be preserved exactly while the substituted function still differs.

The next useful hypothesis is to rank the existing eligible resident pool by
a small, input-conditioned functional sketch. This changes candidate ranking,
not the number of allowed substitutions or the mass and routing guards. The
exact split-gate overlap experiment and the current ten-task practical packet
remain separate and keep their already frozen policies.

## Fixed first ranking experiment

Start from the qualified native-coefficient implementation. Copy the existing
`protected-weights1` policy literally: one substitution, gap `.30`, mass `.12`,
rank `24`, original six coefficient bit patterns. Its `.40` total-mass setting
is ignored by the existing one-substitution implementation. Preserve the
existing missing-slot eligibility and expert exclusion rules.

Collect sixteen native scalar input probes per layer, in calibration-case
order, at the first teacher-token evaluation. Capture the actual normalized
MoE input after the existing router readback completes. Do not add an eval,
sample or KV mutation. The probes are from previously used calibration
contexts and are training data, not a new quality holdout. Record the exact
case, token position and opaque token history for each capture. A disabled
capture control must preserve full raw logits, canonical state and reads.

For every expert, evaluate its real gate/up/activation/down function on each
of those sixteen probes. Use the same quantized weights, activation and
scalar expert kernels as the runtime. Project each 5,120-element output onto
sixteen fixed seeded signed directions, with fixed normalization. Keep the
unweighted output: routing coefficients belong to the unchanged consumer.
Store finite FP16 projections and the FP32 input probes. These projections
are similarity features; they provide no formal quality or output-error bound.

For a current input, choose the probe with greatest normalized input cosine,
breaking ties by probe index. Within the existing router-eligible resident
pool, choose the expert with smallest squared projected-output distance from
the missing native expert, breaking ties by expert ID. Do not widen the pool
or tune thresholds after outcomes. A finite zero-norm input declines the
substitution. An absent, incompatible, truncated or nonfinite map refuses
the requested experimental mode before a forward pass.

This nearest-probe estimator is deliberately a small first experiment. Its
failure would distinguish poor functional features from mere score-gap
selection; it would not establish that expert functions cannot be matched.

## Size and ownership

For 40 layers and 384 experts per layer, the signature payload is
`40 * 384 * 16 * 16 * 2 = 7,864,320` bytes. The input-probe payload is
`40 * 16 * 5120 * 4 = 13,107,200` bytes. Freeze a 24 MiB application-owned
map/probe allowance, with a separate explicit bounded construction workspace.
This excludes allocator padding, engine memory and driver allocations.

The artifact identifies the model path/size/mtime, tensor layout and types,
probe histories, source/build/kernel hashes, projection seed and complete
shape/extent/checksum. Model identity and all tensor bounds are checked only
inside the existing serial pressure guard. Never hash the full GGUF as a
preparation step. Do not reuse a map for a different quantization or model.

Fingerprint construction scans the expert weights once in a frozen order.
Use bounded owned buffers and normal completion/join rules; no second model,
unbounded prefetch or shared live cache mutation is authorized by this design.
Record actual bytes read, construction time and peak owned allocations. That
one-time cost must be included when discussing amortization.

## Evidence required before drawing a conclusion

First review the actual scalar seams and freeze the source, map format,
projection generator, eligibility/ranking code and CPU adversarial fixtures.
Then separately review the focused build and grant the guarded probe/scan
jobs. No scan or runtime implementation is authorized by this document alone.

Require exact feature-off native controls. Compare affinity ranking and
functional ranking with the identical protected policy and native coefficient
bits, retaining their replacement IDs, mass, probe cosine and proxy distance.
Recompute full-vocabulary delta-NLL, KL, total variation and argmax changes
on common teacher trajectories. Use calibration only for diagnosis.

Freeze a new unused teacher set and a new practical task packet before their
first outcomes. Retain every first candidate, wrong answer, invalid output,
tool violation and cap. Generated task success is the primary practical
quality result. Follow any encouraging result with quiet paired timing;
logical read savings and an improved proxy alone are insufficient.

This is a distinct hypothesis from speculative loading of skipped native
experts. The earlier backfill experiment increased reads and left most
prefetches unused; adding backfill to this ranking experiment would obscure
which change caused an effect.
