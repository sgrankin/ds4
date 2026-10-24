# Qwen 3.8 Flash Next: handwritten Zig model owner

2026-10-24, branch `qwen-glm-zig`. Frozen C baseline: `030a040e`
(`before-qwen-glm-ports`). Main remains unchanged. This ports model orchestration
and state management; the existing Metal kernels and Objective-C resource API
remain in use. The installed model is `gguf/Qwen3.8-Flash-Next-Q4.gguf`.

## Module map

All paths below are relative to `src/engine/qwen4/`.

| Component | Native owner |
| --- | --- |
| GGUF configuration, tensor binding and validation | `metadata.zig`, `weights.zig`, `model_entry.zig` |
| Dimensions, placement estimate, graph and pooled-state ownership | `config.zig`, `memory.zig`, `resources.zig`, `state_pool.zig` |
| Embeddings, disk-only ngrams, PLE and image/text positions | `embedding.zig`, `ngrams.zig`, `input.zig` |
| Weight projections, GDN/sparse gated attention, experts and forward traversal | `ops.zig`, `attention.zig`, `moe.zig`, `forward.zig` |
| MTP predictor, rollback and canonical persistence | `mtp.zig`, `snapshot.zig`, `payload.zig` |
| Per-session frontiers, exact/opportunistic acceptance and adaptive depth | `session.zig` |
| Shared-weight session batches, ragged verification and measured draft policy | `batch.zig`, `batch_session.zig` |
| Vision preprocessing/weights/orchestration and encoder primitives | `vision_*.zig`, `vision.zig` |

Actual graph/session records live once in `src/ds4_qwen4_graph.h` and
`src/ds4_qwen4_session.h`; the common frontier, callbacks and pending speculation
records are borrowed directly. C retains common engine/session lifecycle,
transport/backend checks, compatibility generation loops, small view mappings
and an independent CPU diagnostic reference. Native model algorithms call each
other directly. Twenty-one unused
migration exports and 66 obsolete C helpers were removed.

Source audit found no ordinary or tuning-knob route back to C model math.
Disabling session batching selects serialized native session eval; unfused,
legacy matrix, index-selection and expert-grouping knobs choose existing Metal
primitives inside the native graph. The C reference is reached explicitly by
the first-token diagnostic. The compatibility generator also forwards every
model step to the native graph.

Cold metadata, ngram and memory exports are separate objects from the GPU
runtime. This matters for existing fixtures that include `ds4.c`: a model-free
test must not extract a second production C core through a hot graph object.

## Qualification

All model jobs run serially through `speed-bench/ideas/pressure_run.py`.
Qwen settings: `DS4_METAL_NO_RESIDENCY=1`,
`DS4_METAL_NO_MODEL_WARMUP=1`, no SSD-expert flag. Ngrams remain disk-only.
Raw results stay ignored under `.workspaces/model-family-ports-20261023/`.

| Check | Result |
| --- | --- |
| Ordinary CLI generation and top-logprob JSON | Exact frozen/native match |
| Forced-decode frontiers at 2047, 2048 and 2049 tokens | All 744,960 FP32 logits bit-identical |
| Final five checkpoint/snapshot/rewind fixtures | Pass, including persisted chunk cancellation, restore reuse, stale replay and exact-sampling boundary rewind |
| Shared Qwen+GLM seven-case vision matrix, including repeated encodes | Joint total of 12,574,720 floats bit-identical; all 14 guarded runs completed with normal pressure and no swap growth |
| Forced depth3 counting, 48 output tokens | Complete output and trace match; 15 actual three-row verification cycles |
| Forced depth3 prose, 48 output tokens | Complete output and trace match; 23 three-row cycles, including 12 first-draft-accepted/second-draft-rejected restores |
| Model-free host suite and ngram fixture | Pass |
| Focused live ngram/batched-state fixture | Pass: recovery modes0–3 reproduce logits exactly; ordinary/MTP mixed batches, disk failure/recovery, context limits and shared-arena growth pass |
| Canonical payload exchange at 1/3/4/5/129 tokens | Frozen → native → frozen wire bytes and fresh teacher-token logits match exactly; failed-restore invalidation and recovery pass |
| Responses/WebSocket serving | 25 protocol/quality checks pass, including concurrent conversations and response/conversation recovery after restart |
| Complete CLI image conversations | Two different image turns and two text follow-ups complete in both ordinary and MTP modes |

The 2047-token prefill measured 966.79 tokens/sec frozen versus 956.25 native
(1.09% lower). Decode at that frontier measured 54.73 versus 54.57 tokens/sec;
the adjacent frontier rows are also about 54.6 tokens/sec. These are single
matched correctness runs, sufficient to screen for a large regression. They
do not establish an optimization or a statistically meaningful small slowdown.
Final ABBA screens measure 0.03% longer decode time (0.07% on the warm repeat).
The stable 2048-token live append measures 1242.27 versus 1245.23 t/s, or 0.24%
longer native time. First-prefill variability remains substantial in both
binaries; see the [joint report](README.md) for the complete timing limits.

Evidence: `baseline-results/qwen-bench/`, `checks/qwen-final-2k/`,
`checks/qwen-final-state/`, `vision-results/` and `qwen-depth3/` beneath the
ignored workspace above. Additional evidence is in `qwen-payload-exchange/`,
`qwen-responses-3/` and `qwen-vision-prompt/`. The server's automatic SSD/suffix
cache still admits DS4.1 only; Qwen durable Responses restart recovery replays
stored conversation items. The public model payload exchange is a separate
qualified persistence interface.

## Reproduction

Keep the frozen `030a040e` executable at
`.workspaces/model-family-ports-20261023/baseline/bin/ds4`. Build the native
candidate with `nu tools/zig-build ds4 ds4-bench server-test`, then run:

```nu
nu speed-bench/experiments/qwen-glm-zig-20261024/run-qwen-depth3.nu zig-out/bin/ds4
```

The runner clears inherited DS4/PT experiment knobs, runs frozen/native counting
and prose cases under the pressure guard, compares complete outputs and traces,
and requires observed three-row and partial-rejection coverage. Use `--dry-run`
to inspect exact commands. `+reject2` alone also appears when the first draft is
rejected; only `accept +reject2` proves the second snapshot restore path.
Do not add `--dump-logprobs`: that CLI diagnostic uses a scalar loop and bypasses
the speculative packet path being qualified here.

The existing state suite uses these environment variables and selectors:

```nu
with-env {
    DS4_METAL_NO_RESIDENCY: '1'
    DS4_METAL_NO_MODEL_WARMUP: '1'
    DS4_TEST_MODEL: 'gguf/Qwen3.8-Flash-Next-Q4.gguf'
    DS4_TEST_GLM_MTP: '1'
    DS4_TEST_MTP_EXACT: '1'
} {
    python3 -B speed-bench/ideas/pressure_run.py .workspaces/qwen-state-repro -- zig-out/bin/ds4_test --qwen4-prefill-checkpoints --qwen4-restore-reuse --session-snapshot --session-rewind --session-rewind-resample
}
```

Build the separate batch/failure-recovery fixture with
`nu tools/zig-build qwen-ngram-state-test`, then run
`zig-out/bin/test-qwen4-ngram-state gguf/Qwen3.8-Flash-Next-Q4.gguf` under the same
guard and Qwen environment. It covers ordinary/speculative batch interleaving,
draft rejection, snapshot reuse, disk-read failure and recovery, context limits
and shared-arena growth.
