# GLM-5.3 Flash: handwritten Zig model owner

2026-10-24, branch `qwen-glm-zig`. Frozen C baseline: `030a040e`
(`before-qwen-glm-ports`). Main remains unchanged. The ordinary single-device
Metal path now has native model and session owners; the existing Metal kernels
and Objective-C resource API remain in use. The installed model is
`gguf/GLM-5.3-Flash-Q2.gguf`.

## Module map and scope

All paths below are relative to `src/engine/glm53/`.

| Component | Native owner |
| --- | --- |
| GGUF configuration, tensor binding and validation | `metadata.zig`, `weights.zig`, `model_entry.zig` |
| Immutable dimensions/policies, capacity and memory formulas | `config.zig`, `planning.zig`, `memory.zig`, `memory_entry.zig` |
| Graph validation, tensor/view allocation and workspace ownership | `allocation_plan.zig`, `allocation.zig`, `resources.zig` |
| Typed projections, HC mixing and language head | `ops.zig`, `head.zig` |
| KDA recurrence and compact MLA/pooled-indexer attention | `kda.zig`, `attention.zig`, `attention_rows.zig` |
| Routed/shared experts and selected-cache seed policy | `moe.zig`, `cache_seed.zig` |
| Scalar traversal, complete prefill chunk and layer traversal | `executor.zig`, `chunk.zig`, `prefill.zig` |
| Draft layer, recurrence/pool rollback and canonical persistence | `mtp.zig`, `speculation.zig`, `layout.zig`, `payload.zig` |
| Session sync/eval/rewind, speculative acceptance and resident shared-weight batches | `session.zig`, `session_batch.zig` |
| Vision weight binding/encoder and image-row assembly | `vision_weights.zig`, `vision.zig`, `overlay.zig` |

Actual graph and session records live once in `src/ds4_glm_graph.h` and
`src/ds4_glm_session.h`. Native modules borrow the common frontier and pending
speculation records directly. Metadata and pure memory exports are separate
objects from the GPU runtime, so model-free fixtures do not extract the hot
graph and a second C engine core.

C retains common engine/session lifecycle, platform admission, generic model
view mapping, disk page-in/cache services and the Metal resource API. GLM-5.2,
CUDA/ROCm, distributed placement/TP and explicit profiling/diagnostic routes
retain their C implementations. In particular, hidden dumps, stage/expert
profiling, streaming-async profiling and TP debug are deliberate fallbacks;
this is not a claim that every GLM branch has been removed from `ds4.c`.

The ordinary GLM53 Metal path owns the complete chunk envelope in Zig:
tokens/image overlays, embeddings, layer traversal, page-in ordering, final
cache publication and head/readback. It also owns graph geometry and allocation
validation. SSD mapping leaves contain storage mechanics, not layer algorithms.
Expanded full-KV allocation remains disabled as in the baseline. Setting
`DS4_GLM53_DISABLE_INDEXED_PREFILL=1` retains the baseline fresh-prefill rejection
rather than inventing a new cache mode during the rewrite.

## Qualification

All model jobs run serially through `speed-bench/ideas/pressure_run.py` on the
128 GiB M5 Max. The tested GLM configuration uses SSD streaming, a 24 GB expert
cache and `DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP=1`. The unbounded
static-map attempt was pressure-stopped; it is not a completed baseline.
Raw evidence stays ignored beneath `.workspaces/model-family-ports-20261023/`.

| Check | Result |
| --- | --- |
| Forced-decode frontiers at 63, 64 and 65 tokens, eight forced steps each | All 464,640 FP32 logits bit-identical to frozen C |
| Ordinary session snapshot and continued prefill | Both pass; frozen/native continued-versus-cold checks have the same top token 39058, six matching ranks in the top eight and maximum same-rank logit delta 0.298615 |
| Shared Qwen+GLM seven-case vision matrix, including repeated encodes | Joint total of 12,574,720 floats bit-identical |
| Pure model geometry, 84 combinations of context/slice/cache settings | Every memory field, fixed-state estimate and workspace estimate matches the extracted frozen C formulas |
| Model-free host suite | Passes after allocation/dead-C cleanup and shared streaming-header extraction |
| Bounded SSD MTP | Fixed inherited missing nextn mapping; 16 snapshot cycles produce nine single-token and seven double-token cycles, plus one accepted draft during context reuse; no failed-stage diagnostics |
| Canonical payload exchange at 1/3/4/5/129 tokens | Frozen → native → frozen wire bytes and fresh teacher-token logits match exactly; invalid-header preservation and truncated-read invalidation/recovery pass |
| Complete image-conditioned prompt | Image identity reuse, replay, changed-embedding rebuild and restoration pass; eight decoded tokens correctly describe Earth |
| Responses/WebSocket serving | All 25 protocol/quality checks pass, including tools, concurrent conversations and response/conversation restart recovery |
| Partial-prefill cancellation | Untouched checkpoint survives; canceled 17/19-token frontier rejects sampling/persistence; rebuilding matches all 154,880 cold logits exactly |
| Early restore failure | Candidate-only truncation after changed logits invalidates the frontier and recovers exact fresh logits; deeper GPU-state failure remains checked on all arms |

The continued-versus-cold delta above is present in both implementations; it
does not describe a frozen-versus-native difference. The small-frontier runs
report roughly 1.1–1.2 decode tokens/sec in this bounded SSD configuration.
Their native decode rows are slightly lower than the frozen rows, while the
63-token native prefill is faster. The final 2K ABBA campaign measures 0.73%
longer native decode time, with warm prefill at 354.25 versus 355.10 t/s.
See the [joint report](README.md) for the timing limits and build comparison.

MTP needs separate evidence from snapshot success. The inherited failure logs
show a missing model view near 89.25 GiB at the nextn input norm, followed by
scalar fallback. The fixture still reports success with zero two-token cycles.
The fix submits the embedding before mapping the nextn layer, then submits its
head norm before mapping the shared output matrix. Resident/static-map MTP
keeps its existing numerical path. The guarded rerun now demonstrates actual
draft execution and acceptance without mapping failures, qualifying this path.

Evidence directories: `checks/glm-layered-small-baseline/`,
`checks/glm-native-small/`, `checks/glm-layered-state-baseline/`,
`checks/glm-native-state/`, `checks/glm-layered-mtp-baseline/`,
`checks/glm-final-mtp-state/`, `checks/glm-mapped-mtp-state/` and
`vision-results/`. The CPU geometry oracle and its result are local at
`.workspaces/glm53-native-20261023/memory-check.zig` and `memory-check.log`.
`glm-payload-exchange-fixed/`, `checks/glm-vision-prompt*`, `glm-responses/`
and `checks/glm-prefill-cancel*` contain the additional evidence. Resident
multi-session batch math is ported but remains unexercised in this SSD setup.
The server's automatic SSD/suffix cache still admits DS4.1 only; GLM durable
Responses restart recovery replays stored items.

## Reproduction

Keep the frozen `030a040e` executables in
`.workspaces/model-family-ports-20261023/baseline/bin/`. Build the native
candidate with `nu tools/zig-build ds4 ds4-bench server-test`. Clear inherited
DS4/PT experiment settings, then run the same commands for frozen and native:

```nu
with-env {DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP: '1'} {
    python3 -B speed-bench/ideas/pressure_run.py .workspaces/glm-frontier-repro -- zig-out/bin/ds4-bench --model gguf/GLM-5.3-Flash-Q2.gguf --metal --prompt-file speed-bench/experiments/zig-metal-f32-hour26-20261022/prompt.txt --ctx-start 63 --ctx-max 65 --step-incr 1 --ctx-alloc 512 --gen-tokens 8 --teacher-forced-decode --show-output --dump-frontier-logits-dir .workspaces/glm-frontier-repro/logits --ssd-streaming --ssd-streaming-cache-experts '24GB'
}

with-env {
    DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP: '1'
    DS4_TEST_MODEL: 'gguf/GLM-5.3-Flash-Q2.gguf'
    DS4_TEST_SSD_STREAMING: '1'
    DS4_TEST_SSD_STREAMING_CACHE_GB: '24'
} {
    python3 -B speed-bench/ideas/pressure_run.py .workspaces/glm-state-repro -- zig-out/bin/ds4_test --session-snapshot --glm53-continued-prefill
}
```

For the MTP snapshot, add `DS4_TEST_GLM_MTP: '1'` and run
`--session-snapshot`. Inspect the MTP cycle counters and stderr as well as the
exit status. Keep only concise results and reproduction inputs under source
control; logits, snapshots and raw logs remain local.
