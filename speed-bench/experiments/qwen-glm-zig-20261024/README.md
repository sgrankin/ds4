# Native Qwen and GLM Flash ports

2026-10-24. The active single-Metal implementations for Qwen 3.8 Flash Next and
GLM-5.3 Flash now use handwritten Zig model owners. They reuse the downloaded
GGUFs, vision sidecars, upstream Metal kernels and existing resource API.
Baseline: `030a040e` (`before-qwen-glm-ports`). Work is on `qwen-glm-zig`;
main is unchanged.

The [Qwen report](qwen.md) and [GLM report](glm.md) map the native components,
describe remaining C boundaries and provide reproduction commands. Both own
configuration/bindings, graph resources, attention/experts, prefill/decode,
recurrence, MTP, batching, persistence and vision. Shared actual records are
defined once. Common engine/session lifecycle and platform services remain C;
GLM52, distributed backends and explicit legacy diagnostic paths retain C.

## Qualification

- Both families' three full-vocabulary frontiers at 2047/2048/2049 match frozen
  C bit for bit: 1,209,600 floats per paired matrix.
- Seven encoder cases match all 12,574,720 floats, including repeat encodes.
  Complete Qwen image/follow-up conversations pass in ordinary and MTP modes;
  GLM image identity, replay, replacement and restoration pass.
- Each family passes the 25-check real Responses/WebSocket suite, including
  tool calls, concurrent conversations and response/conversation restart recovery.
- Frozen → native → frozen canonical payload exchange passes at positions
  1/3/4/5/129, with exact bytes and freshly computed continuation logits.
  [Payload commands and failure checks](../model-family-ports-20261023/PAYLOAD.md).
- Checkpoint/rewind/MTP, Qwen mixed batches and failed disk-read recovery,
  GLM continued prefill, both GPU kernel fixtures and the full host suite pass.

All model work ran serially under the unchanged pressure guard on the 128 GiB
M5 Max. GLM used a 24 GB expert cache and the existing bounded per-layer mapping.
Completed final campaigns had normal pressure and no swap growth. The initial
GLM global-map pressure stop is retained as a failed baseline configuration.

## Runtime and build measurements

ABBA order uses frozen C, native Zig, native Zig, frozen C with source/build work
idle. Both sides use the same model, prompt, context, chunk and cache settings.

| Case | Frozen C | Native Zig | Interpretation |
| --- | --- | --- | --- |
| Qwen 2048-token append inside one running session | 1245.23 t/s | 1242.27 t/s | 0.24% longer append time |
| Qwen decode, ordinary 2K frontier campaign | 0.44120 s per 24 tokens | 0.44134 s | 0.03% longer; warm repeat 0.07% |
| GLM decode, 2K frontier campaign | 25.89995 s per 24 tokens | 26.08798 s | 0.73% longer |
| GLM warm 2047-token prefill | 355.10 t/s, final control | 354.25 t/s, two candidates | Effectively level; first control was colder |
| Five executable builds, empty project caches | 15.50 s | 14.88 s | No noticeable build regression |

Qwen's first prefill varied from 222–948 t/s across the first two campaigns,
including after a process warm-up. Both binaries varied substantially, so
their first-prefill averages cannot isolate a port effect. The separate
2047→4095 live-session case is stable across all four arms (1241–1247 t/s).
Every compared frontier retains identical FP32 bits. These small campaigns
screen for a noticeable regression; they do not establish a speedup.

The build comparison used fresh `--cache-dir` directories and the existing
toolchain/global caches, building `ds4 ds4-server ds4-agent ds4-bench ds4-eval`
with the normal 12-worker ReleaseFast wrapper. Both builds succeeded.

Raw commands, pressure logs, JSON logits, canonical payloads and timing CSVs
remain ignored under `.workspaces/model-family-ports-20261023/`. Useful paths:
`checks/{qwen,glm}-final-abba-*`, `checks/qwen-warm-final-abba-*`,
`checks/qwen-growth-final-abba-*`, `{qwen,glm}-responses*`,
`{qwen,glm}-payload-exchange*`, `vision-results/`, and `build-performance/`.

The server's automatic SSD/suffix cache still admits DS4.1 only. Qwen/GLM
durable Responses restart recovery replays stored conversation items; their
public canonical model payloads are qualified separately.

## Concrete correctness fixes

Review found three inherited GLM53 issues while porting:

1. Bounded SSD mapping omitted nextn weights. Staged embedding/nextn/head maps
   now permit real MTP drafts; snapshot qualification observed seven accepted
   double-token cycles and one more during context reuse.
2. Canceled nonfinal prefill could expose stale logits as valid. Invalidating
   before mutation makes partial state reject sampling and persistence.
   The focused 17/19-token cancellation fixture rebuilds all 154,880 logits
   exactly; cancellation before work preserves the valid old checkpoint.
3. Restore could overwrite live logits before later row-count validation
   failed. It now invalidates at that first write. A candidate-only cut directly
   after changed logits proves invalidation and exact recovery; the existing
   deeper state-read failure check remains on frozen and native arms.

No wire-format version or intermediate output-head computation was added.
