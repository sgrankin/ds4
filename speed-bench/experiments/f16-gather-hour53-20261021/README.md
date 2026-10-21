# Item53: keep selected compressed KV in F16 through scalar attention

2026-10-21, source base `a2c47483ffd2ffac58726da0f07da5df4cf9accd`.
One-hour budget 02:46–03:46 UTC. This follows the F16 suffix qualification in
[Item51](../f16-suffix-hour51-20261021/README.md).

## Why this experiment

Item51's F16 edits took 152.864 s versus the mean of two F32 controls,
143.463 s (+6.55%). Splitting its existing CSVs changes the interpretation:

| Reused-edit phase, four edits summed | F32 control mean | F16 |
|---|---:|---:|
| Rebuild edited prefix |131.017 s|139.845 s|
| Graft plus fresh tail |12.447 s|13.019 s|

About 94% of the 9.401 s gap occurred rebuilding the prefix. This does not
identify a kernel regression: prefix work includes expert I/O, and the mixed
F32/F16 layout arms also varied. The 6.55% is a whole-edit observation, not a
measured F16 copy or rotary-repair penalty. Item48's independent full-model
F32/F16 bracket had already left latency unresolved within its drift.

Source inspection did find a concrete avoidable round trip: scalar attention
gathered selected F16 cache rows into F32 scratch, then staged those values
back into F16. The candidate copies the same selected rows as 16-bit words
and passes their actual F16 type to existing attention. Its scratch uses the
same existing allocation; no persistent expanded mirror is added. Index/raw
caches, publication, pre-RoPE originals, suffix repair and disk bytes are
unchanged. F32 sessions retain their existing path.

Keeping the selected rows packed also enables existing gathered staging
fusion. DS4.1 never arms the backend's deferred inverse-RoPE hook, so this
change does not select the separate packed reduction/inverse-RoPE kernel.
Its existing attention reduction, BF16 boundary and inverse-RoPE steps remain.
Full-model equality is measured beyond exact row copies.

## Qualification and paired kernel timing

Reuse `tests/test_deepseek41_metal --compressed-kv-f16`: shuffled row IDs,
nonzero source/destination offsets, guard bytes, exact packed copies and F32
expansion. First shader compilation rejected mixed scalar/vector grid
attributes; the corrected kernel uses scalar attributes consistently. The
failed run is retained alongside the passing check.

`--compressed-kv-f16-bench` reuses the real scalar attention API and preallocated
buffers: 1,537 source rows, 128/512 selected rows, 64 heads, width512 and a
128-row raw ring starting at17. One warmup and six alternating pairs, 16 calls
per command batch. It times gather alone and gather plus attention, checks
gather bits exactly and reports attention equality/difference separately.
Like the current DS4.1 caller, this standalone test has no pending inverse-RoPE
fusion; the model runs qualify the actual path and fresh continuations.

## Full-model comparison

Use the existing `speed-bench/decoder_replay_bench`, native routing, replay0,
MTP off, fixed64GiB expert cache, context16,384, prompt8,192 and chunk512.
Main KV is F16 in every arm. Only `DS4_METAL_V41_PACKED_F16_GATHER=0|1`
changes. All arms use one binary, whose hash is recorded per execution.
Ordinary fixture: old/new/new/old. Favorable fixture: old/new, with no closing
control and therefore weaker timing evidence. Each arm starts a fresh engine;
OS/SSD cache state carries over. The ordinary register sits halfway through
the prompt; the favorable register sits near the latest request.

Teacher logits include EOS and use common fixed histories. Independent greedy
answers start from another prefill, so teacher answers cannot condition them.
Report teacher/greedy decode separately from prefix sync, and direct practical
time as greedy sync plus first answer. This is a direct engine measurement,
not HTTP or a persistent-agent speed result.

Chunk512 is the requested setting, not a bound on every GPU dispatch. Current
DS4.1 can choose a wide causal sweep for at least4096 remaining tokens using
its independent carry capacity. The new reader changes scalar attention,
not wide prefill; prefix timing differences in this bracket are not evidence
of a gain from the new reader. Likewise, the1537-token relocation setup and
the8192-token task prefix are different shapes, so their token rates do not
isolate pre-RoPE archive overhead. Enabling originals does not itself force
scalar prefill or add waits inside an existing command batch.

```sh
make -j6 speed-bench/decoder_replay_bench ds4-server ds4-agent
make -j2 tests/test_deepseek41_metal
python3 speed-bench/ideas/pressure_run.py .workspaces/item53-check-guard -- \
  ./tests/test_deepseek41_metal --compressed-kv-f16
python3 speed-bench/ideas/pressure_run.py .workspaces/item53-micro-guard -- \
  ./tests/test_deepseek41_metal --compressed-kv-f16-bench
python3 speed-bench/ideas/pressure_run.py .workspaces/item53-model-guard -- \
  python3 speed-bench/experiments/f16-gather-hour53-20261021/run.py \
  --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  --bench speed-bench/decoder_replay_bench --output .workspaces/item53-model
python3 speed-bench/experiments/f16-gather-hour53-20261021/analyze.py \
  .workspaces/item53-model --output .workspaces/item53-analysis
```

## Initial measurements

| Selected rows | Scope | Old ms/call | Packed ms/call | Saving |
|---|---|---:|---:|---:|
|128|Gather|0.026594|0.020396|23.31%|
|128|Gather + attention|0.103531|0.096896|6.41%|
|512|Gather|0.016000|0.013250|17.19%|
|512|Gather + attention|0.070812|0.067135|5.19%|

All gathered bytes and standalone attention outputs matched exactly. These
wall-clock batch means include encoding/submission; they are not GPU timestamp
durations or whole-model speedups. The two shapes run sequentially, so their
absolute times also include different device warmup/clock conditions.

The final clean repeat, after all builds/model jobs finished, measured:

| Selected rows | Scope | Old ms/call | Packed ms/call | Saving |
|---|---|---:|---:|---:|
|128|Gather|0.034542|0.032604|5.61%|
|128|Gather + attention|0.105396|0.088479|16.05%|
|512|Gather|0.016656|0.014260|14.38%|
|512|Gather + attention|0.069073|0.068302|1.12%|

Outputs again matched exactly. The varying size of the local win is another
reason to retain the whole-model result. A middle repeat overlapped a short
link step and is excluded from latency conclusions; its raw outputs remain.

All six full-model arms returned the expected complete code and stopped.
All captured full-vocabulary teacher logits matched bytewise: zero additional
delta-NLL, KL or argmax changes on these histories (ordinary7/favorable8
teacher positions, including EOS).

| Direct practical time | Old mean | Packed mean | Apparent saving | Control drift |
|---|---:|---:|---:|---:|
|Ordinary ABBA|23.346200 s|22.899819 s|1.91%|+7.04%|
|Favorable single pair|23.747111 s|23.635044 s|0.47%|Unavailable|

The whole-task result is within ordinary control drift. It does not establish
a serving gain. Short answer-only timings are noisy as well; keep all phase
timings rather than attributing prefix variance to the selected-row path.

## Paired continuing decode

To separate decode from prefix I/O, reuse the existing
`speed-bench/metal_decode_schedule_bench`: two sessions in one engine at the
same frontier, old/new order and session assignment alternate each step.
The only harness addition is `--ssd-streaming`, selecting fixed64GiB expert
cache, chunk512 and no full-weight warmup for this model. Each step checks
full-vocabulary equality before choosing the same non-EOS token. This is a
continuing-code throughput/identity test, not an independent task grade.
16 untimed warmup steps amortize first-use setup, then256 timed steps per
variant, including non-EOS argmax. Native routing and F16 remain fixed.

```sh
make -j2 speed-bench/metal_decode_schedule_bench
DS4_V41_COMPRESSED_KV_F16=1 DS4_V41_DECODER_REPLAY_ROWS=0 \
DS4_METAL_V41_RESIDENT_ROUTE_GAP=0 DS4_METAL_V41_RESIDENT_PREFILL_GAP=0 \
DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS=0 \
DS4_METAL_V41_SELECTED_TILE=64 DS4_METAL_V41_SELECTED_LIMIT=1024 \
python3 speed-bench/ideas/pressure_run.py .workspaces/item53-warm-guard -- \
  speed-bench/metal_decode_schedule_bench --ssd-streaming \
  --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  --prefix-tokens 8192 --ctx 16384 --warmup 16 --tokens 256 \
  --candidate-env DS4_METAL_V41_PACKED_F16_GATHER --control-env-value 0 \
  --include-selection
```

The paired run passed 273 full-vocabulary frontier comparisons (35,293,440
float pairs) and 272 paired token selections. Timed control 256 steps took
12.332474s (20.7582 tokens/s); packed 256 steps took 12.342978s
(20.7405 tokens/s), a 0.085% increase in time. This is effectively neutral,
consistent with the selected-row work being a small share of full inference.
The standalone kernel saving is not a measured model-wide improvement.

## Compatibility follow-up and decision

Reuse Item39's original-backed relocation probe with the packed reader on,
both source/destination F16, chunk512 and context8192. The existing probe
supports3/4 original-backed moves; its one-move mode deliberately supports
legacy/no-originals only. An attempted unsupported one-move/originals command
exited2 before loading the model and is retained. The supported three-move
qualification covers right66, left128 and right514, canonical restoration,
fresh teachers and recapture. Its timing is functional evidence only.

All three moves passed canonical/original restoration and retained immutable
source state/originals. Recapture retained originals after every graft.
All 24 fresh native/grafted teacher-logit rows matched the corresponding
Item51 F16 rows bytewise: no additional delta-NLL, KL or argmax changes from
the packed reader. This compares each graft to its previous graft counterpart,
not to native recomputation; the existing stale-prefix approximation remains.

Across all guards, all 669 monitored samples reported normal pressure; peak
wired plus compressed RAM was 94.919 GiB. Maximum swap growth in a single
guard was 1 MiB. No model jobs remain running. Server, agent and focused GPU
test binaries build successfully; the focused typed-copy checks pass.

The initial Item53 decision kept the new reader separately opt-in. Following
review, it is now automatic whenever `DS4_V41_COMPRESSED_KV_F16=1` selects
F16 storage; `DS4_METAL_V41_PACKED_F16_GATHER=0` restores the old reader.
This removes a redundant enable switch: measured outputs stayed exact and
overall performance was neutral. Neither this continuing-decode measurement
nor the noisy task bracket establishes a whole-model gain. Historical
execution evidence retains the settings/binaries actually measured. The
updated paired-decode command explicitly sets the control to0, since merely
unsetting the switch now selects the packed reader. A separate future candidate is fusing
publication's existing quantizer and F16 store, which currently run as two
passes; that is not implemented or measured in this item.

The default-on follow-up rebuilt the server, agent and probes and ran the
ordinary fixture with the reader switch unset. All seven full-vocabulary
teacher rows matched the previous explicit-on run bytewise, and the complete
greedy answer was correct and stopped. [Default-on evidence](evidence/default-on/summary.json).

Compact [execution evidence](evidence), [paired task analysis](analysis/summary.json),
[graft comparison](evidence/graft-comparison.json) and
[pressure summary](evidence/pressure-summary.json) accompany this report.
Raw logit buffers and executables remain under `.workspaces/f16-gather-hour53/`.
