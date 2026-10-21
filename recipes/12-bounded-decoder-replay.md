# Bounded decoder replay: an explicit approximation

**Opt-in qualified on three fixed direct fixtures and one unbatched HTTP tool
workflow. Defaults remain unchanged.** The measured server result is **7.03%
whole-workflow saving**, including nine fewer search-output tokens. The separate
exact 8,192-token direct fixture result is 38.17%, not an HTTP speed promise.
The local opt-in
`DS4_V41_DECODER_REPLAY_ROWS=128` or `256` shortens decoder work during an
already-wide DeepSeek V4.1 prefill. The default 0 retains the existing exact
shrinking dependency suffix. This is separate from suffix relocation and
changes model activations; identical output is not required for the experiment.
The measured fast combination also sets `DS4_V41_DECODER_REPLAY_SELECTED=1`;
replay rows alone did not improve phase 1 timing.

## Algorithm and invariant

Run every encoder row and retain the complete compressed/index publication.
For decoder layers, replay only the final K prompt rows using their original
absolute positions and complete carried residual/pre/post mixture state. For a
replay starting at s, local attention at row i may see only
`max(s, i - 127) ... i`. Compressed/index attention retains its ordinary causal
visibility. Missing earlier raw keys must be excluded from the attention range,
rather than supplied as zero vectors that still receive softmax probability.

```text
encode all prompt rows; publish full compressed/index state
s = prompt_end - K
for each decoder layer:
    restore the carried rows [s, prompt_end)
    compute those rows with raw attention starting at max(s, i - 127)
continue ordinary decoding from the resulting approximate live session
```

The implementation at source base `61bee1df` clips both scalar and batch raw
visibility, including raw-only batch partitions. It does not warm the exact
pre-replay raw dependency window. Full encoder publication is retained even
when encoder work is deferred across sweeps. This is a bounded adaptation of
[DeepInfer's decoder SWA replay](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md#L179-L191),
inspected at `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`; the official cluster
prefill savings are not a prediction of local Metal/Q2 savings.

## Where the local option applies

Only 0, 128, 256 are accepted. Nonzero replay activates in the existing wide
layer-major path when a sweep has at least 8,192 rows and exceeds its encoder
chunk capacity; it does not create a wide sweep from small appends. Disabling
the existing decoder-suffix path also disables this replay. The candidate is
for single-GPU text without MTP/DSpark, TP, images, imatrix or quality mode;
active unsupported configurations reject visibly.

The named `ds4-local` server presets use 512-row chunks and do not reach this
wide path. The separate 2,048-row HTTP configuration also stays below the gate.
The default unbatched server already calls full session sync, so
sufficiently large requests can activate it with no new API: omit
`--batched-session` rather than supplying an invalid 0. The measured HTTP workflow is below; setting the variable alone does not prove
activation. Require the
runtime line `decoder replay rows=... floor=... end=... approximate=1` and inspect
the actual sweep geometry.

Approximation is marked before writes and remains sticky through continuation.
Turning the variable off does not restore exactness to an existing live session.
Canonical payload/layer/snapshot exports and suffix capture are blocked,
including capture with original pre-RoPE tape. Invalidation/recomputation or a
successful canonical load clears that state. This keeps approximate replay out
of ordinary token-keyed prefix caches and relocatable suffix sources.

The persistence restriction is an intentional first-prototype guard, not a
fundamental inability to save these tensors. A separately tagged checkpoint
could restore and continue the same approximate state; restoring it would not
make its activations native or exact. This capability is not implemented.

The current SSD cache's approximate record kind specifically represents an
original-backed, relocatable suffix and requires its pre-RoPE tape. Decoder
replay is a different state kind. Durable replay would need its own tag/profile
and a loader that preserves the approximation identity instead of importing it
as an exact prefix. Matching live continuations work today; ordinary disk
persistence of replay state remains blocked.

## Item47 method

The direct-session driver fixes all three prompts at 8,192 tokens on the verified
1017 IQ2_XXS/Q2_K model, context 16,384 and requested prefill chunk 8,192. Native
expert routing is fixed; MTP and suffix reuse are off. Settings 0,128,256,0 run
sequentially with the same binary under the existing pressure guard, clearing
inherited DS4/PT variables. Each setting covers:

| Case | Fact placement | What it checks |
|---|---|---|
| Ordinary | Current register near the middle | Retrieval beyond the raw replay tail through retained compressed state |
| Favorable | Fresh current register within the final 100 tokens | A nearby fact in or near the reconstructed decoder window |
| Changed | Obsolete register near one-eighth, correction 512 tokens before the query tail | Selection of a newer fact outside the narrow raw replay window |

The code records exact fact/query token offsets. It evaluates the same fixed
teacher answer plus EOS in every arm and measures ΔNLL and `KL(native ||
candidate)` on shared teacher histories. A separate fresh prefill produces the
first independent greedy answer, capped at 48 tokens. Teacher tokens therefore
do not condition or improve the free answer. Fresh means activation recomputation;
expert and OS caches are not flushed between these operations. Preserve first answers, stop/cap
status, runtime activation, approximate flags and export-size checks.

Practical timing is direct greedy prefill plus that first answer's generation,
excluding engine open, separate teacher work and analysis. Compare candidates
with the two native brackets and report bracket drift; plot the two replay
settings against practical quality and time. A 20% teacher-perplexity increase
is a reference point, not a rejection rule. These are three synthetic fact
fixtures, not a persistent-agent, coding or tool-use benchmark.

## Profile-driven phase 2 candidate

The separate ordinary 0/128 diagnostic profile reduced decoder output rows
48,300→5,120 across two sweeps, yet both arms still issued 40 full-layer decoder
pread records. Rounded logs imply approximately 149 GiB of logical pread bytes
per arm, not physical SSD traffic; file cache may serve these requests.
Instrumented map+seed totals were 8.36/9.38 s. Profile wall time is excluded from
the timing curve and does not by itself prove a storage bottleneck.

The follow-up `DS4_V41_DECODER_REPLAY_SELECTED=1` uses the existing selected-
expert batch path for eligible Apple streaming decoder layers at 128/256 replay
rows, avoiding whole-layer read/map/seed while retaining native routing and
mixture weights. Unsupported layers fall back. Encoder state, floor masking and
approximate-state exclusions remain unchanged. This is distinct from resident
expert replacement. The same 0,128,256,0 bracket was repeated with one new
binary; the measured result is below. A separate selected-path profile confirms
20 selected decoder layers and zero whole-decoder-layer pread records over two
sweeps; this is logical read-path evidence, not a physical SSD counter.
[Bounded profile and exact source changes](../speed-bench/experiments/decoder-replay-hour47-20261020/README.md#diagnostic-profile-and-phase-2-method).

## HTTP live-state integration and measured tool workflow

The eligible unbatched HTTP 360 fixture passed all three first tool workflows,
but the initial 128 candidate was 104.99% slower because follow-up requests lost
all cached-token reuse. The suffix-only profile policy gave replay state an
empty live-context profile when suffix reuse was off, causing complete R2/R3
recomputation. Preserve this negative result when reproducing the feature.

Approximate decoder state stays excluded from canonical prefix storage, while
valid matching live continuations can retain it. The narrow fix gives replay
live state an identity scoped by model/forward policy, rows, selected setting
and optional client hint; token/parent checks remain required. The existing
suffix-enabled profile contract and exact-export/suffix-capture exclusions are
unchanged. The fresh 0/128/0 bracket completed with the fixed binary. Its new
owned path renders 8,625 rather than 8,623 input tokens; all arms within each
bracket share that bracket's exact input. Compare candidates with their own
native brackets. The fixed-policy result is summarized below.
[First HTTP result and exact fix](../speed-bench/experiments/decoder-replay-hour47-20261020/README.md#eligible-http360-result-exposed-a-live-continuation-policy-bug).

The fixed-policy HTTP bracket completed all three first search/read/final
workflows with actual file reads and correct answers. Native 52.305368/52.445182 s
(mean 52.375275 s, drift +0.2673%) versus 128 selected 48.691598 s gives **7.03%
whole-workflow saving**, or 6.91% against the faster native control. All first
requests had 8,625 tokens. Candidate R2/R3 visibly retained approximate live
prefixes with 8,673/8,896 cached tokens; native counts were 8,682/8,905. Search
output was 48 rather than 57 tokens, so the saving includes 9 shorter output
tokens and is not a pure prefill gain. No KL is assigned to diverging HTTP
histories. Direct 8,192-token and HTTP 8,625-token scopes remain separate; HTTP
includes a 433-token normally processed tail and real tool round trips.

This qualifies an opt-in for these fixtures, with no whole-agent, coding or
general task-quality claim. The existing batched 512-row preset is unchanged
and does not activate the wide path. Canonical SSD prefix storage and suffix
capture remain blocked for private replay state. Use the existing
[HTTP driver and exact first-output evidence](../speed-bench/experiments/decoder-replay-hour47-20261020/README.md#fixed-http360-result-bounded-whole-workflow-qualification)
to reproduce the measured configuration; incremental schemas were fixed on,
native routing was used and suffix reuse/application SSD KV cache were off.

## Results and replication

Phase 1 completed the 0,128,256,0 bracket. Summed direct prefill+first-answer
time was 69.314 s for the native bracket mean, 70.697 s for 128 (**2.00% slower**)
and 70.391 s for 256 (**1.55% slower**). Aggregate native drift was −0.781%.
All 12 independent first answers passed. On 22 shared teacher rows including EOS,
128 had mean ΔNLL +0.0000226651/KL 0.00000263555; 256 had ΔNLL −0.000000231732/
KL 0.000000371817. Native before/after logits were byte-identical; candidate
argmaxes did not flip. Small quality changes did not yield practical speed here.

Phase 2 fixed selected-expert dispatch and repeated the bracket with one new
binary. Native practical sums were 68.258/68.688 s (+0.631% drift), mean 68.473 s.
Replay 128 took 42.335 s: **38.17% saving** (37.98% versus the faster control).
Replay 256 took 44.970 s: **34.32% saving** (34.12% versus faster control).
All 12 first answers passed; native 22 teacher rows were byte-identical. For 128,
mean ΔNLL was +0.0000325041 nats/token, mean KL 0.00000446931, maximum
KL 0.0000446463. For 256, mean ΔNLL +0.0000128427, mean KL 0.00000116427 and
maximum KL 0.0000142981. Neither candidate changed teacher argmaxes.

This qualifies the opt-in combination on these direct fixed fixtures. 128 is
the faster point; 256 has smaller measured distribution changes. The HTTP result above is a separate whole-workflow package measurement; no
whole-agent or general task-quality claim is made. The
[experiment report](../speed-bench/experiments/decoder-replay-hour47-20261020/README.md)
retains per-case results, worst teacher changes, actual activation/export-size
checks and first outputs. This remains a direct synthetic-fact experiment,
with direct and HTTP scopes kept separate and no whole-agent speed claim.

Use the existing [C probe](../speed-bench/decoder_replay_bench.c) and
[runner](../speed-bench/experiments/decoder-replay-hour47-20261020/run.py), with a
fresh workspace and one guarded model job:

```sh
make -j2 ds4-server speed-bench/decoder_replay_bench
python3 speed-bench/ideas/pressure_run.py .workspaces/item47-rerun-guard -- \
  python3 speed-bench/experiments/decoder-replay-hour47-20261020/run.py \
  --model "$PWD/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf" \
  --output .workspaces/item47-rerun --order 0,128,256,0 --selected
```

The paired [analyzer](../speed-bench/experiments/decoder-replay-hour47-20261020/analyze.py)
reads full logits retained in the ignored experiment workspace. Run it only
after model timing has finished. Publish small metrics and first answers;
large raw logits and binaries do not belong in the recipe evidence.
