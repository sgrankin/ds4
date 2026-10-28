# In-sweep checkpoint correctness qualification

2026-10-28, source `28c98fd7411e30eeb095d3382da59c685b53f5ef`, Apple M5 Max, 128 GiB.
The [installed-model fixture](../../../tests/test_checkpoint_capture.zig) passes
six collected endpoints, 18 teacher rows, two suffix relocations and rejection
of an unsupported approximate source.
The ordinary final forward's entire logit vector is byte-identical with and
without collection. This is a functional/numerical qualification, not a
comparative performance run.

Frozen server, fixture and 27 shaders are under
`.workspaces/checkpoint-capture-20261028/collection/`. Raw evidence is
`native.log`, `native-guard/command.json`, `native-guard/pressure.jsonl` and
`native-guard/result.json` beneath that workspace prefix. The pressure guard
completes with child exit status 0; all 122 pressure samples are normal and
existing swap stays at 23.005 GiB. Separately, `zig-out/bin/test-serving-boundary` passes
all 14 host cases at 18:19:56 UTC; that output was observed directly in the tool
result rather than saved as a separate log.

## What was checked

- One engine/session, the installed 1017 IQ2_XXS/Q2_K checkpoint, context 8192,
  prefill chunk 2048, a 32 GB expert budget, packed main/index KV and original
  pre-RoPE archive. The 3074-token prompt keeps high reasoning enabled.
- Independently obtain ordinary final logits. Build endpoint teachers by
  advancing through 127, 128, 129, 390, 985 and 2247, with two fixed prompt-token
  continuations each; restore before advancing to the next endpoint.
- Collect all six endpoints while prefilling the same complete prompt.
  Every endpoint restores the correct tokens/position, remains canonical,
  and recaptures exactly the same state bytes and original rotary bytes.
  Positions 127/128/129 cover the ring transition and odd compressor carry;
  the later positions exercise a rolled ring and larger packed histories.
- The same immutable 2247-token source supports a left relocation
  `390:2247 → 128:1985` and a right relocation `128:2247 → 390:2509`.
  Each retains three unfinished compressor pairs, evaluates two fresh tail
  tokens, permits private approximate recapture and leaves source hashes intact.
- After relocation, attempting collection from the approximate live state
  returns −1 for ordinary fallback. Position, validity, approximation flag and
  all logits remain unchanged; no output endpoint is published.

Exact restore/recapture and final-forward checks protect ownership and state
assembly. The separately stopped endpoint teachers use different batch shapes;
their logits differ, as recorded below. These checks do not establish identical
endpoint arithmetic, broad task quality, SSD eviction survival, or support for
MTP/drafter collection. The collector retains its ordinary fallback for the
unsupported drafter mode; no drafter was loaded in this run.

## Numerical results

`ΔNLL = NLL(collected) − NLL(reference)` and
`KL(reference || collected)`, in nats. Step 0 scores at the endpoint; steps 1/2
score after one/two fixed teacher tokens. Every distribution is finite.

| Position | Step | ΔNLL | KL | Maximum absolute logit gap |
|---:|---:|---:|---:|---:|
| 127 | 0 | −0.000005170 | 0.000002896 | 1.837899 |
| 127 | 1 | −0.000000299 | 0.000001041 | 2.017276 |
| 127 | 2 | +0.000003107 | 0.000000969 | 1.801393 |
| 128 | 0 | −0.000000424 | 0.000001035 | 2.025629 |
| 128 | 1 | +0.000002878 | 0.000000884 | 1.795110 |
| 128 | 2 | +0.000021596 | 0.000008912 | 3.183246 |
| 129 | 0 | +0.000002841 | 0.000000856 | 1.804171 |
| 129 | 1 | +0.000015901 | 0.000006216 | 2.659678 |
| 129 | 2 | −0.034894573 | 0.005389487 | 2.818019 |
| 390 | 0 | +0.000001171 | 0.000000584 | 3.728886 |
| 390 | 1 | −0.010501363 | 0.004636766 | 2.883141 |
| 390 | 2 | +0.000058041 | 0.000011592 | 2.539576 |
| 985 | 0 | +0.000000057 | 0.000000030 | 1.495624 |
| 985 | 1 | +0.000000180 | 0.000000023 | 1.847399 |
| 985 | 2 | +0.000007098 | 0.000001963 | 2.281085 |
| 2247 | 0 | +0.000001142 | 0.000000854 | 3.936760 |
| 2247 | 1 | −0.000001435 | 0.000000527 | 1.456219 |
| 2247 | 2 | −0.000000010 | 0.000000084 | 2.889436 |

Across these 18 rows: mean ΔNLL **−0.002516070**, range −0.034894573 to +0.000058041;
mean KL **0.000559151**, median 0.000001002, maximum **0.005389487**.
The maximum logit gap is 3.936760, at a row whose KL is 0.000000854. A maximum
coordinate difference and a probability-weighted distribution difference
measure different things; that logit gap is not a KL-sized degradation.

The separate ordinary final-forward row at position 3074 has
**ΔNLL = 0, KL = 0, maximum logit gap = 0, and byte equality = 1**. This row is
excluded from the 18-row endpoint summary above.

## Reproduction

Build from the source revision above, outside a latency campaign:

```nu
nu tools/zig-build checkpoint-capture-test serving-boundary-test
zig-out/bin/test-serving-boundary

# Start with no inherited DS4_* options, as in the recorded run.
let inherited = $env | columns | where {|name| $name | str starts-with 'DS4_' }
hide-env --ignore-errors ...$inherited
with-env {
    DS4_V41_INDEX_FP4: '1'
    DS4_V41_INDEX_SIMD: '1'
    DS4_V41_COMPRESSED_KV_FP4: '1'
    DS4_V41_COMPRESSED_KV_F16: '0'
    DS4_METAL_V41_RESIDENT_ROUTE_GAP: '0'
    DS4_METAL_V41_RESIDENT_PREFILL_GAP: '0'
    DS4_LOCK_FILE: '.workspaces/checkpoint-capture-20261028/native-repeat.lock'
} {
    python3 speed-bench/ideas/pressure_run.py .workspaces/checkpoint-capture-20261028/native-repeat-guard -- caffeinate -i zig-out/bin/test-checkpoint-capture gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --prefill-chunk 2048 --cache-gb 32 out+err> .workspaces/checkpoint-capture-20261028/native-repeat.log
}
```

Use fresh output paths. The recorded run used `native-guard/`, `native.lock`
and `native.log`; the command above uses distinct repeat paths. Shader loading
uses ordinary `zig-out/shaders` discovery; kernels are unchanged.

## Sustained serving and adverse edits

A separate HTTP run using source `28c` passes **25/25 replies** with reasoning
enabled.
Ordinary and favorable layouts each answer 9/9 across eight edits. All 16 edited
imports retain the original RAM-only source (`source_approximate=0`):
ordinary `980:2245` → `426:1691`, favorable `1040:2870` → `486:2316`.
The labels differ from the timing cohort, hence these token offsets.

Corrected and deleted facts each pass 2/2; changing the actual user instruction
passes 2/2. The corrected case performs real prefix padding (`prefix_pad=1`)
and imports 1266 tokens while answering the corrected LARCH code. An additional
request reads the original file from the actual saved initial request and
correctly recovers CEDAR, after the current view answers LARCH.

No capture fails. Hot pages reach 536,836,361 bytes, within 512 MiB. The server
stops and the pressure guard exits with status 0. This demonstrates retention
through the fixed sequence, not indefinite survival or an eight-edit comparative
speedup.
Raw requests/responses, summaries and logs remain in
`.workspaces/checkpoint-capture-20261028/retention/` and `retention-guard/`.
The temporary Nu driver reuses the exact final candidate command, with separate
cache/store directories and port 18359. Timing results remain the separate
five-pair initial-plus-two-edit campaigns.

## Fresh rebuild after approximate state

Serving QA exposed an avoidable fallback when an unrelated replacement prompt
arrived while the live graph was approximate. Final runtime source
`f74180e070516d54c6095f29ad3b9a64846e959e` validates all unsupported-return gates,
then resets that old graph before collecting a genuinely fresh transcript.
A real approximate prefix extension still returns −1 without mutation. Cold
position-zero and ordinary native-extension paths used by the frozen timing
cohort are unchanged; no performance claim is added for this new transition.

The expanded installed-model fixture repeats the original six endpoint,
18 teacher-row, final-forward, relocation and fallback checks, then rebuilds
the original prompt over approximate state. It collects fresh endpoints 128/2247,
preserves the original ordinary final logits byte-for-byte, and restores and
recaptures both new snapshots exactly. The guard exits with status 0. Evidence
is local at
`native-rewrite.log`, `native-rewrite-guard/` and frozen binaries/shaders in
`rewrite/`, beneath `.workspaces/checkpoint-capture-20261028/`.
