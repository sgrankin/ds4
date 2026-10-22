# Native V4.1 DSpark ownership

Item 61 began at 14:26 UTC on 2026-10-22, with a one-hour budget ending at
15:26 UTC. The native owner and its ReleaseFast/Debug builds are integrated;
all five guarded target arms passed by 14:33 UTC. Matching trained tensor and
runtime qualifications subsequently passed as well. This item moves DSpark allocation,
accepted-feature capture, three-stage draft math, bounded target verification
and prefix commit into handwritten Zig. The actual weight/state records have
one shared definition. Ten thin graph facades select the owner; Session policy,
sampling-cycle orchestration and TP receive coordination remain C.

The target-only fixture below cannot establish learned draft outputs,
acceptance rates or speculative throughput. Matching trained weights are now
available for the separate campaign described after the target results.

## Bounded target fixture

The [production-including fixture](verify_commit.c) exercises actual target
verification/commit against the real V4.1 model. Like the existing DSpark
fixture, it includes the actual engine without copying a controller or
inventing a Session layout. Its preserved control is revision `5aa4e391`, native
through the suffix stage with C DSpark ownership. The candidate changes that
owner while retaining the same target execution and payload implementation.

One real engine uses two context-256 target sessions, chunk 256 and a 32 GB SSD
expert-cache target. The first graph receives a real allocated DSpark state
before prefix synchronization. Its borrowed weight record has only valid
three-stage/five-draft-row/Markov-rank metadata and target layers 37/38/39; trained
tensor pointers remain NULL. Allocation, actual target-feature capture,
verification and commit use this state. Seed, draft forward, stage and Markov
inference are never called. The independent second session runs normal target
steps with no DSpark attachment.

At each of prefixes 127 and 129, the fixture independently rebuilds captured
features before verifying six forced target tokens and committing 0, 3 or 6.
It does not restore canonical snapshots to rebuild captures: those snapshots do
not serialize DSpark feature rings. It checks accepted token IDs, live position,
checkpoint validity, Engram history, all three captured-position rings and
canonical frontier size. It records full canonical/cache byte equality and
committed logit differences against the ordinary target session, then evaluates
four fresh common target tokens and records full-logit differences again.

Scalar-versus-batch floating differences are reported as exact equality and
maximum absolute drift, with finite outputs mandatory. No numerical threshold
is used to reject an unmeasured C baseline. C/native fixture outputs can be
compared exactly to distinguish an existing batching difference from a migration
regression. `results.jsonl` is small and self-describing. `logits.bin` has six
little-endian u32 words (`DSV1`, vocabulary size, two prefixes, three keep values,
five observations, two roles), then pairs of F32 vectors in JSONL order:
verified/captured target first and ordinary target second. Step zero is the
committed frontier; steps one through four are freshly computed continuations.
Raw vectors stay in the ignored workspace.

Root owns build integration and pressure-guarded model execution. The
binary is `dspark-target-verify` with `--model MODEL --out EXISTING_DIR` and
optional `--f16 0|1`. This fixture qualifies target
rollback/capture/continuation only; without a trained DSpark GGUF it does not
qualify learned draft outputs, acceptance rates or speculative throughput.

## Serial qualification

[qualify.py](qualify.py) runs F32 control/native and F16 control/native, optionally
followed by one Debug F16 arm. It records commands, binary/helper hashes, elapsed
time, all small JSONL records and the raw-vector hash. It checks successful
execution and expected observation counts, then compares complete vector files
and non-timing records against each storage mode's control. Existing
batch-versus-scalar equality/drift remains a separate observation and is not an
automatic rejection criterion. There is no retry or numerical normalization.

The source-including fixture needs the existing `graph_test_flags` build pattern:
optimized finite-aware C, selected native-engine flags and actual core support.
It requires neither `DS4_TEST_HOOKS` nor a `DS4_NO_GPU` build. Preserve a control
binary before native DSpark activation, then run externally guarded:

```sh
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-dspark-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-engine-dspark-20261022/qualify.py \
    --control-bin-dir .workspaces/zig-engine-20261022/suffix-native/bin \
    --native-bin-dir .workspaces/zig-engine-20261022/dspark-native/bin \
    --debug-bin-dir .workspaces/zig-engine-20261022/dspark-debug/bin
```

The default output is fresh `.workspaces/zig-engine-dspark-rerun`. This ordered
compatibility campaign has no performance-bracket or latency-gain claim.

## Build and qualification status

The frozen C-DSpark control fixture built in 9.836 s. The first candidate
ReleaseFast build passed in 10.325 s with two unused C-reference helper warnings;
the initial Debug build passed in 2.572 s. Narrow unused annotations retained
the old helpers. The final ReleaseFast/Debug rebuilds passed in 8.564/0.620 s,
with empty stderr. These are cached development builds, not clean-build
performance comparisons. [First and final logs](evidence/build/dspark-native-build.json)
are retained. No model or GPU job is launched by the report author.

## Target results

All [five arms](evidence/model/results.json) passed: the two ReleaseFast
control/native storage pairs and native Debug F16. Every complete raw-vector
file has the same SHA-256, and all non-timing records match exactly. Each arm
contains six verified/committed frontiers and thirty paired full-vocabulary
observations, including twenty-four freshly computed continuation rows.

| Storage / implementation | Whole fixture time | Fresh continuation drift | C/native vectors and records |
| --- | ---: | ---: | --- |
| F32 control | 46.894 s | 0 | Identical |
| F32 native | 43.820 s | 0 | Identical |
| F16 control | 42.820 s | 0 | Identical |
| F16 native | 43.924 s | 0 | Identical |
| F16 native Debug | 47.509 s | 0 | Identical to F16 control |

Within each arm, four of the six committed frontiers differ from ordinary
scalar execution by at most `0.00000476837158` in any logit. This existing
batch/scalar difference is identical in C and native. All twenty-four fresh
continuation observations are byte-identical to scalar execution. Cache regions
match at all six frontiers; two complete canonical payloads match, while the
other four contain those different frontier logits. Accepted token IDs, graph
position/checkpoint validity, all three feature-position rings and Engram
history pass at every case. Every recorded logit is finite.

The [guard](evidence/model/guard-result.json) completed with child status zero:
[218 samples](evidence/pressure-summary.json), normal pressure throughout,
38.729 GiB peak wired memory, 9.053 GiB peak compressor and zero measured swap
growth. All [43 selected source hashes](evidence/active-source-sha256.json)
still match the frozen measured source. Raw vectors remain in the ignored
workspace; their complete hashes, compact records, commands and first logs are
retained. [Summary calculations](summary.json) retain the per-arm observations.

These ordered runs establish target-state compatibility, not a speed
improvement. The capture-only target campaign does not substitute for trained
seed/stage/forward/Markov inference or full sampled-cycle qualification.

## Matching trained draft campaign

The [pinned publisher checkpoint](https://huggingface.co/kyuz0/deepseek-v4.1-flash-gguf/blob/20408f699619df7071aa7cda6034e9cef5bf804b/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf)
was downloaded and verified: 7,966,294,016 bytes, SHA-256
`d3efe0b1f6b6fea54f1c58c5cbeb3fded5967b15b64476aed00c95806ad3e3e9`.
Its GGUF architecture is `deepseek41-dspark` with the matching 1017 source
revision, target layers 37/38/39, five draft rows and Markov rank 256. The 81
tensors contain 25 Q8_0, 39 F32, 8 F16 and 9 MXFP4 tensors. In particular, this
publisher uses Q8_0 attention/shared/head weights; it is not an earlier
converter's lossless dense-weight encoding. [Download identity and logs](evidence/draft-download/result.json)
are retained; model bytes stay outside the report.

[qualify_trained.py](qualify_trained.py) reuses the unchanged
`tests/test_deepseek41_dspark.c`. Its tensor-oracle mode executes learned seed,
forward, Markov/confidence and all three stages from controlled captured target
features; it emits 49 complete F32 tensors and checks repeatability. The driver
compares every tensor byte in C/native ReleaseFast and native Debug. Its
separate runtime mode exercises keep0..6 causal verification/commit/rewind,
eight fresh continuations, greedy/exact-sampled cycles, post-verification
payload reload, cancellation and context clamping in C/native ReleaseFast.

The control fixture omits only `DS4_NATIVE_ENGINE_DSPARK` from its actual C
translation unit, keeping the same current native target graph/services and
the original C DSpark body. Both selections retain C Session-cycle policy and
orchestration; the separately prepared native speculative-cycle owner is not
selected here. No archived old engine or copied controller is used.

The [first trained attempt](evidence/trained-first-admission/result.json) failed
in all five commands at the existing common checkpoint-compatibility gate,
before learned inference or any tensor-oracle output. Its message was
`DSpark support model and target checkpoint families differ`. The publisher's
header contains `general.architecture=deepseek41-dspark` and the matching source
revision, but omits `dspark.target_architecture`, which that C gate requires to
identify V4.1. This is a shared admission failure, not trained-oracle evidence
or a native-only regression. First logs and the failed guard status are retained.

The isolated common admission fix accepts that distinct V4.1 architecture when
the redundant target-family field is absent; any explicit family must still
agree. The existing trained fixture adds a cross-family rejection check and
restores the original target shape afterward. No native Drafter or numerical
algorithm changes. The earlier target-only campaign never executes this gate,
so it is not rerun for this metadata fix.

The [second attempt](evidence/trained-second-binding/result.json) passed that
gate but all five commands then failed during common weight binding, still
before learned inference or tensor output. The published Markov tensors use
the upstream `markov_head.embed.weight` and `markov_head.head.weight` names;
the exact-name binder expects `markov_head.markov_w1.weight` and
`markov_head.markov_w2.weight`, as emitted by the existing project converter.
The two failed attempts are retained separately. Neither produced learned
inference or tensor-oracle output.

The subsequent binding fix falls back to upstream `embed`/`head` spellings only
for V4.1 and only when the canonical name is absent. One selected tensor is
counted, and the existing type/shape validation is unchanged. The third campaign
uses this isolated binding change in both implementations. Neither admission
fix changes draft mathematics or a native controller.

### Trained results

The [third trained campaign](evidence/trained/result.json) passed. C,
native ReleaseFast and native Debug each produced all 49 oracle tensors,
19,161,012 bytes per arm. Every complete tensor has identical bytes across all
three implementations. This covers the captured feature-ring seed, learned
forward pass, Markov/confidence outputs, embeddings and the ten recorded
intermediate tensors from each of the three stages. Repeated forward execution
also passes the existing byte-equality assertion. Raw tensor files stay in the
ignored workspace; all sizes/hashes and first logs are retained.

Both actual C/native runtime arms passed the unchanged production-linked
fixture. For every keep value 0 through 6, the verified common prefix has
zero maximum logit difference, rollback/frontier checks pass, and eight fresh
continuation steps satisfy the existing `0.0001` maximum-difference bound.
Eight cycles in each exact-sampling mode preserve accepted-token/checkpoint
contents; post-verification payload reload, cancellation without checkpoint
mutation and clamping at context 2,048 pass. These are synthetic token sequences
and state invariants, not generated coding answers or a draft-acceptance-rate
measurement.

| Trained arm | Whole fixture time | Result |
| --- | ---: | --- |
| C tensor oracle | 0.439 s | 49 tensors, exact |
| Native tensor oracle | 0.375 s | 49 tensors, exact |
| Native Debug tensor oracle | 2.804 s | 49 tensors, exact |
| C runtime | 47.982 s | All existing invariants pass |
| Native runtime | 46.429 s | All existing invariants pass |

The trained target builds passed with empty stderr: initial ReleaseFast/Debug
10.345/0.659 s, admission-fix rebuild 15.973/0.646 s, and alias-fix rebuild
8.711/0.597 s. [All build records](evidence/build/dspark-trained-build3.json)
are retained as cached development measurements. They do not establish build
or runtime speed gains.

The [final trained guard](evidence/trained/guard-result.json) completed with
child status zero and [95 normal-pressure samples](evidence/trained/pressure-summary.json),
54.863 GiB peak wired memory, 11.160 GiB peak compressor and zero measured swap
growth. All [45 final selected source hashes](evidence/trained-source3-sha256.json)
still match the measured snapshot. The [earlier trained snapshot](evidence/trained-source-sha256.json)
and [admission-fix snapshot](evidence/trained-source2-sha256.json) remain separate
from the final alias-fix source. The target-only control/results are preserved
under their original source snapshot and were not silently replaced.

Together these checks qualify the selected native DSpark math/state owner and
its use by the existing C Session runtime. The native speculative-cycle draft
remains inactive, and physical TP is not exercised here. These ordered arms
make no acceptance-rate, generated-answer-quality or speculative-throughput
claim. The report author launched no builds, GPU or model jobs.
