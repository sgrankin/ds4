# Retire original Flash and GLM 5.2

Baseline: qualified native runtime `06128389a665`. This report accompanies the
cleanup commit, after the separately recorded model-constellation/voice design
discussion. The user authorized advancing and pushing main after qualification.
Raw output stays ignored under `.workspaces/model-retirement-20261024/`.

## Scope

- Remove original DeepSeek V4 Flash admission, profile and builtin hotlist,
  Flash-only scheduling/fusion controllers, download targets, API aliases and
  obsolete launchers/model fixtures. This does not remove V4.1 Flash or its
  shared vision/Metal primitives.
- Remove GLM 5.2 admission, download targets, API aliases and builtin hotlist.
  Preserve the shared 79-layer full GLM 5.3 DSA implementation in C, now named
  explicitly and requiring `general.version="5.3"`. Retain its bounded demand-fill
  policy rather than automatically preloading retired-model expert lists.
- Keep V4 Pro's existing C profile, hotlist and previous fallback math. Preserve
  all native DS4.1, Qwen3.8, GLM 5.3 Flash and Flint execution paths and defaults.
- Update active docs/fixtures; retain clearly labelled historical measurements.
  Retire the unbuilt original-Flash CUDA placement-runtime and indexer fixtures;
  keep shared arithmetic fixtures. The CPU placement fixture now tests Pro's
  61 layers and runs in the normal host suite.

Full GLM 5.3's identity/layout was checked against the maintained converter and
the first 4 MiB of the catalog checkpoint:
`antirez/glm-5.3-gguf/GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf`.
Its header identifies `glm-dsa`, string version `5.3` and source revision
`e0b07fd2751b42d5efa199cc02c2b271deadc516`. The response was HTTP 206,
`bytes 0-4194303/211075860864`. No full checkpoint download or live inference
was performed for full GLM 5.3 or Pro; retained support is not a new hardware
qualification claim.

## Reproduce

From the repository root, using the existing downloaded model paths in the
qualification driver and a Python environment containing `websockets`:

```nu
nu tools/zig-build ds4 ds4-agent ds4-server ds4-bench ds4-eval server-test session-state-gpu-test qwen-ngram-state-test payload-test suffix-test dspark-native-oracle
nu tests/verify.nu
nu speed-bench/experiments/native-runtime-20261024/qualify.nu zig-out/bin .workspaces/model-retirement-20261024/qualification
mkdir .workspaces/model-retirement-20261024/dspark-oracle
python3 -B speed-bench/ideas/pressure_run.py .workspaces/model-retirement-20261024/dspark-guard -- zig-out/bin/dspark-native-oracle gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf gguf/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf .workspaces/model-retirement-20261024/dspark-oracle
```

Run one heavy model job at a time, with competing model servers stopped. The
assembled driver uses the existing pressure guard for every child. The host
suite's web-recovery fixture needs permission to bind localhost. Its first
sandboxed run reached that fixture and was denied binding; rerunning with
localhost access passed. No runtime or test threshold was relaxed.

## Results

- All five production executables and the named fixtures build without warnings.
- Full host/script checks pass: retired identity rejection and retained identity
  admission, 132 placement checks, session/SSD/snapshot and routing invariants,
  server/agent/web recovery, 21 download cases and 11 launcher cases.
- GPU state/rollback and Metal kernels pass. Qwen ordinary/MTP batch,
  n-gram failure/recovery and shared-arena checks pass. GLM MTP snapshot,
  context reuse and canceled-frontier recovery pass.
- Four pre-RoPE F16 suffix moves (right 66, left 128, right 514 and zero) pass
  persistence, recapture and fresh continuations, preserving the source state
  and original pre-RoPE values. These verify lifetime/format invariants; suffix
  grafts retain their previously measured approximation.
- All 103 live Responses/WebSocket protocol checks pass, covering streaming,
  tools, concurrent conversations, ephemeral/durable state, disconnect recovery
  and response/conversation recovery after server restart.
- The real 1017 DSpark checkpoint passes matching-family admission and the
  intentional Pro/V4.1 cross-family rejection. All 49 emitted tensors are
  finite and repeated draft forward logits match exactly. This focused oracle
  checks the changed admission boundary; it is not a new sampled-cycle or
  latency campaign.
- All ten guarded children exit zero. The final `make all` is warning-free;
  SHA-256 checks confirm the installed outputs are unchanged from qualification,
  and all five root executables match them. Test servers/workers are stopped.

| Model | Protocol checks | Answer probes |
| --- | ---: | ---: |
| DS4.1 Flash 1017 IQ2/Q2 | 25/25 | 11/14 |
| Qwen3.8 Flash Next Q4 | 25/25 | 14/14 |
| GLM 5.3 Flash Q2, SSD24GB | 25/25 | 14/14 |
| Flint affine INT4 | 28/28 | 16/16 |

DS4.1's complete check and answer records match the previous frozen C control:
the same three disconnect probes produce the wrong `{"surived":false}` answer.
These are inherited quality failures, kept distinct from protocol checks.
The local comparison is `deepseek-comparison.json` in the artifact directory.

Publication advances `main` from `635d5d534e77` through the previously qualified
native runtime `06128389a665`, the separate design journal update, and this
cleanup. Existing slice history remains on its archive bookmark; no raw
benchmark dumps or generated binaries are added to source control.

This is removal/admission cleanup, not a new performance experiment. Native
model arithmetic, precision, cache format and approximation settings are
unchanged. Prior latency evidence remains in the
[native runtime report](../native-runtime-20261024/README.md).
