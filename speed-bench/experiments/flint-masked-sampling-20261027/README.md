# Flint grammar masks on the native GPU sampler

2026-10-27, slice started 04:33 UTC. This is a no-model kernel/ABI qualification,
not a whole-model throughput result. Roughly one MiB of synthetic buffers is
used; no checkpoint, KV bank or inference model is loaded.

The constrained sampler has separate Metal entrypoints. Existing ordinary
sampler kernels/encoders stay unchanged. A bit mask is applied before maximum,
full-vocabulary softmax mass, top-k, top-p, temperature and min-p filtering.
Empty groups contribute zero mass without `-inf - -inf` NaNs; no finite allowed
candidate produces an error rather than a fabricated token.

`flint_session_sample_masked` samples the current frontier without advancing KV:
FP32 after prefill, original BF16 after decode. It retains the resident sampler's
counter seed and logical-position policy. One mask buffer (18,768 bytes) is
allocated lazily per constrained session and included in planned scratch memory.
Unsupported GPU filter settings return -2 for the serving CPU fallback; failure
returns -1. The shared public facade also needs CPU fallback when the optional
overthinking penalty is active, since this GPU sampler reads raw logits.

Both ReleaseFast and Debug fixtures pass:

- Exact full-mask counter parity with the original kernels for greedy, k20/
  nucleus, k128/min-p, and untruncated sampling.
- A prohibited huge logit affects neither selected candidates nor normalization.
- Sparse top-p still uses full allowed-vocabulary mass before top-k: the second
  candidate appears 19/64 times with the fixed seeds.
- A singleton last-vocabulary token with logit -1000 survives otherwise-empty
  groups and NaN logits; all-masked/all-NaN choices fail cleanly.
- BF16 k20/k128 variants, greedy and EOS/handoff invariants.

Two warm ABBA sampler screens with an all-allowed mask, T1/top-p .95/k20:

| Fixture build | Ordinary wall | Masked wall | Ordinary GPU | Masked GPU |
|---|---:|---:|---:|---:|
| ReleaseFast | 242.46 us | 273.70 us | 94.35 us | 123.96 us |
| Debug host | 234.75 us | 244.60 us | 77.73 us | 71.70 us |

Each arm makes 128 selections. The GPU-overhead sign changes between screens;
these give only the small sampler's approximate scale, not a stable performance
claim. Final latency work waits for the user's separate reference server to be
paused. The real constrained decode path still takes scalar steps; asynchronous
forward/CPU-mask overlap and full-model task throughput remain separate work.

Reproduce:

```nu
nu tools/zig-build flint-masked-sampling-test --prefix .workspaces/flint-masked/release
^.workspaces/flint-masked/release/bin/test-flint-masked-sampling
nu tools/zig-build flint-masked-sampling-test -Doptimize=Debug --prefix .workspaces/flint-masked/debug
^.workspaces/flint-masked/debug/bin/test-flint-masked-sampling
```

The executable needs normal local Metal access; the restricted execution sandbox
did not expose a Metal device, so hardware qualification ran outside it. First
build also exposed the fixture package root's shader-embed restriction; its tiny
entry file now lives under `src/`, like the other Flint frontend entries.
Raw binaries remain ignored under `.workspaces/flint-masked-sampling-20261027/`.
