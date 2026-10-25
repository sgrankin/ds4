# Final existing-engine numerical regression

On 2026-10-25, the final owner/frontend candidate matches the frozen pre-Clef
control **byte for byte on all 32 full-vocabulary FP32 logit rows** in this
bounded screen. No model arithmetic fix was necessary.

| Installed model | Vocabulary | Frontiers | Compared rows | ΔNLL / KL / max logit delta |
| --- | ---: | --- | ---: | --- |
| DeepSeek V4.1 Flash 1017 IQ2_XXS/Q2_K | 129,280 | 32, 64 | 8 | 0 / 0 / 0 |
| Qwen3.8 Flash Next Q4 | 248,320 | 32, 64 | 8 | 0 / 0 / 0 |
| GLM-5.3 Flash Q2 | 154,880 | 32, 64 | 8 | 0 / 0 / 0 |
| Flint INT4, M5 Max profile | 150,144 | 32, 64 | 8 | 0 / 0 / 0 |

Each frontier saves the prefill output and three subsequent teacher-forced
decode rows. Prompt IDs, frontier IDs and forced target sequences match before
comparison. Top-1 agreement is 100% and total variation is zero everywhere.
The existing NumPy comparator computes ΔNLL and forward KL; a final direct byte
comparison of every corresponding `.f32` file establishes the stronger exact
result, including float bit patterns.

All eight guarded model processes complete with child 0. Every sampled pressure
level is normal (1), and swap does not grow during any arm. Models run strictly
sequentially after the broad Clef quality server shuts down. Dumps are explicitly
untimed; decode timing columns are zero. This report makes no speed claim.

This protects the concrete ownership migration against a small ordinary
single-model regression. It is not a long-context, multimodal, speculative,
steering, throughput or exhaustive quality test. The separate
[ownership qualification](OWNERSHIP.md) covers actual two-DS4.1-engine lifetime,
the mixed-serving API and the matched short latency screen. Other GGUF families
retain their single-Core admission pending separate concurrency qualification.

## Settings and reproduction

The existing [Nushell driver](engine_regression.nu) reuses `ds4-bench` and the
[teacher-forced comparator](../steering-20261024/compare_decode.py). Context
allocation is 256, frontiers are 32 and 64, each dumps four rows, and CPU threads
are eight. The prompt repeats the same short passage about independent
conversations. DS4.1 uses prefill 64 and a 32 GiB SSD budget; GLM uses a 24 GiB
SSD budget and per-layer decode mapping. Qwen uses its ordinary Metal path;
Flint selects the native M5 Max profile. The driver removes inherited DS4/PT
experimental variables and sets
`DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP=1` for both arms.

```nu
nu speed-bench/experiments/clef-native-20261024/engine_regression.nu .workspaces/clef-native-20261024/control/ds4-bench .workspaces/clef-native-20261024/final-build/bin/ds4-bench --flint-model ../flint/.local/share/rec/models/fdc85248ad77a0fa9fbf0abda8b1525bea8ee74c11ada2ec95eb66113353d677 --output .workspaces/clef-final-logits-20261025
```

Control SHA-256:
`4638ee4f858cf5ff06386974f8825b33fadb28f86babbfb5a0b740819ebbb2f6`.
Candidate SHA-256:
`fe147ad334ca48c934029be2bf9529021bc78499d88c516ca8a771440fc93de5`.
The candidate includes the final scoped Host/Metal ownership, worker captures,
referencecounted process admission and engine-aware frontend metadata calls.
Commands, binary identities, guards, raw arrays and comparison summaries remain
ignored in `.workspaces/clef-final-logits-20261025/`.

One log difference is an accounting correction: Flint now reports about
1,908 MiB of planned session buffers rather than the old generic estimate of
30.5 MiB at context 256. Its existing decoder allocates global KV in 65,536-token
blocks, even for a small logical limit; the native estimate includes that actual
rounded capacity, local rings, checkpoint and lazy prefill workspace. The
ownership work does not introduce that allocation or change the requested
logical context. No source change was needed for this observation.
