# Residual-stream steering

A bank contains one hidden-width FP32 row per trunk layer. Capture the final
prompt token's residual after the FFN/full transformer block (`residual`, the
builder default), or after attention (`attn_residual`). For HC models, capture
averages the four residual streams to learn one direction; runtime editing
applies that direction independently to each stream. Flint has one stream.
The residual convention is implemented for Metal inference.

Given a bank row `v`, residual `R` and configured strength `s`:

| Mode | Runtime edit | Strength convention |
| --- | --- | --- |
| `project` (default) | `R -= s * d * dot(d, R)`, where `d = v / norm(v)` | Per selected layer |
| `add` | `R += (s / L) * v` | Divided across `L` nonzero bank rows |

Project mode normalizes every nonzero row at loading. Positive strength
suppresses the component **already present** along that axis; negative strength
amplifies the existing component. Replacing `d` with `-d` leaves projection
unchanged. Whether removing or amplifying it encourages the learned target
style is empirical; projection does not carry addition's target orientation.
Add mode preserves
learned vector magnitude. With a target-minus-control bank, positive additive
strength moves toward the target residual; negative strength moves away.

Zero rows select no layer. `L` counts the actual nonzero rows, so selecting
fewer layers concentrates additive strength across those layers. The same
`1/L` convention applies separately to attention and FFN strengths. Projection
strength remains per layer as the selection changes.

## Run a bank

```bash
./ds4 --model /path/to/model --dir-steering-file dir-steering/out/style.f32 \
  --dir-steering-mode project --dir-steering-ffn 1 --dir-steering-attn 0 \
  --temp 0 -n 128 -p 'Explain why databases use indexes.'

./ds4 --model /path/to/model --dir-steering-file dir-steering/out/style-add.f32 \
  --dir-steering-mode add --dir-steering-ffn 1 --dir-steering-attn 0 \
  --temp 0 -n 128 -p 'Explain why databases use indexes.'
```

File-only defaults are project mode, FFN strength 1 and attention strength 0.
The FFN flag edits the residual after the full block; the attention flag edits
its intermediate residual. Both can be enabled independently. A zero strength
preserves the normal computation; a bank may be loaded at zero for later live
FFN changes. Changes affect subsequent computation; existing KV retains the
influence of its earlier settings.

Mode and bank are chosen at engine startup; live `/steer` changes the FFN
strength in the selected mode. DS4.1 and Flint edit the ordinary BF16 residual
publication and round the edited result back to BF16. Disabled/zero steering
uses the original computation.

Automatic KV SSD/suffix reuse cannot be combined with a loaded steering bank,
including a zero-strength preload. The agent disables that cache; the server
rejects an incompatible cache setup. Start a steered server without
`DS4_STATE_CACHE_DIR` and suffix-cache options. Expert-weight SSD streaming works.
Per-token timing therefore does not quantify the loss of those KV cache hits
for a persistent agent; cache compatibility is separate work.

The runtime reads the raw `.f32` bank. Its JSON sidecar documents extraction;
pass the desired runtime mode explicitly. The new recipes use residual banks.
Rebuild directions learned at the previous module-output capture points.

## Build a bank

The builder learns `mean(target residual) - mean(control residual)` from paired
prompt lists. Project mode writes unit directions. Add mode writes the raw
mean differences, preserving their magnitude; it performs no pair-wise
normalization or orthogonalization.

| Builder profile | Bank shape |
| --- | --- |
| `deepseek-v4.1-flash` | `40 x 5120` |
| `flint` | `49 x 2048` |
| `glm-5.3-flash` | `45 x 4096` |
| `qwen3.8-flash-next` | `48 x 2560` |

Predictor layers are excluded. Retained V4 Pro uses `61 x 7168`; full GLM 5.3
uses `78 x 6144`. Those two have no extraction profile here. Full GLM's runtime
primitives/shape have been tested; its full checkpoint is not installed and
live inference remains unqualified.

```bash
python3 -B dir-steering/tools/build_direction.py \
  --profile flint --ds4 ./ds4 --model /path/to/flint-int4 \
  --good-file dir-steering/examples/succinct.txt \
  --bad-file dir-steering/examples/verbose.txt \
  --mode add --component residual --layers '12-23,30' \
  --out dir-steering/out/flint-style-add.json --ctx 512
```

Use `--mode project` for projection, or `--component attn_residual` to learn an
attention-residual bank. `--layers all` is the default. Selection uses zero-based
indices and inclusive ranges; unselected rows are written as zeros. Capture
still gathers the full bank, making alternate selections possible offline.
Apply an attention-residual bank with `--dir-steering-attn STRENGTH` and
`--dir-steering-ffn 0`.
Start with a small paired prompt list and grade held-out prompts before
extracting a larger bank or choosing a strength.

Flint uses its automatic thinking template. The builder sends no forced
thinking flag for it and rejects `--think`. Other profiles support `--think`
or use their direct-answer prompt by default.

Capture files are `dump_residual-L_pos0.bin` or
`dump_attn_residual-L_pos0.bin`: one little-endian FP32 hidden-width row per
layer. Here `pos0` labels the prompt capture. The final prompt row survives
because extraction uses the existing prefill-only `--dump-logits` diagnostic
and performs no decoding. Prompt files, stderr and raw captures remain beneath
`.workspaces/steering-capture/`; `--work-dir DIR` selects another parent. Each
unique workspace is printed and recorded in the sidecar.

Capture subprocesses run from the repository root for Metal source discovery,
including when `--ds4` names an isolated build. V4.1 uses a 32GB streaming expert
cache with replay/resident approximations disabled. GLM Flash uses a 24GB cache,
disables static decode mapping and receives no unsupported chunk override.
GLM capture is bounded to `--ctx <= 2048` for its native chunk. Keep extraction
separate from other heavy model jobs and retain the existing pressure guard.

## Choose strengths and layers

Use fixed-token timing and held-out behavioral tasks. Small measured overhead
is acceptable; retain prefill, live-append and decode results separately. Sweep
strength as well as layer selection, recording output quality and useful
behavior. A synthetic bank establishes arithmetic correctness, not semantic
control. Historical word counts from the retired original Flash model do not
qualify the current residual hooks.

```bash
nu dir-steering/tools/run_sweep.nu --ds4 ./ds4 --model /path/to/flint-int4 \
  --direction dir-steering/out/flint-style-add.f32 --mode add \
  --prompts dir-steering/examples/eval_prompts.txt --scales '-1,0,0.5,1,2'
```

The sweep prints outputs for inspection. Use the
[fixed-token driver](../speed-bench/experiments/steering-20261024/README.md)
for latency and full-distribution comparisons. No current strength is a
universal recommendation: layer selection, capture component and mode change
what that number means.

An initial [held-out behavioral check](../speed-bench/experiments/steering-behavior-20261024/README.md)
found useful Flint points with the four-pair succinct/verbose bank: project
`+0.1` shortened all four final answers (16% fewer words in total), add `+1`
shortened all four (36% fewer), and add `-1` lengthened all four (25% more).
The answers stayed coherent and retained the key facts; extra numerical errors
remained in the lightning example, including in the baseline. This is four
greedy prompts, not a preset or a universal strength recommendation. Larger
amplification/addition points collapsed into repetition.

## Combining banks offline

Additions at the same hook combine linearly. For banks `v_i` with nonzero-row
counts `L_i` and desired strengths `s_i`, form
`w = sum((s_i / L_i) * v_i)`. Save `w` as FP32 and run it in add mode with
strength `L_w`, the nonzero-row count of the merged bank. This reproduces the
summed additive edit mathematically, with ordinary FP32 rounding. Skip an
all-zero result. Banks must have the same shape and capture hook.

Projection banks generally cannot be merged exactly by summing directions.
Several projections may remove a higher-dimensional subspace; projecting along
the sum removes one direction. Sequential nonorthogonal projections also depend
on order. Keep those as separate experiments.

## Sources and conventions

- [Upstream ds4's original pipeline](https://github.com/antirez/ds4/commit/3f7e5c93ed46f47fc15d80d69f48763d50418149)
  supplied the contrastive extraction and directional-steering foundation.
  Its module-output intervention differs from this residual convention.
- [Directional ablation](https://arxiv.org/abs/2406.11717) and
  [activation addition](https://arxiv.org/abs/2308.10248) describe the underlying
  interventions. A working hook alone does not validate a useful refusal/style
  direction or reproduce either paper's complete experimental setup.
- Layer-budgeted addition follows our oMLX implementation
  (`omlx/patches/steering.py` in the local oMLX checkout). The complete formulas
  and HC interpretation are specified above; oMLX is not a runtime dependency.
