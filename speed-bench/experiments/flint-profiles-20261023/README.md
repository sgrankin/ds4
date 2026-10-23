# Flint hardware profiles — 2026-10-23

Native revision: the commit containing this report (`flint-metal` bookmark).
Source reference: `recursive-oss/rec`
`e53663b262af4cb1ca498e596ca8b1a2d79e0800` (Apache-2.0). Checkpoint: Recursive's
group-64 affine INT4 export, revision
`3b93f9053b530488405ae9fca51d4900d6ce21f0`, with INT8 embeddings.

**All measurements here use Apple M5 Max / 128 GiB.** Selecting `m4-pro` or
`m5-pro` on it exercises that profile's source and launch shape, but cannot
qualify actual Pro residency or predict its performance. Physical M4 Pro and
M5 Pro runs remain outstanding.

## Implementation and selection

The default `auto` selection recognizes the Metal device names `Apple M4 Pro`,
`Apple M5 Pro` and `Apple M5 Max`. Unknown devices fail before model loading.

| Profile | Resident / MLP groups | Prefill backend | Chunk cap |
| --- | ---: | --- | ---: |
| `m4-pro` | 36 / 32 | Portable SIMD, without M5 NAX | 2048 |
| `m5-pro` | 36 / 32 | NAX | 4096 |
| `m5-max` | 76 / 64 | NAX | 4096 |

An explicit profile may underfill a larger supported device, but may not exceed
its resident-group envelope or request M5 NAX on M4. An explicit smaller prefill
chunk is retained; larger requests are capped. Existing session ownership,
serialized resident packets, KV layout and Responses protocol are shared.

The CPU fixture checks selection, defaults and unsafe-override rejection.
GPU/model results below are recorded separately; compiling a shader alone does
not establish numerical correctness or task quality.

## Completed profile measurements on M5 Max

Fixed inputs are the same **21-token short prompt** and **798-token context
prompt** used in the [initial native qualification](../flint-native-20261023/README.md).
Context capacity is 262144 and output allowance is 128 tokens. The requested
chunk size is 4096, capped at 2048 for M4 Pro.
Each case has three fresh-prefix repetitions; the table averages warm repeats
1 and 2. Greedy uses temperature 0; default filtering uses temperature 1,
top-k 20, top-p .95, min-p 0. Jobs run sequentially under the pressure guard.

| Profile | Case | Prefill ms | Decode tokens/s | Request ms |
| --- | --- | ---: | ---: | ---: |
| M4 Pro | Short, greedy | 36.28 | 233.38 | 580.46 |
| M4 Pro | Short, default filtering | 36.29 | 230.13 | 588.16 |
| M4 Pro | Context, greedy | 478.43 | 218.50 | 1059.66 |
| M4 Pro | Context, default filtering | 478.45 | 216.24 | 1065.75 |
| M5 Pro | Short, greedy | 37.74 | 236.22 | 575.38 |
| M5 Pro | Short, default filtering | 37.46 | 234.30 | 579.50 |
| M5 Pro | Context, greedy | 216.66 | 222.18 | 788.27 |
| M5 Pro | Context, default filtering | 216.63 | 218.06 | 799.04 |
| M5 Max | Short, greedy | 37.29 | 245.26 | 555.13 |
| M5 Max | Short, default filtering | 37.28 | 244.62 | 556.47 |
| M5 Max | Context, greedy | 217.27 | 232.24 | 764.17 |
| M5 Max | Context, default filtering | 217.41 | 227.40 | 775.94 |

The Pro rows are **36-group schedules running on M5 Max**, not Pro hardware
results. Their decode is roughly 4–6% slower here than the 76-group Max schedule.
M4's portable SIMD prefill takes about 2.2 times as long on the 798-token case.
Decode rate is `(generated_tokens - 1) / decode_seconds`. These two warm repeats
per arm do not establish universal speed differences.

Eight teacher-forced prefixes compare all 150144 vocabulary entries at identical
token IDs with the **frozen, previously qualified native M5 Max backend**:

| Current profile | Mean delta-NLL, nats/token | Mean KL(reference to current), nats | Top-1 matches |
| --- | ---: | ---: | ---: |
| M4 Pro | -0.014220 | 0.00150505 | 8/8 |
| M5 Pro | 0 | 0 | 8/8 |
| M5 Max | 0 | 0 | 8/8 |

The M5 Pro and current M5 Max teacher rows are byte-identical to that native
reference. Both M5 profiles also reproduce all 128 generated token IDs in each
of the four cases. M4's free-running replies first differ after 123, 95, 42 and 123
matching tokens respectively in the table's case order. Its negative delta-NLL
means those eight target tokens became slightly more likely; it does **not**
establish improved model quality. This is a small two-prompt numerical probe,
not a coding benchmark. The separate original-MLX comparison remains in the
initial report and uses a different baseline.

All three guarded profile matrices completed successfully. M4's model-free GPU
fixture also passes matrix/expert tails, causal and sliding-window attention
with sinks, token-65536 KV bank crossing, wrapped local-ring publication, and
sampler ordering/handoff/EOS on M5 Max.

## Scalar, serving and regression checks

Both Pro profiles reproduce **all 128 greedy token IDs** between scalar and
eight-token packet execution for each short/context prompt and each of three
repetitions: **12/12 continuations match exactly**. The M4 arithmetic differences
above therefore do not arise from an incorrectly advanced packet state in these
fixtures.

Each Pro profile passes the existing **29/29 WebSocket checks and 17/17 answer
probes**, including the official SDK, actual tools, independent concurrent
conversations, cancellation, reconnect, durable restart and clean shutdown:

| Profile on M5 Max | Parent input tokens | Continuation input tokens | Correctly credited cached tokens |
| --- | ---: | ---: | ---: |
| M4 Pro | 4855 | 5006 | 4855 |
| M5 Pro | 4855 | 5011 | 4855 |

These cases cross each profile's chunk boundary and then extend the prompt after
decoding. They check the restored prefill snapshot and recomputed generated tail,
in addition to ordinary fresh requests. Cached-token accounting excludes the
tail that was recomputed.

The full `nu tests/verify.nu` host, server, storage, agent and script regression
suite passes. All planned local cross-validation checks are complete. **Physical
M4 Pro and M5 Pro qualification remains outstanding**: their real residency,
memory pressure and latency require those machines. This work does not extend
the earlier full-context qualification boundary.

## Reproduction and evidence

Use [the Flint instructions](../../../docs/FLINT.md) to build and download.
Reuse the token fixtures produced by
[`flint_reference.py`](../../ideas/flint_reference.py), then select a profile
before the native model argument:

```nu
let profile = 'm5-pro'
let model = './gguf/flint-int4'
let reference = '.workspaces/flint-baseline-20261023'
let frozen_native = ($reference | path join 'teacher-native')
let output = '.workspaces/flint-profiles-20261023'
python3 speed-bench/ideas/pressure_run.py ($output | path join 'reproduce-guard') -- ./zig-out/bin/flint-smoke --flint-profile $profile $model ($reference | path join 'short-prompt.json') ($output | path join 'reproduce-short') default
python3 speed-bench/ideas/pressure_run.py ($output | path join 'quality-guard') -- ./zig-out/bin/flint-smoke --flint-profile $profile $model ($reference | path join 'teacher-suite.json') ($output | path join 'reproduce-teacher') quality
python3 tools/flint-quality.py ($reference | path join 'teacher-suite.json') $frozen_native ($output | path join 'reproduce-teacher')
python3 speed-bench/ideas/pressure_run.py ($output | path join 'compile-guard') -- ./zig-out/bin/flint-smoke --flint-profile m4-pro --compile-only
python3 speed-bench/ideas/pressure_run.py ($output | path join 'scalar-reproduce-guard') -- ./zig-out/bin/flint-smoke --flint-profile $profile $model ($reference | path join 'short-prompt.json') ($output | path join 'reproduce-short-scalar') scalar
```

`teacher-native` is the saved native Max dump from the initial qualification;
`teacher-reference` is the original MLX dump and answers a different comparison.

Raw evidence remains local and ignored in `.workspaces/flint-profiles-20261023/`:

- `matrix.nu`, `{m4-pro,m5-pro,m5-max}-matrix-guard/{command,result}.json`
  and corresponding matrix logs.
- `{m4-pro,m5-pro,m5-max}-{short,context}-{greedy,default}-{0,1,2}.json`
  and initial FP32 logits.
- `{m4-pro,m5-pro,m5-max}-teacher-*.f32` and quality logs;
  `m4-pro-quality-comparison.jsonl` and `m5-max-quality-comparison.jsonl`
  use the frozen native Max baseline.
- `m4-pro-compile2.log` and `m5-pro-compile.log` for shader/KV/sampler smoke;
  `m4-pro-fixture.log` additionally checks portable SIMD prefill arithmetic.
- `{m4-pro,m5-pro}-{short,context}-scalar-{0,1,2}.json`, `scalar.nu` and
  `scalar-guard/result.json` for the scalar control.
- `ws-{m4-pro,m5-pro}/result.json` and corresponding guard/log files for the
  complete serving checks; `host-tests.log` records the regression suite.

The earlier M5 Max serving qualification remains documented in the initial
report; it is not silently counted as a new profile run.
