# Native Clef versus local references — 2026-10-25

The scheduling chart's four-second helper is full dense27B Clef at2881 input
tokens, sharing the GPU with Flint. It is not short Flash latency. A fresh
same-checkpoint solo screen finds a real long-input gap to MLX:3.608s vs3.147s,
14.7% slower. Short full-Clef requests are within6% either way.

## Fresh matched full-Clef screen

M5 Max128GB; parent serving revision`60e7b671b830`, unchanged native binary
`.workspaces/clef-native-20261024/final-build/bin/clef-smoke`, SHA256
`74a4b002f5461bb02915dac3bfafca301dd0035c5e8d9fe1150d70ff84935686`.
Both runs load pinned`gguf/clef-int4`; native GPU head enabled, layer quantum0;
MLX uses the bundle's reference and FP32 head. One guarded model process at a
time, no builds. Both guards finish normally, pressure normal, swap growth0.

Each native case runs four times; discard its first call and take the remaining
three's median. The existing reference oracle warms each case and measures
three combined calls. Native includes record encoding; reference excludes it.
Native encoding is under1ms here. Model load and HTTP are excluded in both.
Native runs before reference, without a closing control: this is a small screen,
not a precise universal throughput ranking.

| Input tokens | Native warm | MLX warm | Native difference |
| ---: | ---: | ---: | ---: |
|169 |212.72ms |225.41ms |−5.6% |
|266 |349.41ms |332.86ms |+5.0% |
|157 |220.14ms |228.05ms |−3.5% |
|238 |284.35ms |290.08ms |−2.0% |
|2881 |3608.28ms |3146.74ms |+14.7% |

The long native backbone takes3577.75ms and its head29.57ms on the median-total
call. Separately synchronized MLX stages report3194.78ms backbone /14.02ms head;
those stage medians are different passes, not a decomposition of its combined
3146.74ms. Most elapsed time and the remaining gap are in the backbone.

All seven top decisions match; maximum probability difference is0.000510.
Exact tokens match each case. These are implementation comparisons at the same
INT4 weights, not an INT4-versus-unquantized quality evaluation.

## Other existing comparisons

- [Flash](HEAD.md):67–108ms native vs65–100ms local MLX on four157–266-token
  records, about3–8% slower. Native includes preparation; MLX excludes encoding.
- [Omni](OMNI.md):62–79ms native vs70–86ms local MLX on four161–262-token
  records, about8–12% faster, with the same timing-boundary qualification.
- [Cleffa's published M5 Max results](https://github.com/zknpr/cleffa#speed):
  full Clef2525ms at2235 tokens and5353ms at4510 tokens. Same hardware class,
  different inputs and BF16 weights; this supports a similar latency scale,
  not a matched claim that our INT4 implementation beats it.
- [Cloudflare's updated hosted results](https://blog.cloudflare.com/clef-faster-cheaper-multimodal/):
  full Clef305ms median at about3400 input tokens. Hosted hardware/workload is
  different; our implementation does not achieve that latency on this Mac.

Dense long-input matrix multiplication is a sensible next profiling target.
INT4 saves weight storage but does not guarantee proportionally faster prefill.
Reusing decision prefix/features is a separate unimplemented opportunity.

## Reproduction

Use the first four records in`requests.jsonl`. The fifth is the scheduling
fixture: state`A customer was billed twice and needs a refund.\n` repeated250
times; question`Which team should handle the request?`; billing criterion
`Payments, charges and refunds`, technical criterion`Broken software`.
Reuse `tests/mixed-serving-wire.py`'s payload or its saved request body.

Repeat each native input four times in JSONL and run the existing`clef-smoke`
with`DS4_CLEF_TIMING=1 DS4_CLEF_FULL_HEAD_GPU=1 DS4_CLEF_LAYER_QUANTUM=0` under
`speed-bench/ideas/pressure_run.py`. Run the unique five JSONL records through
`oracle.py gguf/clef-int4 REQUESTS OUTPUT --repeats 3` in the existing reference
venv, also guarded. The exact inputs, invocation and raw output remain local in
`.workspaces/clef-speed-comparison-20261025/`; its`run.nu` orchestrates these
existing tools sequentially. No new benchmark harness or runtime change.
