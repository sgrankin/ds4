# Native Clef components

The public setup and API guide is [docs/CLEF.md](../../../docs/CLEF.md).
`engine.zig` owns one loaded model, tokenizer, head choice and scratch mutex.
The decision registry in `src/clef_entry.zig` keeps those engines independent;
`src/server/decisions.zig` translates the two wire formats without model code.

- `tokenizer.zig` and `record.zig` reproduce the published STATE/schema record,
  spans, lexical options, media tokens and three-axis positions.
- `backend.zig` owns safetensors mappings, tensor banks, dense hybrid execution
  and dispatch. `backend_omni.zig` implements the distinct sparse Omni thinker.
- `backend_head_full.zig` is the ordinary GPU learned head. `head.zig` is the
  FP32 Accelerate diagnostic oracle; `backend_head.zig` retains the measured
  intermediate split. Serving loads only its selected head representation.
- `image.zig`, `audio.zig`, `video.zig` prepare CPU inputs. `vision.zig` and
  `audio_encoder.zig` own the native towers. `media_prepare.zig` owns per-request
  decoded inputs and copies tower output before reusable scratch changes.
- `media_source.zig` uses standard base64/libcurl. `media_codec.c` is the narrow
  in-memory FFmpeg boundary; it has no model policy or hidden file/network I/O.

Model requests are one prefill over all required hidden rows followed by the
joint head, rather than autoregressive generation. The head needs state rows,
question/option spans and selected lexical embedding rows. Ordinary generative
KV-prefix reuse alone therefore does not implement reusable decision features.

## Sources and credit

Record/head policy follows Cloudflare's published
[Clef-Flash](https://huggingface.co/Cloudflare/clef-flash/tree/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f)
and [Clef-Omni](https://huggingface.co/Cloudflare/clef-omni/tree/df2d7c9505e62ba8a527b147513a4fa66a77dc5d)
source. The [pinned checkpoint report](../../../speed-bench/experiments/clef-native-20261024/README.md)
records the MLX-community affine quantizations and licenses shipped with them.
The native code preserves BF16 learned weights and uses FP32 activations where
qualified against the reference.

Selected attention, GatedDeltaNet, vision/audio leaves and CPU-head formulation
adapt [Cleffa](https://github.com/zknpr/cleffa/tree/24fc91e28d154ff8dadfa58462a30b437413c3f4),
whose MIT notice is retained at
[src/metal/clef/LICENSE.cleffa](../../metal/clef/LICENSE.cleffa).
Matrix kernels reuse the existing [Flint/MLX shader source closure](../../metal/flint/prefill/README.md),
with its Apple MIT and Recursive Apache notices. Adapted kernels identify their
source beside the code. Production inference does not link MLX or invoke Python.

Golden JSON fixtures retain exact IDs, spans and media positions from the pinned
record sources. Raw model outputs and benchmark data remain local and ignored.
The [qualification report](../../../speed-bench/experiments/clef-native-20261024/README.md)
links individual numerical, task, latency and ownership results.
