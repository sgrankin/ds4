# Clef media frontend checkpoint

Status on 2026-10-25: CPU media preparation, record placement and native vision/
audio towers are wired into the production engine source. The numerical
component checks below pass; real image/audio/video decisions are being
qualified separately. Source integration alone is not an end-to-end quality
claim. See [the vision report](vision.md) for learned vision and record results.

The native CPU frontend is implemented in `src/models/clef/`:

- `image.zig`: existing Iris PNG/JPEG codecs, RGB conversion, model geometry,
  uint8 antialiased bicubic resize and canonical HF patch layout.
- `audio.zig`: mono 16 kHz PCM16/float32 WAV, batched centered Whisper STFT,
  128 Slaney log-mel channels, per-clip lengths and CNN/attention layout.
- `video.zig`: frame selection, pixel budgets, temporal pairs and timestamps.
- `codec.zig` + `media_codec.c`: narrow libav memory decode, common encoded
  audio downmix/resampling and sampled RGB video with its audio track.

Python is used only for offline numerical fixtures. Production media receives
already-fetched bytes; the codec does not open files or secondary network URLs.
The server's source reader owns HTTP/base64. Caller-supplied sample/frame limits
fail visibly rather than silently truncating media. Decoder output reports real
frame timestamps and audio start time; record/model adapters own their placement.
Encoded audio is decoded to mono FP32 at 16 kHz; already-decoded sample arrays
must use that rate. Video decoding samples at 2 fps. The decoder's optional
frame-pixel limit may downscale large source frames; the serving engine sets it
to 16 megapixels so ordinary inputs reach the model's own preprocessing budget
without an earlier resize. The low-level helper default is 262,144 pixels.

Image patch order is `[temporal-pair, block-y, block-x, merge-y, merge-x]`, with
one row containing `[channel, temporal-tap, patch-y, patch-x]`. All three model
checkpoints use patch 16, temporal 2 and spatial merge 2: 1536 values per patch.
Images repeat their frame in both temporal taps; video pairs retain distinct
frames and duplicate only an odd final frame. Dense video budgets total pixels;
Omni uses its published per-frame bounds. The data structures carry actual grids
and token counts into the GPU towers.

`test-clef-media` uses production Zig methods. The numerical fixture covers:

| Check | Result |
| --- | --- |
| PNG RGB decode | Byte-exact |
| Image resize/patches, 393,216 floats | Bit-exact to frozen Cleffa CPU oracle |
| Position interpolation versus NumPy linspace | Max 2.56e-6, RMS 8.5e-7 |
| PCM16 decode, 20,000 samples | Bit-exact |
| Whisper frontend, 128×125 | Max 3.64e-5, RMS 2.33e-6 versus independent NumPy processor |
| Three video frames, 49,152 patch floats | Bit-exact to independent HF reshape/transpose |
| FFV1 video sampled at 2 fps | Four 64×64 RGB frames exact; times 0/.5/1/1.5 seconds |
| Same video shifted to first PTS 2 s | Identical selected frames at 2/2.5/3/3.5 s; audio origin 2 s |
| Encoded 44.1 kHz audio → 16 kHz mono | All 32,000 samples bit-exact to FFmpeg CLI |

The frontend and test compile without warnings. These are component results;
modality support also requires the learned tower, correct token placement and a
real end-to-end decision. Reference code is credited to MIT-licensed Cleffa
revision 24fc91e2; its notice is preserved in `src/metal/clef/LICENSE.cleffa`.
The audio Hann/Slaney constants retain their pinned numerical source comments.

Reproduction:

```nu
nu tools/zig-build clef-media-test --prefix .workspaces/clef-native-20261024/build -j2
nu speed-bench/experiments/clef-native-20261024/media.nu
```

The reference-only environment, Cleffa clone and FFmpeg must be installed as in
the main qualification report. Raw numerical outputs remain ignored under
`.workspaces/clef-media-20261024/`. The reproduction driver builds the reference
image leaf and uses standard Pillow/NumPy/Whisper processing and FFmpeg commands;
it does not introduce another serving harness.

The native encoded-media bridge requires FFmpeg development headers/libraries
and `pkg-config` (on this Mac, Homebrew `ffmpeg` and `pkgconf`). The build obtains
flags with `pkg-config` for `libavformat`, `libavcodec`, `libavutil`,
`libswresample` and `libswscale`; it does not hardcode a Homebrew cellar version.
PNG/JPEG still images reuse the existing Iris codecs. The codec bridge receives
memory bytes, disables secondary URL/file protocols, and uses software decoding.

CPU token/position fixtures cover dense images, multiple images and video, plus
Omni images, audio and synchronized video/audio. They derive tokens from pinned
official record encoding, and positions from the published model-specific
implementations. Actual learned vision/audio qualification is reported separately.

## Native Omni audio encoder

The learned audio tower is now native Zig/Metal in `audio_encoder.zig` and
`src/metal/clef/audio.metal`. It keeps the published BF16 weights, FP32 operands,
three convolution stages, sinusoidal positions, 32 bidirectional transformer
layers and the 2048-wide output projection. Each audio clip retains its own
attention windows; CNN padding is shared across clips in the request. The native
backend loads the tower lazily and returns borrowed feature rows for media
injection. No Python runs in serving.

On the 20,000-sample chirp fixture (128×125 mel frames, 17 output tokens), native
features have cosine 0.99999869 to the pinned MLX FP32 tower, RMS error 5.18e-5
and maximum absolute error 3.14e-4. All 34,816 values are finite. The published
BF16-weight tower with FP32 mel input gives identical reference output bytes on
this fixture. This is a numerical component check; it does not establish audio
task quality or long-record performance.

Initial native warm time was 50.87 ms. Inspection found a 64-row MPP tile with
a host divisor of 16, issuing redundant groups. Correcting dispatch reduced
time to 40.14 ms; selecting the existing vision path's 16-row FP32 tile then
reduced it further. A quiet mirrored 16/64/64/16 rerun measured
22.450/39.941/39.857/22.354 ms, about 44% faster with the retained 16-row tile.
All four outputs are bit-identical. No layer-command batching was added.

| Encoder input | Native FP32 × BF16 | MLX FP32 | MLX BF16 weights / FP32 input |
| --- | ---: | ---: | ---: |
| 125 mel frames, 17 tokens | 22.40 ms | 21.54 ms | 27.82 ms |
| 1000 mel frames, 130 tokens | 34.09 ms | 28.21 ms | Not measured |

Each number is a median over eight warm calls, excluding load, CPU mel work,
and the decision backbone/head. Native short-input time is approximately 4%
above the FP32 reference; the long input is about 21% above it. These are small
component screens, with different processes for native/reference. The longer
fixture spans ten CNN chunks, exercising scratch reuse beyond the eight-chunk
batch and attention's 104-token window. All 266,240 output values are finite;
reference cosine is 0.99999869, RMS error 5.03e-5, maximum absolute error 3.45e-4.
It repeats prepared mel data and is a layout/numerical check, not a speech task.

```nu
nu tools/zig-build clef-audio-smoke --prefix .workspaces/clef-native-20261024/build -j2
# After generating the media fixtures above, run one job at a time:
python3 speed-bench/ideas/pressure_run.py .workspaces/clef-media-20261024/audio-native-guard -- .workspaces/clef-native-20261024/build/bin/clef-audio-smoke gguf/clef-omni-int4 .workspaces/clef-media-20261024/audio-mel.f32 125 .workspaces/clef-media-20261024/audio-native.f32
python3 speed-bench/ideas/pressure_run.py .workspaces/clef-media-20261024/audio-reference-fp32-guard -- .workspaces/clef-artifacts-20261024/venv/bin/python speed-bench/experiments/clef-native-20261024/audio_oracle.py gguf/clef-omni-int4 .workspaces/clef-media-20261024/audio-mel.f32 125 .workspaces/clef-media-20261024/audio-reference-fp32 --fp32
```

Numerical outputs, timings and guard logs remain under
`.workspaces/clef-media-20261024/`: `audio-recheck-{a1,a2,b1,b2}` are the quiet
mirrored controls; `audio-final-comparison.json` contains the longer numerical
comparison. All qualification guard children exited zero.

Actual speech task inputs can be generated without audible playback:

```nu
nu speed-bench/experiments/clef-native-20261024/speech_fixture.nu
```

It uses the Mac's current voice at 180 words/minute to make two 16 kHz mono
PCM16 WAV files: `spoken-code.wav` says "The access code is seven." and
`spoken-code-three.wav` says "The access code is three." A meaningful paired
decision asks the same question with the same choices for both files, without
including the expected code in request text. These generated waveforms remain
local and ignored. Speech task results belong with the end-to-end report.
