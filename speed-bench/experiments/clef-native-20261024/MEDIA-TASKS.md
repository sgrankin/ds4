# Native media serving checkpoint

All three models pass controlled visual decisions through the native HTTP
server. Omni also passes contrasting speech and audiovisual decisions. This is
a small task-path qualification, separate from the component numerical oracles
in [MEDIA.md](MEDIA.md) and [vision.md](vision.md), and from a public benchmark.

Candidate SHA-256:
`fa67e1d5f3a0ec19fc28b7da5110867b6dc5f6beb699a0d7ecef4ab0ac8b921b`.
It uses the qualified full GPU joint head by default, skips the duplicate CPU
head and prepares head pipelines during startup. All three pinned models remain
loaded together. No Python/MLX inference runs in this process.

## Actual checks

For **Flash, full and Omni**, identical questions distinguish red and blue
images. OpenAI Decisions also accepts a question-specific `input_image`, rather
than only images in the shared input. Four-frame red→blue videos answer blue at
the end, both from explicit frames/timestamps and an encoded H.264 MP4.

For **Omni**, two otherwise identical recordings say “The access code is three”
or “The access code is seven.” Identical requests offer three/seven/nine; both
answers follow the actual audio. Both codes also pass alongside the video,
answering the code and final color together. This works with explicit frames
plus an audio source, and with encoded MP4/AAC soundtracks. There are11 requests
in the final Omni driver and5 each for Flash/full, all successful. The earlier
Flash run also covered a standard input-image part and a TypeSafe image asset.

These generated color/code tasks establish changing-input behavior and actual
codec/tower/placement/serving integration. They do not establish general speech,
temporal reasoning, audiovisual grounding or visual benchmark quality.

## Indicative end-to-end latency

Warm-ish final-run client wall times include HTTP, base64/codec/preprocessing,
tokenization, towers, backbone and head. These are individual functional checks,
not mirrored latency medians; first-image and first-tower calls are colder.

| Input | Flash | Full | Omni |
| --- | ---: | ---: | ---: |
| Second contrasting image |113ms |305ms |150ms |
| Explicit frame video |104ms |294ms |248ms |
| Encoded silent video |103ms |294ms |249ms |
| Spoken code, two inputs |— |— |145–157ms |
| Explicit audiovisual, two inputs |— |— |295–296ms |
| Encoded audiovisual, two inputs |— |— |295–298ms |

Cold startup/first media pipelines cost more and are visible in the local logs.
The server exposes per-stage timings with `DS4_CLEF_TIMING=1`, including CPU
media preparation and tower time separately from text backbone time.

## Reproduce

```nu
nu tools/zig-build ds4-server
nu speed-bench/experiments/clef-native-20261024/media_task_fixtures.nu
python3 speed-bench/ideas/pressure_run.py .workspaces/clef-media/guard -- ./zig-out/bin/ds4-server --model gguf/clef-flash-int4 --decision-model gguf/clef-int4 --decision-model gguf/clef-omni-int4 --port 19311 --ctx 64000
# From another terminal, one driver at a time:
nu tests/test_clef_media.nu --model clef-flash --vision-only --output .workspaces/clef-media/flash
nu tests/test_clef_media.nu --model clef --vision-only --output .workspaces/clef-media/full
nu tests/test_clef_media.nu --model clef-omni --output .workspaces/clef-media/omni
```

The fixture recipe writes speech to files silently using the macOS speech
service. Sandboxing that service can return success with zero samples; a simple
`ffprobe` duration check now exposes that actual failure. The native media
decoder correctly returned400 for the empty test file. No model failure was
hidden by substituting a text answer or a label in request state.

Raw fixtures, answers and stage logs are ignored under
`.workspaces/clef-media-20261024/`, `.workspaces/clef-media-serving-20261024/` and
`.workspaces/clef-native-20261024/media-serving.log`. After all three models and
media towers were used, process RSS was about76.9GiB (includes mapped checkpoint
pages as well as model buffers). This is a combined-process observation, not
per-model minimum RAM. Pressure stayed normal and existing swap stayed21.785GiB
with no growth; the guard and server exited0 after orderly shutdown.
