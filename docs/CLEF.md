# Clef decision models

Clef-Flash, Clef and Clef-Omni use native Zig/Metal inference. These models answer
bounded questions in one prefill plus their learned joint head; they do not
generate a response token by token. Python and MLX are used only as offline
references. The M5 Max is the measured hardware target.

```nu
nu download-model.nu clef-flash
nu tools/zig-build ds4-server
./ds4-local --server --bin-dir zig-out/bin clef-flash
```

Replace `clef-flash` with `clef` or `clef-omni` for the larger bundles. Downloads
include the tokenizer, learned head and media weights, and verify pinned sizes
and SHA-256 hashes. On-disk sizes are 6.19,16.31 and19.77GB respectively; these
are not peak memory requirements. The ordinary DeepSeek model symlink is retained.

Both `POST /v1/systemone` and `POST /v1/decisions` call the same native engine.
`GET /v1/models` lists only loaded models. Decisions are stateless and do not
stream; Responses HTTP/WebSocket remains the conversational API for generative
models such as DeepSeek and Flint.

A generator can also invoke an already loaded helper inside a Response using
the [hosted decision tool](HOSTED-TOOLS.md), with no client-side tool handler.

```nu
http post --content-type application/json http://127.0.0.1:8000/v1/decisions {
    model: clef-flash,
    input: 'The customer was charged twice and wants the duplicate refunded.',
    questions: [{
        type: choice, name: department,
        instructions: 'Who should handle the request?',
        choices: [
            {value: billing, description: 'Payments and refunds'},
            {value: technical, description: 'Software faults'}
        ]
    }]
}
```

OpenAI clients can call `client.decisions.create(...)` with this endpoint as
their `base_url`. Choice values retain their string/boolean types; question
names may be repeated or omitted. Supported questions are predicate, choice and
score. SystemOne uses `noul`, `choice` and `score`, with a named `questions`
object and `state` instead of `input`. Box extraction and multipart assets are
not implemented. OpenAI confidence is the model's maximum answer probability;
it does not reproduce a hosted model's calibration. SystemOne uses the current
TypeSafe chance-corrected choice and score-dispersion formulas.

## Fast paths and repeated state

Dense Flash/full choose their fast kernels automatically. Short inputs use the
quantized GEMM; at 2,048 processed tokens and above, each projection expands once
into a reusable BF16 matrix before MPP multiplication with FP32 accumulation.
This uses an additional 192 MiB of scratch for Flash or 340 MiB for full Clef,
allocated on the first long request. The checkpoint stays INT4. The direct
BF16 SwiGLU path also avoids an intermediate FP32 matrix.

Named `ds4-local` profiles enable one in-memory text-prefix entry in each loaded
Flash/full helper. Repeated state can answer changed questions without running
the whole backbone again. Changed prefix tokens miss; the learned head still
runs for every request. Short inputs, media and Omni use their ordinary paths.
The measured entry retained about 321 MiB for Flash or 651 MiB for full Clef at
the 2,881-token test size; larger contexts retain more memory.

Direct `ds4-server` users enable this with `DS4_CLEF_PREFIX_CACHE=1`. Set it to
`0` to disable it in a named profile. This entry belongs to the loaded engine
and lives in RAM; it is not persisted into the DeepSeek SSD cache.

See the [kernel measurements](../speed-bench/experiments/clef-native-20261024/KERNEL-PERFORMANCE.md)
and [prefix replay results](../speed-bench/experiments/clef-native-20261024/PREFIX-CACHE.md)
for quality, memory, timings and reproduction commands.

## Media

PNG/JPEG sources may be HTTP(S) URLs, base64 data URLs or objects containing
`base64`/`data_base64`. OpenAI `input_image` parts and SystemOne image `assets`
are accepted. Media is prefixed with a short label; image references replace
the original byte payload in rendered evidence. This labeled-prefix policy is
a local mapping, rather than TypeSafe's arbitrary inline asset expansion.

Native extensions on either endpoint expose each model's other inputs:

```nu
{
    model: clef-omni,
    state: 'Inspect the recording.',
    images: ['data:image/png;base64,...'],
    audio: ['data:audio/wav;base64,...'],
    videos: [{
        frames: ['data:image/png;base64,...' 'data:image/png;base64,...'],
        timestamps: [0 0.5],
        audio: 'data:audio/wav;base64,...'
    }],
    questions: {event: {type: noul, instructions: 'Did the speaker say stop?'}}
}
```

Videos also accept encoded HTTP(S)/base64 sources. Encoded media uses a small
in-memory FFmpeg decoder, samples video at2fps, preserves source timestamps and
resamples audio to16kHz mono. Explicit audio samples use
`{samples:[...],sample_rate:16000}`. Flash/full accept images and silent video;
Omni additionally has an audio encoder and synchronized video/audio placement.
FFmpeg development libraries and `pkg-config` are build dependencies.

The request context limit is16,384 tokens for Flash/full and64,000 for Omni,
capped by `--ctx`. Images and video frames consume context tokens too. Towers
are loaded lazily on the first media request. Source bodies are limited to64MiB;
invalid media returns a visible client error rather than becoming text evidence.

The [qualification report](../speed-bench/experiments/clef-native-20261024/README.md)
separates component numerical checks from actual media task checks, and records
the exact checkpoint/runtime revisions. A small smoke suite is not a full
Decision Index score.

## Alongside a generative model

```nu
./zig-out/bin/ds4-server --model gguf/flint-int4 --decision-model gguf/clef-flash-int4
```

Use the installed Flint directory, or a DeepSeek GGUF with its ordinary SSD
settings, as `--model`. The qualified DeepSeek/helper setup uses an explicit
expert-cache budget:

```nu
./zig-out/bin/ds4-server --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --ssd-streaming --ssd-streaming-cache-experts 32GB --ctx 8192 --prefill-chunk 512 --decision-model gguf/clef-flash-int4
```

Automatic GGUF cache admission still budgets one engine; it does not subtract
other loaded models. Use explicit per-engine cache budgets when co-hosting.
The tested two-DS4.1-engine API fixture uses8GiB per cache. This implementation
does not add an aggregate automatic memory allocator.

Repeat `--decision-model` to load more decision models.
The models keep separate weights and scratch while sharing the Metal runtime.
Submitted work from different owners is serialized; requests can prepare input
and wait concurrently. Contention therefore increases latency without requiring
model unload/reload. Current serving still has one generative model per process.

## Sources and measurements

The native implementation follows Cloudflare's published record/head and model
code, the pinned MLX-community affine INT4 checkpoints, and adapts credited
MIT-licensed [Cleffa](https://github.com/zknpr/cleffa) Metal leaves. Existing
Flint tensor operations are reused where their shapes and packing contracts fit.
See [model artifacts, numerical comparisons and task scores](../speed-bench/experiments/clef-native-20261024/README.md),
[vision](../speed-bench/experiments/clef-native-20261024/vision.md),
[media frontends](../speed-bench/experiments/clef-native-20261024/MEDIA.md) and
[mixed serving](../speed-bench/experiments/clef-native-20261024/mixed-serving.md).

The [broader task report](../speed-bench/experiments/clef-native-20261024/BROAD_QUALITY.md)
includes Flash's2,000-request sample and all three28-case BLINK samples. The
[cooperative scheduling report](../speed-bench/experiments/clef-native-20261024/SCHEDULING.md)
plots the optional layer-group tradeoff for long helpers; default0 preserves
the ordinary single-command prefill.
