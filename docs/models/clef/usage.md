# Clef setup and API

Clef-Flash, Clef and Clef-Omni answer bounded questions through the native
Zig/Metal server. Build with FFmpeg development libraries and `pkg-config`, in
addition to the [native build prerequisites](../../../src/README.md#build-and-run).
Download and start Flash with:

```nu
nu download-model.nu clef-flash
nu tools/zig-build ds4-server
./ds4-local --server --bin-dir zig-out/bin clef-flash
```

The final `clef-flash` argument selects the downloaded bundle and its model ID
for requests. Replace it in both commands with `clef` or `clef-omni` to download
and start another bundle; use that same ID in requests. The downloads include the
tokenizer, learned head and media weights, and verify pinned sizes and SHA-256
hashes. Their respective on-disk sizes are 6.19, 16.31 and 19.77 GB; these are
not peak memory requirements. The ordinary DeepSeek model symlink is retained.
For a direct decision-only server, pass the bundle directory as `--model`:

```nu
./zig-out/bin/ds4-server --model gguf/clef-flash-int4
```

## Decision requests

`POST /v1/decisions` and `POST /v1/systemone` use the same native engine.
`GET /v1/models` lists loaded models. Decisions are stateless and do not stream;
Responses HTTP/WebSocket is the conversational API for generative models.
Clients may compose the APIs and manage their own tool execution.

The Decisions endpoint accepts an ordered array of questions:

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

An illustrative response, with hypothetical probabilities and token count, is:

```json
{
  "model": "clef-flash",
  "answers": [{
    "type": "choice", "name": "department", "choice": "billing",
    "confidence": 0.8,
    "probabilities": [
      {"value": "billing", "probability": 0.8},
      {"value": "technical", "probability": 0.2}
    ]
  }],
  "usage": {"input_tokens": 100, "output_tokens": 0, "total_tokens": 100,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0}}
}
```

SystemOne uses a named questions object and `state`:

```nu
http post --content-type application/json http://127.0.0.1:8000/v1/systemone {
    model: clef-flash,
    state: 'The customer was charged twice and wants the duplicate refunded.',
    questions: {
        department: {
            type: choice,
            instructions: 'Who should handle the request?',
            criteria: {billing: 'Payments and refunds', technical: 'Software faults'}
        }
    }
}
```

For the same hypothetical probabilities and token count, SystemOne returns:

```json
{
  "model": "clef-flash",
  "answers": {"department": {
    "type": "choice", "choice": "billing", "confidence": 0.6,
    "probabilities": {"billing": 0.8, "technical": 0.2}
  }},
  "usage": {"input_tokens": 100, "output_tokens": 0}
}
```

Decisions supports `predicate`, `choice` and `score`; SystemOne calls the
predicate type `noul` and also supports `choice` and `score`. SystemOne choice
criteria are a keyed object and score criteria are an ordered array. Decisions
question names may be repeated or omitted, and choice values retain their
string or boolean types. OpenAI clients can call `client.decisions.create(...)`
with this server as their `base_url`.

Predicates return the probability of true; scores are probability-weighted mean
level indices. Decisions choice and score confidence is the maximum answer
probability, not a reproduction of hosted calibration. SystemOne uses
chance-corrected choice confidence and score-dispersion confidence. See the
[implementation](implementation.md#requests-and-answers) for the mapping and
an example. Box extraction and multipart assets are unsupported.

## Media and limits

PNG and JPEG sources may be HTTP(S) URLs, base64 data URLs or objects containing
`base64` or `data_base64`. Decisions accepts `input_image` parts and SystemOne
accepts image `assets`. The local adapter places media in a labeled prefix,
replacing extracted image payloads with references in rendered evidence; it
does not implement arbitrary inline asset expansion.

Both endpoints also accept native `images`, `audio` and `videos` extensions.
This is a `POST /v1/systemone` body; replace the placeholder data URLs with real
base64 media before submitting it:

```nu
http post --content-type application/json http://127.0.0.1:8000/v1/systemone {
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

Videos also accept encoded HTTP(S) or base64 sources. FFmpeg decodes encoded
media in memory, samples video at 2 frames per second, preserves source
timestamps and resamples audio to 16 kHz mono. Explicit audio samples use
`{samples: [...], sample_rate: 16000}`. Flash and full Clef accept images and
silent video; Omni also accepts audio and synchronized video with audio. Media
towers load lazily on the first media request.

The complete request record, including questions and media tokens, is limited
to 16,384 tokens for Flash/full or 64,000 for Omni, further capped by `--ctx`.
The server rejects an overlong record rather than truncating it. HTTP bodies and
individual media sources are limited to 64 MiB, with further decoded-media
limits. Invalid or unsupported input returns a client error; oversized media
returns 413, fetch failure 502 and an overlong context 422. Execution failures
can return 500. See [media and failure boundaries](implementation.md#media-and-failure-boundaries).

## Repeated state and co-hosting

Named `ds4-local` profiles enable one in-memory text-prefix entry per loaded
Flash/full engine. Repeating a long state with changed questions can reuse the
prefix; changed prefix tokens miss and the head still runs for every request.
Short inputs, media and Omni use their ordinary paths. Direct `ds4-server` users
enable the cache with `DS4_CLEF_PREFIX_CACHE=1`; set it to `0` to disable it in
a named profile. The entry lives in RAM, not the DeepSeek SSD cache. At the
measured 2,881-token size it retained about 321 MiB for Flash or 651 MiB for
full Clef; larger contexts retain more. The [prefix replay report](../../../speed-bench/experiments/clef-native-20261024/PREFIX-CACHE.md)
describes the tested workload, memory and results.

To load a decision helper alongside one generative model, first download both
bundles, then start the server with their directories:

```nu
./zig-out/bin/ds4-server --model gguf/flint-int4 --decision-model gguf/clef-flash-int4
```

Select `clef-flash` in decision requests to `/v1/decisions` or `/v1/systemone`;
select the loaded generative model in Responses requests. `GET /v1/models`
shows the available model IDs.

Use the installed Flint directory, or a DeepSeek GGUF with its ordinary SSD
settings, as `--model`. The qualified DeepSeek/helper setup specifies an expert
cache budget:

```nu
./zig-out/bin/ds4-server --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --ssd-streaming --ssd-streaming-cache-experts 32GB --ctx 8192 --prefill-chunk 512 --decision-model gguf/clef-flash-int4
```

Automatic GGUF cache admission budgets one engine and does not subtract other
loaded models; use explicit per-engine cache budgets when co-hosting. The tested
two-DS4.1-engine API fixture uses 8 GiB per cache. Repeat `--decision-model` to
load more helpers. They retain separate weights and scratch while sharing the
Metal runtime. Submitted work from different owners is serialized, so contention
can increase latency; current serving has one generative model per process.

For long helpers, `DS4_CLEF_LAYER_QUANTUM=1` optionally yields between layer
groups to let another GPU owner run. The default `0` keeps a single prefill
command; yielding can increase helper latency. The [scheduling report](../../../speed-bench/experiments/clef-native-20261024/SCHEDULING.md)
measured one M5 Max co-serving workload with two repetitions per setting, not a
universal setting. See [Apple Silicon execution](apple-silicon.md) for the
mechanism and [mixed serving](../../../speed-bench/experiments/clef-native-20261024/mixed-serving.md)
for its evidence.

The [qualification report](../../../speed-bench/experiments/clef-native-20261024/README.md)
separates numerical checks from media task checks and records checkpoint and
runtime revisions; a small smoke suite is not a full Decision Index score. The
[broader quality report](../../../speed-bench/experiments/clef-native-20261024/BROAD_QUALITY.md),
[vision](../../../speed-bench/experiments/clef-native-20261024/vision.md),
[media frontends](../../../speed-bench/experiments/clef-native-20261024/MEDIA.md)
and [kernel measurements](../../../speed-bench/experiments/clef-native-20261024/KERNEL-PERFORMANCE.md)
provide workload and performance details. The [reference manual](reference-manual.md)
explains the model contract.
