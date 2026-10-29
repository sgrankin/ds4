# Using Qwen 3.8 Flash Next

This guide covers the local Qwen 3.8 Flash Next profile. `qwen4exp` is its GGUF architecture and source identifier, not a claim that arbitrary Qwen checkpoints are compatible. For the model contract, execution and backend details see the [reference](reference-manual.md), [runtime](implementation.md) and [Apple Silicon](apple-silicon.md) manuals.

## Build, download and run

On Apple Silicon, install the [Metal build prerequisites](../../METAL.md#build-and-run), Zig and Nushell; maintained wrappers are tested with Nushell 0.116.1. From the repository root:

```sh
nu tools/zig-build ds4-server
./download-model.nu qwen38-q2
./zig-out/bin/ds4-server --model gguf/Qwen3.8-Flash-Next-Q2.gguf --ctx 8192 --prefill-chunk 1024
```

The Zig build places programs in `zig-out/bin`; `make` builds and copies the main programs to the repository root. Use the same model options with `ds4` or `ds4-agent`. The downloader also updates `ds4flash.gguf`. Downloads and exact artifact names are in the [model catalogue](../../MODELS.md#qwen38-flash-next).

| Download | File size | Main and MTP weights |
| --- | ---: | ---: |
| `qwen38-q2` | 137.10 GiB | 41.73 GiB |
| `qwen38-q4k` | 165.11 GiB | 69.74 GiB |

Each file includes the original 95.37 GiB BF16 n-gram table, read on demand from disk without a sidecar or preload. Keep it on a fast local SSD. Weight sizes are not total process memory: context, session state and runtime buffers need additional space. The 8K context and 1024-token chunk above are a conservative starting point for Q2 on a 64 GB Mac; Q4 needs more memory. Q2 uses calibrated IQ2_XXS gate/up and Q2_K down experts, while Q4 uses calibrated Q4_K gate/up and MXFP4 down experts.

The old DGX Spark instructions used `make cuda-spark` and an 8192-token default prefill chunk. They describe upstream CUDA work, not a maintained or locally qualified CUDA build in this Metal-focused tree. The [historical Spark results](../../../gguf-tools/quality-testing/README.md#7-qwen38-spark-baseline) and [checkpoint benchmark](../../../speed-bench/qwen38-checkpoints/README.md) record their specific workloads; older two-run throughput figures are not a current performance comparison.

## Send a request and continue

With the server running, connect a WebSocket client to `ws://127.0.0.1:8000/v1/responses` and send:

```json
{"type":"response.create","model":"qwen3.8-flash-next","store":false,"reasoning":{"effort":"none"},"input":"Remember code=EMBER-73. Reply OK."}
```

Read events through `response.completed`, which contains the answer and a response ID. On the same connection, send a continuation using that ID:

```json
{"type":"response.create","model":"qwen3.8-flash-next","store":false,"previous_response_id":"resp_FROM_COMPLETED_EVENT","reasoning":{"effort":"none"},"input":"What code did I ask you to remember?"}
```

Read the next `response.completed` for the answer. `store=false` keeps this chain in connection-local memory; a reconnect requires replay unless the parent was durably stored. See the [Responses contract](../../../src/serving/RESPONSES.md) for HTTP, persistence and tool-result handling.

## Reasoning, tools and sampling

The server exposes `qwen3.8-flash-next`, `qwen3.8-flash-next-chat` and `qwen3.8-flash-next-reasoner` aliases. `--nothink` disables thinking. Tool calls use the native `<tool_call><function=...><parameter=...>` format; the [reference](reference-manual.md#prompt-reasoning-and-tools) explains rendering and tool results.

The embedded multi-token predictor proposes drafts that the target model verifies before committing tokens. Add `--mtp` to use it. Without it, ordinary decoding remains available; `ds4-bench` measures ordinary decoding and does not accept `--mtp`. Default MTP accepts matching greedy drafts even at nonzero temperature, so it does not promise ordinary sampling's distribution. With `--mtp-exact-sampling`, a sampled single session uses exact speculative sampling; sampled batched requests instead use ordinary batched decode, while temperature-zero requests can still speculate. See [sampling semantics](../../SPECULATIVE_DECODING.md).

On Metal, independent conversations can share GPU weight work while retaining separate session state. Use `ds4-server --batched-session N` to group eligible sessions. Floating-point reduction order can change replies, and a seed need not reproduce a reply across batching schedules. Speculative batches above 16 sessions, images, steering and other unsupported conditions use ordered fallback; the [runtime manual](implementation.md) describes eligibility and state. Disk checkpoints include recurrent state; an older rewind replays the retained prefix on the next evaluation. Image identity and embeddings are not stored in the session payload, so do not assume an image session can be restored from that payload alone.

The native context is 262,144 tokens. Set `DS4_QWEN4_YARN_FACTOR=2` or `DS4_QWEN4_YARN_FACTOR=4` to enable static YaRN for longer contexts; this can reduce quality on shorter prompts. Context allocation still requires sufficient memory.

## Images

Download the separate encoder and pass it at startup:

```sh
./download-model.nu qwen38-vision
./zig-out/bin/ds4-server --model gguf/Qwen3.8-Flash-Next-Q2.gguf \
  --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf --ctx 8192 --prefill-chunk 1024
```

The CLI accepts `/read image.png`; the server accepts `image_url` parts. Images are resized to multiples of 32 pixels with a minimum of 64 visual tokens and a default maximum of 1024. Set `DS4_QWEN4_IMAGE_MAX_TOKENS` to any value from 64 through 16383 to lower or raise the maximum; values outside that range silently use 1024. Live KV reuse uses image fingerprints. This is image input, not a claim of support for other modalities. See the [server image API](../../SERVER.md#images) and [runtime state limits](implementation.md#rewind-snapshots-and-reuse).

## Conversion

Follow [GGUF conversion](../../../gguf-tools/README.md#qwen38-flash-next) for the HF checkpoint and calibrated quantization. `gguf-tools/qwen4_pack_to_qwen4exp.py` migrates old DS4 fast packs to the active schema. Finish older main-only conversions with `gguf-tools/qwen4_native_ngrams.py` using the original HF n-gram shards. Expanding an already quantized table to BF16 cannot recover lost precision.

## Validation

The [release qualification](../../../QA_BEFORE_RELEASES.md#18-qwen38-flash-next) defines the model-backed gates and their scope. Model-free checks include:

```sh
make test-qwen4-kernels test-qwen4-ngrams test-frontends
make -C gguf-tools quants-shared
python3 -m unittest discover -s gguf-tools/tests -p test_qwen4_pack.py
python3 -m unittest discover -s gguf-tools/tests -p test_qwen4_native_ngrams.py
uv run --with numpy python -m unittest discover -s tests -p test_qwen4_vision_ref.py
uv run --with numpy --with pillow python -m unittest discover -s tests -p test_jpeg_decode.py
```

### State and replay

The n-gram state fixture requires model weights and exercises failed disk reads and recovery through prefill, decode and MTP:

```sh
nu tools/zig-build qwen-ngram-state-test
./zig-out/bin/test-qwen4-ngram-state /path/to/MODEL.gguf
```

For checkpoint, MTP and logit replay, build the server test first and run the model-backed checks:

```sh
make ds4_test
DS4_TEST_MODEL=/path/to/main-with-mtp.gguf \
  DS4_TEST_GLM_MTP=1 DS4_TEST_MTP_EXACT=1 \
  ./ds4_test --qwen4-prefill-checkpoints --qwen4-restore-reuse \
    --session-snapshot --session-rewind --session-rewind-resample
python3 tests/test_qwen4_checkpoint_replay.py --model /path/to/main-with-mtp.gguf
python3 tests/test_qwen4_mtp_limits.py --model /path/to/main-with-mtp.gguf
python3 tests/test_qwen4_logit_dump.py --model /path/to/main-with-mtp.gguf
```

These check state replay and continuation; the logit test requires NumPy. Different prefill and decode schedules can round differently and exchange nearly tied experts, so full logits need not match across schedules. Include task quality, not state replay alone. With the server started at `--ctx 49152`, the following checks sixteen facts in a 31K-token story and a correction turn that reuses the prefix:

```sh
python3 tests/test_server_story.py --url http://127.0.0.1:8000 \
  --model qwen3.8-flash-next --output .workspaces/qwen-story
```

### Continuation quality

For official continuation fixtures, create `.workspaces` if needed and build `make quality-score`, then run from the repository root:

```sh
gguf-tools/quality-testing/score_official MODEL.gguf \
  gguf-tools/quality-testing/data/qwen38-flash-alibaba-100/manifest.tsv .workspaces/qwen-short.tsv 4096
gguf-tools/quality-testing/score_official MODEL.gguf \
  gguf-tools/quality-testing/data/qwen38-flash-alibaba-long/manifest.tsv .workspaces/qwen-long.tsv 32768
```

Repeat each scorer command with `--quality` appended after the context argument; for the long set also repeat with `--continued-prefill 1` and `--continued-prefill 256` appended. These fixtures use the no-thinking template; use `--rendered-prompt` instead when a fixture already contains the complete chat template. The [quality guide](../../../gguf-tools/quality-testing/README.md#7-qwen38-spark-baseline) describes collection settings and hosted-checkpoint limitations.

### Vision

For vision parity, build the encoder fixture and compare its GGUF weights with the CPU Hugging Face reference:

```sh
nu tools/zig-build vision-encoder-test
uv run --with numpy --with torch --with torchvision --with pillow \
  --with safetensors --with transformers --with gguf tests/qwen4_vision_ref.py \
  --binary ./zig-out/bin/test-qwen4-vision-encoder \
  --snapshot /path/to/HF-checkpoint \
  --mmproj gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --image tests/vision-fixtures/qwen38/orbit.png \
  --image tests/vision-fixtures/qwen38/maple.png \
  --image tests/vision-fixtures/glm53/diagram.png \
  --image tests/vision-fixtures/glm53/text.png \
  --image tests/vision-fixtures/glm53/earth.jpg \
  --image tests/vision-fixtures/glm53/screenshot.png \
  --json-report .workspaces/qwen38-vision-parity.json
```

Transformers must include `qwen4_exp`; repeat `--image` for more cases. The original single-image `--out` embedding dump is also supported. Implementation parity uses a minimum per-token cosine of 0.99 and gates the default exit status; `--require-quality` also fails on the GGUF-versus-original quality comparison. Passing implementation parity alone does not establish original-checkpoint image quality.

For CLI image turns and text follow-ups in ordinary and MTP modes, build `make ds4` and run:

```sh
uv run tests/test_qwen4_cli_vision.py --model /path/to/main-with-mtp.gguf \
  --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --image /path/to/first.png --image /path/to/second.png
```

This checks completion and saves responses and diagnostics; it does not grade image content.

### Agent and steering

Directional steering covers all 48 trunk layers; see [steering controls](../../../dir-steering/README.md). Model-backed steering and agent recovery checks are:

```sh
python3 tests/test_qwen4_steering.py --model /path/to/main-with-mtp.gguf
make ds4_agent_test
./ds4_agent_test --full-context-save /path/to/main-with-mtp.gguf
make test-web-recovery
```

An agent can save a full text transcript without a KV payload, then restart with a larger `--ctx`, `/switch` to it and `/compact` to recover room. Image sessions retain their save restriction. These tests qualify only the exercised configurations, not every model, schedule or image task.
