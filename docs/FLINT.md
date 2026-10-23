# Flint on Metal

The native backend loads Flint's original affine INT4 safetensors directory and
serves it through the shared Responses HTTP/WebSocket server. Zig owns the model
and sessions; the existing Objective-C Metal driver executes the kernels.
Serving does not load Python or the MLX runtime.

The first supported checkpoint is Recursive's group-64 INT4 export, with INT8
embeddings, from [recursive-oss/rec](https://github.com/recursive-oss/rec) revision
`e53663b262af4cb1ca498e596ca8b1a2d79e0800`. Keep `config.json`, `tokenizer.json`,
`flint-int4.json`, and its safetensors shards together. This loader accepts that
directory format, not GGUF. The resident decoder is qualified on **M5 Max only**;
other chips need a separately qualified schedule.

## Serve

From Nushell, set the path to your existing Flint model directory:

```nu
nu tools/zig-build ds4-server flint-smoke
let model = '/path/to/flint-int4-model'
./ds4-local --bin-dir ./zig-out/bin --server flint --model $model --port 8001
```

The `flint` preset selects two conversation slots, a 262144-token context,
4096-token prefill chunks, and durable Responses history at
`.workspaces/response-store`. It removes
inherited DeepSeek SSD/suffix settings. Override its ordinary arguments, for
example `--ctx 8192`, or set `DS4_SERVER_RESPONSE_STORE_DIR` for a separate
history directory. `--bin-dir` is optional when using the root executables.
The server model ID is `flint`; `flint-int4` is also accepted. Sampling defaults
are temperature 1, top-k 20, top-p 0.95 and min-p 0, matching the original harness.

The built-in `ds4-agent` does not yet support Flint's tool format and rejects
this model. Use an agent connected to the server. The plain `ds4 --model DIR`
CLI also supports native Flint inference.

Connect to `ws://127.0.0.1:8001/v1/responses` and send:

```json
{"type":"response.create","model":"flint","store":true,"input":"Remember code=EMBER-73. Reply OK.","max_output_tokens":256}
```

After `response.completed`, send only the next input and its response ID:

```json
{"type":"response.create","model":"flint","store":true,"previous_response_id":"resp_FROM_COMPLETED_EVENT","input":"What code did I ask you to remember?","max_output_tokens":256}
```

Stored response/conversation history survives restart when the response-store
directory is retained. Multiple sockets can use independent conversation slots.
GPU work is serialized across those sessions: the resident kernel's threadgroup
barriers require exclusive execution. The default mixed-prefill quantum is
1024 tokens for Flint; `--mixed-prefill-quantum` can change it.

## Capabilities and state

- Text, native tokenizer/prompt rendering, model-controlled reasoning, tool
  schema rendering, and structured tool-call parsing use the shared protocol.
  Omit `reasoning` effort and `enable_thinking` controls; Flint rejects those
  switches because its template lets the model decide when to reason.
- Generation uses packets of up to 8 tokens on the GPU. Greedy and common
  filtered sampling use the native packet path; unsupported sampling settings
  use scalar evaluation. The native random stream differs from MLX's keys.
- Prefix reuse is in RAM. Each session retains its own BF16 KV and a bounded
  local-ring/logit prefill checkpoint. After generation, extending or rewinding
  within that continuation restores the checkpoint and recomputes the tail
  through prefill, following the original harness's arithmetic distinction.
- Flint does not yet have model-state SSD checkpoints, suffix relocation,
  vision, or grammar/schema masking. Durable Responses history is independent
  of these tensor-cache capabilities.

Weights are resident. Global KV allocates in 65,536-token banks, with about
1.78 GiB for the first bank per active session; prefill scratch and the local
checkpoint add to that. Startup logs report the planned per-session buffers.
The model context limit is 262,144 tokens.

## Reproduce the comparison

Run one model job at a time under the existing pressure guard and pause other
work during latency measurements. Results belong in `.workspaces/`, not source
control. Use Flint's MLX-enabled Python with `rec_mlx` importable; for the source
checkout, set `PYTHONPATH` to its `engine/src` directory. The reference runtime
is used only for this independent comparison.

```nu
let model = '/path/to/flint-int4-model'
let flint_python = '/path/to/flint-python'
let output = '.workspaces/flint-comparison'
$env.PYTHONPATH = ($env.PWD | path join '..' 'flint' 'engine' 'src')
python3 speed-bench/ideas/pressure_run.py .workspaces/flint-reference-guard -- $flint_python speed-bench/ideas/flint_reference.py --model $model --output $output --quality
```

The driver sets `MLX_METAL_FAST_SYNCH=1` before import, then calls the original
`Engine`, `Sampling`, and empty-tool grammar. It writes two prompt fixtures,
three fresh-prefix runs per greedy/default mode with 128 generated tokens,
and optional eight teacher-forced FP32 vocabulary rows. Repeat 0 includes cold
pipeline effects; compare the two warm repeats separately. Decode rate uses
`(generated_tokens - 1) / decode_seconds`, matching the original harness.

Run the native executable against those identical token IDs, sequentially:

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/flint-native-short-guard -- ./zig-out/bin/flint-smoke $model ($output | path join 'short-prompt.json') ($output | path join 'native-short')
python3 speed-bench/ideas/pressure_run.py .workspaces/flint-native-default-guard -- ./zig-out/bin/flint-smoke $model ($output | path join 'short-prompt.json') ($output | path join 'native-short-default') default
python3 speed-bench/ideas/pressure_run.py .workspaces/flint-native-context-guard -- ./zig-out/bin/flint-smoke $model ($output | path join 'context-prompt.json') ($output | path join 'native-context')
python3 speed-bench/ideas/pressure_run.py .workspaces/flint-native-quality-guard -- ./zig-out/bin/flint-smoke $model ($output | path join 'teacher-suite.json') ($output | path join 'teacher-native') quality
python3 tools/flint-quality.py ($output | path join 'teacher-suite.json') ($output | path join 'teacher-reference') ($output | path join 'teacher-native')
```

Quality capture uses positions 16/32/64/96 in each reference greedy continuation.
It measures the same target token and full vocabulary at each fixed prefix;
delta-NLL and KL do not depend on whether free-running replies later diverge.
Eight positions from two prompts are a focused numerical probe, not a broad
coding-quality or long-context evaluation. Native serving parses tool output
but does not reproduce the reference grammar masks or MLX's random sequence.

The [prefill source notes](../src/metal/flint/prefill/README.md) and
[resident decoder notes](../src/metal/flint/README.md) document the kernel
boundary, BF16 stages, source attribution and licenses.
