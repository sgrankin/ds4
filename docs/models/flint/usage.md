# Using Flint on Metal

Flint loads the original INT4 safetensors directory, not a GGUF file. Native
profiles support M4 Pro, M5 Pro and M5 Max; physical hardware qualification so
far is on M5 Max only. The [Apple Silicon manual](apple-silicon.md#profiles-and-evidence)
describes the profiles and qualification limits. The [reference manual](reference-manual.md)
and [runtime implementation](implementation.md) explain the model and its native execution.

## Download and serve

From the repository root on supported Apple Silicon, use Zig 0.17, Apple Clang
and the macOS SDK; project scripts use Nushell (0.116.1 is tested). See the
[native build guide](../../../src/README.md#build-and-run) and [Metal setup](../../METAL.md#build-and-run)
for prerequisites. Download the pinned checkpoint and build the server and
comparison executable:

```nu
./download-model.nu flint
./tools/zig-build ds4-server flint-smoke
```

The build installs `ds4-server` and `flint-smoke` in `zig-out/bin`; the
launcher below uses that directory. `flint-int4` is an equivalent download target. The command downloads five
safetensors shards and seven metadata and tokenizer files into `gguf/flint-int4`,
pinned to Hugging Face revision `3b93f9053b530488405ae9fca51d4900d6ce21f0`.
It checks each file's size and SHA-256; rerun it to resume. `DS4_GGUF_DIR`
changes the parent directory, and the download leaves `ds4flash.gguf` unchanged.
Keep the metadata, tokenizer and shards together when using an existing directory.

From Nushell, start the server with the downloaded directory:

```nu
let model = './gguf/flint-int4'
./ds4-local --bin-dir ./zig-out/bin --server flint --model $model --port 8001
```

The `flint` preset requests two conversation slots, a 262144-token context,
prefill chunks up to 4096 tokens, and durable Responses history in
`.workspaces/response-store`. It removes inherited DeepSeek SSD and suffix
settings. Override ordinary arguments, for example with `--ctx 8192`, or set
`DS4_SERVER_RESPONSE_STORE_DIR` to choose a separate history directory.
`--bin-dir` is optional when using root executables. The server model ID is
`flint`; `flint-int4` is also accepted. Sampling defaults to temperature 1,
top-k 20, top-p 0.95 and min-p 0, matching the original harness. The plain
`ds4 --model DIR` CLI also supports native Flint inference.

The context limit is 262144 tokens. Weights are resident, and each active
session allocates global KV in 65536-token banks: the first bank is about
1.78 GiB, in addition to prefill scratch and the local checkpoint. Startup logs
report planned per-session buffers. Durable Responses history is independent of
tensor-cache support; Flint does not yet support model-state SSD checkpoints,
suffix relocation or vision. See [runtime state](implementation.md) for details.

## Select a hardware profile

`--flint-profile auto` is the default for `ds4`, `ds4-server` and the server
preset. It recognizes the Metal device names `Apple M4 Pro`, `Apple M5 Pro` and
`Apple M5 Max` and rejects other devices. To select it explicitly:

```nu
./ds4-local --bin-dir ./zig-out/bin --server flint --model $model --flint-profile auto --port 8001
```

| Profile | Default and maximum prefill chunk |
| --- | ---: |
| `m4-pro` | 2048 tokens |
| `m5-pro` | 4096 tokens |
| `m5-max` | 4096 tokens |

`--prefill-chunk` can lower the chunk size; larger values are capped at the
profile maximum. The preset's 4096-token request therefore uses 2048 on M4 Pro.
An explicit `--flint-profile m4-pro`, `m5-pro` or `m5-max` can select a smaller
schedule on a larger supported device: M5 Max can run either Pro profile, and
M5 Pro can run M4 Pro. The engine rejects a larger resident grid and an M5 NAX
profile on M4. Such overrides exercise the code but do not establish performance
or residency on physical Pro hardware. See the
[profile qualification](../../../speed-bench/experiments/flint-profiles-20261023/README.md).

## Connect an agent

The built-in `ds4-agent` rejects Flint because it does not support its tool
format. Use an external agent connected through Responses HTTP or WebSocket.
For example, connect to `ws://127.0.0.1:8001/v1/responses` and send:

```json
{"type":"response.create","model":"flint","store":true,"input":"Remember code=EMBER-73. Reply OK.","max_output_tokens":256}
```

After `response.completed`, send the next input with the response ID from that
event as its parent. Take the ID from the event's `response.id` field:

```json
{"type":"response.create","model":"flint","store":true,"previous_response_id":"resp_FROM_COMPLETED_EVENT","input":"What code did I ask you to remember?","max_output_tokens":256}
```

Stored response and conversation history survives restart if the response-store
directory is retained. Multiple sockets can use independent conversation slots;
GPU work is serialized across sessions. Flint defaults to a mixed-prefill quantum
of 1024 tokens; `--mixed-prefill-quantum` changes it. For request behavior,
reasoning and tool handling, see the [runtime implementation](implementation.md)
and [shared server contract](../../SERVER.md#constrained-output-and-client-tools).

## Reproduce the reference comparison

Run one model job at a time under the existing pressure guard, and pause other
work during latency measurements. Keep results under `.workspaces/`. The
reference comparison requires Flint's MLX-enabled Python with `rec_mlx`
importable. For a source checkout at `../flint`, set `PYTHONPATH` to its
`engine/src` directory:

```nu
let model = '/path/to/flint-int4-model'
let flint_python = '/path/to/flint-python'
let output = '.workspaces/flint-comparison'
$env.PYTHONPATH = ($env.PWD | path join '..' 'flint' 'engine' 'src')
python3 speed-bench/ideas/pressure_run.py .workspaces/flint-reference-guard -- $flint_python speed-bench/ideas/flint_reference.py --model $model --output $output --quality
```

The [reference driver](../../../speed-bench/ideas/flint_reference.py) sets
`MLX_METAL_FAST_SYNCH=1` before import and uses the original engine, sampler
and empty-tool grammar. It writes two prompt fixtures, three fresh-prefix runs
per greedy/default mode with 128 generated tokens, and eight optional
teacher-forced FP32 vocabulary rows. Repeat 0 includes cold pipeline effects;
the original report compared the two warm repeats. Decode rate is
`(generated_tokens - 1) / decode_seconds`. This historical three-run procedure
reproduces that qualification; it is not a new five-sample performance
comparison. Follow the current [measurement methodology](../../lab/methodology.md)
for new comparisons.

Run the native executable against the identical token IDs, sequentially:

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/flint-native-short-guard -- ./zig-out/bin/flint-smoke $model ($output | path join 'short-prompt.json') ($output | path join 'native-short')
python3 speed-bench/ideas/pressure_run.py .workspaces/flint-native-default-guard -- ./zig-out/bin/flint-smoke $model ($output | path join 'short-prompt.json') ($output | path join 'native-short-default') default
python3 speed-bench/ideas/pressure_run.py .workspaces/flint-native-context-guard -- ./zig-out/bin/flint-smoke $model ($output | path join 'context-prompt.json') ($output | path join 'native-context')
python3 speed-bench/ideas/pressure_run.py .workspaces/flint-native-quality-guard -- ./zig-out/bin/flint-smoke $model ($output | path join 'teacher-suite.json') ($output | path join 'teacher-native') quality
python3 tools/flint-quality.py ($output | path join 'teacher-suite.json') ($output | path join 'teacher-reference') ($output | path join 'teacher-native')
```

The quality capture compares the target token and full vocabulary at positions
16, 32, 64 and 96 in each reference greedy continuation. Delta-NLL and KL use
fixed prefixes even if free-running replies diverge. Eight positions from two
prompts are a focused numerical probe, not broad coding-quality or long-context
coverage. The original qualification used native tool parsing without grammar
masks and does not measure the later XGrammar integration. Native and MLX random
streams differ. See the [original report](../../../speed-bench/experiments/flint-native-20261023/README.md),
[masked sampler fixture](../../../speed-bench/experiments/flint-masked-sampling-20261027/README.md)
and [constrained-packet qualification](../../../speed-bench/experiments/constrained-packets-20261027/README.md).

## Validate a profile

These commands exercise a schedule and fixture today; they do not by themselves
reproduce the historical performance qualification. First complete the reference
comparison above in the same Nushell session: it defines `$model` and `$output`
and creates `$output/short-prompt.json`. Alternatively, set those variables to
an existing model directory and a reference output directory containing that
fixture. Place the profile selector before the model argument:

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/flint-m5-pro-guard -- ./zig-out/bin/flint-smoke --flint-profile m5-pro $model ($output | path join 'short-prompt.json') ($output | path join 'm5-pro-short') default
```

The no-model fixture compiles the selected shaders and checks KV publication,
sampling and M4 portable matrix and attention arithmetic. The `scalar` mode
provides a one-token-at-a-time greedy control for the packet path:

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/flint-m4-fixture-guard -- ./zig-out/bin/flint-smoke --flint-profile m4-pro --compile-only
python3 speed-bench/ideas/pressure_run.py .workspaces/flint-m4-scalar-guard -- ./zig-out/bin/flint-smoke --flint-profile m4-pro $model ($output | path join 'short-prompt.json') ($output | path join 'm4-pro-short-scalar') scalar
```

The [WebSocket qualification driver](../../../tests/test_responses_websocket.py)
accepts `--flint-profile` and forwards it
to both initial and restarted servers. Its Flint cases include a prompt longer
than 4096 tokens followed by an append after decoding, as well as tools and
independent conversations. The [profile report](../../../speed-bench/experiments/flint-profiles-20261023/README.md)
records commands, results and remaining physical hardware checks.
