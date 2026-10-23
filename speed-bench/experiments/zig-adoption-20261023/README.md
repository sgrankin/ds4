# Adopt the Zig implementation

Parent: `5a1ac350` (`zig-rewrite` / `zig-engine`). Cleanup is a separate change;
`main` stays at `ec5424de5562`. No numerical algorithm or default approximation
setting is intentionally changed.

## Ownership

All five program entry points are Zig: CLI, server, Agent, benchmark and Eval.
Server signal flags have one Zig owner, shared with the C signal/I/O helpers.
Qwen and GLM retain their C model engines, selected from GGUF metadata, behind
these same frontends. Real Metal objects/ARC and pipeline catalogues remain
Objective-C; MSL and third-party libraries remain in their original languages.

Removed the `-Dnative=false` migration build, `DS4_NATIVE_*` selectors and duplicate
C implementations already replaced by Zig. Deleted unused C CLI/benchmark/Eval,
Web, SSD configuration, Engram, prompt-prefix and state/KV-cache implementations.
Historical implementations remain in version history. C model/platform paths
still used by retained engines remain. This is not a port of Qwen/GLM to Zig.

Make now delegates to `tools/zig-build` and copies the five executables to the
repository root for existing launchers. GGUF quantizers remain standalone; the
continuation scorer links the same production engine through `quality-score`.

The proposed `zig/` to `src/` directory move is deliberately separate and has
not happened in this change. A subsequent commit renames the current tree to
`src/`; historical source paths below and in earlier reports refer to their
recorded revisions.

## Qualification

M5 Max, macOS 27, Zig 0.17 / Apple Clang 21. Frozen parent binaries and raw
commands/logs/results are local under `.workspaces/zig-adoption-cleanup-20261023/`.

- All five executables build, return successful `--help`, and define one `main`.
- 22 existing ReleaseFast host fixtures pass: agent/server, request boundary,
  WebSocket framing, session/cache/store, suffix transactions and rollback,
  tokenizer/GGUF, images, Web recovery and resident-route helpers.
- Seven focused Debug fixtures pass, including native suffix transaction tests.
- The simplified `make all` and `make test` workflows pass end to end, including
  Eval case/extractor checks and the public host/frontend fixtures.
- Real DeepSeek, GLM and Qwen continuations match the frozen build exactly,
  including every reported top-20 logit/logprob value. The short prompt is
  `Reply with OK.`; each model emits `OK` then its own stop token. This is a
  smoke/parity check, not broad model-quality or performance evidence.
- Qwen kernel suite and GLM-5.3 KDA GPU suite pass.
- Live Responses/WebSocket integration passes 26/26 required checks, including
  SDK use, actual tool retrieval, concurrent sockets, durable restart and clean
  shutdown. Answer quality is 12/15, retaining the same pre-existing three
  `survived` spelling/boolean errors documented in the earlier WebSocket report.

Known failure retained: default `test-deepseek41-metal` stops at quantization
mode 3, index 5428 (`11.25 != 7.5`) in **both** frozen and cleaned binaries.
Quantization code/shader and launch geometry are unchanged. An exact FP4 rounding
midpoint is sensitive to reciprocal multiplication under Metal fast math; CPU
arithmetic reproduces the observed alternate rounding, but this is an explanation
candidate, not compiler-level proof. No kernel fix is bundled into this cleanup.

One fully resident GLM baseline attempt was stopped by the pressure guard. The
process exited; the monitor recorded a timeout while waiting for SIGKILL. GLM's
supported SSD path then completed without pressure stops. Failed initial fixture
invocations with missing scratch arguments/directories are retained locally;
corrected commands pass in fresh directories.

## Reproduce

Build the parent in its own checkout to preserve a reference installation, then
build this revision:

```sh
tools/zig-build --prefix .workspaces/zig-adoption-cleanup-20261023/after \
  install server-test agent-test serving-boundary-test websocket-frames-test \
  agent-context-suffix-test suffix-host-test qwen-kernels-test glm53-kda-test
make
make test
```

The model comparisons use `ds4 --model gguf/MODEL --metal --ctx 512 --nothink
--temp 0 --tokens 8 -p 'Reply with OK.' --dump-logprobs OUTPUT.json` plus:

| Model | Additional settings |
| --- | --- |
| `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` | `--ssd-streaming --ssd-streaming-cache-experts 24GB --prefill-chunk 128` |
| `GLM-5.3-Flash-Q2.gguf` | `--ssd-streaming --ssd-streaming-cache-experts 24GB` |
| `Qwen3.8-Flash-Next-Q4.gguf` | Environment `DS4_METAL_NO_RESIDENCY=1 DS4_METAL_NO_MODEL_WARMUP=1`; no SSD flag. Its 95.37 GiB n-gram table remains disk-only. |

Run one model job at a time via `python3 -B speed-bench/ideas/pressure_run.py
FRESH_GUARD_DIRECTORY -- COMMAND...`. Compare the parsed before/after JSON
objects with Python's standard `json` module. Keep these dumps local.

For live integration, reuse the existing campaign and its dependencies:

```sh
python3 -B speed-bench/ideas/pressure_run.py FRESH_GUARD_DIRECTORY -- \
  .workspaces/responses-ws-venv/bin/python -B tests/test_responses_websocket.py \
  --server .workspaces/zig-adoption-cleanup-20261023/after/bin/ds4-server \
  --output FRESH_RESULT_DIRECTORY --sdk
```

See [the earlier WebSocket report](../responses-websocket-20261023/README.md)
for dependency setup, contract scope and existing answer-quality controls.
