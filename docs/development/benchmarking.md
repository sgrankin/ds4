# Benchmarking tools

Choose the question, baseline and timed scope first. The [lab methodology](../lab/methodology.md)
owns sample counts, alternating order, quality interpretation and reporting;
this page explains the tools. Compare matching checkpoints, quantization,
context and sampling. Record whether weights are resident, streamed or
distributed, and distinguish cold startup from a warm expert or OS cache.

## Context sweeps

```sh
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128
```

Each prefill rate covers only the newly added interval; generation is a fixed
greedy, non-EOS probe. The program normally restores a memory snapshot after
each probe. Network TP, pipeline runs and snapshots beyond its memory limit
instead replay the prefix; replay time is not continued-prefill speed.
`--step-mul F` selects exponential spacing. CSV includes prefill and generation
rates and snapshot size where available. The prompt is the cleaned public-domain
*I Promessi Sposi* in [speed-bench](../../speed-bench/README.md).

The native V4.1 [prefill ceiling](../../src/engine/deepseek41/layout.zig) depends
on configured context: 2048 below 8192 tokens, 4096 below 16384, and otherwise
8192. Explicit chunks and workspace/disable settings can lower it; a server's
scheduling quantum can also split the actual forward calls. Retained legacy
Flash uses 4096, CUDA TP historically used 2048, and long PRO prompts may use
8192. GLM selects
its own chunks and rejects `--prefill-chunk`. The strict DeepSeek API-vector
test pins 2048; that does not set the default for every benchmark.

When comparing sessions, report client latency and aggregate throughput, and
identify ordered fallback. Compare speculative decode with ordinary decode;
a draft can cost more on an unpredictable prompt. A comparison of resident TP
against streamed single-host inference includes a capacity effect as well as
parallelism. The retired original Flash Q2 historical sweeps are in
[M5 Max CSV](../../speed-bench/m5_max.csv) and [Spark CSV](../../speed-bench/gb10.csv),
with [M5 chart](../../speed-bench/m5_max_ts.svg) and [PRO Q2 M3 Ultra chart](../../speed-bench/pro_model_m3_ultra_ts.svg).
They are not current-model performance claims.

## Paired quality probe

`speed-bench/ideas/quality_probe.c` teacher-forces a frozen official
continuation. A baseline run stores full raw logits; a candidate reads those
records and performs one forward pass per position. Token IDs and a hash of the
complete prompt and truncated continuation must match the reference. NLL uses
the unfiltered temperature-one softmax; KL compares the complete vocabulary.
The probe also reports total variation, argmax changes, per-prompt deltas and
tail KL. A raw byte comparison separately checks disabled-mode parity, because
KL cannot detect a constant shift of all logits. Each row is scored before its
target token is fed; `decode_ms` evaluates that token and produces the next
row's distribution. Decode-only changes start with exact prefill.

The `screen.tsv` and `heldout.tsv` manifests contain separate non-thinking
continuations from tracked V4.1 fixtures. They screen changes but cannot
establish thinking or tool-task quality. Optional policy logits require a build
with `-DDS4_IDEAS_POLICY_HOOK` against the penalty runtime; these logits precede
temperature and filtering, and are not the sampled distribution.

```sh
# Both inputs must be archived C checkouts with the compiled .o files.
nu speed-bench/ideas/build_probe.nu ARCHIVED_C_CHECKOUT speed-bench/ideas/artifacts/quality-probe-base
nu speed-bench/ideas/build_probe.nu ARCHIVED_C_CANDIDATE_CHECKOUT speed-bench/ideas/artifacts/quality-probe-resident
./speed-bench/ideas/artifacts/quality-probe-base --self-test
python3 speed-bench/ideas/pressure_run.py speed-bench/ideas/artifacts/screen-pressure -- \
  python3 speed-bench/ideas/run_screen.py speed-bench/ideas/artifacts/resident-screen \
  --baseline speed-bench/ideas/artifacts/quality-probe-base \
  --candidate speed-bench/ideas/artifacts/quality-probe-resident \
  --manifest speed-bench/ideas/screen.tsv --cases 16 --tokens 32
```

Build before final timing. The runner freezes binaries and external Metal
sources, records hashes and settings, checks disabled-mode and repeat-baseline
byte parity, and retains per-position scores. Large binary references stay
ignored locally. Probe timings exclude metric arithmetic and reference I/O but
remain instrumented: use them for screening, then measure profile-off latency
and practical tasks. Prompt-bootstrap intervals are descriptive and do not
replace independent task coverage.

## Metal dependency profiles

`DS4_METAL_ENCODER_TIMELINE=PATH` records command-buffer and encoder intervals,
but it changes the schedule it observes. The scheduler closes ordinary reused
encoders between acquisitions so each pass can receive counter samples. Each
command buffer also owns sample storage, and completion logging shares a mutex
with pipeline submission. A candidate that submits fewer command buffers can
therefore remove profiling overhead as well as runtime work. Use the trace to
identify ordering, overlap and candidate dependencies; an instrumented gap
reduction is not a predicted saving in ordinary execution. The implementation
is in [command scheduling](../../src/metal/command_schedule.zig) and the
[Metal timeline hooks](../../src/metal/ds4_metal.m).

For a lighter check, use `DS4_METAL_CB_TIMES=1` without the encoder timeline,
alongside a bounded CPU sample. This preserves ordinary encoder grouping and
reads completed command buffers' driver and GPU timestamps. Detailed rows stop
after 400 waits, so prefill can consume the allowance. The finish trace's
`encode` field includes intervening flushes and waits; it is not pure CPU
encoding time. Its relative `gpu start` field also compares a Metal timestamp
with a host monotonic clock; observed clock offsets can make that value invalid
as a queue-delay estimate. Use durations within one clock domain instead.
`DS4_METAL_GPU_BUSY_PROFILE=1` reports accumulated command-buffer
GPU spans, which are not hardware occupancy. These diagnostics help locate
waiting, while profile-free alternating runs establish performance.

## Guarded execution

Keep useful workspaces under `.workspaces/` and raw results under ignored
`speed-bench/` paths. Check for other heavy jobs, then wrap every model-opening
command, even a tokenizer diagnostic:

```sh
python3 speed-bench/ideas/pressure_run.py speed-bench/ideas/artifacts/RUN-pressure -- \
  COMMAND ARGUMENTS
```

The guard serializes cooperating invocations, monitors pressure and swap, and
cleans up its owned process groups after interruption or sustained pressure. It
cannot stop an unguarded competing job. Retain failed runs and guard logs; large
RAM use alone is not failure. The guard uses a file lock and records its observations alongside the run. Its
tests cover sampling, process cleanup, probe timeouts and PID reuse.
