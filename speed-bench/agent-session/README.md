# Agentic session benchmark

This is a synthetic coding session through the real `ds4-agent` tool loop,
paired with a fixed token replay for controlled SSD performance experiments.
It complements short interactive prompts; it does not replace them.

The fixture is a standard-library Python payment reconciliation project. Three
user turns require inspecting code, observing failing tests, fixing money,
refund and retry handling, adding a date-filter API with tests, and producing
checked reports from 275 CSV records. Independent held-out checks run outside
the editable project after every turn. Existing test definitions and data must remain
unchanged; additional tests may be added. Task success, tool calls, context growth, turn wall time, prefill and
decode times are recorded. Model reasoning is enabled, matching agent defaults.

## Live task

From the repository root:

    python3 speed-bench/agent-session/live.py /tmp/agent-session-RUN

The runner copies the fixture, executable and Metal sources, isolates agent KV
files, fixes datetime/timezone and seed, and uses greedy sampling. Tools run in
the copied project. This runs actual model-selected commands; use a disposable
workspace. It never points tools at the ds4 checkout. `--env NAME=VALUE` enables
an experimental path; `--cache-gb N` sets the total expert-cache target.
`--prefill-chunk N` controls the agent prefill allocation; the replay equivalent
is `--candidate-env DS4_REPLAY_PREFILL_CHUNK=N`. The timeout is 1200 seconds
by default. Validation time is excluded from turn
wall time, but startup is reported separately. Tool and host work are included
in turn wall time; model timings alone are not time to task completion.

The live run must pass all three task stages and execute at least six tool calls
before emitting summary.json and replay.txt. A failure is a failed benchmark,
not a faster result. Live variants may take different valid tool paths: compare
success and elapsed time, and do not claim identical-workload speedups unless
the recorded token streams match.

## Fixed session replay

    make speed-bench/agent-session/replay
    python3 speed-bench/agent-session/replay_abba.py /tmp/session-ABBA \
      --replay /tmp/agent-session-RUN/replay.txt \
      --candidate-env DS4_METAL_V41_BATCH_SHARED_OVERLAP=1

Repeat `--candidate-env NAME=VALUE` to test a combination against a control
with all those flags unset. Record single-factor studies before combinations.

Each replay retains the recorded system prefix, every user/tool-result prefill
boundary, and every generated token. Generated tokens are teacher-forced one
at a time through the ordinary decode API. It preserves evolving KV and expert
cache state within the session. Separate-process ABBA freezes executable,
shaders and tokens; it aborts if full-vocabulary logits at any phase boundary
or the complete final continuation snapshot differ. Large binary comparison
artifacts are hashed and deleted after checking; hashes and timings persist.

The replay excludes tools, sampling, tokenization and terminal rendering; use
it to attribute model runtime, and confirm useful gains with live tasks and
interactive probes. It does not simulate pauses, saved-session restoration,
images, compaction or speculative decoding. Trace extraction rejects compaction
and incomplete token blocks rather than silently fabricating a workload.
The token recording is specific to the DS4.1 tokenizer/model configuration.

The checked recording is `session-v1.txt`, SHA256
310159cecfb6688d65967cf30279e1626c15571dbedd02838549f0d1ff14f368.
It was captured from the passing live baseline on an M5 Max /128 GiB with
DeepSeek-V4.1-Flash-Q2, SSD streaming, context100000 and automatic cache sizing.
It includes20 tool calls (7 reads,6 shell commands,3 edits,3 directory lists,
and1 write),14 generation rounds, an1859-token system prefix,
3371 generated tokens and final context8710. Tool/user suffixes range62-826
tokens. Baseline user-turn wall times158.705/93.843/39.166s, total291.714s;
startup17.420s is separate. Model prefill94.280s and decode196.339s account for
nearly all turn time; the remainder includes tools and host overhead.
See baseline-live-v1.json and baseline-checks-v1.json for independent success
checks. This is a small synthetic working session, not coverage of long-context
compaction, every tool, or arbitrary coding-task quality.
`test_trace_session.py` checks extraction boundaries and rejection of
incomplete/unsupported traces.
