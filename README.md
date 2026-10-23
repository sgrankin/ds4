<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

**DwarfStar** aims to be the best way to run a few excellent large
language models on consumer hardware (that is, hardware that people
can actually own). To reach this goal, we are building
a small native inference engine optimized first for
**DeepSeek V4 Flash** (including the experimental vision model),
**DeepSeek V4.1 Flash** (Metal, and text inference on CUDA),
and additionally **GLM 5.2 and 5.3**, **GLM 5.3 Flash** and
**DeepSeek V4 PRO**, and **Qwen3.8 Flash Next** (Metal and CUDA). The code is self-contained and
deliberately narrow, not a general GGUF runner: you need to use the
GGUF files the project produces, that are part of the project
itself.

We test things in integration: model loading, prompt rendering,
tool calls, KV state, the HTTP server, and the coding agent are built and tested together.
The repository also includes tools and data for GGUF, imatrix, quality, and speed.

## Supported hardware

* **Metal**, the primary target, on Macs with 64 GB (for Qwen Flash next) and 96 GB or more (for DeepSeek and GLM). Smaller machines can use SSD streaming. SSD streaming is also needed in order to run very large models such as full GLM 5.x (not Flash) on 128GB systems.
* **NVIDIA CUDA**, the DGX Spark is our main gaol. DwarfStar also supports multi-GPU systems that are not supported by other backends, for instance it can run DeepSeek v4 Flash on Ada Lovelace cards.
* **ROCm** on Strix Halo systems such as the Framework Desktop.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

**Model support is intentionally opportunistic**. The project follows the best open
weights for useful local machine sizes, especially 128 GB laptops and 256/512 GB
workstations. A model may be removed when a better replacement arrives.

# So, what can I do with this software?

* You can run a very capable models in your consumer hardware, a MacBook, a DGX Spark, or a Strix Halo for example. Even if you have not enough RAM, with SSD streaming, you can run it at a decent speed.
* You can use multiple CUDA cards as a multi-user LLM server. Ada Lovelace, including L40S, is supported: newer models can run here even when their other inference implementations require newer GPUs. Our eight-L40S Flash setup has reached about 126 t/s aggregate generation with 16 sessions.
* Using two 128 GB Macs connected with RDMA, you can run 4-bit DeepSeek Flash or GLM 5.3 Flash with tensor parallelism. Larger GLM 5.2 quants need larger machines, such as Mac Studios.
* You can also use pipeline paralellism to glue together multiple systems to sum their RAM and run larger models.

## Motivations

* Capable open-weight models now fit on high-end personal machines.
* DeepSeek V4 Flash and PRO, GLM 5.2, tolerate aggressive routed-expert quantization.
* Compressed KV caches and fast local SSDs make long contexts practical.
* The idea of an inference system specialized for a few models.

# AI full disclosure

* This software is developed with **strong assistance from AI coding agents** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The software is currently very fast changing. Consider it beta quality.
Before each release, a big QA run is executed, however instabilities
and regressions are definitely possible.

# How to use this project?

I (Salvatore) believe that the way projects should be shipped and used changed because of AI. The main differences today are:

1. With AI, users can modify the software in significant ways with low efforts, costs, and even lacking deep domain knowledge about the task they want to accomplish. For instance, a DwarfStar user with a specific hardware setup can ask a coding agent to improve the inference speed of this software for the specific hardware setup, asking the model to reach the maximum prefill and generation speed without impacting correctness, and also asking to do a deep QA pass.
2. Similiarly, because of "1", software may be shipped in a different way than before. It must be more a working template for the biggest use cases, without trying to cover every possible setup. If DwarfStar showcases a few good implementations of tensor parallel execution, the code will work as a rail for implementing the same feature in specific conditions, for a new model, and so forth.

So, while this project attempts to be usable for the featured models and the most common hardware setups, I ask you, if you have access to coding agents, to consider using coding agents as an interface to discover the project, make modifications, create personalized setups. This way you can likely do more than what we ship, and certain things that are not documented or implemented, and that you require, are potentially very easy to achieve.

## Start Here

```sh
git clone https://github.com/antirez/ds4.git
cd ds4
```

Choose your build. The platform guides cover prerequisites, memory sizing,
and hardware-specific setups:

| Platform guide | Build |
| --- | --- |
| [Metal on Apple Silicon](docs/METAL.md) | `make` |
| [DGX Spark](docs/DGX_SPARK.md) | `make cuda-spark` |
| [Strix Halo / Framework Desktop](docs/STRIX_HALO.md) | `make strix-halo` |
| [One or more CUDA cards, including Ada/L40S](docs/CUDA_MULTI_GPU.md) | `make cuda-generic` |

This branch also offers a Metal build with handwritten Zig modules:
`tools/zig-build`, with programs in `zig-out/bin`. See the
[Zig module guide](src/README.md) for scope, model backends, presets and measurements.

For a first run on a 96 or 128 GB machine, download DeepSeek V4 Flash Q2:

```sh
./download_model.sh ds4f-q2
```

Downloads go in `gguf/`. Repeat the command to resume an interrupted download.
Leave memory for the context and runtime buffers as well as the model.
See [other models](docs/MODELS.md) or use [SSD streaming](docs/SSD_STREAMING.md)
on a smaller Mac.

## Everyday Use

Once built and with a model downloaded:

```sh
./ds4
./ds4 -p "Explain Redis streams in one paragraph."
./ds4-agent
./ds4-server --ctx 32768
```

The default model is `ds4flash.gguf`, a link updated by main-model downloads.
Pass `-m FILE` to choose explicitly. Commands normally run from the repository
root; use `--chdir /path/to/ds4` when launching elsewhere.

The server listens at `http://127.0.0.1:8000` by default; see [serving](docs/SERVER.md)
for API access and multiple sessions.

The interactive CLI keeps a multi-turn conversation. Use `/help`, `/read FILE`,
`/ctx N`, and `/quit`. Ctrl+C interrupts generation and returns to the prompt.
Run each binary with `--help` for its full options.

### Native coding agent

`ds4-agent` runs inference directly, without a separate HTTP server. It keeps
the token history and live model state together, shows prefill progress, and
uses the model's native tool format. DeepSeek and GLM have their own templates.

Our [recipe journal](recipes/README.md) describes the local improvements, measurements,
tradeoffs and unsuccessful experiments, with steps for adapting them to other runtimes.

Local V4.1 opt-ins can be enabled with the small `ds4-local` wrapper:

```sh
./ds4-local persistent-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
./ds4-local persistent-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --live-context-file .workspaces/agent-context.txt
./ds4-local resident-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
./ds4-local --server context-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
./ds4-local --server persistent-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
./ds4-local --server resident-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
```

`persistent-fast` combines approximate suffix reuse with resident-expert
substitutions during decoding and selected prefill, in either the direct agent
or the server. `context-fast` enables suffix reuse with native expert routing;
`resident-fast` uses up to three resident-expert substitutions with native
routing weights. All use a 64 GB expert-cache target, context 8192 and prefill
chunk 512. Later CLI arguments override those defaults. The script contains
the environment settings explicitly and leaves thinking mode unchanged.

The direct agent compacts automatically under context pressure, or on `/compact`:
it generates a task-state summary and keeps a recent verbatim tail. With
`context-fast` or `persistent-fast`, it attempts suffix reuse during that rebuild
without requiring a live context file. The latest user message and everything
after it remain fresh. When no suitable checkpoint or aligned span exists, it
recomputes normally. Add `--live-context-file PATH` for CLM-style editing: the
agent receives instructions and short editing recipes, and can rewrite its
mirrored conversation using ordinary file tools. Accepted edits update the
active context after tool execution or before the next user turn.

The mirror labels conversation spans with header lines such as
`[[DS4_TURN a3c91e704b82d6f0 role=assistant]]`. IDs are random and stay stable
during the running agent. Keep each ID with its turn when editing or moving it;
the file determines conversation order. New turns receive fresh IDs.
Restarting may assign different IDs. These headers are editing handles: they
are removed before the conversation is evaluated by the model. The protected
system/tool prefix stays unlabelled. Tool calls and their results can occur
within an assistant span.
Literal header-shaped lines in message bodies have a backslash display escape;
preserve that escape when keeping the text.

The editing workflow is to list the headers, inspect the selected older span
with a bounded read, then archive useful facts into a checkable file before
deleting redundant history or replacing it with a concise note. Preserve the
protected prefix, latest user request, chat delimiters and untouched text.
The recipes edit selected ranges in the existing file rather than printing the
whole context or retyping retained conversation. Batch related edits to specific
older IDs, then continue the task after the successful edit receipt.
Built-in compaction and agent-directed edits are separate ways to shorten
context; compaction itself does not automatically archive discarded facts into
files.

The [direct-agent qualification](speed-bench/experiments/agent-package-hour41-20261009/README.md)
exercises left/right SSD-backed grafts, fresh inventory/archive reads, and real
built-in compaction without a mirror file. It records a favorable edit-workflow
timing screen and the alignment fallbacks separately.

Selected-prefill resident routing is a further experimental opt-in:
`DS4_METAL_V41_RESIDENT_PREFILL_GAP=.90` uses the existing rank, mass and
substitution-count limits while preserving prefill's native mixture weights.
The [prefill-only screen](speed-bench/experiments/recipe-prefill-hour37-20261008/README.md)
measured a 10.6% warm-append saving and a 6.0% coding-task saving against bracket
means at three substitutions, with all first-code checks passing. It also
changes selected tails within initial prefills; full sweeps remain native.
The `persistent-fast` preset enables this knob alongside decoder
substitutions and suffix reuse. Set `DS4_METAL_V41_RESIDENT_PREFILL_GAP=0` to
disable its prefill substitutions. Other presets leave it unset. See the
screen for commands, timing drift and the likelihood tradeoff.
The agent `context-fast` and `persistent-fast` presets also enable
`DS4_AGENT_CONTEXT_SUFFIX_PREROPE=1` for repeated
aligned relocations. The earlier `DS4_AGENT_CONTEXT_SUFFIX_PREROPE=0` path
requires a live context file and native decoder routing.
This retains the normalized rotary lanes as original BF16 values before RoPE
and cache quantization, then applies the destination RoPE once. Private
approximate checkpoints can become relocation sources; they remain excluded
from exact-prefix restore and canonical KV export. This avoids accumulating
rotation and quantization error, while retaining the original context influence
in reused activations. The latest user message and at least 128 tail tokens are
evaluated freshly; unsupported geometry or missing originals rebuilds normally.
`DS4_AGENT_CONTEXT_SUFFIX_RIGHT=1` permits even right shifts after insertions
or growing replacements. This flag is off by default, and `persistent-fast`
enables it; set `DS4_AGENT_CONTEXT_SUFFIX_RIGHT=0` for left shifts only.
The server `context-fast` preset uses the same pre-RoPE implementation with an
opt-in fresh tail of 64 tokens; set `DS4_SERVER_CONTEXT_SUFFIX_FRESH_TAIL=128`
or `256` to retain more fresh context. It also keeps the complete latest user
message fresh. Reused states retain earlier conditioning, so measure output
quality for the intended edits. Server matching normally permits even left
shifts. `DS4_SERVER_CONTEXT_SUFFIX_RIGHT=1` also permits even right shifts
after insertions or growing replacements, using immutable pre-RoPE originals.
The server flag is off by default; the `persistent-fast` preset enables
it. Set `DS4_SERVER_CONTEXT_SUFFIX_RIGHT=0` to retain left shifts only. It uses
a separate private context-cache identity;
unsupported geometry or missing originals falls back to normal evaluation.
For example:

```sh
DS4_SERVER_CONTEXT_SUFFIX_RIGHT=1 ./ds4-local --server context-fast \
  --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
```

The [faithful-compaction measurements](speed-bench/experiments/recipe-compaction-hour38-20261008/README.md)
saved11.7–12.4% whole-task time on two complete READ/edit/WRITE/FINAL coding
workflows, with every first implementation passing161 checks. The
[right-shift experiment](speed-bench/experiments/recipe-rightshift-hour39-20261008/README.md)
records expansion tasks and the replay tradeoffs. Server reuse requires a `prompt_cache_key`
identity hint; Conversations supplies its ID by default. `resident-fast`
explicitly disables suffix reuse. The `persistent-fast` preset
combines left/right suffix reuse with resident substitutions in both decoding
and selected prefill. The
[combined fresh-task confirmation](speed-bench/experiments/combined-fresh-hour40-20261008/README.md)
saved 23.5% whole-task time against bracket means, or 21.5% against the faster
control, on previously unused interval-grouping and dependency-layer tasks.
All ten first writes passed 161 checks each. A repeated combined run matched
code and token usage and changed time by 0.04%; native timing drift was 4.93%.
These workflows used caller-supplied READ/WRITE/FINAL reasoning budgets 0/128/1;
the preset leaves thinking policy to the caller. Routing added a 12.2% saving
over suffix reuse alone on this workload. These measurements used the HTTP
server; they do not establish the direct agent's latency. This small screen
supports the opt-in package, not a general accuracy or speed guarantee. The
[combined four-session screen](speed-bench/experiments/suffix-resident-package-hour26-20261004/README.md)
passed all facts and reads; its additional whole-time saving was small relative
to control drift. The [combined CLM coding screen](speed-bench/experiments/clm-resident-package-hour28-20261004/README.md)
saved5.10% task time with all saved first codes passing161 checks; the ledger
benefited while LRU was slightly slower. These are additional routing gains
within each measured configuration, not percentages to add across reports.
The [fresh-task substitution curve](speed-bench/experiments/fresh-resident-curve-hour33-20261004/README.md)
qualified limits1,2,3 with all first codes passing161 checks. Limit3 repeated
code, public output, usage and routing counters exactly; its task mean was
187.3seconds, compared with193.2/194.2seconds for1/2. This supports the preset
limit3 on these tasks; the1/2 difference is small relative to measured drift.
Choose `context-fast` to keep native expert routing.

Two additional server opt-ins are available for complete full-input Responses
histories: `DS4_SERVER_CONTEXT_SUFFIX_CANONICAL=1` materializes native warming
requests from their tokenized history, and `DS4_SERVER_CONTEXT_SUFFIX_PAD=1`
may insert one newline in an old prefix when it enables a longer aligned graft.
Padding keeps the latest user message fresh and changes actual input usage by
one token. Server `context-fast` and `persistent-fast` enable both; set either
variable to0 to compare the earlier policy. Their combination saved20.56%
whole launcher wall in the [four-session qualification](speed-bench/experiments/parity-suffix-hour24-20261004/README.md),
with all edited facts correct and one retained warm-format failure.

All server presets default to localhost and two active slots. Later CLI flags
override defaults, for example `--batched-session 1 --port 8001`. They also set
`DS4_SERVER_RESPONSE_STORE_DIR` to `.workspaces/response-store`; override that
environment variable to choose another durable store. Stored Responses support
text/function-tool continuation through `previous_response_id`, GET and DELETE;
Conversations supports create/get/delete and text/function/reasoning item
append/list/get/delete. The [three-edit Conversation qualification](speed-bench/experiments/repeated-conversations-hour30-20261004/README.md)
passed all24 real reads/replies across native brackets and suffix reuse, saving
36.7% whole workflow and39.6% edited time. Four later grafts used approximate
pre-RoPE originals from SSD after restart; changed archive codes were read
correctly each round. Whole savings include an exact-cache warmup benefit.
The [eight-deletion follow-up](speed-bench/experiments/repeated-conversations-hour31-20261004/README.md)
used one Conversation, matched warm costs and saved58.2% whole/66.3% edited
time. All27 reads/replies passed, including seven approximate SSD-source
grafts and current archive codes after restart. This qualifies the measured
eight-edit workflow; it does not guarantee unlimited error-free relocations.
These histories survive KV eviction; they do not promise exact KV reuse after restart. See the [Responses report](speed-bench/experiments/responses-hour11-20261004/README.md),
[Conversations report](speed-bench/experiments/conversations-hour13-20261004/README.md)
and [installed SDK/Pi smoke](speed-bench/experiments/server-confirmation-hour17-20261004/client-results.md)
for the supported subset and limits.
The [actual Pi coding-agent qualification](speed-bench/experiments/pi-agent-preset-hour34-20261004/README.md)
ran the installed CLI through the `persistent-fast` wrapper. Its real config
read, first code write and final reply completed; the first code passed161
checks with no retry or cap. This qualifies one client task, not a CLM edit or
a comparative latency result.
The [combined-preset SDK smoke](speed-bench/experiments/sdk-preset-hour36-20261004/README.md)
also passed streamed Responses, stored-parent continuation/GET, Conversations
CRUD and the installed Pi provider tool loop through the actual wrapper.

The [fresh coding screen](speed-bench/experiments/fresh-clm-coding-hour32-20261004/README.md)
tested previously unused deduplication and token-bucket tasks through actual
read/edit/write/final calls. All eight first implementations passed161 checks
each without caps. With fixed request reasoning budgets, `context-fast` settings
saved11.6% task time and `persistent-fast` settings17.3% against native brackets;
the latter saved6.5% against suffix-only. These are explicit-setting package
measurements on two tasks, with1.8% native timing drift, rather than additive
gains or a general accuracy guarantee.

The presets enable automatic checkpoint caching in `.workspaces/context-cache`.
Server defaults are hot0 and SSD4096 MiB, matching the recent qualification;
agent defaults remain hot512 and SSD8192 MiB. Override
`DS4_STATE_CACHE_DIR`, `DS4_STATE_CACHE_HOT_MB` and `DS4_STATE_CACHE_SSD_MB`
to choose another directory or capacity; hot size512 enables a host RAM tier,
and hot size0 uses SSD only. The
[hot-capacity screen](speed-bench/experiments/hot-cache-hour21-20261004/README.md)
did not establish a latency benefit from the extra host tier. The
[SSD quota curve](speed-bench/experiments/ssd-budget-hour29-20261004/README.md)
kept all four tested suffix sources at512 MiB, with1.2% additional whole time;
128 MiB evicted them and added20.7%. All edited facts remained correct.
These capacities describe this four-history workload; the default retains more
room for other histories. Direct `ds4-agent` / `ds4-server` invocations enable
this tier by setting the directory.
Captured native prefixes and private pre-RoPE suffix originals survive restart.
Immutable compressed-attention pages share storage by content; raw sliding
windows, logits and unfinished compression pairs remain endpoint state. Writes
reach SSD before returning; hot and disk LRU limits evict unpinned entries.
The manifest becomes visible after its chunks, and a directory has one process
owner. A server's clients share that owner; independent agent processes need
separate directories.

Sessions keep their own mutable GPU graphs. `ds4-server --batched-session 2`
can interleave two resident sessions and batch decoding while inactive
conversation checkpoints live on SSD. This cache does not reduce the GPU memory
needed by each active slot. Exact-prefix lookup excludes approximate suffix
states. The cache currently supports native text V4.1 on Metal, with identity
including model file, context and forward-policy settings.

The [fresh WRITE-budget curve](speed-bench/experiments/fresh-thinking-curve-hour35-20261004/README.md)
kept unrestricted read-stage thinking and the persistent cache/routing settings
fixed. WRITE budgets1 and32 both passed all322 checks, saving8.5% and3.8% task
time against128 brackets; control drift was3.0%. Budget1 is a measured post-read
option on these two contracts, rather than a general reasoning default.

Thinking mode stays unchanged by the presets. Responses callers can opt into
`"ds4_reasoning_budget": 128` or `256` on a post-read request, alongside their
chosen `reasoning.effort`. Omitting it or using 0 leaves thinking unrestricted
on that request. Later thinking can consume the savings or hit the output cap,
as the fresh-task confirmation shows.
This is a soft, once-per-request thinking closure for native DS4.1 text, not
automatic WRITE detection or a persistent conversation setting. Unsafe parser
state, later reopening or recovery can exceed the budget; nonzero disables MTP
for that request, and forced closure counts toward the output cap. The small
[server thinking screen](speed-bench/experiments/server-thinking-hour16-20261004/README.md)
and [fresh-task confirmation](speed-bench/experiments/server-confirmation-hour17-20261004/README.md)
measure its scope. Keep tool definitions and task instructions stable across
tool turns: the [stable-instructions screen](speed-bench/experiments/stable-instructions-hour18-20261004/README.md)
saved40.49% across two completed coding tasks by preserving ordinary prefix
reuse. This includes shorter prompts and different generated histories; it is
not an additional percentage to add to the reasoning or suffix results. See also the
[measurements and limits](speed-bench/experiments/20261003-idea-review.md).

For a completion summary after an already successful write, the
[FINAL policy screen](speed-bench/experiments/final-thinking-hour20-20261004/README.md)
kept high effort and used budget1, saving24.55% whole-task time with accurate
summaries on both tasks. Switching that request to effortnone lost prefix reuse
and was slower. This does not establish a budget for harder final reasoning.
With those same request budgets, the
[resident package screen](speed-bench/experiments/resident-combined-hour22-20261004/README.md)
saved9.70% with up to three substitutions and3.58% with one; both coding tasks
passed161 checks in all arms. Suffix reuse was off in that comparison.

An external CLM harness can send its complete edited Responses `input` with
`store: false` and a stable `prompt_cache_key` per history. Retain tool call IDs,
actual tool results and unchanged task instructions when removing old context.
The server selects a compatible private suffix and evaluates the new query
freshly; missing or unsupported sources fall back to ordinary evaluation.
Use `previous_response_id` for appending to server-owned history. To edit that
history, submit full input instead, or edit the corresponding Conversations
items. A cache key scopes reuse; it does not make changed prefixes equivalent.

The installed Pi Responses provider accepts the local model/base URL and sends
full input with `store: false`. Its `sessionId` becomes `prompt_cache_key` when
cache retention is enabled. Its `onPayload` hook can add a budget chosen by the
caller for this request:

```js
const options = {
  sessionId: "persistent-agent-1",
  cacheRetention: "short",
  temperature: 0,
  onPayload: payload => ({ ...payload, ds4_reasoning_budget: budget }),
};
```

Keep effort and tool definitions stable across the loop. The measured example
uses budget0 for READ,128 after the read for WRITE, and1 only for a short
completion summary. These are caller decisions. The installed provider's
streaming read/tool-result path was exercised in the linked client smoke;
the payload hook above is source-verified, not a complete Pi agent benchmark.

On the actual full-input CLM coding loop, removing obsolete notes after a real
config read and enabling suffix64 saved11.81% task time, with all six first
implementations passing161 checks. The
[coding report](speed-bench/experiments/clm-coding-hour27-20261004/README.md)
keeps the original grader-path error and records pre-graft READ divergence;
the result measures the configuration package rather than pure graft cost.

Use `/hints on` for occasional, brief explanations of the programming choices
behind the work, and `/hints off` to stop them. Changes take effect at the next
conversation boundary without rebuilding the cached context. New and resumed
sessions start with hints off.

Sessions are stored in `~/.ds4/kvcache`:

| Command | Action |
| --- | --- |
| `/save` | Save the current session |
| `/list` | List saved sessions |
| `/switch <sha>` | Resume a session |
| `/del <sha>` | Delete a saved session |
| `/strip <sha>` | Keep text and title, removing the large KV payload |

Compatible local KV snapshots avoid rebuilding the prompt. Stripped sessions
and network TP restores require prefill. Sessions containing images cannot yet
be saved. Saved conversations and traces may contain private information.

For Pi, OpenCode, Codex CLI, or Claude Code, use `ds4-server` instead and follow
the [client setup guide](docs/CLIENTS.md).

### Models, images, and speculation

[Models and vision](docs/MODELS.md) lists the supported downloads and memory
requirements. DeepSeek Vision Experimental uses a different checkpoint from
Flash 0731; GLM 5.3 Flash and Qwen3.8 Flash Next add vision to the same text
model through a separate encoder.

DeepSeek V4.1 Flash text and vision run on Metal; text also runs on a DGX Spark.
Q2 runs with SSD streaming on one 128 GB Mac or Spark, or resident across two
Macs or two Sparks using RDMA. Q4 needs SSD streaming or a 512 GB Mac.
Engram tables remain on disk in every mode, so use a fast
local SSD. See the [model guide](docs/MODELS.md#deepseek-v41-flash) for downloads
and setup.

With the matching encoder passed as `--vision FILE`, use `/read image.png`
in the CLI or `view_image` in the native agent.

Qwen3.8's smaller Q2 release has **41.73 GiB** of main/MTP weights,
with imatrix IQ2_XXS gate/up experts and padded Q2_K down projections.
It is the starting option for 64 GB Macs.
The GGUF also contains 95.37 GiB of original BF16 n-grams, read directly
from disk rather than loaded into RAM. Keep it on a fast SSD. Start with 8K context:

```sh
./download_model.sh qwen38-q2
./ds4 --ctx 8192 --prefill-chunk 1024
```

The download fetches one 137.10 GiB file and updates `ds4flash.gguf`.
Add `--mtp` for speculative decoding. The larger
`qwen38-q4k` target is also available. Download the optional vision encoder
with `./download_model.sh qwen38-vision` and pass it with `--vision`.
See [Qwen setup](docs/QWEN38_FLASH_NEXT.md) for details.

Speculative decoding is opt-in. GLM and Qwen use `--mtp`; V4 Flash DSpark needs a matching
support GGUF. It can improve generation, but not every workload benefits.
Read [speculative decoding](docs/SPECULATIVE_DECODING.md) for setup and the
difference between default opportunistic sampling and `--mtp-exact-sampling`.

### Output and power

Thinking is enabled by default. Use `--nothink` or `/nothink` for direct
answers, and `--think` or `/think` to enable it again.
For V4.1, `ds4` and `ds4-agent` also accept
`--think-level 25` or `/think 25`: 1 to 100 sets the reasoning effort, and
0 disables thinking. `--think` selects 75, `--think-max` selects 100.
Changing the level in a conversation rebuilds its cached prefix.
The normal sampling defaults are temperature 1, top-p 1, and min-p 0.05;
`--temp 0` selects greedy output.

For DeepSeek V4, `--power N` trades throughput for lower sustained GPU load.
The default is 100. V4.1 and GLM currently require `--power 100`.

DeepSeek V4 Flash and GLM 5.3 Flash also support directional steering. Load a
vector with `--dir-steering-file FILE`; `/steer F` adjusts its scale for
subsequent tokens in a local CLI or agent session, without rebuilding the
existing KV cache. See [steering documentation](dir-steering/README.md).

`--prefix-file FILE` preloads complete `USER:` / `ASSISTANT:` pairs before
the live conversation. A turn marker must start a line, roles must alternate,
and the last turn must be `ASSISTANT:`.

## Capability Evaluation

`ds4-eval` runs embedded capability regression tests against a real GGUF.
These are DwarfStar integration checks, not official leaderboard scores.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
./ds4-eval -m ds4flash.gguf --suite hard-smoke
./ds4-eval -m ds4flash.gguf --suite hard --retry-incomplete
```

The default suite is `core`; `--suite all` runs core and hard cases.
`--list-cases` lists tests without loading a model. `--plain` selects
non-interactive output, and `--regrade-trace FILE` scores an existing trace
without generating again. Sources and licenses are in [EVAL_DATA.md](EVAL_DATA.md).
For inference correctness and release checks, read [testing](docs/TESTING.md).

## Speed

This recorded DeepSeek V4 Flash Q2 sweep uses an M5 Max with 128 GB RAM,
2048-token continued-prefill intervals, and 128 greedy generation tokens per
frontier. It is a baseline, not a fresh benchmark of every commit.

![M5 Max Flash Q2 throughput](speed-bench/m5_max_ts.svg)

See [performance and benchmarking](docs/PERFORMANCE.md) for the full numbers,
DGX Spark results, comparison conditions, and benchmark commands.

## Detailed Guides

- [Models and vision](docs/MODELS.md): Flash, PRO, GLM, Qwen, and matching encoders.
- [Qwen3.8 Flash Next](docs/QWEN38_FLASH_NEXT.md): model setup, MTP, vision, and validation.
- [SSD streaming](docs/SSD_STREAMING.md): run larger than RAM and size the cache.
- [Inference across machines](docs/DISTRIBUTED.md): two-Mac TP/RDMA and layer pipelines.
- [Speculative decoding](docs/SPECULATIVE_DECODING.md): DSpark, GLM and Qwen MTP, and sampling.
- [Serving](docs/SERVER.md): APIs, images, batching, and disk KV caches.
- [Coding agent clients](docs/CLIENTS.md): Pi, OpenCode, Codex CLI, and Claude Code.
- [Performance](docs/PERFORMANCE.md): reproducible measurements and recorded baselines.
- [Testing and development](docs/TESTING.md): regression tests, debugging, and model-building tools.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request.

## Logo

The DwarfStar logo was designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino, whose human touch made
it rock.
