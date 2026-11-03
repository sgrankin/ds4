# The local agent

`ds4-agent` runs inference directly in an interactive process; no HTTP server is required. It holds its conversation and model session together, executes its own file and shell tools, and can save sessions. This is distinct from the server, where the client executes function calls. For model and hardware setup see the [DeepSeek guide](../models/deepseek/README.md).

```sh
./tools/zig-build ds4-agent
./ds4-local --bin-dir ./zig-out/bin persistent-fast \
  --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  --live-context-file .workspaces/agent-context.txt
```

The [launcher](../../ds4-local) selects opt-in DeepSeek settings. `context-fast` enables approximate suffix reuse with native expert routing. `resident-fast` substitutes up to five resident experts during decode and keeps native routing weights; `persistent-fast` combines suffix reuse with these substitutions in decode and selected prefill. `persistent-packed` is an alias. These profiles target a 64 GB expert cache, an 8192-token context and a 512-token prefill chunk; later command-line arguments override the defaults. They use packed main/index caches and compact SSD checkpoints, and do not force thinking on or off. Server `persistent-fast` additionally selects semantic boundary capture; the direct agent does not acquire those server defaults. These approximations change inference, so measure the quality and latency of your task. The [recipe index](../../recipes/README.md) records mechanisms and evidence.

## Editing the live context

The optional mirror file gives the agent a working view it can edit with its ordinary file tools. Accepted edits are imported after tool execution or before the next user turn. The protected system/tool prefix is unlabelled. Each conversation span has a handle such as `[[DS4_TURN a3c91e704b82d6f0 role=assistant]]`; tool calls and results can be inside an assistant span. IDs are randomly generated and stable during a running session, including retained turns across edits and compaction. New turns receive new IDs, and a restart may assign different IDs. Headers are editing handles and are removed before model evaluation. A literal header-shaped body line has a backslash display escape, which must be retained.

A useful editing sequence is to list headers, read a bounded range around selected older IDs, and save useful facts to a checkable archive file before deleting redundant material or replacing it with a concise note. Keep each ID with its turn when moving it; the file order is conversation order. Preserve the protected prefix, latest user request, delimiters and untouched text. Edit ranges in place rather than retyping the retained transcript, then continue after the edit receipt. The [context owner](../../src/agent_context.zig) validates and prepares the edit; the runtime commits it only after successful preparation. Suffix reuse can accelerate an eligible edit but is approximate, and missing or unsuitable sources fall back to recomputation. The [qualification](../../speed-bench/experiments/agent-package-hour41-20261009/README.md) records tested workflows and fallbacks.

### A bounded context-edit recipe

For example, after an investigation has superseded an old file dump, ask the
agent to do the following:

1. List only turn headers in `.workspaces/agent-context.txt` and select the old
   dump by its stable handle.
2. Read a bounded range inside that span to confirm it is the intended material.
3. Save any still-useful facts, paths and evidence references to an archive file.
4. Replace the selected body with a short archive reference. Keep its header,
   role, chat delimiters, protected prefix and latest user request intact.
5. Continue the task after the edit receipt; do not repeatedly reread the full
   mirror looking for the same text.

The accepted receipt reports before/after token counts and whether suffix reuse
was approximate or absent. A rejected edit reports a problem instead of
committing a partial working view. The receipt is evidence of the import result;
editing the file alone does not establish that the active context changed.

## Compaction

Under context pressure, or on `/compact`, the agent privately generates a task-state summary and rebuilds the context with a recent verbatim tail. It can attempt suffix reuse during that rebuild even without a mirror file. The latest user message and subsequent content remain fresh. Compaction does not automatically write discarded facts to an archive; explicitly save facts that need later verification. If a partial tool call is interrupted by compaction, it is not executed. Oversized tool results get a compaction attempt and may still be rejected if they cannot fit. See [compaction planning](../../src/agent_compaction.zig) and [turn execution](../../src/agent/turn.zig).

## Tools

The agent provides bounded file reading, search and editing, and shell execution; long reads can be continued with `more`. Running shell jobs remain external to compaction and can be checked or stopped. `/help` shows the current commands.

## Saving and resuming sessions

`/save` stores the current session, `/list` and `/switch SHA` select saved sessions, `/del SHA` deletes a saved session, `/strip SHA` removes the KV payload so a later switch rebuilds by prefill, and `/new` starts from the system prompt. Checkpoints and the live context are not substitutes for a factual archive. Approximate-context sessions can be saved as transcript without KV and are re-evaluated on reopening; saved state must also match the model and quantization. See [session handling](../../src/agent/worker_sessions.zig) and [checkpoint loading](../../src/agent/checkpoint.zig).

Sessions live in `~/.ds4/kvcache`. Compatible local KV snapshots can avoid prefill; stripped sessions and network tensor-parallel restores rebuild it. Sessions containing images cannot currently be saved. Saved conversations and traces can contain private information. `--live-context-file` is limited to text chat and cannot be combined with `--vision` or raw prompts. With a matching encoder, the native agent's `view_image` tool can inspect an image.

## Prompts, hints and steering

`/hints on` requests brief programming explanations at the next conversation boundary; `/hints off` disables them, and new or resumed sessions start off.

Load a model-compatible directional steering vector with `--dir-steering-file FILE`;
`/steer F` changes the scale for subsequent tokens without rebuilding existing
KV. See the [steering guide](../../dir-steering/README.md) for supported models,
vector extraction and projection or additive semantics.

`--prefix-file FILE` preloads alternating `USER:` and `ASSISTANT:` turns, with markers at the start of lines and a final assistant turn. `-p` and `--prompt-file` supply an initial prompt; raw prompts require non-interactive mode and cannot be combined with a prefix file. Run `ds4-agent --help` for the complete options.

## Web and image tools

The direct agent also has web search and page-visit tools. The web runtime may request interactive approval to start a visible Chrome browser; the tool returns an error if startup is denied or cannot be approved. The server does not host these tools for API clients. See the [agent executor](../../src/agent/executor.zig) and [web qualification](../../speed-bench/experiments/zig-web-20261022/README.md).
