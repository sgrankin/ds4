# Native Metal and shared protocol qualification

Run commands from the repository root. The current supported build is native
macOS Metal; [build and test](../build-and-test.md) explains prerequisites,
fixture targets and Debug limitations. These checks supplement, rather than
replace, the relevant model qualification.

## Build and model-free checks

```sh
jj status
make
make test
make test-frontends test-session-state test-quality-api test-download-model
./ds4 --help
./ds4-server --help
./ds4-agent --help
./ds4-eval --self-test-extractors
```

Save complete build output and investigate compiler warnings. Check help for
readable wrapping. Use `git diff --check` where appropriate. `make clean` removes
installed outputs but retains Zig caches; ensure the selected sources were
actually rebuilt after replacing a tree. `make test` is model-free. Run
`make test-metal-command-memory` after command-buffer or copy changes with
Metal API validation: repeated batches must preserve output and keep CPU heap
bounded after warmup. After M5 dense-kernel changes, run
`make test-metal-dense-mpp`: its Q8 decode and Q4_0/Q4_K prefill checks compare
against CPU dot products and cover repeats, partial tiles and output tails.
Check long runs for temporary driver objects accumulating
per token. Run the relevant parser and agent fixtures with sanitizers when the
build supports them; Debug alone is not a sanitizer build.

After parser, loader, snapshot or cache changes, reject oversized declared
lengths before allocation or payload reads, truncated and overflowing GGUF or
safetensors dimensions, and non-finite values in integer JSON fields without
undefined behavior. Exercise repeated owned fields in malformed requests and
confirm a following valid request works. Replay at least 4,096 assistant/tool
pairs through the applicable validators; acceptance should match short replay
and work should remain linear. These are test objectives; the former C-era
sanitizer and distributed fixture commands must be re-established for the
current build before claiming those checks passed.

### Artifact and download checks

Test `download-model.nu` in a disposable directory using `make test-download-model`;
check resume, size and checksum failures, split assembly, naming and symlinks,
retained catalog entries and rejection of removed targets. Verify advertised model
names against the catalog.

## Continuation quality and storage

Build the production-linked scorer with `make quality-score`; use its
[fixture instructions](../../../gguf-tools/quality-testing/README.md) and
checkpoint-matched manifests. Compare the candidate with the last known good
using `python3 gguf-tools/quality-testing/compare_scores.py OLD.tsv NEW.tsv`.
Retain summary and API agreement lines and investigate material NLL, first-token
or API ordering changes. Quality tradeoffs can be intentional; document their
practical effects rather than assuming a numerical threshold determines value.
Compare resident and SSD execution on the same fixture where both fit.

For SSD streaming, test automatic sizing, an impossible cache request and a
one-slot cache. The memory guard must reduce an oversized target; one slot must
use direct per-layer reads and complete. Record effective cache, decode map and
planned memory. Exercise cold and repeated prompts, cache hits and evictions;
compare complete logits and deterministic output across cache policies at fixed
prefill geometry. Cache changes must not change routing counts or precision.
Run `MTL_DEBUG_LAYER=1 make test-metal-ssd-experts` after cache changes. Monitor
actual wired memory and swap as well as planned allocations. A stuck filesystem
read surviving cancellation requires independent filesystem investigation, not
just a longer timeout or a pass on another host.

For disk KV, start with an admitted context and a disposable directory, for
example `./ds4-server --ctx 100000 --kv-disk-dir .workspaces/qa-kv --kv-disk-space-mb 8192`.
Repeat a request for a hit, force eviction, and check that the new entry and
useful anchors survive. Restore a disk checkpoint twice and check it remains on
disk. Reject corrupt or incompatible model, quantization, context and KV-layout
checkpoints. Test agent `/strip` followed by `/switch`: prefill must rebuild a
sane history.

## Server and agent

### Streaming and malformed requests

Exercise Chat Completions, Responses and Anthropic, streaming and non-streaming,
with thinking enabled and disabled. Check discovery for the loaded model,
keepalive during long prefill, and cancellation while queued, prefilling and
decoding; a subsequent valid request must work. Check reasoning closure and
recovery of a complete tool call inside unclosed reasoning without leaking
reasoning into the answer or inventing continuation. Split tool delimiters at
byte boundaries, including literal markers and escaped wrappers in arguments.
Truncated arguments must remain partial and never execute as a complete call. Preserve the distinction between `length` and client `stop`. Recovery must respect the original output
budget. Use the current [Responses contract](../../serving/responses.md) for
stored ancestry and full-history replay semantics.

### Replay and vision

Run a real coding client through several read/edit/build/test rounds, a harmless
tool failure and recovery, beyond 4K tokens. Capture `--trace` and inspect tool
IDs, replay, prefix matches and continued prefill. Unchanged history should reuse
its prefix; editing an earlier message must invalidate its changed suffix.
Compare continuation with fresh full-history replay. Traces can contain private
conversation data. For vision changes, exercise changed, removed and reordered
images, image tool results, concurrent sessions and hidden-reasoning replay;
unchanged images may reuse the prefix, incompatible images must not. The
`tests/test_server_vision_cache.py` and `tests/test_server_vision_agent.py`
drivers exercise these paths against an idle server.

### Agent and compaction

Manually test agent help, session save/list/switch/history/compact/new/delete and
strip; interrupt generation, prefill and long tools, then continue. Queue input
while busy and verify tool execution precedes the queued message. Exercise exact
edits and `--edit-upto` ambiguity, shell output truncation, failures and job
stop/status, web approval and errors, and multiline TUI editing. Run
`tests/test_agent_compaction.py` with a matching model at a small and an admitted
long context: independently build and test its output, restore a full previous
session, cancel compaction and continue. Unfinished tool calls must not run and
unfinished text must survive. Test speculative stop rewinds separately.

For parser or speculation changes, run actual GLM MTP and matched DeepSeek
DSpark coding-client loops in both default and `--mtp-exact-sampling` modes.
Discard unseen speculative suffixes after tool or stop markers. When the parser
changes sampling mode, exact sampling must resample; opportunistic decoding
should retain verified greedy drafts. Check tool-result continuation through all
three APIs and retain acceptance, failures and traces. The agent and server
implement the mode transition in `src/agent/turn.zig` and
`src/server/generation.zig`; a fixture alone does not replace the real loop.

## Performance

Use [benchmarking](../benchmarking.md) and the [methodology](../../lab/methodology.md):
match checkpoint, workload, context, cache and settings; alternate at least five
measured samples per side, retain all samples and report variation. Include
startup, peak memory, initial and continued prefill, first decode and steady
decode where relevant; record restoration method and distinguish logical reads
from physical SSD traffic. For batching report aggregate and per-session speed.
Inspect generated answers and speculation acceptance so repetition does not look
like a speedup. Test advertised context allocations and `--power 100` and
`--power 50` where practical. Historical three-run or single-run figures are
context, not a current comparative gate.
