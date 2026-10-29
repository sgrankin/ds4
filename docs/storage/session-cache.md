# Inference checkpoints and conversation storage

Logical history and model state have different owners. A client can resend its conversation; Responses can additionally retain logical ancestry. Inference checkpoints hold evaluated model state to avoid repeating prefill. They are disposable and do not replace an archive of task facts. Expert weights have a separate [cache](expert-streaming.md).

## Choosing a store and reusing state

`DS4_STATE_CACHE_DIR` enables the chunked RAM/SSD state cache described below.
The separate `--kv-disk-dir` option enables the legacy disk KV store; neither
setting enables the other. Logical response history has a third setting,
`DS4_SERVER_RESPONSE_STORE_DIR`, and can remain available after inference-cache
eviction.

On an eligible text request, the server first checks live session reuse. It may
load a legacy disk prefix on a miss; during synchronization, the state cache can
then supply a longer compatible exact prefix. Canonical suffix policy can
bypass the legacy text load, and approximate suffix grafting has its own
eligibility checks. These are conditional sources of acceleration, rather than
tiers of one lookup. See [generation](../../src/server/generation.zig) and
[prefix restoration](../../src/server/persist.zig) for the handoffs.

## Checkpoint representation and lifecycle

The cache automatically persists usable evaluated endpoints; it is not limited
to a user's explicit “save session”. A prefix can be restored or an older
suffix source discovered after slot eviction or process restart. Conversation
history and the agent's factual archive remain separate responsibilities.

In compressed attention, an endpoint consists of cumulative KV/index arrays
and recurrent/frontier state: tokens, raw windows, pending compression pairs,
logits and endpoint headers. The store splits cumulative arrays at 2048-token
pages, hashes immutable byte chunks and stores the remaining endpoint state
separately. Pre-RoPE originals are immutable chunks too. It reassembles the
existing native payload for GPU import rather than using another device layout.

```text
put_durable(endpoint):
    partition native state and originals into immutable byte chunks
    write absent chunks to SSD
    publish endpoint manifest last
    insert eligible chunks into bounded host LRU

get(source):
    acquire source lease (pins manifest/chunks)
    admit materialization scratch
    read each chunk from hot RAM or SSD-backed file
    reassemble native payload + originals
    import into this session; release lease
```

Every successful durable put commits synchronously to SSD. Transient puts retain
an endpoint only in the bounded hot tier and require a nonzero hot budget. They
can share existing durable pages, but their manifest is not restart-persistent.
Semantic intermediate checkpoints use this transient path. The hot tier is a host LRU;
the SSD tier evicts unpinned endpoints and unreferenced chunks. Startup scans
manifests. One directory has one process owner; multiple server clients share
that owner under the existing inference lock. Active sessions still have their
own mutable GPU graphs. This storage layer is not GPU paged attention or a new
concurrent GPU scheduler.

Prefix lookup requires a compatible model/forward-policy profile and token IDs.
It returns native or explicitly typed decoder-replay checkpoints; private
suffix-approximate sources remain excluded. A replay restore stays approximate.
Our profile includes model
file identity/stat metadata, context, routed quantization, chunk/quality mode
and sorted `DS4_METAL_*` environment settings, with replay K/selected policy added
for replay configurations. It is not a full 365 GB weight-file
digest. Suffix discovery can enumerate private approximate originals, with
the caller still checking latest-User freshness and relocation geometry.
See [host cache API](../../src/ds4_state_cache.h),
[implementation](../../src/state_cache.zig) and [profile adapter](../../src/ds4_state_cache_runtime.h).


The qualified adapter is native DeepSeek V4.1 text on Metal; tensor parallel and directional steering are excluded. `DS4_STATE_CACHE_DIR` enables this store, `DS4_STATE_CACHE_HOT_MB` bounds its host tier, and `DS4_STATE_CACHE_SSD_MB` bounds SSD payloads. One process owns a directory. Metadata and temporary materialization storage are additional. A persisted source may be served from hot RAM or the OS file cache, so selection does not prove physical SSD traffic. The [state-cache recipe](../../recipes/03-state-cache.md) covers typed replay, compatibility and measurements.

## Explicit prompt boundaries


DS4.1 text requests accept OpenAI-style explicit checkpoint hints on text
content blocks in Chat Completions and Responses:

```json
{
  "model": "deepseek-v4.1-flash",
  "prompt_cache_options": {"mode": "explicit"},
  "input": [{
    "role": "developer",
    "content": [{
      "type": "input_text",
      "text": "Stable project context goes here.",
      "prompt_cache_breakpoint": {"mode": "explicit"}
    }]
  }, {"role": "user", "content": "What should we change next?"}]
}
```

The default `implicit` mode retains automatic caching and honors marked ends.
`explicit` mode permits inference-cache reads only at marked prefix endpoints
and writes at the last four marked ends. Historical marks remain eligible for
lookup; more than four marks are accepted. With no marks in the resolved
conversation, explicit mode disables inference-cache reuse and writes. Logical
Response/Conversation storage remains independent, including `store:true` and
`previous_response_id` recovery. Private replay preserves marked blocks.

A boundary ends a prefix; changed earlier text still prevents exact reuse of
later state. It does not independently cache a suffix. The metadata adds no
model-visible text. Encoding splits at marked ends so stored token IDs are
actual prefixes of the complete encoded request; moving a mark can change a
token merge at that boundary. SSD restoration also verifies the restored IDs
against the current encoded prompt before accepting the hit.

Cache budgets and minimum checkpoint sizes still apply. Explicit controls
currently require DS4.1 text; marked images and other model families return a
clear unsupported error. Ordinary Flint caching is unchanged. A tool-result-only
request using explicit controls needs full replay, `previous_response_id`, or a
Conversation reference so it can recover without unmarked live KV. TTL/prewarm
options are not implemented. These breakpoint fields are documented provider
extensions to core OpenResponses.

## Legacy disk KV and tool replay


Disk caching saves useful prefixes across slot reuse and server restarts:

```sh
./ds4-server --ctx 100000 \
  --kv-disk-dir .workspaces/ds4-kv --kv-disk-space-mb 8192
```

Clients can resend complete conversation histories. The server first tries
the live token prefix, then compatible rendered-text prefixes from disk, and
prefills the new suffix. With multiple slots, each slot has its own live state;
disk is the additional persistence layer, not the only way to retain sessions.

TP cache loading rebuilds the saved token prefix on both ranks. It is not an
instantaneous restoration of both GPUs. Pipeline loading redistributes the
saved layer state over its route.

Defaults are intended to avoid saving fragile token boundaries. For unusual
workloads, the controls are `--kv-cache-min-tokens`,
`--kv-cache-cold-max-tokens`, `--kv-cache-continued-interval-tokens`,
`--kv-cache-boundary-trim-tokens`, and `--kv-cache-boundary-align-tokens`.
Check `./ds4-server --help` for their defaults.

Quantization variants may share compatible prefixes. Add
`--kv-cache-reject-different-quant` for same-quant reuse only.
Cache files contain prompt text and model state: treat the directory as
private. It is disposable; stop the server before clearing it.

## Tool history and debugging


For DeepSeek, the server preserves sampled DSML tool blocks and assigns
unguessable tool IDs. Replaying those IDs avoids retokenizing a differently
formatted JSON history. The bounded replay map can be stored in cache files.
When exact replay is unavailable, canonical rendering may require rebuilding
part of the prefix.

`--tool-memory-max-ids` bounds this map.
`--disable-exact-dsml-tool-replay` disables it for diagnostic comparisons.
Use `--trace .workspaces/ds4-trace.txt` to record prompt rendering, cache decisions,
generated text, and tool-parser events. Traces can contain sensitive content.

Cache formats are implementation details. The current header and extension
definitions are in [ds4_kvstore.h](../../src/ds4_kvstore.h) and
[kvstore.zig](../../src/kvstore.zig); model-specific payload handling is in
[the DeepSeek payload owner](../../src/engine/deepseek41/state_payload.zig).

## Semantic endpoints and edited contexts

Server `persistent-fast` enables DeepSeek V4.1 Metal semantic boundaries and anchor capture during prefill. Intermediate endpoints are bounded RAM-only sources; restart or eviction loses them. An eligible unchanged suffix can be relocated after an edit, but its activations still reflect the old prefix, so reuse is approximate. Missing sources fall back to recomputation. Client controls, tool schemas and model policy participate in eligibility. The client owns the materialized edited view and its factual archive. See the [checkpoint recipe](../../recipes/16-semantic-context-checkpoints.md) for restrictions and evidence.

The DeepSeek fast presets use `.workspaces/context-cache` unless overridden. Direct-agent defaults are 512 MiB hot and 8192 MiB SSD; server defaults are 4096 MiB SSD and either 512 MiB hot for `persistent-fast` or zero hot for the other fast presets. Independent agent processes need separate cache directories. Active session graphs still consume GPU memory independently of this store. Durable native prefixes and eligible private suffix sources can survive restart; semantic intermediate endpoints cannot. See the [semantic checkpoint experiment](../lab/journal/2026-10-28-deepseek-semantic-checkpoints.md) for measured cases and the distinction between preset and tested settings.
