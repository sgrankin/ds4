# Persistent hot-RAM and SSD checkpoint storage

## Store a complete endpoint, share only immutable bytes

The cache automatically persists usable evaluated endpoints; it is not limited
to a user's explicit “save session”. A prefix can be restored or an older
suffix source discovered after slot eviction or process restart. Conversation
history and the agent's factual archive remain separate responsibilities.

In compressed attention, an endpoint consists of cumulative KV/index arrays
AND recurrent/frontier state: tokens, raw windows, pending compression pairs,
logits and endpoint headers. Split cumulative arrays at 2048-token pages, hash
immutable byte chunks, and store the remaining endpoint state separately.
Pre-RoPE originals are immutable chunks too. Reassemble the existing native
payload for GPU import instead of inventing another device layout.

```text
put(endpoint):
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

Every successful put commits synchronously to SSD. The hot tier is a host LRU;
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
See [host cache API](../ds4_state_cache.h),
[implementation](../ds4_state_cache.c) and [profile adapter](../ds4_state_cache_runtime.h).

## Typed replay persistence and format boundaries

Item50 adds a resumable replay kind without changing the old native or suffix
encodings. The `DS4SC01` manifest distinguishes native (0), original-backed
suffix (1) and decoder replay (2). Replay state has a versioned 32-byte `DRP1`
envelope around the existing `DSV4` v2 payload, recording replay K and the
selected-expert setting. The dedicated loader checks those settings and retains
the approximate/replay identity. Canonical exports and suffix capture stay
blocked; replay records cannot appear in suffix discovery. See the
[replay recipe](12-bounded-decoder-replay.md) and
[Item50 qualification](../speed-bench/experiments/replay-persistence-hour50-20261021/README.md)
for the SSD reopen, cross-layout fresh-logit and real server-restart qualification.

The new reader still accepts old native/suffix records. Downgrading to an older
reader makes replay records unknown: its startup cleanup prunes those manifests
and unreferenced chunks, so replay must be recomputed. These are disposable KV
records, not the agent's factual archive. Keep one process owner per directory.

The design is sufficient for this cache: immutable hashed chunks, atomic
manifest publication and explicit state kinds/versions permit this bounded
extension. It is not a generic portable database. The adapter still hardcodes
DS4.1 layer counts, dimensions, vocabulary and frontier layout; a different
architecture needs a deliberate adapter/format change. Disk payloads retain F32
values even with the opt-in F16 runtime main cache, so runtime allocation savings
do not halve persisted bytes. A replay's inner `DSV4` bytes are storage inside
its private envelope, not permission to load the state as canonical/exact.

## Knobs and scope

`DS4_STATE_CACHE_DIR` enables storage. `DS4_STATE_CACHE_HOT_MB=0` uses disk-backed
chunk materialization; `DS4_STATE_CACHE_SSD_MB` bounds stored payloads. Presets
use `.workspaces/context-cache`: server hot 0/SSD4096 MiB; agent hot 512/SSD8192 MiB.
Metadata and caller-owned materialization scratch are additional, separately
accounted resources. Source leases are released before close. Native DS4.1
text/Metal is the qualified adapter; TP/directional steering are excluded.

There are three distinct caches: expert weights in RAM, application checkpoint
chunks in RAM/SSD, and the OS file cache. `source_ssd=1` means a persisted source
was selected; its chunks may have been served from hot RAM or the OS cache.
Do not label that flag proof of a physical NVMe miss.

## Evidence and limits

| Experiment | Observation | Interpretation |
|---|---|---|
| [Item9](../speed-bench/experiments/state-cache-hour9-20261003/README.md), fix `bf0d20cb` | Two endpoints 63.58 MiB used 51.08 MiB distinct files;12 native materializations preserved bytes/originals across reopen | 19.65% byte sharing on these sources, not a universal dedup ratio |
| Item9 hot 0/1/64 MiB | First reassembly 81.45/78.65/64.36 ms; warm 79.45/77.50/.60 ms | Endpoint byte-reassembly microbenchmark; puts 192–199 ms; no physical-cold cache flush |
| [Item21](../speed-bench/experiments/hot-cache-hour21-20261004/README.md), `fe5d6e26` | Hot512 served 173/173 pre-restart chunk reads from RAM but did not beat the matched faster hot 0 behavior | No established whole-task latency gain from extra hot RAM |
| [Item29](../speed-bench/experiments/ssd-budget-hour29-20261004/README.md), `91c9cf21` | SSD512 MiB retained 4/4 suffix sources, +1.23% whole wall vs 4096 bracket;128 MiB lost all 4, +20.69% whole/+53.88% edited wall | Capacity matters through retaining useful sources; not a universal minimum |

Item29 had four histories/two slots, one edit each and a restart, native routing,
hot 0, Q2/cache 64/ctx 8192/chunk 512. All changed-file facts and 32 fresh reads passed;
one 128 MiB warm reply failed READY-only formatting before any graft. Keep that
failure and the differing output cost in the capacity comparison.

The first Item9 agent run answered correctly but grafted only once: a long exact
prefix candidate displaced the genuinely shifted suffix, then failed because
destination 0 was inadmissible to the graft synchronizer. Excluding destination 0
from suffix candidates fixed activation while leaving ordinary prefix lookup.
The corrected hot 0 run observed 3/3 SSD grafts, including a private approximate
source. This is an integration lesson: separate exact-prefix candidates from
relocation candidates before choosing the winner.

## Replication and credit

Start with two overlapping endpoints; close/reopen the cache, materialize both,
and compare native bytes and originals. Hold one source lease while advancing
another session. Then interleave more histories than resident slots, edit them,
restart, update external facts, and verify fresh retrieval. Sweep quota and hot
size on the same workload, recording endpoint loss, actual grafts, eviction,
owned bytes and task wall. A smaller read count caused by source loss is not a
performance improvement.

Credit [oMLX's V4.1 delta-cache design](https://github.com/jundot/omlx/blob/main/omlx/cache/deepseek_v41_delta.py)
for separating cumulative compressed state from endpoint recurrent state.
The [DeepSeek report](https://arxiv.org/abs/2609.19969) provides persistent-KV/SWA
background. [ds4-recipe](https://tangled.org/astrra.space/ds4-recipe), chapter 12,
describes a different checkpoint/delta design and broader source discovery.
We reuse native DS4 payloads and immutable chunks; no oMLX/recipe throughput or
binary-format compatibility is claimed.
