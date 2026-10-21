# Item50: private decoder-replay checkpoints survive restart

2026-10-21, 01:40–01:52 UTC; one-hour cap 02:40. **SSD reopen and a real
server restart passed. Replay state retains its approximate identity.** Source
base `c360d3789a8e64c8c4612eaeb51751f8eca12396`; binary hashes and commands are
retained in [checks](evidence/checks/checks.json) and each execution record.

## Small format extension

Keep immutable hashed chunks, bounded host/SSD tiers, and atomic endpoint
publication. Extend the existing DS4SC01 manifest kind field from native0 and
suffix1 to replay2; its 96-byte header and 64-byte segment records are unchanged.
A replay payload uses a 32-byte little-endian DRP1 v1 envelope around the
unchanged DSV4 v2 body: magic, version, replay kind, K, selected flag, body length
and a reserved zero. Cache profiles include configured K/selected policy, and
the core loader independently checks both. Save records the policy actually
applied to the session. Configuration is fixed for a process/session lifetime.

The private loader preserves the sticky approximate/replay flags. Canonical
exports and suffix capture remain blocked. Replay entries support matching
token-prefix resume but never suffix lookup/enumeration and carry no pre-RoPE
archive. Existing native/suffix encodings remain readable. An older binary
prunes unknown replay records on downgrade; this is recomputable KV, separate
from permanent agent facts. F16 receiving sessions still use canonical F32
disk values. No generic tensor container, database or migration system was added.

## Qualification

The [cache checks](../../../tests/test_state_cache.c) exercise native/suffix
compatibility, replay byte identity across close/reopen, policy isolation,
suffix exclusion, unknown-kind rejection, sharing, pinning and quotas. Server
and agent model-free suites and both production binaries build/pass.

The [focused model probe](../../replay_checkpoint_bench.c) uses the verified
1017 IQ2_XXS/Q2_K model, K128/selected1, native routing, context16384/chunk8192,
expert cache64GiB. It prefills8192 and evaluates one teacher token, saves a
63,479,384-byte odd-frontier private checkpoint, closes/reopens a hot0 SSD
cache, and loads into a fresh F16 session. All bytes survive materialization
and private resave. After the next shared teacher token, all129,280 finite
fresh logits are byte-identical to the uninterrupted F32 session. Different
profile and suffix lookup miss; canonical export and suffix capture remain
blocked. [Probe result](evidence/probe/summary.json).

The [HTTP driver](serve.py) reuses Item47/46 workflow code. Two distinct server
PIDs use the same fixture path/request, shared SSD directory, hot0/SSD4096MiB,
K128/selected1 and suffix reuse off. Both first workflows discover a reader,
read the actual owned file and produce the correct archived/current facts.
Cold execution writes three replay endpoints; the fresh server restores8616
tokens on its first request and executes no replay sweep. It then handles real
new tool results and computes the final answer. All three endpoints remain,
with96,095,024 stored bytes,195 chunks and no evictions. Restart reads87 chunks.
[First requests, outputs and logs](evidence/serve/result.json).

Workflow wall was47.713s cold and18.628s restarted, excluding server startup.
This is one ordered restart qualification, not an ABBA throughput estimate;
OS file caching is not flushed. HTTP `cached_tokens` currently reports0 for
that SSD hit because existing usage accounting runs before the state-cache
loader. The actual replay-hit log, fresh-process identity and continuation
grade establish restoration; the counter is a known separate reporting defect.

The probe's20 and server pair's70 pressure samples were all normal. Peak
wired+compressed was87.221/100.424GiB, maximum swap growth2.69/0MiB. One model
job ran at a time under the unchanged guard; private cache files stay in the
workspace and are excluded from published evidence.

```sh
make -j2 ds4-server ds4-agent speed-bench/replay_checkpoint_bench
mkdir -p .workspaces/item50-probe
python3 speed-bench/ideas/pressure_run.py .workspaces/item50-probe-guard -- \
  speed-bench/replay_checkpoint_bench \
  --model "$PWD/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf" \
  --out .workspaces/item50-probe
python3 speed-bench/ideas/pressure_run.py .workspaces/item50-server-guard -- \
  python3 speed-bench/experiments/replay-persistence-hour50-20261021/serve.py \
  --model "$PWD/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf" \
  --output .workspaces/item50-server
```

[Portable cache recipe](../../../recipes/03-state-cache.md) and
[replay recipe](../../../recipes/12-bounded-decoder-replay.md). The cache design
credits oMLX; bounded replay follows the separately pinned DeepInfer research
in those recipes. This item changes persistence, not replay mathematics.
