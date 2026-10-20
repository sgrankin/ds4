# Recent exact context frontiers, 2026-10-02

Status: **native regression passed under the main pressure guard at 17:06:48
UTC**, following source recovery after the host restart. The prior aligned
engine benchmark and CLM feature remain independently validated in parent
commits 93c38b7d and 72ebbc1c. This new regression validates cache state and
fallback behavior, rather than a task-level speedup or general agent quality.

The prototype extends opt-in live-context editing with recent complete
checkpoints. `DS4_AGENT_CONTEXT_FRONTIERS=2` (up to 4) enables capture at the
evaluated system prefix and completed assistant/tool boundaries.
`DS4_AGENT_CONTEXT_FRONTIER_MB` defaults to 128 MiB and accepts 1..1024 MiB.
The public serialized-byte count is checked before allocating a snapshot.
Old entries are evicted to obey both the count and byte budgets, including
the staged new allocation and exact token-prefix copies.

After a context edit, the largest cached prefix that exactly matches the new
tokens is restored. Everything after that prefix is evaluated normally.
The existing `DS4_AGENT_COMPACT_PREFIX_CACHE` bootstrap opt-in remains a
separate fallback. No suffix KV is reused under a changed earlier context.
Checkpoint arithmetic follows its saved prefill boundaries; a cold rebuild
with other boundaries may differ, so this feature remains opt-in.

Entries must match the active engine and session context size. Unsupported
model families, image-bearing contexts, TP/distributed execution, and
directional steering disable and release the cache. Reset, thinking-mode
change, and shutdown also release it. Empty or invalid checkpoints are skipped.
Failed loads invalidate the live graph and drop the entry before fallback.

The native regression was compiled alone and relinked to the stable main-root
core objects after the focused root build completed at 16:57:35 UTC, avoiding
a duplicate engine build. Header differences are additive function declarations,
with matching struct ABIs. Its SSD expert-cache configuration is explicitly
64 GiB. The root whole-agent build and ordinary CPU suite also passed around
17:00 UTC; `root-context-build.log` and `root-agent-test.log` preserve provenance.
Ordinary build: `make -j2 ds4-agent ds4_agent_test tests/test_agent_context_frontiers`.
The native regression captures 2048/2304/2816-token boundaries, verifies
eviction and longest-prefix selection, then compares complete restored and
512-token suffix states with partition-matched replay. It also checks wrong
prefix/context, corrupt payload fallback, image exclusion, and a 1 MiB cap.
Run it only through the parent pressure guard and a granted serial model slot.

The passed regression retained two checkpoints under 64 MiB and selected the
2304-token exact prefix. Restoration took 6.610 ms; normal evaluation of the
512-token changed suffix took 17.130 s. Complete serialized state, including
all vocabulary logits and durable DS41 graph spans, matched the saved prefix
and partition-matched fresh replay bit-for-bit. The corrupt-payload case
invalidated the cache and successfully rebuilt the requested token transcript;
that fallback check compares tokens, not logits against another partition.
Count eviction, wrong-prefix/context rejection, image exclusion, and the
1 MiB skip-before-allocation case also passed. `native-result.json` and
`native-test.log` preserve the exact assertions and command provenance.

The guarded run lasted approximately 156 seconds: all pressure levels were 1,
swap did not grow, peak wired RAM was 77.35 GiB and peak compressor RAM 9.15 GiB.
There were no adverse samples or pressure stops. The measured restore/suffix
times are regression diagnostics, not a controlled speed comparison.

No DS4/model process was launched by this agent near the 16:00 UTC panic
window. Its previous CLM inference finished at 15:51:45 UTC; the slot was
released at 15:52:41 UTC. The unfinished frontier code had never been built
or executed before the restart.
