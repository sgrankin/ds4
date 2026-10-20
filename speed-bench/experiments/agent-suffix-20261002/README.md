# Opt-in live-context suffix reuse

`DS4_AGENT_CONTEXT_SUFFIX_REUSE=1` enables one approximate suffix graft when
an accepted `--live-context-file` edit has a compatible retained exact source.
The default is off. Resident rerouting is explicitly incompatible: an absent
or zero `DS4_METAL_V41_RESIDENT_ROUTE_GAP` is required. Native controls use a
cleared DS4 environment and a fixed forward profile.

The agent selects an existing exact frontier that matches a prefix of the
old transcript and ends in the unchanged token suffix. Source/destination
starts must be even, and displacement must be zero or an even left shift.
At least 64 tokens are reused, and the default minimum fresh tail is 128.
`DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL=64` is an explicit lower override;
accepted values are 64 through 4096. It is a minimum: all tokens after the
selected source endpoint are evaluated freshly. A full checkpoint is never
silently shortened to construct a source. Trace fields report actual reused
and fresh counts, endpoints, bytes, and whether the latest user turn is
entirely beyond the graft endpoint.

With this mode enabled, the existing exact-frontier feature defaults to two
slots when `DS4_AGENT_CONTEXT_FRONTIERS` is absent. An explicitly configured
slot count, including zero, is respected. `DS4_AGENT_CONTEXT_FRONTIER_MB`
remains128 MiB by default. Source and rollback snapshots, retained snapshot
capacities and their owned token prefixes, the edited-prefix allocation,
8 MiB native serialization scratch, conservative token-load scratch, and
stdio/struct overhead are accounted before graph mutation. Actual owned
capacities are rechecked after capture. A source is owned before any restore
can evict or move the frontier entry.

The original exact checkpoint is captured before source loading. B′ is
evaluated freshly from the best unchanged exact frontier, C is grafted,
and the remaining tail is evaluated normally. Cancellation or a graft error
restores the original checkpoint; failure to restore invalidates it and
rejects the token transaction. Interruptions stop the tool pipeline before
its acknowledgement is committed. Unsupported states, insertions, odd
displacements, short/absent sources, insufficient budget, or an already
approximate session use the existing exact edit path. A second edit of an
approximate session explicitly invalidates/restores an exact prefix before
replay, even if the changed text is after the live endpoint. Images, TP,
distributed sessions, steering and core speculation gates retain exact
fallback. `DS4_AGENT_COMPACT_PREFIX_CACHE=1` can restore the validated system
checkpoint when no retained unchanged exact frontier survives.

Approximate graph state is private. Exact frontier capture rejects it before
the duplicate-token fast path, and core canonical payload/snapshot export is
blocked. Named agent sessions use the existing zero-KV text format, preserving
transcript/title metadata without a forced model replay during save. Reopening
such text while the live session is approximate first invalidates it, then
evaluates the saved text freshly. If rendered text would reinterpret ordinary
literal control spellings as controls, saving fails before overwriting an
existing file. The live mirror and transcript remain available.

This mode is not a forgetting mechanism. The separately validated Record X
fixture still raises the old code's sequence probability after a 128-token
fresh tail, although greedy output is correct. See the compact evidence in
`9951aedc`, `record-x-wrap-manifest.json`, and `RECORD_X_WRAP_RESULTS.md` in
the neighboring suffix-reuse experiment. Combined resident+suffix policy
behavior is unvalidated and is not enabled here.

The ASan+UBSan model-free suite exercises the actual agent helper and importer
with mocked session/tokenizer seams: same/even-shift selection, source
eviction/memmove,12 transaction/error/cancellation modes, byte-identical undo,
fail-closed recovery, minimum 128 / override 64 selection, resident CLI rejection,
low-budget preflight, approximate export blocking, zero-KV save/reopen,
literal-control rejection without overwrite, and exact second-edit fallback.
The complete agent builds and the ordinary `ds4_agent_test` suite passes.

`speed-bench/agent_suffix_context_smoke.py` prepares and drives a three-turn
real agent run with an isolated cache and files: old-code retrieval, neutral
notes to retain an earlier exact source, and a tool-driven live-context edit.
The new answer appears only in the isolated updater script and edited context.
It checks the original 2718 control, actual approximate reuse with >=128 fresh
tokens, a wholly fresh latest query, a 3141 final answer, protected/outside-byte
preservation, and zero-KV named persistence. `--prepare-only` opens no model.
Actual execution needs an explicit root GPU slot and the MAIN pressure guard.

Native semantic/reuse checks now pass in the actual production three-turn
agent: old answer2718, one earlier-record edit, final answer3141,556 reused
tokens and a wholly fresh latest query. The original runner rc3 is preserved
because it also expected a save that headless EOF never requested. See the
[production result](../../ideas/evidence/agent-suffix-native-three-turn-20261002/NATIVE_THREE_TURN_RESULTS.md).
A separate inspectable native frontend invoked the actual save function,
producing zero-KV text while preserving the approximate graph/logits/checkpoint
bytes and avoiding replay. Actual reopen remains unrun; see the
[native save result](../../ideas/evidence/agent-suffix-native-persistence-20261002/NATIVE_PRIMARY_RESULTS.md).
These results do not establish general task quality, complete-turn speedup,
or multi-edit approximate cache fidelity. The later integrated runtime also
passes native-off phase-end vector and graph parity.
