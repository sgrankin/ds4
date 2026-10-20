# CLM live-context file, 2026-10-02

`ds4-agent --live-context-file FILE` exposes the current complete rendered chat
as a text file. The agent receives a system instruction describing its use.
It can use the existing read, edit, write or bash tools to replace redundant
history with notes or delete irrelevant observations. Edits are imported after
all tool calls in a round complete, before their observations are appended.
Edits made while the agent is idle are also imported at the start of the next
user turn, before automatic compaction, mirror refresh or the new user message
is appended. A rejected idle edit visibly stops that turn and leaves the
existing transcript and incoming user message uncommitted.
The mirror is refreshed at generation/tool boundaries and final completion.

The initial system/tool/preloaded-prefix section must remain byte-for-byte
unchanged. Its original tokens are retained rather than reparsed, because
tool instructions may quote special-token spellings as ordinary text.
Edits also retain the original opaque tokens in the longest unchanged body
prefix and suffix. Only the changed middle is tokenized, after widening to
whole token and UTF-8 boundaries. The result must render to exactly the edited
file bytes before the transaction can commit. Literal assistant/DSML examples
outside that middle therefore retain their original meaning. A preserved
literal control spelling inside the reparsed middle causes conservative
transactional rejection; this can occur with multiple separated edits, which
should instead be made in smaller tool rounds. Newly typed control spellings
use the rendered-chat parser in the changed span.
NUL-containing, oversized, missing, and altered
prefix edits leave the existing transcript untouched and produce an explicit
rejection in the tool observation, or a turn error for an idle edit. The
ordinary next session sync evaluates the resulting context.
`DS4_AGENT_COMPACT_PREFIX_CACHE=1` can restore the
validated bootstrap prefix before that sync. This uses exact prefix state;
by default no old suffix KV is shifted or retained under a different history.
The separate opt-in `DS4_AGENT_CONTEXT_SUFFIX_REUSE=1` path can reuse old suffix
state approximately; see the [storage-backed agent measurements](../agent-suffix-hour6-20261003/README.md)
for settings, actual fresh-tail sizes and the exact-rebuild fallback.

This is opt-in and text-only: `--raw-prompt` and `--vision` cannot be combined
with it. The file is the live transcript, not an archive or an externally
polled editor. Edits must finish inside a tool round or while idle before the
next user turn is submitted. Reasoning remains in the mirror until deliberately
edited or compacted. Automatic stripping is absent.
Changing history can change task behavior; passing the smoke does not establish
general agent quality or a task-level speedup.

Build with `make -j2 ds4-agent ds4_agent_test tests/test_agent_live_context`.
The ordinary unit suite checks CLI exposure and existing agent behavior,
including that a rejected idle file is handled before user append or model work.
`tests/test_agent_live_context MODEL` checks the actual model's chat-token
round trip, accepted deletion, unchanged mirror, opaque literal assistant/DSML
prefix/suffix preservation, a UTF-8 character edit, transactional ambiguity
rejection for separated edits, protected prefix, NUL,
missing-file and context-budget rejection. It opens the model but performs no
token evaluation; still schedule its Metal initialization with other runs.

`speed-bench/clm_live_context_smoke.py --model MODEL --output FRESH_DIRECTORY`
runs two native agent turns: first store a secret code alongside a redundant
blob, then ask the agent to edit the live file and return the preserved code.
It independently checks an accepted token-count reduction, removed blob,
retained fact and correct final answer, with isolated cache files and trace.
Use a scheduled GPU slot; its default timeout is four minutes.

The Q2 native format test passed (`native-format-test.log`, 1985-token system
prefix). The real two-turn smoke passed in 90.81 seconds: one Python/bash
call edited 3478 tokens down to 2460, deleting 1018 redundant history tokens.
The secret fact survived and the final assistant returned `CODE=2718`.
There was one bootstrap-cache hit and no rejected edits. The following
prefill reused 1989 tokens and evaluated a 541-token suffix in 15.326 seconds.
`e2e/result.json`, the complete trace, and before/after mirrors preserve the
evidence. This is an operational smoke, not a controlled latency comparison
or proof that automatic context editing preserves quality on other tasks.

The lossless edit-splice regression also passed in a later guarded native run
(`literal-native-test.log`): 1985-token system prefix and 2014-token edited
transcript, with no token evaluation. It covered exact deletion identity,
opaque assistant/DSML prefixes and suffixes, a UTF-8 character replacement,
byte round-trip, and transactional rejection when two separated edits would
reparse a preserved literal marker. Protected-prefix, NUL, missing-file and
context-budget rejection also passed. `literal-manifest.json` records the
frozen helper, source, root-core and model provenance; `literal-guard` contains
the exact guarded command, result and pressure samples. The root whole-agent
build and ordinary CPU suite passed around 17:00 UTC as well.
