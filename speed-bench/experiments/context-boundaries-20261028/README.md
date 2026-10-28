# Semantic checkpoint boundaries for selective context edits

Implementation checkpoint, 2026-10-28. **Combined source24f86d2d passes CPU
qualification and serial/parallel functional smoke runs. The semantic-end floor
fix awaits qualification; five-pair performance remains pending. No default promotion.**
The design is [context compaction and checkpoints](../../../docs/CONTEXT_COMPACTION_AND_CHECKPOINTS.md).
This slice supplies server cache support for a client/harness that materializes
selective edits; it does not add server-hosted compaction tools or a scheduler.

## Implemented first slice

- DeepSeek rendering records semantic message/tool-invocation byte ends outside
  model text. Tokenization observes already-produced token bytes and selects
  the existing token end at or before each semantic end. A crossing BPE token
  stays fresh; ends are deduplicated. This does not split BPE or insert markers.
  The all-controls admission end still rounds up, keeping pending instructions
  out of earlier task-profile sources.
- Explicit cache breakpoints retain their existing independent policy. Automatic
  capture defaults to at most four recent boundaries, spaced by 2,048 tokens.
- At selected evaluated ends, capture immutable pages in bounded hot RAM.
  Existing SSD/hot chunks are shared. Unbacked-page eviction retires whole
  unpinned endpoints; incomplete sources are never returned. A later durable put
  can promote the same endpoint to SSD. Final ordinary endpoints stay durable.
- Optional ongoing-task reuse binds each source to ordered actual system,
  developer and user controls plus rendered tool declarations/response schema,
  not tool-result wrappers, literal role spellings or an arbitrary older match.
  Only captures after all actual system/developer/user controls were evaluated
  receive that identity. A late control therefore conservatively excludes
  earlier checkpoints even when its payload and ordered identity are unchanged
  by moving it around assistant/tool content. Tool results do not extend this
  admission frontier. The v3 identity excludes older experimental aliases.
  New, changed or deleted controls/declarations change the identity; the latest
  actual user token start remains the fallback freshness cutoff. The current one-span matcher remains.
- Capture logs separate serialization/copy time from page hashing/publication;
  source logs distinguish transient and durable sources. Source staging remains
  bounded and is configurable for longer contexts.

This still performs full capture/serialization/hash work. Shared hot pages and
avoiding per-boundary SSD publication are not an append-tail GPU collector.
Measure after physical main-FP4 integration before adding such a collector.
Typed main/index canonical/private versions are accepted; decoder-replay
checkpoints remain on their existing separate policy.

## Configuration

Experimental settings are explicit and ordinary defaults remain unchanged:

| Setting | First comparison |
| --- | --- |
| `DS4_SERVER_CONTEXT_BOUNDARIES` | `0` control / `1` candidate |
| `DS4_SERVER_CONTEXT_SUFFIX_ONGOING_USER` | Separate `0`/`1` freshness comparison |
| `DS4_SERVER_CONTEXT_BOUNDARY_LIMIT` | `4` default; maximum `16` |
| `DS4_SERVER_CONTEXT_BOUNDARY_MIN_TOKENS` | `2048` default; `512` denser trial |
| `DS4_STATE_CACHE_HOT_MB` | Explicit bounded budget, e.g. `512`; `0` disables boundary capture |
| `DS4_SERVER_CONTEXT_SUFFIX_REUSE` | Existing required suffix feature |
| `DS4_SERVER_CONTEXT_SUFFIX_STAGING_MB` | `512` default; increase explicitly only with measured memory headroom |
| `DS4_SERVER_CONTEXT_CAPTURE_PROFILE` | `1` to record existing endpoint costs even with boundaries off |

Keep the model/quant, ordinary prefill chunk, expert budget, main/index formats,
suffix padding/right-shift policy, reasoning and sampling fixed across arms.
The minimum64-token span and even relocation geometry still apply; failed
eligibility falls back to ordinary evaluation. Eligibility/coverage are results,
not reasons to silently alter a measured prompt.

## Qualification and reproduction

Combined source24f86d2d passed14 serving-boundary cases, both prompt-cache policy
tests, the full cache transient/packed/restart fixture, `ds4_test --server`, and
typed DSML/Flint output checks. Current fixture coverage:

- Native serving boundary fixture: unchanged rendered bytes, typed boundaries,
  control identity stable across tool-result edits and changed by new or
  changed user turns, system/developer instructions and tool declarations,
  including repeated identical user text. Late system/developer placement keeps
  the actual-user fallback cutoff separate from the all-controls admission end.
- Production cache ABI fixture: transient restart loss, pin pressure, bounded
  whole-source eviction, same-key promotion, durable restart, native/private
  v4/v5 packed-main spans and old formats.
- `ds4-server` CPU build/typecheck; no model/GPU run by this implementation agent.

`run.nu` is HTTP-only; the caller owns one pressure-guarded server and its
lifetime. It retains the initial request and eight materialized edits as local
JSON archives. Each edit trims an early dump and changes a later result while
retaining investigation text between them. This tests availability of an
intermediate source rather than a multi-span planner. Prior measured short
answers are not appended to the synthetic investigation. Cases include ordinary,
favorable, corrected/deleted facts and new/changed/deleted user instructions.
Ordinary retains the serial assistant-investigation/tool-result layout. Favorable
uses one assistant turn calling A/C/D in parallel, followed by three separate
tool results: trimmed A, unchanged C with three times the configured rows, and
changed D with the final authoritative registry appended. It ends directly
after D. The shared User wrapper begins before A, putting C beyond the legacy
cutoff; useful C reuse requires an intermediate source and ongoing-task policy.
Requests use `store:false`: the original dump is retained in the client's
`edit-0.request.json`, rather than a newly added server-hosted transcript log.
With `--archive-control`, the client reads that actual file through the standard
JSON reader and supplies its original dump as a fresh tool result. The separate
retrieval check is retained outside all eight-edit timing totals; use the
`corrected` case to make its original code differ from the current one.

```nu
nu speed-bench/experiments/context-boundaries-20261028/run.nu http://127.0.0.1:18345 MODEL .workspaces/context-boundary-fixtures --fixtures-only
nu speed-bench/experiments/context-boundaries-20261028/run.nu http://127.0.0.1:18345 MODEL .workspaces/context-boundary-run --case ordinary --rows 72 --edits 8
nu speed-bench/experiments/context-boundaries-20261028/run.nu http://127.0.0.1:18345 MODEL .workspaces/context-boundary-retrieval --case corrected --rows 24 --edits 1 --archive-control
```

For the small `rows24` smoke, set boundary spacing512 so the retained C endpoint
is eligible; fewer than2048 tokens may follow it. Use `rows72` and the default
four boundaries/spacing2048 for the initial five-pair workflow comparison.
Retain startup/cold population as well as all eight edits. A fresh C boundary
must log `transient=1` without an ordinary same-token publication, then be chosen
with `source_ssd=0`. Final ordinary endpoint publication remains durable.

An `ongoing_user=1` event alone does not establish extra coverage: candidate
policy uses the actual-user cutoff, while legacy-off can reuse part of ordinary
C before a later synthetic tool-result User wrapper. Compare actual source and
destination geometry. Favorable places C after the sole result-group User
wrapper, making this policy difference explicit. Separately isolate freshness
with the parallel-results diagnostic:

```nu
nu speed-bench/experiments/context-boundaries-20261028/freshness-fixtures.nu .workspaces/context-boundary-fixtures .workspaces/context-freshness-fixtures
```

Post its `initial.request.json`, then `edited.request.json`, to one fresh guarded
server for each arm. Keep boundaries enabled and all settings fixed, changing
only `DS4_SERVER_CONTEXT_SUFFIX_ONGOING_USER=0/1`. The canonical render opens one
User wrapper before both parallel results. The retained second result is after
that legacy cutoff; the final assistant cue remains valid. Logs must demonstrate
off-policy fallback and on-policy `ongoing_user=1` import, with the same actual
user/tools. Both answers should be `CODE=CEDAR-4826`. This is a coverage/quality
smoke; it supplies no comparative latency claim without five samples per side.

`run-five.nu` owns one sequential server child at a time under the existing
pressure guard. It takes a frozen binary with adjacent `shaders/`, clears
inherited `DS4_*` variables, records the exact server arguments/settings, and
runs five fresh processes per side in AB/BA/AB/BA/AB order. Both arms use main
and index FP4, HOT512, 32GB experts, context32768, configured prefill8192, native routing,
suffix reuse/padding/right shifts and high reasoning. No MTP head is loaded.
Only the semantic-boundary/ongoing-user bundle changes. Both arms retain the
production server's two-session batched mode: its ordinary coordinator quantum
is2048, so an8192 configured chunk does not mean an uninterrupted8K prefill.
The larger chunk avoids512-sized model chunks within that quantum;
semantic-boundary splitting costs remain included. `--chunk` changes the
configured setting in both arms, without changing scheduler policy.

Each process runs ordinary and favorable workflows under distinct cache keys
and fixed per-case instruction labels. The labels are identical across arms
but prevent the second case from reusing a long exact prefix from the first.
Both initial contexts are cold; the second shares warm model weights. Startup
time is recorded separately. Each case retains initial population plus eight
edits, with no archive-retrieval request in its timed totals. Process logs live
under `processes/pair-N/ARM`; case data preserve the existing analyzer layout.

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/context-boundary-guard -- nu speed-bench/experiments/context-boundaries-20261028/run-five.nu FROZEN/bin/ds4-server .workspaces/context-boundary-five
```

For a smoke, add `--pairs 1 --rows 24 --edits 1 --spacing 512 --chunk 2048`. `--dry-run`
retains exact manifests and request fixtures without loading a model or opening
HTTP connections. `--case corrected` (or another existing case) selects one
later control instead of the ordinary/favorable pair. Completed five-pair runs
invoke the existing analyzer separately for each case; smaller smoke runs make
no comparative performance claim.
`--limit` changes the candidate boundary cap, default4 and maximum16, for later
frequency/coverage comparisons. `execution-order.json` retains actual process
start/finish order rather than only the planned AB/BA schedule.

Case summaries retain every HTTP sample and mean, median, sample standard
deviation and range. With `--processes PATH`, the same analyzer also reads the
known capture/suffix lines in each process's `server.stderr`. It saves all
capture and source events with line numbers, actual imported/fresh counts,
source positions, endpoint durability flags, fallback reasons, serialization and
publication times, and observed hot/SSD page-byte peaks. These process totals
combine ordinary/favorable; they are not attributed to a single case. Peaks
are cache counters sampled at captures, rather than OS allocation or IO bytes.
The analyzer fails visibly if completed workflows have no capture log records,
or selected boundaries have no transient capture records. Failed captures are
retained as measurements, alongside successful ones.
The wire log field `source_ssd` means the chosen endpoint has been durably
published. A durable endpoint can still be materialized from hot RAM; its flag
does not prove an SSD read. Aggregates therefore use `durable_sources` and
`ram_only_sources`. A zero flag does identify a RAM-only transient. Graft
geometry and imported/fresh counts describe actual reuse; global cache-read
counters include candidate enumeration and cannot identify useful graft hits.
These logs do not expose per-endpoint eviction.

## Diagnostic history

The first live launch stopped after the successful ordinary control because
redirected Nu child execution left a stale `LAST_EXIT_CODE`. Both responses
completed correctly; artifacts remain under
`.workspaces/context-compaction-20261028/live-smoke`. The driver now uses the
standard `complete.exit_code`, retaining child stdout/stderr and exit status.

The corrected old-layout smoke under `live-smoke-fixed` completed both cases
and arms, all eight answers correct. It used one pair, rows24, one edit,
spacing512 and chunk2048. Candidate chose RAM-only C sources: ordinary imported
1265 tokens (`982:2247` to `428:1693`), and the older serial favorable imported
1865 (`982:2847` to `428:2293`), both with `ongoing_user=1` and no padding.
The390-token task checkpoint was exact-restored before each candidate edit.
These are functional diagnostics, not comparative performance measurements.
The favorable layout was subsequently changed to the parallel form above.
Its first smoke also completed all four answers correctly, but semantic A/C
ends were dropped by the exact-byte-end mapper. Native token4697 is `>\n\n`,
straddling `</tool_result>` and the next separator. Candidate still imported
1457 tokens from an ordinary durable quantum endpoint (`1042:2499` to
`488:1945`), so the gap was a lost semantic C endpoint, not total reuse failure.
The mapper now floors to the preceding token end; that fix needs a repeated
functional smoke before the five-pair comparison. Selection policy is unchanged.

Use five fresh server samples per side in alternating order. Store each workflow
under `pair-N/control` or `pair-N/candidate`; the launcher must retain actual
order, settings, source snapshot and server logs. Then:

```nu
python3 speed-bench/experiments/context-boundaries-20261028/analyze.py .workspaces/context-boundary-five-pairs
```

Report initial population separately from edited requests and their combined
whole-workflow time. Retain all samples, variation, answer correctness, output
length, actual source selection/imported coverage, capture costs and hot/SSD
bytes. Small slower means within normal variation are neutral, particularly if
coverage or memory improves. No inference gain or broad quality equivalence is
claimed by the CPU fixtures or calculated row sizes.
