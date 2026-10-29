# Semantic checkpoint boundaries for selective context edits

Implementation checkpoint, 2026-10-28. **The floor-qualified build passes CPU,
serial/parallel, changed-fact/control, archive, protocol and actual restart
checks. The complete 15-process Pareto campaign makes two repeated edits 38.0%
faster with fuller4, with 81.8% more initial population and 2.14% longer whole
time at that horizon. All 90 replies pass. A separate eight-edit diagnostic
passes 18/18 replies and retains the original RAM-only C through all 16 edits.
Server `persistent-fast` now enables fuller4/spacing512 with HOT512; ordinary
defaults and other presets receive no new boundary defaults.** See
[five-round results and plot](RESULTS.md).
The design is [context compaction and checkpoints](../../../docs/lab/journal/2026-10-28-deepseek-semantic-checkpoints.md).
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

Server `persistent-fast` enables boundaries/ongoing reuse, limit4, spacing512
and HOT512 MiB; explicit environment overrides win. Its existing 64GB expert
target, resident substitution policy and configured512 chunk remain unchanged.
The benchmark instead holds native routes/32GB and configured8192 chunk
(ordinary coordinator quantum2048). Use `--prefill-chunk 8192` for the tested
configured chunk; that alone is not an exact benchmark configuration. The agent
and other presets receive no new semantic-boundary defaults.

Keep the model/quant, ordinary prefill chunk, expert budget, main/index formats,
suffix padding/right-shift policy, reasoning and sampling fixed across arms.
The minimum64-token span and even relocation geometry still apply; failed
eligibility falls back to ordinary evaluation. Eligibility/coverage are results,
not reasons to silently alter a measured prompt.

## Qualification and reproduction

The floor-qualified build passed14 serving-boundary cases, all three prompt-cache policy
tests, the full cache transient/packed/restart fixture, `ds4_test --server`, and
typed DSML/Flint output checks. Current fixture coverage:

Final installed source `9b826389c9e22c508a895610994145fd2b9529cd` adds the
reasoning-enabled identity and padding-adjusted control-end guards: 14 serving
cases and 4 policy tests pass,
and all five installed executables build/help successfully. The frozen latency
cohort remains `d5b8e8a3`; it never exercises those changed transitions.

- Native serving boundary fixture: unchanged rendered bytes, typed boundaries,
  control identity stable across tool-result edits and changed by new or
  changed user turns, system/developer instructions and tool declarations,
  including repeated identical user text. Late system/developer placement keeps
  the actual-user fallback cutoff separate from the all-controls admission end.
- Production cache ABI fixture: transient restart loss, pin pressure, bounded
  whole-source eviction, same-key promotion, durable restart, native/private
  v4/v5 packed-main spans and old formats.
- `ds4-server` CPU build/typecheck; no model/GPU run by this implementation agent.

### Completed live-quality qualification

Frozen `d5b8e8a3`, normal high reasoning and native routes, main/index FP4,
HOT 512 MiB, no DSpark/MTP head. These are correctness/coverage checks, not
comparative latency samples. Both same-runtime freshness arms keep boundaries
enabled and change only `DS4_SERVER_CONTEXT_SUFFIX_ONGOING_USER`:

| Probe | Observed result |
|---|---|
| Ongoing policy off | Correct `CODE=CEDAR-4826`; 410 tokens reused exactly, no aligned suffix source |
| Ongoing policy on | Correct same answer; imports 1,688 C tokens (`1002:2690`→`448:2136`), `ongoing_user=1`; this source is durable (`source_ssd=1`) |
| Floor-mapped parallel C | All four initial/edited answers correct; captures C at 2872 as RAM-only and imports 1,830 tokens (`1042:2872`→`488:2318`), `source_ssd=0` |
| Changed fact | Initial CEDAR→edited LARCH; both replies correct |
| Deleted fact | Initial CEDAR→edited NONE; both replies correct |
| New user instruction | Initial CEDAR→new MAPLE instruction; both replies correct |
| Changed user instruction | Initial CEDAR→changed MAPLE instruction; both replies correct |
| Deleted user instruction | Initial PINE→remaining CEDAR authority; both replies correct |
| Client archive retrieval | Rereads original dump from actual saved `edit-0.request.json`, supplies it as a fresh tool result, correctly answers CEDAR while current edited view answers LARCH |
| Stateful Responses protocol | Stored first response answers READY; `previous_response_id` continuation answers MAPLE |
| Actual process restart | Transient C unavailable; edited request logs `no-aligned-source`, cached=0, recomputes and correctly answers LARCH |
| Eight-edit retention, final installed server | Ordinary 9/9 and favorable 9/9 replies correct; all 16 edited grafts reuse the original RAM-only C (`source_approximate=0`), no capture errors; functional only |
| Final padded control path | CEDAR→LARCH2/2 correct; actual `prefix_pad=1`,666-token graft `692:1358`→`426:1092`; CPU fixture checks shifted control-admission end |
| Final thinking-control transition | Medium→high2/2 correct with thinking on throughout; high request cached0 / `no-aligned-source`; none/medium distinction checked by CPU renderer only |

The five fact/user cases pass all 10 replies; archive and protocol controls are
additional checks. Generated reasoning can differ: these probes grade semantic
answers, not identical output IDs or full-distribution quality. Restart checks
loss of an optimization rather than loss of the client archive. Durable final
endpoints survive independently; this probe does not claim that every edited
view has a reusable durable suffix source.

Evidence remains local under `.workspaces/context-compaction-20261028/`:
`floor-off/`, `floor-on/quality/quality-results.json`, each case's request,
response and result files, `floor-on/quality/protocol/`, and `floor-on-restart.log`.
`floor-on/quality/restart-edited.result.json` records the correct LARCH answer
with 0/3114 cached input tokens. The original source geometry is in
`floor-off.log`/`floor-on.log`; RAM C floor geometry and transient capture are in
`parallel-floor-smoke/processes/pair-1/candidate/server.stderr`.
All owned quality servers have stopped and their unchanged guards exit 0.
The final installed run's retention/control artifacts are in
`retention-eight/` and `retention-eight.log`. Its shutdown summary has no pinned
endpoints and536,007,540 hot bytes within512 MiB;9 aggregate endpoint evictions
also include later QA and do not reveal per-source eviction history.

The completed latency campaign is separate: 15 fresh processes, five each for
off / sparse1 / fuller4, alternating forward/reverse order, spacing 512, rows=24,
ordinary serial and favorable parallel workflows with cold population plus
**two edits**. Raw data and guard are
`.workspaces/context-compaction-20261028/pareto-fast/` and `pareto-fast-guard/`.
The preceding rows=72/eight-edit `pareto/` run was interrupted as calibration
because its full cost exceeded the checkpoint budget; it is retained separately
and is not stitched into the fresh cohort. A separate eight-edit retention
diagnostic passes all 18 replies and 16 original-C imports; any eight-edit speed
projection is calculated, not a
measured five-pair comparison. [RESULTS.md](RESULTS.md) retains all measured
rounds, variation/order, actual source geometry and capture/storage costs.
No speed claim comes from the smokes above. See the [portable recipe](../../../recipes/16-semantic-context-checkpoints.md)
for mechanism and reproduction without the historical design discussion.

`run.nu` is HTTP-only; the caller owns one pressure-guarded server and its
lifetime. It retains the initial request and configured edits (default 8;
current timing trial 2) as local
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
retrieval check is retained outside all timed edit totals; use the
`corrected` case to make its original code differ from the current one.

```nu
nu speed-bench/experiments/context-boundaries-20261028/run.nu http://127.0.0.1:18345 MODEL .workspaces/context-boundary-fixtures --fixtures-only
nu speed-bench/experiments/context-boundaries-20261028/run.nu http://127.0.0.1:18345 MODEL .workspaces/context-boundary-run --case ordinary --rows 72 --edits 8
nu speed-bench/experiments/context-boundaries-20261028/run.nu http://127.0.0.1:18345 MODEL .workspaces/context-boundary-retrieval --case corrected --rows 24 --edits 1 --archive-control
```

For the small `rows24` smoke, set boundary spacing512 so the retained C endpoint
is eligible; fewer than2048 tokens may follow it. The time-bounded five-round
campaign explicitly uses `--rows 24 --edits 2 --spacing 512`, retaining the
useful early prefix anchor. The larger rows72/eight-edit run remains a stopped
calibration, not a performance result. Runtime and launcher spacing defaults
remain 2048; this trial is not a default promotion or an additional spacing sweep.
Retain startup/cold population as well as both edits. A fresh C boundary
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
inherited `DS4_*` variables, and records the exact server arguments/settings.
It reads the prefix's required `source-revision.txt` as `runtime_revision` and
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
time is recorded separately. The current campaign retains initial population
plus two edits per case, with no archive-retrieval request in timed totals.
The driver still defaults to eight edits; pass the explicit current settings
below. Process logs live
under `processes/pair-N/ARM`; case data preserve the existing analyzer layout.

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/context-compaction-20261028/pareto-fast-guard -- nu speed-bench/experiments/context-boundaries-20261028/run-five.nu FROZEN/bin/ds4-server .workspaces/context-compaction-20261028/pareto-fast --include-sparse --spacing 512 --rows 24 --edits 2
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
`--include-sparse` adds one fixed third arm: `sparse` enables boundaries/ongoing
reuse with limit1, alongside ordinary `candidate` limit4 and disabled `control`.
Five rounds alternate `[control, candidate, sparse]` and its reverse, giving five
fresh samples per arm in15 processes. It uses the same two workflows and all
other settings. This fixed comparison requires `--limit 4`; effective per-arm
limits and actual execution order are retained in the manifest and event log.
The analyzer keeps all15 process samples and writes separate
`summary-candidate.json` and `summary-sparse.json` comparisons against control,
alongside the combined summary. It measures the population/amortization tradeoff
of retaining extra early prefix anchors rather than assuming they are wasteful.
This run measures the observed initial-plus-two-edit horizon; a later eight-edit
retention diagnostic is separate. Extending the two-edit timing curve to eight
edits is an explicitly calculated projection, not an eight-edit five-pair result.
The three concrete points are off, sparse1 and fuller4 at the same explicit
spacing512 trial; control remains unchanged.

Case summaries retain every HTTP sample and mean, median, sample standard
deviation and range. With `--processes PATH`, the same analyzer also reads the
known capture/suffix lines in each process's `server.stderr`. It saves all
capture and source events with line numbers, imported-span/planned-tail counts,
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
Wire `reused` becomes `imported_tokens`: the transplanted matching span only.
Wire `fresh` becomes `planned_tail_tokens`, equal to edited length minus
destination_end; it excludes the prefix. Wire `graft_fresh` becomes
`graft_tail_tokens`, the tail evaluated inside the graft before the final fresh
tail. `destination_start` is prefix geometry; some of that prefix may be
exact-restored, as the retained raw prefix-hit log lines show. API cache-write
counts also include prefix tokens even when restored. These counters do not
identify total forward-token work; whole HTTP workflow time is primary.

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
The mapper now floors to the preceding existing token end. Its repeated
`parallel-floor-smoke` completed all four answers correctly. All nine semantic
ends were retained; candidate selected three checkpoints, including C at2872,
and imported1830 C tokens (`1042:2872` to `488:2318`) from a RAM-only endpoint
with `ongoing_user=1` and no padding. The crossing `>\n\n` token remains fresh;
rendered bytes and ordinary token IDs are unchanged. The three CPU cache-policy
tests and14 boundary cases pass. This is functional evidence, not a comparative
performance result; the subsequent five-sample population/two-edit campaign
is reported separately in [RESULTS.md](RESULTS.md). The eight-edit retention
diagnostic is separate. Selection policy is unchanged.

Post-campaign review found two admission-only gaps: medium and disabled
thinking shared numeric effort0, and successful prefix padding did not shift
the all-controls endpoint before publishing another source. Final source
`9b826389` hashes the enabled bit and shifts that endpoint; focused CPU and
live checks above pass. Rendering/tokenization and inference math are unchanged.
The frozen high-reasoning, unpadded `d5b8e8a3` timing cohort exercises neither
changed transition, so its five-round results remain separate and applicable
to that measured configuration.

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
