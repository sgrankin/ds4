# Local stack audit, 2026-10-20

Read-only revision audit; no jj mutation, fetch, rewrite, build, test or model job.
The JSON inventories beside this note record the pinned revisions and file-level
comparisons. Working-copy edits after the pinned tip are outside this audit.

## Recommendation

For ROOT's whole-file reconstruction, replace the long integration/experiment
history with **six dependency-ordered commits**, preserving the complete final
tree and keeping private, unpublished experiment branches separately reachable.
Separate runtime consumers from historical results; do not indiscriminately
squash everything into one runtime change or delete evidence.

Build this shorter stack against the old common base first, compare its final
tree with the saved tip, then rebase those feature commits onto fetched upstream.
This separates accidental feature loss from upstream conflict resolution.

## Pinned graph and net changes

| Item | Observed value |
| --- | --- |
| Initial working revision | `030ae74983d6df9526716d3add86cc05189497a8`, empty |
| Last nonempty first-parent revision | `6110237291321b6e93a989a5b706a26d07497ea4` |
| Initial `main@upstream` | `5379c1c9ac31`, stale until ROOT fetch |
| Actual common base | `0aaea5a238fb41a35106a551e73c8409dfb751ac` |
| Local ancestry after common base | 248 commits, including 15 merges |
| First-parent stack | 198 commits, including 11 merges |
| Initial upstream-only ancestry | 30 commits after the common base |
| First-parent commits touching runtime/tests | 48; 39 also mix benchmark paths |
| First-parent commits with no runtime/test paths | 150 |
| Net local changed paths | 15,101: 15,047 under `speed-bench/`, 54 elsewhere |
| Visible heads | 153: 122 empty, 31 nonempty, including unrelated remote branches |

`main` and `main@origin` initially point at `07e8e1315407`, partway through the
local stack; not every revision is unpublished. Remote model-family branches
are not cleanup candidates for this task.

**Use the actual common base for the local net diff.** Comparing the tip directly
against `5379c1c9` misleadingly lists upstream CUDA/TP/DSPark/test additions as
local removals. Against `0aaea5a2`, the local delta has no CUDA/ROCm/TP changes;
it changes the Metal/runtime/agent/server files, 25 tests, three shaders, usage
docs and benchmark material. Upstream preservation must be assessed separately
during the rebase.

## Preferred six-commit file grouping

This order avoids splitting the large shared `ds4.c` implementation into fragile
partial features. It is a dependency review, not an intermediate-build result.

| Order | Commit / complete file groups |
| --- | --- |
| 1 | **Backend and cache foundation:** `ds4.c/.h`, `ds4_gpu.h`, `ds4_deepseek41_gpu.h`, `ds4_metal.m`, `ds4_engram.c`, `ds4_resident_route.h`, `ds4_resident_native_weights.h`, `ds4_suffix_layout.h`, `ds4_suffix_match.h`, `ds4_state_cache.c/.h`, `ds4_state_cache_runtime.h`, `ds4_response_store.h`, `ds4_conversations.h`, changed `metal/{dense,dsv41,norm}.metal`, common `ds4_cli.c`/`ds4_bench.c`/`ds4_help.c`, final `Makefile`, backend/cache tests. |
| 2 | **Agent consumer:** complete `ds4_agent.c`, `tests/ds4_agent_test.c`, `tests/test_agent_{compact_prefix,live_context,context_frontiers,context_suffix_mock}.c`. |
| 3 | **Server consumer:** complete `ds4_server.c` and relevant server/store/suffix tests; standalone store headers were supplied in1. |
| 4 | **Presets and user-facing docs:** `ds4-local`, README, `docs/CLIENTS.md`, AGENTS and ignore rules. |
| 5 | **Reusable benchmark source and frozen inputs:** non-result helpers/drivers/fixtures under `speed-bench/`. |
| 6 | **Retained experiment evidence:** reports/results/refusals/logs/manifests/source copies and original large references under `speed-bench/experiments/` and `speed-bench/ideas/evidence/`. |

`Makefile` must arrive with the cache source: the final normal agent/server link
already includes `ds4_state_cache.o`. Adding that standalone object before its
consumers use it is harmless. Its final line1120 also declares the two new store
headers as prerequisites of normal server objects, so those headers belong in1
even while the old server does not include them. New benchmark targets are explicit targets, not
default `all` dependencies. Agent-including tests belong with the complete new
agent, even though their rules already exist in the foundation Makefile. The
public API additions used by the new consumers are supplied by commit1; the old
agent/server consumers use the earlier subset. No intermediate build was run.

## Semantic feature map for reviewing the foundation and consumers

The commit IDs are navigation anchors, not instructions to cherry-pick whole mixed commits.
Several files, particularly `ds4.c`, `ds4_agent.c` and `ds4_server.c`, require
splitting by feature rather than assigning the whole file to one row.

| Order | Feature boundary | Main files / existing anchors |
| --- | --- | --- |
| 1 | Native Metal SSD execution and tuning | Exact scalar queue/tails, selected tiles, overlap, workspace/cache lifetime and Engram controls: `ds4.c`, `ds4_metal.m`, `ds4_gpu.h`, `ds4_engram.c`, `metal/{dense,norm}.metal`; early `ca5238f2`–`d6750e03`. Preserve remaining opt-in diagnostic controls in this group, or a separate explicit experimental-controls commit if review becomes unwieldy. |
| 2 | Bounded resident routing with native coefficient bits | `ds4_resident_route.h`, `ds4_resident_native_weights.h`, scalar call sites and selected-prefill residency; integration `6fa3fe01`, native weights `f24e2272`, prefill `9c6918f0`. Keep original flags/defaults and all current sub1/2/3 controls. |
| 3 | Immutable suffix/session primitives | Canonical versus private diagnostic state, pair/pre-RoPE originals, prefix loading, left/right relocation and rollback: `ds4.c/.h`, `ds4_suffix_layout.h`, GPU declarations, `metal/dsv41.metal`; `6fa3fe01`, `52278b7b`, `4cfff6cb`. |
| 4 | Bounded shared RAM/SSD state cache | `ds4_state_cache.{c,h}`, `ds4_state_cache_runtime.h`, owned references/materialization and profile binding; `584dd8d4`. Retain exact/approx separation and single-process directory ownership. |
| 5 | Direct-agent editable context and compaction | Live mirror/protected prefix, stable opaque turn IDs, frontiers/latest-user freshness, cache-backed suffix selection, `/compact` and idle import: `ds4_agent.c`; `dddd2c0d`, `aac9b33b`, `f612a187`, `bf0d20cb`, `e203b806`, `d2952200`, `f1f0c756`. |
| 6 | Durable Responses and Conversations history | `ds4_response_store.h`, `ds4_conversations.h`, server endpoints, replay/previous IDs, pagination/conflict publication; `b57dd59f`, `0392c260`, `3fb450f7`. Keep public API limits documented. |
| 7 | Server suffix reuse and scoped continuation | `ds4_suffix_match.h`, server source capture/cache/profile gates, fresh-user policy, canonical input, one-newline parity padding and approximate visible continuation; `a892cf09`, `a56347a8`, `7adb0d85`, `00da420b`, `91c9cf21`, right shifts from `4cfff6cb`. |
| 8 | Native sampling controls and request thinking budget | Existing opt-in marker penalty/native sample accessors plus server `ds4_reasoning_budget`, one actual close/eval, reasoning usage and speculative bypass; `cffb558f`/`4d1e4fd5`, `bc6896ae`. Keep diagnostics/failed penalty evidence distinct from the request-budget feature. |
| 9 | Supported presets and usage documentation | `ds4-local`, README, `docs/CLIENTS.md`, help: named agent/server profiles, shader paths, SSD defaults and current compatibility recipe; `f24e2272`, `0b698113`, `292cef47`, `a01c86dd`, `e203b806`. |
| 10 | Reusable benchmark/qualification tooling and frozen inputs | Current helpers, drivers, graders and input fixtures under `speed-bench/`, with ordinary test dependencies. This is source, not disposable output. |
| 11 | Historical experiment results and provenance | Retain reports, raw small evidence, failures/refusals, original source copies, manifests and large-payload references. Include project ignore/experiment guidance changes here or in the tooling commit. Keep outside runtime feature diffs. |

The feature map is useful for checking that six whole-file commits retain all
current behavior; it does not require nine runtime commits or hunk extraction.
The SLRU introduction/removal (`89370fba`/`1f757780`) is an example where the
runtime net change is zero: `ds4_slru.h` is absent in both base and tip. Preserve
its measured regression as evidence without replaying a temporary runtime path.

## Supported surface checked in the current README and wrapper

- `context-fast`: native routing plus suffix reuse. `resident-fast`: bounded
  native-weight resident substitutions, suffix off. `persistent-fast`: suffix
  reuse plus decoder and selected-prefill resident substitution.
- Direct agent: editable mirror with random 16-hex turn IDs, complete protected
  prefix, built-in summary compaction and cache-backed pre-RoPE left/right reuse;
  latest-user freshness and ordinary fallback remain important behavior.
- Server: opt-in original-backed suffix reuse, immutable SSD checkpoints,
  full-input canonical/padding policy, scoped approximate continuation, partial
  Responses/Conversations persistence and request-local thinking budget.
- Cache: immutable shared content chunks, exact versus approximate endpoints,
  hot/disk quotas, restart and one owner per cache directory. Server clients can
  interleave under that owner; this is not cross-process GPU-page sharing.
- The presets do not choose thinking budgets. Private functional matching,
  original-reader/backfill/CLM isolation experiments are not implicitly promoted
  by the wrapper. Their source must remain available independently.

## Head cleanup: exact evidence, not description guesses

**Safe structural simplification:** all 122 empty heads have one parent and an
identical Git tree to that parent. Abandoning just such a tip discards no files;
its parent becomes the head. Named workspaces still need deliberate handling.
This does **not** make the parent experiment or all its ancestors disposable.
The exact cleanup list is `empty-heads.tsv`: full commit/parent/tree IDs, workspace
name/path and any matching main-stack tree. There are 167 registered workspaces;
113 empty heads have a workspace and nine do not. Preserve the default workspace
and ROOT's saved-tip reference while simplifying old protective workspaces.

Especially clear redundant heads:

| Head | Evidence / disposition |
| --- | --- |
| `7721cf5472f1`, `4efab5397826` | Empty-root children; both tree `4b825dc642cb6eb9a060e54bf8d69288fbee4904`. No source. |
| `aaf54720db5c` | Nonempty root snapshot, but its **entire tree equals existing ancestor `9231ab9a1760`**. Redundant snapshot head; retain the real ancestor/reference. |
| `fac5ff422ada` | Four added evidence paths; all four mode/blob identities already match the current tip. No exclusive file content in that change. These are the practical40 refusal/manifest/runtime proof and framework diagnostic. |
| `e8e89f28`, `b518b610`, `7bcad5f5`, `477c7bf5`, `3423da8f`, `479559f3`, `122ec5f5`, `9cb56165`, `3ba1c756`, `868fbbe0` | Empty side tips whose full trees also equal an ancestor already in the main stack. The inventory records exact matching ancestor IDs. |

**Do not abandon these source-bearing branches merely because old:**

- `60e2d7451eef` is another detached snapshot, but has **127 paths absent from
  the current tree**, including reasoning-budget source, policy/runner tests and
  retained refusal/build evidence. It is not the duplicate `aaf54720` case.
- Functional/private source: `a45b75e5b6bb` has 612 absent paths and distinct
  functional C/header/capture/scanner code; `7dcad98c56f1` has 104 absent paths.
  Empty protective children in that family retain source-bearing parents too.
- Agent pair-tape/private semantics: `81e3c23c53f7` / `8034181355c6` retain private
  `ds4_suffix_pair_tape.h`, agent/core variants and substantial unique artifacts.
- Backfill/late/pool heads (`b6321e33`, `a11fc27c`, `5cf0fe41`, `5e08a432`,
  `ff1f0550`, `bdf68a16`) retain distinct native hooks and private headers.
- `475a61f5`, `80b70a4d`, `c5fe18f4`, `8e59ca97`, `a1df8530` and several smaller
  study heads retain evidence not present at the same paths in the current tree.
  An import elsewhere is not proof that every source/refusal was preserved.
- Remote Laguna/GLM/model-family heads and the bookmarked legacy archive
  `archive/ds41-experiments-2026-09-16` are outside this cleanup.

**Empty merges require special care.** `b58344d4358c` and `c1e4724c2f2c` report
Jujutsu `empty=true` because they make no resolution changes over their merged
parents. They still introduce feature bytes relative to the first parent.
For example `b58344d4` imports three-expert routing from `b9408c3b`. Preserve
their net contribution when linearizing; the single-parent-head test above is
deliberately narrower.

## Minimal preservation checks for ROOT's rewrite

1. Keep the old tip and source-bearing private heads reachable before rewriting.
   Avoid rebasing all 153 heads; only the selected main feature stack needs the
   upstream integration now.
2. Reconstruct the shorter stack at `0aaea5a2` and compare final tree identity
   with `030ae749` (`0dd71e4b9acf5ed742a099942c7e74c23000ced3`). No file renames,
   evidence deletion or binary regeneration is needed to shorten history.
3. Rebase the short feature stack onto fetched upstream. Preserve upstream
   CUDA/ROCm/TP changes and resolve overlapping Metal/agent/server/API hunks;
   do not apply the old whole-tree snapshot over new upstream.
4. Attach relevant checks to runtime commits and keep historical outcome files
   byte-identical. Embedded historical commit IDs remain historical; do not
   rewrite old manifests to pretend their experiments ran on the new revision.

## Inventory files and limitations

- `ancestor-commits.json`: all 248 local ancestor IDs/parents/empty flags.
- `first-parent-inventory.json`: 198 commits, per-parent file/numstat lists,
  tree IDs and runtime/test versus benchmark classification.
- `local-net-diff.json`: authoritative local delta from the actual common base.
- `net-diff.json`: direct diff against old upstream, kept only to make the
  misleading upstream-deletion comparison explicit.
- `heads.json`: 153 heads, tree/parent identity and nonempty-head net path
  comparisons with current tip. Missing/different paths are conservative
  preservation flags, not semantic judgments about an old experiment.
- `empty-heads.tsv` and `workspaces.json`: exact structural cleanup candidates
  and current workspace ownership, without any forget/abandon operation.

No current runtime tests were executed and no upstream fetch was performed by
this audit. Counts refer to the pinned initial graph, not ROOT's later changes.
