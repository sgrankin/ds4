# C-first Zig checkpoint-cache pilot

Budget: 2026-10-22 03:27:33–04:27:33 UTC. Base is C `main`
`ec5424de5562269cbe6952cdad7408989df293de`; the previous behavior-qualified
translation remains at `zig-rewrite` (`108f90cf2267d19387f7263c5c12d529e258461f`).
Work is isolated on `zig-modular`; `main` is unchanged.
Pilot revision: `3b5fdffe6e804131696f2e2ca92ac3fbc751d448`.

This starts again from C, retaining the useful handwritten checkpoint module.
The engine, frontends, serializers and Metal host are C/Objective-C. Four
previously qualified narrow autorelease-pool changes and the tokenizer-only
shared mapping fix are present in both controls. No translated engine, server,
platform-header dump or duplicated Zig test implementation is carried over.

## Source review

The native module is 903 lines including its public boundary, versus 759 C
implementation lines plus the separate header. This is not a line-count win.
It is a bounded implementation whose algorithms and boundary can be reviewed
without traversing SDK declarations or C promotion expressions.

All meaningful C implementation and header explanations were restored beside
the corresponding code: serialized callers, one process owning a directory,
immutable shared chunks, independent hot/disk limits, profile identity, pinned
references, publication order and rollback, caller-owned materializations,
replay-prefix eligibility and replay exclusion from suffix relocation.
Private helper and publication names were clarified. The thirteen exported
signatures and byte formats are unchanged; libc allocations deliberately cross
the C boundary. An independent source review found no concrete ABI, ownership,
format or replay-selection mismatch in the inspected paths.

## Qualification

The existing `tests/test_state_cache.c` links the actual selected cache object.
C and native each pass in Debug and ReleaseFast with assertions enabled.
Its independent fixture exercises deduplication, publication failure, pinning,
eviction, restart, exact/suffix/replay classification and byte materialization.

The reused three-request HTTP workflow performs actual `tool_search`, discovers
`read_note`, reads a local file and returns the file's archived code and receipt
plus the current request key. Each direction starts a second process using the
other implementation and requires a real SSD hit, identical first request,
matching cached-token accounting and another fresh tool/file read. Hot caching,
suffix reuse, decoder replay and resident substitutions are disabled; context
is 16,384, prefill chunk 512, MTP draft 1.

- C writer → native reader: **pass**, restoring **8,622 tokens** from SSD.
- Native writer → C reader: **pass**, restoring **8,622 tokens** from SSD.
- The first 64 GiB expert-cache attempt was stopped by the pressure guard after
  sustained macOS pressure level 2. It is retained as a pressure-stop result,
  not a cache compatibility failure. The successful retry uses 48 GiB.

Artifacts and exact commands are under `.workspaces/zig-restart-20261022/`.
The runner is `serve.py`; both directions have `execution.json`, wire requests,
tool/read results and pressure logs. The passing direction's workflow times
were 52.43 seconds cold and 12.90 seconds after restart. That comparison shows
cache reuse, **not a speedup caused by Zig**.

The [recorded runner](serve.py) is also kept beside this report. At the pilot
revision, build controls
with `tools/zig-build --prefix .workspaces/control` and the native arm with
`tools/zig-build -Dnative-state-cache=true --prefix .workspaces/native`. Run
`python3 -B speed-bench/ideas/pressure_run.py NEW_GUARD_DIR -- python3 -B
speed-bench/experiments/zig-modular-hour1-20261022/serve.py --model MODEL
--server CONTROL_SERVER --restart-server NATIVE_SERVER --output NEW_RESULTS_DIR
--expert-cache 48GB --timeout 600`, then swap the two servers for the reverse
direction. The guard and result directories must be fresh. Use the actual model
and executable paths, and keep other model jobs stopped.

## Build cost

Initial build system uses Zig's C compiler, independent frontend objects and a
shared core. On this machine, warmed full-code rebuilds took 11.922 seconds
with C cache and 12.038 seconds with native cache. A current isolated Apple
Make clean build took 9.497 seconds. Zig no-op builds average 0.173/0.177
seconds; cache-only edits average 0.389/0.409 seconds; CLI-only edits average
0.465/0.472 seconds. First-ever builds including build-runner setup were
13.785/13.573 seconds. All source bytes were restored after the content-edit
measurements.

The native cache adds little build cost relative to the same Zig/C build, but
the C compiler choice created a noticeable full-rebuild regression. We retained
a small standard build graph using Apple Clang for remaining C/ObjC files;
native modules still compile with Zig. Warm full-code rebuilds are now
10.023 seconds with C cache and 9.900 seconds with native cache, versus Make's
9.497 seconds. Native-cache no-op, cache-only and CLI-only averages are
0.161, 0.402 and 0.350 seconds respectively. These observations do not establish
a general compiler speedup, but remove most of the measured regression.

Clang depfiles supply real project-header dependencies. Changing only
`ds4_state_cache.h` rebuilt cache/server/agent in the C arm and server/agent in
the native arm; neither rebuilt the engine. All temporary edits were restored
by hash. All four existing cache fixture combinations also pass under the final
graph. The final real restart/tool-retrieval check also passes, restoring
8,619 tokens from SSD (the fixture's different path changes its token count).

Decision at **03:44:34 UTC**, inside the hour: retain the bounded cache module,
the C-first build and the Apple C compiler choice. Proceed to KV persistence;
do not revive the bulk translation. This qualifies the cache and exercised
workflow, not a whole native engine rewrite.

Raw timings, commands, hashes and fixture outputs are in `build/` beneath the
workspace artifact directory, and the final graph is in `apple-build/`. These
are single full-rebuild observations and
short incremental samples, not a broad compiler benchmark.
