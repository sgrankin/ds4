# Native KV persistence through public-header bindings

Budget: 2026-10-22 03:39–04:39 UTC. Base is the qualified checkpoint-cache pilot
`3b5fdffe6e804131696f2e2ca92ac3fbc751d448` on `zig-modular`. `main` and the
previous bulk translation remain unchanged.

The handwritten KV store replaces `ds4_kvstore.c`; frontends, session payload
serialization and graph execution remain C/Objective-C. The thirteen checkpoint
cache APIs from the preceding item remain native too.

## Source and ABI

`zig/c_api.h` is a declaration-only bridge. Standard Zig `addTranslateC` puts
generated declarations in the compiler cache, behind a `c_api` import. Public
C headers define the actual records, enums, callbacks and opaque engine/session
types. No generated SDK declarations or translated implementation bodies are
checked in. This removes the previous port's handwritten copies of eight ABI
records and its manual libc/engine declarations.

The 851-line implementation preserves all 37 exports and format/diagnostic
strings. It restores the C rationale for byte-prefix filenames versus exact
payload tokens, model identity and payload ABI bytes, tokenization boundaries,
alignment, decaying hits, eviction anchors, superseded contexts, budget reserve,
staging, trailer hooks and exact loaded-token history. Eviction arithmetic now
explicitly uses optimized float mode to match the C build's fast-math policy.
This corrects an incidental difference in the earlier handwritten port.

The private text builder uses standard `ArrayList(u8)` and `c_allocator`, replacing
manual capacity growth. `toOwnedSliceSentinel` transfers C-free-compatible
NUL-terminated strings. Raw spans use slices after the C pointer/length boundary;
public records retain their original ownership. Independent review found no
concrete ABI, staging, trailer, buffer or load-order defect in the inspected paths.

## Qualification

All five C and native programs build. Existing `tests/ds4_test.c --server` calls
the actual selected KV implementation through the production C server wrappers;
the server implementation is included once by that existing fixture, with no
copied Zig test body. It and `tests/test_state_cache.c` pass for C/native in both
Debug and ReleaseFast: **eight successful fixture runs**. Assertions remain live.

The [recorded restart runner](serve.py) disables `DS4_STATE_CACHE_DIR`, replay,
suffix reuse and resident substitutions, forcing the older KVC file path. It
requires a genuine nonempty KVC payload, actual legacy cache-hit log, matching
HTTP cached-token accounting and tool_search → read_note → fresh file read →
checked final JSON in each distinct process. Context is 16,384, prefill chunk 512,
expert cache 48 GiB and MTP draft 1, under the existing pressure guard.

- C writer → Zig reader: **pass**, restoring **8,192 tokens**. Cold/restart
  workflows take 51.17/28.72 seconds; this compares cache reuse, not Zig speed.
- Zig writer → C reader: **pass**, restoring **8,192 tokens**, with another
  successful fresh tool/file retrieval.

Exact build/run commands, source/binary hashes, wire requests, actual file reads,
cache headers and pressure logs are in `.workspaces/zig-kvstore-20261022/`.
Initial all-program builds take 11.29 seconds for C and 11.65 seconds for native;
these include build setup and are not repeated compiler benchmarks.

`tools/zig-build` now selects the qualified native host modules by default;
`-Dnative=false` builds the C reference. The checkpoint-specific pilot flag is
replaced by this single package switch. Use `--prefix` to retain both arms.
Build existing fixtures with `state-cache-test server-test`; run `ds4_test
--server`, and give `test-state-cache` a fresh directory argument.

Decision at **03:51 UTC**, inside the hour: retain the module and single native
package switch. This qualifies persistence and its actual callers; it does not
qualify a whole native server or engine.
