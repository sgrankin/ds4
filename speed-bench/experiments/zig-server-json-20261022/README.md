# Item75b — Native primitive JSON cursor

Handwritten Zig implements the nine existing whitespace/literal/string/number/
integer/boolean/skip/raw/minify actions. Request/value/protocol callers retain
one API and the original accepted grammar. Strings use standard ArrayList and
Unicode encoding; strtod/ctype use the real SDK. Cursor failure positions,
duplicate fields, raw byte acceptance, surrogate replacement, embedded NUL,
nonfinite/clamped integers, literal prefixes and the256-container ceiling retain
the original contracts. The shared minifier has one selected implementation.
No Server/Engine/Session state or record mirror enters this owner.

Integration/qualification17:27–17:29UTC starts at e4ff5ee8. C primitive controls
with other native owners, native ReleaseFast/Debug and full C fallback all build
cleanly and pass the unchanged production `ds4_test --server` group. All five
commands/results and final source/binary hashes are in results.json/evidence/.
There were no compiler/runtime failures in this item and no new test cases.
The existing broad suite includes surrogate/lone/bad-low escapes, nonfinite
values, duplicate fields, nesting256/257, tools, prompt/replay and the previous
allocated-empty stop regression. Earlier request ownership fixes are retained.

```
tools/zig-build ds4-server server-test -Dnative=true -Doptimize=ReleaseFast \
  --prefix .workspaces/json-repeat/native
.workspaces/json-repeat/native/bin/ds4_test --server
```

Use -Dnative=false for the full C reference and -Doptimize=Debug for native
runtime checks. DS4_NATIVE_SERVER_JSON selects the package; Make retains C.
This is parser behavior qualification, with no model or performance claim.
Generated replies and each remaining protocol controller follow separately;
aggregate actual HTTP generation is measured after their integration.
