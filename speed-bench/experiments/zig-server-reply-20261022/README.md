# Item75c — Completed reply parsing and recovery

Handwritten Zig owns the eight complete generated-reply/search/attribute/repair
actions (659 lines plus28-line sole export facade), replacing977 selected C
body lines. DSML legacy/short/XML/V4.1, GLM and Qwen parsing retains reasoning
boundaries, nested parameters, raw tool replay, schema-forced strings and
wrapper-only recovery. Actual protocol constants move once into a shared header;
original C controllers remain the fallback. No Server/Session record mirror.

Integration17:30–17:34UTC starts at12bbdb90. C parser controls with other native
owners, native ReleaseFast/Debug and full-C fallback build and pass the full
production `ds4_test --server` suite. Results and exact commands are recorded.
Before activation, source review found allocated-empty ownership loss in both
call-vector append and repair-buffer adoption; retain a nonnull allocation even
at length0. The existing repair case now preallocates both buffers. The first
fixture compile used an unavailable xcalloc helper; replaced with standard
calloc plus its existing assertion. A stale binary run after that failed build
is explicitly excluded from qualification; final control builds/fixtures pass.
No assertions weakened and no model/performance claim.

```
tools/zig-build ds4-server server-test -Dnative=true -Doptimize=ReleaseFast \
  --prefix .workspaces/reply-repeat/native
.workspaces/reply-repeat/native/bin/ds4_test --server
```

Use -Doptimize=Debug for runtime checking and -Dnative=false for the full C
reference. OpenAI/Anthropic live controllers follow as separate items; actual
HTTP generation follows their aggregate integration.
