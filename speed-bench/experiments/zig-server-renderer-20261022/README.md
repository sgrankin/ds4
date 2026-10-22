# Item75 — Native prompt/replay renderer and tool codecs

Handwritten Zig owns full/live V4.1 prompt assembly, Anthropic result-span
splitting, stable tool-result ordering, reasoning visibility, replay history,
argument ownership/schema ordering and shared DSML/GLM/Qwen tool codecs. Actual
json_arg/json_args records move once into a real header; messages/buffers/schema
records remain their existing actual definitions. Borrowed keys use a standard
StringHashMap. The minifier is one shared primitive action, currently backed by
the existing C parser; the next JSON owner replaces it without another copy.
Server, Engine, Session, durable store and live-KV policy stay outside this owner.

Integration/qualification17:16–17:26UTC starts at0ad06be9. All five final
production-linked `ds4_test --server` selections pass: C renderer/native other
owners in ReleaseFast/Debug, native renderer ReleaseFast/Debug, and full C
fallback. Existing tests cover full/live V4.1 order, split Anthropic results,
reasoning, exact sampled DSML replay, literal closing controls/entities,
namespaced schemas and GLM/Qwen schema-ordered arguments. Commands, actual final
source/binary hashes and successful/failed runs are in results.json/evidence/.

The unchanged Qwen visible-checkpoint case exposed an owning-buffer bug: a
generic empty-slice helper discarded a nonnull C allocation while retaining its
positive capacity. Appending then wrote address0x1. LLDB located the path at
ArrayList.appendSlice → Text.append/string → qwenCalls → build_qwen_assistant_suffix
→ test_qwen_tool_visible_checkpoint_boundary. The helper now preserves allocated
empty pointers and uses static empty storage only for NULL+zero.

A focused audit found the same adoption pattern in the existing native input,
schema and message owners. A valid request replacing its stop field reproduced
the failure before their fix. One case in the existing server fixture now checks
`stop:"old", stop:"new"` while storage is retained, using the actual logical
decoder and C parser reference. All affected ownership helpers preserve the
original allocation; no grammar, normalization or duplicate-field policy changed.
Both C renderer controls were rebuilt after this fix, omitting only the renderer
frontend define; evidence/control-build.zig records that selection and the
original build file was restored bytewise. No product build option was added.

First compilation also caught one local shadowing a method; it was renamed.
Native ReleaseFast/Debug and C fallback builds finish cleanly. Initial segfaults,
the duplicate-stop reproduction and corrected runs remain as evidence. No test
assertion or tolerance was relaxed.

Reproduce from the repository root:

```
tools/zig-build ds4-server server-test -Dnative=true -Doptimize=ReleaseFast \
  --prefix .workspaces/render-repeat/native
.workspaces/render-repeat/native/bin/ds4_test --server
tools/zig-build server-test -Dnative=false -Doptimize=ReleaseFast \
  --prefix .workspaces/render-repeat/fallback
.workspaces/render-repeat/fallback/bin/ds4_test --server
```

The native package selects DS4_NATIVE_SERVER_RENDER; Make retains C rendering.
Primitive JSON, generated-reply recovery and OpenAI/Anthropic protocol output
follow as separate owners, then the actual guarded HTTP workflow is repeated.
Request materialization, model generation, Conversation policy and routing
remain C. This component qualification makes no model-quality or speed claim.
