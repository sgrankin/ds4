# Native grammar compiler and token identity

Implemented 2026-10-27 04:11–04:31 UTC as a bounded native integration slice.
All 11 native fixtures pass in ReleaseFast and Debug. The two composed-format
cases invoke the actual serving format builder for DS41 thinking/auto/strict
and Flint auto/strict, accept complete calls with literal delimiter data, and
separately accept ordinary answers. No model runtime speed claim is made.
This qualifies the compiler boundary, not end-to-end model decoding or speed.
There is no model job, Python/MLX runtime, GPU dependency or external binary.

XGrammar v0.2.8 (`97787376faee5ed8466cfad57c99855e4ce2f6aa`) is vendored
unchanged under `src/vendor/xgrammar`, excluding TVM FFI and Python. The build
compiles C++17 sources with the same Xcode SDK/Clang dependency tracking used by
the existing Zig build, then statically links them. Public opaque C handles
cross into `src/grammar.zig`: a vocabulary compiler/cache, immutable compiled
grammars, and a matcher per generation. The compiler cache is bounded at 128MiB.

`grammar-test` exercises JSON Schema/regex/structural compilation, vocabulary
bytes including NUL and split UTF-8, masks, rejected tokens, rollback through
EOS, compiler/compiled/matcher lifetime and native error propagation. First
build exposed the C enum's unsigned backing type; corrected before qualification.

The model-aware styles `ds4_deepseek41_xml` and `ds4_glm_xml` reuse XGrammar's
schema-to-XML converter. A narrow AST pass changes generated XML wrappers to
dedicated token IDs and changes the unrestricted raw-string stop rule to token
exclusions. It protects value-rule subgraphs, so marker spellings inside strict
enum/const and nested JSON values remain literal data. The same byte spelling
cannot stand in for a syntax control. Root model framing uses TokenFormat IDs.

Token-aware compiler construction assigns controls private nonempty invalid-UTF8
pieces internally. Empty pieces cannot work: pinned XGrammar rejects those IDs
before token matching. Private pieces prevent dedicated controls from satisfying
ordinary text terminals, while explicit Token(id) nodes still accept them. Engine
tokenization and decoded output are untouched. Both unrestricted and bounded/
pattern strings, a strict enum, a referenced const, Flint delimiter text and a
non-strict nested JSON parameter are covered by the native fixture.

Reproduce from the repository root:

```nu
nu tools/zig-build grammar-test --prefix .workspaces/native-grammar/release
^.workspaces/native-grammar/release/bin/test-grammar
nu tools/zig-build grammar-test -Doptimize=Debug --prefix .workspaces/native-grammar/debug
^.workspaces/native-grammar/debug/bin/test-grammar
```

Raw build/debugging artifacts stay ignored under
`.workspaces/xgrammar-native-20261027/`; the pinned reference clone is
`.workspaces/xgrammar-reference-20261027/`. Model-specific request semantics,
unsupported strict-schema guarantees, real sampling masks/speculation and
whole-request performance need their separate serving qualification.

References: [XGrammar source](https://github.com/mlc-ai/xgrammar/tree/v0.2.8),
[native XML conversion](https://github.com/mlc-ai/xgrammar/blob/v0.2.8/cpp/converter_ext/deepseek.cc),
[Flint wrapper](https://github.com/recursive-oss/rec/blob/e53663b262af4cb1ca498e596ca8b1a2d79e0800/engine/src/rec_mlx/grammar_builder.py).
