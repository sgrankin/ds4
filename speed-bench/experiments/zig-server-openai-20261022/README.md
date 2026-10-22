# Item75d — OpenAI responses and live projection

Handwritten Zig owns text/chat response envelopes, SSE role/content/reasoning/
tool/terminal events, cache-aware usage and live sampled-DSML projection. The
524-line OpenAI owner and207-line shared byte/text owner compose through one
65-line facade with15 actual exports, replacing772 selected C function lines.
Stream records move once into their actual header. Anthropic still uses C
controllers, including selected shared UTF-8/parameter helpers. The sampled
transcript, model generation and durable cache publication keep their owners.

Integration17:34–17:36UTC starts at4f406211. Otherwise-native C controls,
native ReleaseFast/Debug and full-C fallback all build cleanly and pass the
unchanged full `ds4_test --server` group. The real socket cases cover all DSML
forms, literal closing markers, split UTF-8, reasoning, suppression, multiple
calls, live IDs and cache accounting. No new test body or failure in this item.
Exact commands, final hashes and outputs are retained in results/evidence.
No model or speed claim; actual HTTP generation follows Anthropic integration.

```
tools/zig-build ds4-server server-test -Dnative=true -Doptimize=ReleaseFast \
  --prefix .workspaces/openai-repeat/native
.workspaces/openai-repeat/native/bin/ds4_test --server
```

Use -Doptimize=Debug for runtime checks and -Dnative=false for the C reference.
