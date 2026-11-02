# GLM qualification

Qualify full GLM 5.3 and GLM 5.3 Flash independently: they have different model shapes, attention and MTP paths. The current native Mac implementation is described in the [Flash manual](../../models/glm53-flash/README.md). Commands below run from the repository root; model-backed runs need the matching GGUF and the [pressure guard](../../../speed-bench/ideas/pressure_run.py). Record missing hardware and reference checks rather than treating a smoke test as qualification.

## Full GLM 5.3

On a machine large enough for the retained Q2 model, check greedy thinking and no-thinking prompts, family reporting, resident and SSD-streamed first-token/logprob sanity, and selected layer prefix and expert budget. Compare ordinary decode with integrated MTP using `--mtp-timing`: greedy text must agree and speculation must complete cleanly; ordinary decoding remains the default. Exercise chat, Responses and Anthropic requests including thinking and SSE. Compare a non-tool prompt and a complete tool-call/result transition byte for byte with the model template, including system message, reasoning effort, schema, reasoning and observation. The fixed one-tool fixture was 901 bytes, SHA-256 `f1718f3ebb1c41532bcd5eedd9ebd5c84ae930b018e31962d8743efdbc5affc3`; token counts alone do not establish template agreement. `./ds4_test --server` checks the schema and delimiters without a model where that binary is available.

The 512 GB Metal long-context smoke used `tests/glm_long_context_smoke.nu`: at the advertised context its continuation had to start with `>` and contain no known corruption markers. The retained two- and four-session oracle required `family=glm native_shared=0 native_qkv=0`, exact mixed prefill/decode and no DeepSeek row-grid activation. Use a matching full 5.3 fixture; GLM 5.2 vectors do not establish its quality.

### Full-model RoPE and streaming prefill

On an idle host with external memory monitoring, run `python3 tests/test_glm_rope_prefill.py --model /path/to/GLM-5.3-Q2.gguf`. The retained script runs bounded SSD streaming, compares every finite next-token logit against the general attention path within `1e-5`, and requires the answer `Gamma`. The script and current source retain the control, but this check has not been run here; verify the binary and full checkpoint are available. Flash has no RoPE contribution and cannot catch this regression.

Test an 8K prompt followed by a 4K append in a 16K session with automatic and explicit cache targets. This crosses compact-indexer warmup after earlier layer mappings have been released. Compare full logits and monitor memory throughout. For `ds4-bench`, use `DS4_BENCH_FORCE_SNAPSHOT=1` if the extra snapshot fits; otherwise prefix replay changes cache state before the append. Record the restoration method. Retain the exact tokenized prompt and independent reference logits for reproduction; the old QA did not specify these inputs. The timing archive `tests/test-vectors/glm-openrouter/prompts/long_memory_archive.txt` is present, but its first 2500 bytes omit the final question. The historical full Q2 model checksum was `059b36accd4c9acf73099da9f703b574d627869d619b7c4c316aa856e33d472e`; add the final question and require Gamma for an answer check, not just timing.

## Flash qualification

### Continuation and primitive checks

Run Q2 and Q4 matching 100-case continuation fixtures after graph, quantization, attention, KDA, mHC, TP or cache changes. The manifest exists at `gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv`. Build the current scorer with `make quality-score`, then run:

```sh
./zig-out/bin/score-official /path/to/GLM-5.3-Flash-Q2.gguf \
  gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv \
  .workspaces/glm53-q2.tsv 4096
```

Repeat with the Q4 checkpoint. The 2026-10-28 QA snapshot at `70537d44` records Q2 mean NLL 0.458030488, first-token 89/100 and mean greedy prefix 7.37; Q4 0.299917952, 90/100 and 9.66. The M3 Ultra Q4 variant with Q8 KDA projections, embeddings and output head recorded 0.300804038, 90/100 and 9.48. Measurement revisions were not separately identified. These are comparison references, not automatic rejection thresholds.

See the [scorer instructions](../../../gguf-tools/quality-testing/README.md#4-score-gguf-variants) for the long rendered fixture, 8192-token context and missing API logprobs; the [fixture index](../../../gguf-tools/quality-testing/README.md#1-tracked-fixture-sets) identifies checkpoint-specific PRO sets. For attention changes compare the default path with a scalar control on the same eight rendered long cases. The current source inspection did not establish a selectable scalar control for this fixture, so this remains a qualification objective until a suitable control is verified; do not claim it passed. `make quality-score` installs `zig-out/bin/score-official` and copies the compatibility executable to `gguf-tools/quality-testing/score_official`; either path can be used after that build.

The current model-free KDA test is `make test-glm53-kda`. It covers BF16 projections, pool construction and expansion, grouped scoring, causal visibility and recurrent prefill versus decode. Also check selected attention against independent references, including FP16/FP32 padding, empty selections and continued causal attention.

### Selector boundaries

The selector chooses 512 complete four-token pools, or 2,048 token positions, plus an incomplete tail of at most three tokens. Dense attention is equivalent through 2,051 visible tokens; selection starts at 2,052. Construct prompts with `--dump-tokens`, and cover 2,048–2,056 and 4,096–4,100, including every pool remainder. Require finite logits, the same greedy token and a multi-token exact-output task. Repeat across allocation sizes, streaming and prefill chunk sizes; padding must be masked in batch. A 4K-plus actionable instruction must be recovered in a tool or exact-output task. Coherent short output alone does not check selection.

### Memory admission

Session admission must allocate owned DSA and indexer pools/tails, KDA recurrent state and supported prefill workspace before accepting a request; the first prefill must not grow per-layer caches. Check 4,096-token and long sessions. A historical 100K M5 Max compact Q2 plan was approximately 94.09 GiB: 89.87 model, 1.11 compact history and 3.11 fixed graph buffers. Do not restore the old 2.75 GiB expanded per-head KV as a presumed quality fix; repeat continuation quality after changing compact representations or crossover.

### Batch and snapshot replay

For two and four sessions below 2,052 tokens, compare native batch and isolated runs with full-logit tolerance 0.001 and identical selected tokens; test serial rollback at zero tolerance. Above 2,051 require the ordered fallback unless a sparse native full-vocabulary oracle establishes correctness. Snapshot across the sparse boundary, restore and continue within `1e-6` of uninterrupted top logits. Repeat with integrated MTP: replay 16 cycles including one- and two-token outcomes, match committed tokens and final top-eight logits, then resync to the long prefix and continue. Measure continued prefill as actual appends to one live session, including small tool results and following decode, with latency as well as throughput.

### MTP, steering and M3

Check ordinary, opportunistic MTP and `--mtp-exact-sampling` on practical tasks, recording accepted and rejected drafts and speed. For steering, a zero 45 x 4096 vector must match unsteered output at both hooks; reject a 46-row file with expected 737,280-byte size. Use the [steering guide](../../../dir-steering/README.md) for vector construction and compare ordinary, MTP and two-session server runs. A held-out target/control sweep must remain coherent. On M3 Max use the generic non-NAX Q2 path and repeat the boundary, continuation, snapshot and session checks; Q4 requires bounded streaming, not full residency.

## Vision

Text-only tests do not cover image preprocessing, embeddings or image-aware cache identity. The vision sidecar `GLM-5.3-Flash-Vision-Encoder.gguf` had SHA-256 `ae23e14c6979e889051b2e4a39351abcdafb161e18e606fae4d8c40095a4bf3a`. With a fixed PNG and Q2 model, test three encoder repetitions for finite repeatable embeddings. Prompt replay must reuse an unchanged image, rebuild on fingerprint change, and change logits when embeddings are zeroed at fixed token positions. With a 4,096-token session, invalidation and restoring the embedding must reproduce full image-conditioned logits within `1e-6`. Run `python3 tests/test_compare_glm53_vision_embeddings.py` for invalid reference and candidate coverage.

Compare a fixed Metal embedding with other backends: cosine at least 0.995, mean absolute error at most 0.001 and maximum error at most 0.06. With a vision server, `python3 tests/run_glm53_vision_quality.py` requires 6/6; include photos, screenshots, diagrams, text, spatial questions and unrelated-image controls. Check complete answers through CLI, server and agent against the official service, not just plausible prose. Expose raster inputs alone to the agent. Test image turns followed by text and MTP, five images beyond 4K, and a real read/edit/test loop. Image observations must be user-role multimodal turns; text-only observations remain tool-role.

Exercise RGB, RGBA, grayscale and palette PNG, baseline/progressive JPEG, EXIF orientation and malformed, huge or decompression-bomb inputs under sanitizers. Test chat data URIs, Responses `input_image` and Anthropic base64, two images and later image turns. Local paths, file or remote URLs, malformed base64, unsupported media, over 256 images and bodies over 64 MiB must return 4xx without filesystem reads or network access.

## Historical platform and physical TP plans

The source QA at revision `70537d44` (2026-10-28) specified physical full GLM TP on two machines over TCP and RDMA, Q2/Q4 owned experts, a long prompt beyond 2,048, 100 teacher-forced tokens and the 100-case fixture against an unsplit graph. Unsupported routed types must reject before evaluation; token-prefill is a diagnostic. For Flash Q4 it specified explicit TB5 RDMA, 50/50 residency and graduated 10K/25K/50K admission (about 102.64 GiB per rank at 50K), stable swap, responsive SSH and clean exits. The 25K task includes an inert archive and recoverable tool failure; the 50K task has 30K live tokens and a verified source repair. The recorded devices were `rdma_en1` and `rdma_en6`, GID 1; TCP fallback did not satisfy that specific gate. Vision TP additionally requires one leader encoding and matching multimodal KV. These are test plans, not evidence of a current pass; see [distributed qualification](distributed.md).

The old eight-GPU CUDA full-GLM plan required explicit host permission, ordinary layer placement, cache/workspace admission, no CPU spill and tier-switching prefill. Spark Flash Q2 checks covered primitives, continuation, boundary, snapshots, MTP and server separately on each host; Q4 was not a single-Spark target. An accepted `.180` 100-case reference was NLL 0.461783551, 90/100 first tokens and 7.49 mean greedy prefix. These platform procedures and the historical long-fixture NLLs 0.661721610 ROCm and 0.680093110 CUDA (6/8 first tokens each) are from the 2026-10-28 QA snapshot, without a separately identified measurement revision; they are not current native Metal gates.
