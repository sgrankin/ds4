/* Exercise private session bookkeeping without loading a model. The GPU build
 * also runs speculative rollback through the real pooled-indexer kernel. */
#include "../src/ds4.c"
#include <assert.h>

/* Synthetic legacy profile for private state bookkeeping. Production model
 * metadata is owned by the native GLM adapter. */
static const ds4_shape DS4_SHAPE_GLM53 = {
    .name = "GLM 5.3 Flash",
    .family = DS4_MODEL_FAMILY_GLM_DSA,
    .variant = DS4_VARIANT_GLM53,
    .n_layer = 46,
    .n_embd = 4096,
    .n_vocab = 154880,
    .n_head = 64,
    .n_head_kv = 1,
    /* Raw DSA cache row: 512 latent dimensions and no RoPE tail. */
    .n_head_dim = 512,
    .n_value_dim = 256,
    .n_rot = 0,
    .n_lora_q = 1536,
    .n_expert = 288,
    .n_expert_used = 8,
    .n_expert_shared = 1,
    .n_ff_exp = 2048,
    .n_ff_dense = 12288,
    .n_indexer_head = 32,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 2048,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .n_nextn_predict = 1,
    .n_leading_dense = 3,
    .n_kv_lora = 512,
    .n_key_mla = 256,
    .n_value_mla = 256,
    .n_kda_head = 64,
    .n_kda_head_dim = 128,
    .n_kda_conv = 4,
    .rms_eps = 1.0e-5f,
    .hc_eps = 1.0e-6f,
    .expert_weight_scale = 2.5f,
    .swiglu_clamp_exp = 10.0f,
    .rope_orig_ctx = 1048576,
    .kda_gate_lower_bound = -5.0f,
};

static void test_vision_prefix(void) {
    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    s->frontier.checkpoint_valid = true;
    s->frontier.checkpoint.len = 100;
    ds4_vision_span images[2] = {
        {.token_start = 100, .embedding = {.token_count = 10, .fingerprint = {1}}},
        {.token_start = 150, .embedding = {.token_count = 10, .fingerprint = {2}}},
    };
    assert(ds4_session_vision_prefix_matches(s, NULL, 0));
    assert(ds4_session_vision_prefix_matches(s, images, 1));
    assert(!ds4_session_vision_state_matches(s, images, 1));
    images[0].token_start = 99;
    assert(!ds4_session_vision_prefix_matches(s, images, 1));
    ds4_vision_identity old = {.token_start = 50, .token_count = 10, .fingerprint = {1}};
    s->frontier.checkpoint_images = &old;
    s->frontier.checkpoint_image_count = 1;
    images[0].token_start = 50;
    assert(ds4_session_vision_state_matches(s, images, 1));
    assert(ds4_session_vision_prefix_matches(s, images, 2));
    assert(!ds4_session_vision_state_matches(s, images, 2));
    assert(!ds4_session_vision_prefix_matches(s, NULL, 0));
    images[0].embedding.fingerprint[0] ^= 1;
    assert(!ds4_session_vision_prefix_matches(s, images, 2));
    images[0].embedding.fingerprint[0] ^= 1;
    images[0].token_start++;
    assert(!ds4_session_vision_prefix_matches(s, images, 2));
    images[0].token_start--;
    images[0].token_start = 7;
    assert(ds4_session_rebase_vision_state(s, images, 1));
    assert(images[0].token_start == 50);
    images[0].token_start = 7;
    assert(!ds4_session_rebase_vision_state(s, images, 2));
    assert(images[0].token_start == 7);
    images[0].embedding.fingerprint[0] ^= 1;
    assert(!ds4_session_rebase_vision_state(s, images, 1));
    assert(images[0].token_start == 7);
    images[0].embedding.fingerprint[0] ^= 1;
    images[0].embedding.token_count++;
    assert(!ds4_session_rebase_vision_state(s, images, 1));
    images[0].embedding.token_count--;
    images[0].token_start = 50;
    images[1].token_start = 99;
    assert(!ds4_session_vision_prefix_matches(s, images, 2));
    ds4_vision_identity pair[2] = {
        {.token_start = 50, .token_count = 10, .fingerprint = {1}},
        {.token_start = 70, .token_count = 10, .fingerprint = {2}},
    };
    s->frontier.checkpoint_images = pair;
    s->frontier.checkpoint_image_count = 2;
    images[0].token_start = 7;
    images[1].token_start = 8;
    images[1].embedding.fingerprint[31] ^= 1;
    assert(!ds4_session_rebase_vision_state(s, images, 2));
    assert(images[0].token_start == 7 && images[1].token_start == 8);
    images[1].embedding.fingerprint[31] ^= 1;
    assert(ds4_session_rebase_vision_state(s, images, 2));
    assert(images[0].token_start == 50 && images[1].token_start == 70);
    assert(ds4_session_vision_state_matches(s, images, 2));
    ds4_vision_span swapped[2] = {images[1], images[0]};
    assert(!ds4_session_rebase_vision_state(s, swapped, 2));
    assert(!ds4_session_vision_prefix_matches(s, swapped, 2));
    s->frontier.checkpoint_valid = false;
    assert(!ds4_session_vision_prefix_matches(s, images, 2));
    assert(!ds4_session_rebase_vision_state(s, images, 2));
    free(s);
}

/* Fingerprint-prefix reuse: positions may differ (hidden reasoning shifts them
 * and rebase repairs them), appended images are allowed, but a fingerprint or
 * row-count change is never reusable.  This predicate gates the server's live
 * reuse probe, so a mismatch must never look reusable. */
static void test_vision_fingerprint_prefix(void) {
    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    s->frontier.checkpoint_valid = true;
    s->frontier.checkpoint.len = 100;
    ds4_vision_identity ckpt = {.token_start = 50, .token_count = 10,
                                .fingerprint = {1}};
    s->frontier.checkpoint_images = &ckpt;
    s->frontier.checkpoint_image_count = 1;

    ds4_vision_span images[2] = {
        {.token_start = 50, .embedding = {.token_count = 10, .fingerprint = {1}}},
        {.token_start = 90, .embedding = {.token_count = 10, .fingerprint = {2}}},
    };
    assert(ds4_session_vision_fingerprint_prefix_matches(s, images, 1));
    /* Appended image: allowed, as long as the historical one still matches. */
    assert(ds4_session_vision_fingerprint_prefix_matches(s, images, 2));
    /* Shifted historical position (rebase's job), same identity: still reusable. */
    images[0].token_start = 7;
    assert(ds4_session_vision_fingerprint_prefix_matches(s, images, 1));
    assert(ds4_session_vision_fingerprint_prefix_matches(s, images, 2));
    /* Different pixels: never reusable. */
    images[0].embedding.fingerprint[0] ^= 1;
    assert(!ds4_session_vision_fingerprint_prefix_matches(s, images, 1));
    assert(!ds4_session_vision_fingerprint_prefix_matches(s, images, 2));
    images[0].embedding.fingerprint[0] ^= 1;
    /* Different row count is a different image to the model. */
    images[0].embedding.token_count++;
    assert(!ds4_session_vision_fingerprint_prefix_matches(s, images, 1));
    images[0].embedding.token_count--;
    /* Dropping the historical image (or any image) is not a prefix extension. */
    assert(!ds4_session_vision_fingerprint_prefix_matches(s, NULL, 0));
    assert(!ds4_session_vision_fingerprint_prefix_matches(s, images + 1, 1));
    /* A text-only request cannot reuse an image-conditioned checkpoint. */
    assert(!ds4_session_vision_fingerprint_prefix_matches(s, images, 0));
    /* Invalid checkpoints never match. */
    s->frontier.checkpoint_valid = false;
    assert(!ds4_session_vision_fingerprint_prefix_matches(s, images, 2));
    assert(!ds4_session_vision_fingerprint_prefix_matches(NULL, images, 2));
    free(s);
}

static void test_rewind(void) {
    ds4_engine e = { .core.backend = DS4_BACKEND_CPU };
    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    s->engine = &e;
    s->frontier.ctx_size = 1024;
    for (int i = 0; i < 260; i++) ds4_tokens_push(&s->frontier.checkpoint, i);
    s->frontier.checkpoint_valid = true;
    s->frontier.mtp_draft_valid = true;
    s->frontier.checkpoint_images = calloc(1, sizeof(*s->frontier.checkpoint_images));
    assert(s->frontier.checkpoint_images);
    s->frontier.checkpoint_image_count = 1;
    ds4_vision_identity *images = s->frontier.checkpoint_images;

    ds4_session_rewind(s, 260);
    assert(s->frontier.checkpoint_valid && s->frontier.mtp_draft_valid);
    ds4_session_rewind(s, 300);
    assert(s->frontier.checkpoint.len == 260 && s->frontier.checkpoint_valid);
    const int boundaries[] = {259, 256, 255, 128, 127, 4, 3, 0};
    for (size_t i = 0; i < sizeof(boundaries) / sizeof(*boundaries); i++) {
        s->frontier.checkpoint_valid = true;
        ds4_session_rewind(s, boundaries[i]);
        assert(s->frontier.checkpoint.len == boundaries[i]);
        assert(!s->frontier.checkpoint_valid && !s->frontier.mtp_draft_valid);
        assert(s->frontier.checkpoint_images == images && s->frontier.checkpoint_image_count == 1);
        assert(ds4_session_common_prefix(s, &s->frontier.checkpoint) == 0);
        assert(ds4_session_argmax(s) == -1);
        assert(ds4_session_argmax_excluding(s, 1) == -1);
        assert(ds4_session_argmax_ignoring_eos(s, DS4_THINK_NONE) == -1);
        uint64_t rng = 1;
        assert(ds4_session_sample(s, 1, 0, 1, 0, &rng) == -1);
        char err[256] = "";
        int accepted[2];
        assert(ds4_session_eval(s, 1, err, sizeof(err)) != 0);
        assert(strstr(err, "synchronized checkpoint"));
        assert(ds4_session_eval_speculative(s, 1, 2, -1, 1, 0, 1, 0,
                    &rng, accepted, 2, err, sizeof(err)) == -1);
    }
    ds4_session_rewind(s, -1);
    ds4_session_rewind(NULL, 0);
    ds4_session_free(s);
}

static void test_session_memory(void) {
    const uint64_t gib = UINT64_C(1) << 30;
    ds4_engine e = { .core.backend = DS4_BACKEND_METAL,
                     .placement_session_count_hint = 4 };
    assert(ds4_engine_glm_graph_budget(&e, 2*gib) == 8*gib);
    e.session_resources.glm_session_count = 1;
    e.session_resources.glm_session_graph_bytes = 3*gib;
    assert(ds4_engine_glm_graph_budget(&e, 2*gib) == 9*gib);
    e.session_resources.glm_session_count = 4;
    e.session_resources.glm_session_graph_bytes = 8*gib;
    assert(ds4_engine_glm_graph_budget(&e, 3*gib) == 11*gib);
    e.placement_session_count_hint = 0;
    assert(ds4_engine_glm_graph_budget(&e, 3*gib) == 11*gib);
    assert(ds4_engine_glm_graph_budget(&e, UINT64_MAX) == UINT64_MAX);
    e.core.backend = DS4_BACKEND_CUDA;
    assert(ds4_engine_glm_graph_budget(&e, 3*gib) == 3*gib);

    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    e.core.backend = DS4_BACKEND_CPU;
    s->engine = &e;
    s->glm.reserved_graph_bytes = 2*gib;
    ds4_session_free(s);
    assert(e.session_resources.glm_session_count == 3 && e.session_resources.glm_session_graph_bytes == 6*gib);
    e.core.backend = DS4_BACKEND_METAL;
    e.placement_session_count_hint = 4;
    assert(ds4_engine_glm_graph_budget(&e, 2*gib) == 8*gib);
}

static void test_payload_tokens(void) {
    FILE *fp = tmpfile();
    assert(fp);
    char err[256] = "";
    const uint32_t ids[] = {71, 19, 900, 4};
    for (size_t i = 0; i < 4; i++)
        assert(payload_write_u32(fp, ids[i], err, sizeof(err)) == 0);
    /* Opaque cache bytes are skipped, but the caller's trailer stays unread. */
    for (int i = 0; i < 65539; i++) assert(fputc('x', fp) != EOF);
    const long end = ftell(fp);
    assert(fputs("TRAILER", fp) >= 0);
    rewind(fp);
    ds4_tokens tokens = {0};
    assert(payload_read_tokens_for_rebuild(fp, 4, end,
                                           &tokens, err, sizeof(err)) == 0);
    assert(tokens.len == 4);
    for (int i = 0; i < 4; i++) assert(tokens.v[i] == (int)ids[i]);
    assert(ftell(fp) == end && fgetc(fp) == 'T');
    ds4_tokens_free(&tokens);
    rewind(fp);
    assert(payload_read_tokens_for_rebuild(fp, 4, 15,
                                           &tokens, err, sizeof(err)) != 0);
    assert(ftell(fp) == 0 && tokens.len == 0);
    assert(payload_read_tokens_for_rebuild(fp, 0, end,
                                           &tokens, err, sizeof(err)) != 0);
    rewind(fp);
    assert(payload_write_u32(fp, DS4_N_VOCAB, err, sizeof(err)) == 0);
    rewind(fp);
    assert(payload_read_tokens_for_rebuild(fp, 4, end,
                                           &tokens, err, sizeof(err)) != 0);
    assert(tokens.len == 0);
    rewind(fp);
    assert(payload_write_u32(fp, ids[0], err, sizeof(err)) == 0);
    rewind(fp);
    assert(payload_read_tokens_for_rebuild(fp, 4, end + 8,
                                           &tokens, err, sizeof(err)) != 0);
    ds4_tokens_free(&tokens);
    fclose(fp);
}

static void test_snapshot_bytes(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    const uint32_t saved_ratio = g_ds4_compress_ratios[0];
    g_ds4_shape = DS4_SHAPE_FLASH;
    g_ds4_shape.n_layer = 1;
    g_ds4_shape.n_head_dim = 4;
    g_ds4_shape.n_vocab = 8;
    g_ds4_compress_ratios[0] = 0;
    ds4_engine e = {.core.backend = DS4_BACKEND_CPU};
    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    s->engine = &e;
    s->frontier.ctx_size = 8;
    s->frontier.prefill_cap = 1;
    s->frontier.checkpoint_valid = true;
    s->frontier.logits = calloc(DS4_N_VOCAB, sizeof(float));
    assert(s->frontier.logits);
    kv_cache_init(&s->cpu_cache, 8, 0);
    for (int i = 0; i < 3; i++) ds4_tokens_push(&s->frontier.checkpoint, i);
    /* Every byte of the final float is nonzero, including the last byte. */
    const uint32_t bits = UINT32_C(0x3f9e0652);
    for (int i = 0; i < 3 * 4; i++)
        memcpy(s->cpu_cache.layer[0].raw_kv + i, &bits, sizeof(bits));
    ds4_session_snapshot snapshot = {0};
    const int lengths[] = {1, 3, 2, 3};
    for (size_t i = 0; i < sizeof(lengths) / sizeof(*lengths); i++) {
        s->frontier.checkpoint.len = lengths[i];
        s->cpu_cache.layer[0].n_raw = lengths[i];
        char err[192] = {0};
        FILE *fp = tmpfile();
        assert(fp);
        assert(ds4_session_save_payload(s, fp, err, sizeof(err)) == 0);
        const long bytes = ftell(fp);
        assert(bytes > 0);
        assert(ds4_session_save_snapshot(s, &snapshot, err, sizeof(err)) == 0);
        assert(snapshot.len == (uint64_t)bytes);
        rewind(fp);
        for (long j = 0; j < bytes; j++) {
            const int expected = fgetc(fp);
            if (expected != snapshot.ptr[j])
                fprintf(stderr, "snapshot byte %ld/%ld: expected=%d actual=%d\n",
                        j, bytes, expected, snapshot.ptr[j]);
            assert(expected == snapshot.ptr[j]);
        }
        assert(fgetc(fp) == EOF);
        fclose(fp);
    }
    ds4_session_snapshot_free(&snapshot);
    ds4_session_free(s);
    g_ds4_shape = saved_shape;
    g_ds4_compress_ratios[0] = saved_ratio;
}

static void test_text_observations(void) {
    ds4_engine e = {0};
    ds4_vocab *v = &e.core.vocab;
    char bytes[256][5] = {{0}};
    v->n_vocab = 260;
    v->token = calloc((size_t)v->n_vocab, sizeof(*v->token));
    assert(v->token);
    v->token_to_id = ds4_token_map_create(v->n_vocab);
    v->merge_rank = ds4_token_map_create(0);
    assert(v->token_to_id && v->merge_rank);
    /* Fixture vocabulary for GPT-2's printable-byte alphabet. The maps borrow
     * these stack keys until vocab_free below; no private BPE helper is needed. */
    uint32_t escaped = 256;
    for (int i = 0; i < 256; i++) {
        const bool printable = (i >= 33 && i <= 126) ||
            (i >= 161 && i <= 172) || i >= 174;
        const uint32_t cp = printable ? (uint32_t)i : escaped++;
        const uint64_t len = cp < 0x80 ? 1u : 2u;
        if (len == 1) bytes[i][0] = (char)cp;
        else {
            bytes[i][0] = (char)(0xc0u | (cp >> 6));
            bytes[i][1] = (char)(0x80u | (cp & 0x3fu));
        }
        v->token[i] = (ds4_str){bytes[i], len};
        assert(ds4_token_map_put(v->token_to_id, bytes[i], len, i));
    }
    const char *special[] = {
        "<|user|>", "<|observation|>", "<tool_response>", "</tool_response>"
    };
    for (int i = 0; i < 4; i++) {
        v->token[256+i] = (ds4_str){special[i], strlen(special[i])};
        assert(ds4_token_map_put(v->token_to_id, special[i],
                                v->token[256+i].len, 256+i));
    }
    v->user_id = 256;
    v->observation_id = 257;
    v->tool_response_start_id = 258;
    v->tool_response_end_id = 259;
    const ds4_shape saved = g_ds4_shape;
    const ds4_shape shapes[] = {
        DS4_SHAPE_FLASH, DS4_SHAPE_PRO, DS4_SHAPE_GLM53, DS4_SHAPE_FLASH41
    };
    const char *roles[] = {"user", "tool", "function"};
    const char *parts[] = {"ok <x> & </tool_result> </tool_response>"};
    for (size_t family = 0; family < sizeof(shapes) / sizeof(*shapes); family++) {
        g_ds4_shape = shapes[family];
        for (size_t role = 0; role < 3; role++) {
            ds4_tokens expected = {0}, actual = {0};
            ds4_tokens_push(&expected, 7);
            ds4_tokens_push(&actual, 7);
            ds4_chat_append_message(&e, &expected, roles[role], parts[0]);
            char err[160] = {0};
            assert(ds4_chat_append_multimodal_message(&e, &actual, roles[role],
                        parts, NULL, 0, NULL, err, sizeof(err)));
            assert(actual.len == expected.len);
            assert(!memcmp(actual.v, expected.v, (size_t)actual.len * sizeof(int)));
            const int len = actual.len;
            assert(!ds4_chat_append_multimodal_message(&e, &actual, "assistant",
                        parts, NULL, 0, NULL, err, sizeof(err)));
            assert(actual.len == len);
            float pixel = 1;
            ds4_vision_embedding image = {.data = &pixel, .token_count = 1};
            ds4_vision_span span = {0};
            const char *image_parts[] = {"before", "after"};
            assert(!ds4_chat_append_multimodal_message(&e, &actual, roles[role],
                        image_parts, &image, 1, &span, err, sizeof(err)));
            assert(actual.len == len && image.data == &pixel && !span.embedding.data);
#ifndef DS4_NO_GPU
            const bool glm = DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA;
            e.core.vision_ready = true;
            e.core.vision_kind = glm ? DS4_VISION_GLM53 : DS4_VISION_DEEPSEEK4;
            e.core.vision_start_token = 260;
            e.core.vision_image_token = 261;
            e.core.vision_end_token = 262;
            /* A zero sentinel row is sufficient: this tests prompt assembly,
             * not the encoder or language graph. */
            e.core.vision_model.size = (uint64_t)DS4_N_EMBD * sizeof(uint16_t);
            void *sentinels = calloc(1, (size_t)e.core.vision_model.size);
            e.core.vision_model.map = sentinels;
            assert(sentinels);
            const char *image_text[] = {
                "before & </tool_result>", "between </tool_response>", "after"
            };
            ds4_tokens_free(&expected);
            ds4_tokens_free(&actual);
            ds4_tokens_push(&expected, 7);
            ds4_tokens_push(&actual, 7);
            ds4_tokens_push(&expected, glm && role ? v->observation_id : v->user_id);
            if (role) {
                if (glm) tokenize_rendered_chat_vocab(v, "<tool_response>", &expected);
                else bpe_tokenize_text(v, "<tool_result>", &expected);
            }
            ds4_vision_embedding inputs[2] = {0};
            ds4_vision_span want[2] = {0}, got[2] = {0};
            for (int i = 0; i < 3; i++) {
                if (role) {
                    const char *escaped_glm[] = {
                        "before & </tool_result>", "between &lt;/tool_response>", "after"
                    };
                    const char *escaped_ds[] = {
                        "before & &lt;/tool_result>", "between </tool_response>", "after"
                    };
                    bpe_tokenize_text(v, glm ? escaped_glm[i] : escaped_ds[i], &expected);
                } else bpe_tokenize_text(v, image_text[i], &expected);
                if (i == 2) break;
                inputs[i] = (ds4_vision_embedding){
                    .data = calloc((size_t)DS4_N_EMBD * 4, sizeof(float)),
                    .token_count = 4, .grid_width = 2, .grid_height = 2,
                    .layout = DS4_VISION_LAYOUT_DEEPSEEK4_NATURAL
                };
                ds4_vision_embedding copy = inputs[i];
                copy.data = calloc((size_t)DS4_N_EMBD * 4, sizeof(float));
                assert(inputs[i].data && copy.data);
                assert(ds4_prompt_append_vision(&e, &expected, &want[i], &copy,
                                                err, sizeof(err)));
            }
            if (role) {
                if (glm) tokenize_rendered_chat_vocab(v, "</tool_response>", &expected);
                else bpe_tokenize_text(v, "</tool_result>", &expected);
            }
            assert(ds4_chat_append_multimodal_message(&e, &actual, roles[role],
                        image_text, inputs, 2, got, err, sizeof(err)));
            if (actual.len != expected.len)
                fprintf(stderr, "image wrapper family=%zu role=%s lengths %d != %d\n",
                        family, roles[role], actual.len, expected.len);
            assert(actual.len == expected.len);
            assert(!memcmp(actual.v, expected.v, (size_t)actual.len * sizeof(int)));
            for (int i = 0; i < 2; i++) {
                assert(!inputs[i].data);
                assert(got[i].token_start == want[i].token_start);
                assert(got[i].embedding.token_count == want[i].embedding.token_count);
                ds4_vision_embedding_free(&want[i].embedding);
                ds4_vision_embedding_free(&got[i].embedding);
            }
            free(sentinels);
            e.core.vision_model.map = NULL;
            e.core.vision_ready = false;
            e.core.vision_kind = DS4_VISION_NONE;
#endif
            ds4_tokens_free(&expected);
            ds4_tokens_free(&actual);
        }
    }
    g_ds4_shape = saved;
    vocab_free(v);
}

#ifndef DS4_NO_GPU
static void test_glm_attention_budget(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_GLM53;
    ds4_glm_gpu_graph *g = calloc(1, sizeof(*g));
    assert(g);
    g->glm53 = true;
    g->compact_cache_cap = 16384;
    const uint32_t capacities[] = {1024, 4096, 8192, 16384};
    for (size_t i = 0; i < sizeof(capacities) / sizeof(*capacities); i++) {
        g->ctx_cap = capacities[i];
        assert(glm_graph_dense_compact_attention_limit(g) == 2051);
        g->full_kv_cache = true;
        assert(glm_graph_dense_compact_attention_limit(g) == 2051);
        g->full_kv_cache = false;
    }
    free(g);
    g_ds4_shape = saved_shape;
}

enum { POOL_DIM = 128, POOL_ROWS = 12, MODEL_BYTES = 16384 };
static float pool_raw[POOL_ROWS * POOL_DIM], pool_gates[POOL_ROWS * POOL_DIM];

static void pool_update(ds4_glm_gpu_graph *g, ds4_gpu_tensor *x,
                         ds4_gpu_tensor *gate, void *model, int pos, int rows) {
    assert(ds4_gpu_tensor_write(x, 0, pool_raw + pos * POOL_DIM,
                                rows * POOL_DIM * sizeof(float)));
    assert(ds4_gpu_tensor_write(gate, 0, pool_gates + pos * POOL_DIM,
                                rows * POOL_DIM * sizeof(float)));
    assert(ds4_gpu_glm53_indexer_pool_update_tensor(
        g->layer_indexer_key_cache[3], g->layer_indexer_tail_k[3],
        g->layer_indexer_tail_gate[3], x, gate, model, MODEL_BYTES,
        0, 4096, 8192, pos, rows, 16, POOL_DIM, 4, 1e-6f, false));
    assert(ds4_gpu_synchronize());
}

static void test_glm_spec_rollback(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_GLM53;
    void *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON, -1, 0);
    assert(model != MAP_FAILED);
    for (int d = 0; d < POOL_DIM; d++) ((float *)model)[d] = 1;
    for (int t = 0; t < POOL_ROWS; t++) for (int d = 0; d < POOL_DIM; d++) {
        pool_raw[t * POOL_DIM + d] = sinf((t + 1) * (d + 1) * 0.1f);
        pool_gates[t * POOL_DIM + d] = cosf((t + 2) * (d + 1) * 0.03f);
    }
    assert(ds4_gpu_init());
    assert(ds4_gpu_set_model_map(model, MODEL_BYTES));
    ds4_glm_gpu_graph *g = calloc(1, sizeof(*g));
    assert(g);
    ds4_native_glm_graph_configure(g, &g_ds4_shape);
    g->glm53 = true;
    g->layer_end = 3;
    for (int il = 0; il < 3; il++) {
        g->layer_kda_conv_state[il] = ds4_gpu_tensor_alloc(64*sizeof(float));
        g->layer_kda_recurrent_state[il] = ds4_gpu_tensor_alloc(128*sizeof(float));
        assert(g->layer_kda_conv_state[il] && g->layer_kda_recurrent_state[il]);
        assert(ds4_gpu_tensor_fill_f32(g->layer_kda_conv_state[il], 7, 64));
        assert(ds4_gpu_tensor_fill_f32(g->layer_kda_recurrent_state[il], 11, 128));
    }
    const uint64_t tail_bytes = 4 * POOL_DIM * sizeof(float);
    g->layer_indexer_key_cache[3] = ds4_gpu_tensor_alloc(tail_bytes);
    g->layer_indexer_tail_k[3] = ds4_gpu_tensor_alloc(2 * tail_bytes);
    g->layer_indexer_tail_gate[3] = ds4_gpu_tensor_view(
        g->layer_indexer_tail_k[3], tail_bytes, tail_bytes);
    const uint64_t bytes = glm53_graph_spec_state_bytes(g);
    assert(bytes == 3*(64+128)*sizeof(float) + 2*tail_bytes);
    g->mtp_state_backup = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(POOL_ROWS*POOL_DIM*sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(POOL_ROWS*POOL_DIM*sizeof(float));
    assert(g->layer_indexer_key_cache[3] && g->layer_indexer_tail_gate[3] &&
           g->mtp_state_backup && x && gate);

    for (int start = 3; start < 7; start++) {
        assert(ds4_gpu_tensor_fill_f32(g->layer_indexer_tail_k[3], 0, 8*POOL_DIM));
        for (int p = 0; p < start; p++) pool_update(g, x, gate, model, p, 1);
        assert(glm53_graph_copy_spec_state(g, true));
        for (int p = start; p < POOL_ROWS; p++) pool_update(g, x, gate, model, p, 1);
        float expected[3*POOL_DIM], replay[3*POOL_DIM];
        assert(ds4_gpu_tensor_read(g->layer_indexer_key_cache[3], 0,
                                   expected, sizeof(expected)));
        assert(glm53_graph_copy_spec_state(g, false));
        pool_update(g, x, gate, model, start, 2);
        for (int il = 0; il < 3; il++) {
            assert(ds4_gpu_tensor_fill_f32(g->layer_kda_conv_state[il], -1, 64));
            assert(ds4_gpu_tensor_fill_f32(g->layer_kda_recurrent_state[il], -1, 128));
        }
        assert(glm53_graph_copy_spec_state(g, false));
        for (int p = start; p < POOL_ROWS; p++) pool_update(g, x, gate, model, p, 1);
        assert(ds4_gpu_tensor_read(g->layer_indexer_key_cache[3], 0,
                                   replay, sizeof(replay)));
        assert(memcmp(expected, replay, sizeof(expected)) == 0);
        for (int il = 0; il < 3; il++) {
            float state[128];
            assert(ds4_gpu_tensor_read(g->layer_kda_conv_state[il], 0,
                                       state, 64*sizeof(float)));
            for (int j = 0; j < 64; j++) assert(state[j] == 7);
            assert(ds4_gpu_tensor_read(g->layer_kda_recurrent_state[il], 0,
                                       state, sizeof(state)));
            for (int j = 0; j < 128; j++) assert(state[j] == 11);
        }
    }
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(gate);
    glm_graph_free(g);
    free(g);
    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);
    g_ds4_shape = saved_shape;
}
#endif

#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
static void test_spark_dspark_policy(void) {
    assert(ds4_gpu_init());
    if (!ds4_gpu_device_is_spark()) {
        ds4_gpu_cleanup();
        return;
    }
    const ds4_shape saved_shape = g_ds4_shape;
    const bool saved_vision = g_ds4_flash_vision_exp;
    g_ds4_shape = DS4_SHAPE_FLASH;
    g_ds4_flash_vision_exp = false;
    ds4_engine *e = calloc(1, sizeof(*e));
    ds4_session *s = calloc(1, sizeof(*s));
    assert(e && s);
    ds4_tensor gate = {.type = DS4_TENSOR_MXFP4};
    ds4_tensor down = {.type = DS4_TENSOR_MXFP4};
    e->core.backend = DS4_BACKEND_CUDA;
    e->tp.active = true;
    e->core.weights.layer[DS4_N_LEADING_DENSE].ffn_gate_exps = &gate;
    e->core.weights.layer[DS4_N_LEADING_DENSE].ffn_down_exps = &down;
    s->engine = e;
    unsetenv("DS4_DSPARK_SCHEDULER_WINDOW");
    unsetenv("DS4_DSPARK_SCHEDULER_MIN_AVG_MILLI");
    unsetenv("DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS");
    unsetenv("DS4_DSPARK_SEED_BATCH");
    assert(ds4_dspark_spark_tp_policy(e, true));
    assert(ds4_dspark_scheduler_window(s) == 16);
    assert(ds4_dspark_scheduler_min_avg_milli(s) == 1000);
    assert(ds4_dspark_scheduler_tail_min_tokens(s) == 4);
    assert(ds4_dspark_scheduler_tail_min_tokens(NULL) == 10);
    setenv("DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS", "7", 1);
    assert(ds4_dspark_scheduler_tail_min_tokens(s) == 7);
    setenv("DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS", "0", 1);
    assert(ds4_dspark_scheduler_tail_min_tokens(s) == 0);
    unsetenv("DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS");
    setenv("DS4_DSPARK_SEED_BATCH", "0", 1);
    assert(ds4_dspark_scheduler_tail_min_tokens(s) == 10);
    unsetenv("DS4_DSPARK_SEED_BATCH");
    setenv("DS4_DSPARK_SCHEDULER_WINDOW", "7", 1);
    setenv("DS4_DSPARK_SCHEDULER_MIN_AVG_MILLI", "1800", 1);
    assert(ds4_dspark_scheduler_window(s) == 7);
    assert(ds4_dspark_scheduler_min_avg_milli(s) == 1800);
    unsetenv("DS4_DSPARK_SCHEDULER_WINDOW");
    unsetenv("DS4_DSPARK_SCHEDULER_MIN_AVG_MILLI");
    e->tp.active = false;
    assert(!ds4_dspark_spark_tp_policy(e, false));
    assert(ds4_dspark_scheduler_tail_min_tokens(s) == 10);
    assert(ds4_dspark_scheduler_window(s) == 4);
    assert(ds4_dspark_scheduler_min_avg_milli(s) == 1500);
    e->tp.active = true;
    e->core.ssd_streaming = true;
    assert(!ds4_dspark_spark_tp_policy(e, true));
    assert(ds4_dspark_scheduler_tail_min_tokens(s) == 10);
    e->core.ssd_streaming = false;
    e->core.dspark_exact_sampling = true;
    assert(!ds4_dspark_spark_tp_policy(e, true));
    assert(ds4_dspark_scheduler_tail_min_tokens(s) == 10);
    e->core.dspark_exact_sampling = false;
    g_ds4_flash_vision_exp = true;
    assert(!ds4_dspark_spark_tp_policy(e, true));
    assert(ds4_dspark_scheduler_tail_min_tokens(s) == 10);
    g_ds4_flash_vision_exp = false;
    gate.type = DS4_TENSOR_IQ2_XXS;
    down.type = DS4_TENSOR_Q2_K;
    assert(ds4_dspark_spark_tp_policy(e, true));
    assert(ds4_dspark_scheduler_window(s) == 16);
    assert(ds4_dspark_scheduler_min_avg_milli(s) == 1000);
    assert(ds4_dspark_scheduler_tail_min_tokens(s) == 4);
    e->core.ssd_streaming = true;
    assert(!ds4_dspark_spark_tp_policy(e, true));
    assert(ds4_dspark_scheduler_window(s) == 4);
    assert(ds4_dspark_scheduler_min_avg_milli(s) == 1500);
    e->core.ssd_streaming = false;
    e->core.dspark_exact_sampling = true;
    assert(!ds4_dspark_spark_tp_policy(e, true));
    assert(ds4_dspark_scheduler_window(s) == 4);
    e->core.dspark_exact_sampling = false;
    g_ds4_flash_vision_exp = true;
    assert(!ds4_dspark_spark_tp_policy(e, true));
    g_ds4_flash_vision_exp = false;
    down.type = DS4_TENSOR_Q8_0;
    assert(!ds4_dspark_spark_tp_policy(e, true));
    assert(ds4_dspark_scheduler_window(s) == 4);
    assert(ds4_dspark_scheduler_tail_min_tokens(s) == 10);

    /* The support model can be split even when target experts are owned
     * whole. Reject unmeasured policies before allocating optional halves. */
    down.type = DS4_TENSOR_Q2_K;
    ds4_tensor output = {.type = DS4_TENSOR_Q8_0, .dim = {4096, DS4_N_VOCAB}};
    ds4_tensor draft_gate = {.type = DS4_TENSOR_IQ2_XXS, .ndim = 3,
                             .dim = {4096, 2048, 256}};
    ds4_tensor draft_down = {.type = DS4_TENSOR_Q2_K, .ndim = 3,
                             .dim = {2048, 4096, 256}};
    ds4_tensor dense = {.type = DS4_TENSOR_Q8_0, .dim = {4096, 2048}};
    e->core.dspark = true;
    e->core.support_kind = DS4_SUPPORT_DSPARK;
    e->core.weights.output = &output;
    e->core.dspark_weights.n_stages = 1;
    e->core.dspark_weights.block_size = 6;
    e->core.dspark_weights.markov_rank = 128;
    e->core.dspark_weights.stage[0].markov_w1 = &dense;
    e->core.dspark_weights.stage[0].markov_w2 = &dense;
    ds4_layer_weights *draft = &e->core.dspark_weights.stage[0].block;
    draft->ffn_gate_exps = draft->ffn_up_exps = &draft_gate;
    draft->ffn_down_exps = &draft_down;
    draft->attn_q_b = draft->attn_output_a = draft->attn_output_b = &dense;
    draft->ffn_gate_shexp = draft->ffn_up_shexp = draft->ffn_down_shexp = &dense;
    unsetenv("DS4_TP_DISABLE_DSPARK_SPLIT");
    assert(!e->cuda_tp_expert_split);
    assert(tp_dspark_split_eligible(e, true, true));
    assert(tp_dspark_split_eligible(e, false, true));
    assert(!tp_dspark_split_eligible(e, true, false));
    e->core.dspark_confidence_threshold = 0.4f;
    assert(!tp_dspark_split_eligible(e, true, true));
    assert(tp_dspark_split_eligible(e, false, true));
    e->core.dspark_confidence_threshold = 0;
    e->core.dspark_exact_sampling = true;
    assert(!tp_dspark_split_eligible(e, true, true));
    e->core.dspark_exact_sampling = false;
    e->core.ssd_streaming = true;
    assert(!tp_dspark_split_eligible(e, true, true));
    e->core.ssd_streaming = false;
    g_ds4_flash_vision_exp = true;
    assert(!tp_dspark_split_eligible(e, true, true));
    g_ds4_flash_vision_exp = false;
    down.type = DS4_TENSOR_Q8_0;
    assert(!tp_dspark_split_eligible(e, true, true));
    down.type = DS4_TENSOR_Q2_K;
    setenv("DS4_TP_DISABLE_DSPARK_SPLIT", "1", 1);
    assert(!tp_dspark_split_eligible(e, true, true));
    unsetenv("DS4_TP_DISABLE_DSPARK_SPLIT");
    e->core.backend = DS4_BACKEND_METAL;
    assert(!tp_dspark_split_eligible(e, true, true));
    assert(ds4_dspark_scheduler_tail_min_tokens(s) == 10);
    free(s);
    free(e);
    g_ds4_shape = saved_shape;
    g_ds4_flash_vision_exp = saved_vision;
    ds4_gpu_cleanup();
    puts("Spark DSpark policy: measured layout, exclusions and overrides: ok");
}
#endif

/* Lockstep TP drafter VERIFY flags and the worker's proposal check. */
static void test_dspark_tp_verify_flags(void) {
    const int proposal[5] = {11, 12, 13, 14, 15};
    const int seeded[6] = {7, 11, 12, 13, 14, 15};
    /* A seed block trimmed to its seed carries no proposal row: unmarked,
     * so the worker verifies the seed without a check. */
    assert(dspark_tp_verify_flags(false, true, 1, 1) == 0u);
    assert(dspark_tp_verify_flags(true, true, 1, 1) == DS4_TP_VERIFY_HEAD_SPLIT);
    assert(dspark_tp_block_matches(dspark_tp_verify_flags(false, true, 1, 1),
                                   seeded, 1, false, proposal, 0));
    /* Full and trimmed seed blocks. */
    const uint32_t full = dspark_tp_verify_flags(false, true, 1, 6);
    assert(full == (DS4_TP_VERIFY_DSPARK_TP | DS4_TP_VERIFY_DSPARK_SEED));
    assert(dspark_tp_block_matches(full, seeded, 6, true, proposal, 5));
    assert(dspark_tp_block_matches(full, seeded, 2, true, proposal, 5));
    assert(!dspark_tp_block_matches(full, seeded, 6, false, proposal, 5));
    assert(!dspark_tp_block_matches(full, seeded, 6, true, proposal, 4));
    assert(!dspark_tp_block_matches(full, seeded, 1, true, proposal, 5));
    int changed[6];
    memcpy(changed, seeded, sizeof(changed));
    changed[4] = 99;
    assert(!dspark_tp_block_matches(full, changed, 6, true, proposal, 5));
    assert(dspark_tp_block_matches(full, changed, 4, true, proposal, 5));
    /* Seed decoded first: the block is the proposal itself (an EOS can trim
     * it to one row). */
    const uint32_t plain = dspark_tp_verify_flags(false, true, 0, 1);
    assert(plain == DS4_TP_VERIFY_DSPARK_TP);
    assert(dspark_tp_block_matches(plain, proposal, 1, true, proposal, 5));
    assert(!dspark_tp_block_matches(plain, seeded, 1, true, proposal, 5));
    /* Leader-local drafts are never checked. */
    assert(dspark_tp_verify_flags(true, false, 1, 6) == DS4_TP_VERIFY_HEAD_SPLIT);
    assert(dspark_tp_block_matches(0u, changed, 6, false, proposal, 0));
    /* Every produced flag set is one the TP layer accepts, and a marked block
     * always holds a proposal row. */
    for (int head = 0; head < 2; head++) {
        for (int tp = 0; tp < 2; tp++) {
            for (int seed = 0; seed < 2; seed++) {
                for (int n = 1; n <= 6; n++) {
                    const uint32_t f = dspark_tp_verify_flags(head, tp, seed, n);
                    assert(!(f & DS4_TP_VERIFY_DSPARK_SEED) || (f & DS4_TP_VERIFY_DSPARK_TP));
                    assert(!(f & DS4_TP_VERIFY_DSPARK_TP) || (tp && n > seed));
                    assert(((f & DS4_TP_VERIFY_HEAD_SPLIT) != 0) == (head != 0));
                    assert((f & ~DS4_TP_VERIFY_FLAGS) == 0u);
                }
            }
        }
    }
}

/* Memory kept for the hinted sessions before the TP drafter halves are built. */
static void test_dspark_tp_split_reserve(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_FLASH;
    const uint64_t gib = UINT64_C(1) << 30, mib = UINT64_C(1) << 20;
    ds4_dspark_weights dw;
    memset(&dw, 0, sizeof(dw));
    dw.n_stages = 3;
    dw.block_size = 5;
    dw.target_layer_count = 3;
    ds4_context_memory mem;
    memset(&mem, 0, sizeof(mem));
    mem.prefill_cap = 2048;
    mem.raw_cap = 2304;
    /* The reference Flash capture state (about 111 MiB) bounds from above. */
    const uint64_t one = tp_dspark_split_reserve_bytes(&mem, &dw, 1);
    assert(one >= 4 * gib + 111 * mib && one <= 4 * gib + 113 * mib);
    const uint64_t capture = one - 4 * gib;
    mem.total_bytes = 3 * gib;
    assert(tp_dspark_split_reserve_bytes(&mem, &dw, 1) == 7 * gib + capture);
    assert(tp_dspark_split_reserve_bytes(&mem, &dw, 8) == 4 * gib + 8 * (3 * gib + capture));
    assert(tp_dspark_split_reserve_bytes(&mem, &dw, 0) ==
           tp_dspark_split_reserve_bytes(&mem, &dw, 1));
    mem.total_bytes = UINT64_MAX / 2;
    assert(tp_dspark_split_reserve_bytes(&mem, &dw, 3) == UINT64_MAX);
    /* Larger prefill or raw caps only grow it. */
    mem.total_bytes = 0;
    mem.prefill_cap = 4096;
    const uint64_t wider = tp_dspark_split_reserve_bytes(&mem, &dw, 1);
    assert(wider > one);
    mem.raw_cap = 4608;
    assert(tp_dspark_split_reserve_bytes(&mem, &dw, 1) > wider);
    g_ds4_shape = saved_shape;
}

/* Public Session policy must distinguish adjusted selection from raw diagnostic
 * logits, and cached/prefilled thinking must not spend the generated threshold. */
static void test_session_sampling_policy(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_FLASH41;
    g_ds4_shape.n_vocab = 8;
    ds4_engine e = { .core.backend = DS4_BACKEND_CPU,
        .core.vocab = {.assistant_id = 6, .think_start_id = 4, .think_end_id = 5,
                  .eos_id = 7, .endoftext_id = -1},
        .core.overthink = {.penalty = .75f, .think_only = true, .after = 1,
                      .ids = {2, -1, 8}, .count = 3} };
    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    s->engine = &e;
    s->frontier.logits = malloc(8 * sizeof(float));
    s->frontier.sample_probs = malloc(8 * sizeof(float));
    assert(s->frontier.logits && s->frontier.sample_probs);
    const float raw[] = {-9, -9, 2, 1.5f, -9, -9, -9, -9};
    assert(ds4_session_set_logits(s, raw, 8) == 0);
    ds4_tokens_push(&s->frontier.checkpoint, 6);
    ds4_tokens_push(&s->frontier.checkpoint, 4);
    s->frontier.overthink_prompt_len = 2;
    s->frontier.checkpoint_valid = true;
    assert(ds4_session_argmax(s) == 2); /* Prefilled thoughts are not generated. */
    assert(!s->frontier.sampling_logits);
    ds4_tokens_push(&s->frontier.checkpoint, 0);
    assert(ds4_session_argmax(s) == 3);
    assert(ds4_session_argmax_excluding(s, 3) == 2);
    float adjusted[8], copied[8];
    assert(ds4_session_copy_sampling_logits(s, adjusted, 8) == 8);
    assert(adjusted[2] == 1.25f && adjusted[3] == 1.5f);
    assert(ds4_session_copy_logits(s, copied, 8) == 8);
    assert(!memcmp(raw, copied, sizeof(raw)));
    ds4_token_score top[2], single;
    assert(ds4_session_top_logprobs(s, top, 2) == 2 && top[0].id == 2);
    assert(ds4_session_token_logprob(s, 2, &single) == 1);
    assert(single.logit == raw[2] && single.logprob == top[0].logprob);
    uint64_t actual_rng = 17, expected_rng = 17;
    int expected = ds4_sample_logits(adjusted, 8, .7f, 0, .9f, .05f, &expected_rng);
    assert(ds4_session_sample(s, .7f, 0, .9f, .05f, &actual_rng) == expected);
    assert(actual_rng == expected_rng);
    ds4_tokens_push(&s->frontier.checkpoint, 5);
    assert(ds4_session_argmax(s) == 2); /* Closed thinking disables the policy. */
    const float controls[] = {-9, -9, 1, 7, 8, -9, -9, 9};
    assert(ds4_session_set_logits(s, controls, 8) == 0);
    assert(ds4_session_argmax_ignoring_eos(s, DS4_THINK_NONE) == 3);
    assert(ds4_session_argmax_ignoring_eos(s, DS4_THINK_HIGH) == 4);
    assert(ds4_session_common_prefix(s, &s->frontier.checkpoint) == 4);
    s->frontier.checkpoint_valid = false;
    assert(ds4_session_common_prefix(s, &s->frontier.checkpoint) == 0);
    assert(ds4_session_argmax(s) == -1);
    assert(ds4_session_copy_sampling_logits(s, copied, 8) == 0);
    /* Raw diagnostic/copy/set entry points historically lack a valid gate. */
    assert(ds4_session_copy_logits(s, copied, 8) == 8);
    assert(ds4_session_top_logprobs(s, top, 2) == 2 && top[0].id == 7);
    assert(ds4_session_copy_logits(s, copied, 7) == 0);
    assert(ds4_session_set_logits(s, raw, 7) == 1);
    free(s->frontier.sampling_logits);
    free(s->frontier.logits);
    free(s->frontier.sample_probs);
    ds4_tokens_free(&s->frontier.checkpoint);
    free(s);
    g_ds4_shape = saved_shape;
}

/* Protocol boundaries differ: GLM role IDs stop generation; Qwen/DS41
 * roles do not. Sampling still applies penalties, while diagnostics stay raw.
 * Owned Core metadata must win over the legacy reference snapshot. */
static void test_session_sampling_families(void) {
    const ds4_shape saved = g_ds4_shape;
    const ds4_model_family families[] = {
        DS4_MODEL_FAMILY_DEEPSEEK41, DS4_MODEL_FAMILY_QWEN4_EXP, DS4_MODEL_FAMILY_GLM_DSA
    };
    for (size_t family = 0; family < sizeof(families)/sizeof(families[0]); family++) {
        ds4_engine engine = {0};
        engine.native_core = true;
        engine.core.configuration.shape = saved;
        engine.core.configuration.shape.family = families[family];
        engine.core.configuration.shape.n_layer = 1;
        engine.core.configuration.shape.n_vocab = 12;
        engine.core.vocab = (ds4_vocab){.n_vocab=12, .eos_id=11, .endoftext_id=10,
            .system_id=2, .user_id=3, .assistant_id=4, .observation_id=5,
            .think_start_id=6, .think_end_id=7};
        g_ds4_shape.family = DS4_MODEL_FAMILY_DEEPSEEK4;
        g_ds4_shape.n_vocab = 5; /* Deliberately stale compatibility metadata. */
        float logits[] = {-10,-10,90,80,70,60,50,49,40,39,101,100};
        float probabilities[12], adjusted[12], copied[12];
        ds4_session *session = calloc(1, sizeof(*session));
        assert(session);
        session->engine = &engine;
        session->frontier.logits = logits;
        session->frontier.sample_probs = probabilities;
        session->frontier.checkpoint_valid = true;
        const bool glm = families[family] == DS4_MODEL_FAMILY_GLM_DSA;
        for (int role = 2; role <= 5; role++) {
            assert(ds4_token_is_stop(&engine, role) == glm);
            assert(ds4_token_is_stop_for_think_mode(&engine, role, DS4_THINK_HIGH) == glm);
        }
        assert(!ds4_token_is_stop(&engine, -1));
        assert(ds4_token_is_thinking_control(&engine, 6));
        assert(!ds4_token_is_stop_for_think_mode(&engine, 6, DS4_THINK_HIGH));
        assert(ds4_token_is_stop_for_think_mode(&engine, 6, DS4_THINK_NONE));
        assert(ds4_session_argmax_ignoring_eos(session, DS4_THINK_HIGH) == (glm ? 6 : 2));
        assert(ds4_session_argmax_ignoring_eos(session, DS4_THINK_NONE) == (glm ? 8 : 2));
        engine.core.overthink = (ds4_overthink_policy){.penalty=100, .think_only=false,
            .ids={2,6}, .count=2};
        assert(ds4_session_argmax_ignoring_eos(session, DS4_THINK_HIGH) == (glm ? 7 : 3));
        assert(ds4_session_copy_sampling_logits(session, adjusted, 12) == 12);
        assert(adjusted[2] == -10 && adjusted[6] == -50 && logits[2] == 90 && logits[6] == 50);
        ds4_token_score scores[2];
        assert(ds4_session_top_logprobs(session, scores, 2) == 2 && scores[0].id == 10 && scores[1].id == 11);
        assert(ds4_session_token_logprob(session, 11, &scores[0]) == 1 && scores[0].id == 11);
        assert(ds4_session_copy_logits(session, copied, 11) == 0);
        assert(ds4_session_set_logits(session, logits, 11) == 1);
        assert(ds4_session_set_logits(session, logits, 12) == 0);
        session->frontier.checkpoint_valid = false;
        assert(ds4_session_argmax(session) == -1);
        assert(ds4_session_top_logprobs(session, scores, 2) == 2);
        assert(ds4_session_copy_logits(session, copied, 12) == 12 && !memcmp(copied, logits, sizeof(logits)));
        free(session->frontier.sampling_logits);
        free(session);
    }
    g_ds4_shape = saved;
}

int main(void) {
    test_session_sampling_families();
    test_session_sampling_policy();
    test_dspark_tp_verify_flags();
    test_dspark_tp_split_reserve();
    test_vision_prefix();
    test_vision_fingerprint_prefix();
    test_rewind();
    test_session_memory();
    test_payload_tokens();
    test_snapshot_bytes();
    test_text_observations();
#ifndef DS4_NO_GPU
    test_glm_attention_budget();
    test_glm_spec_rollback();
#endif
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    test_spark_dspark_policy();
#endif
    puts("session state tests: ok");
    return 0;
}
