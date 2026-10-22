/* Header-only sparse fixtures: no real model or GPU allocation. */
#include "../ds4.c"
#include "../ds4_engram.h"
#include <assert.h>
#include <sys/wait.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

static void check_unmapped(const void *ptr, size_t page) {
#ifdef __APPLE__
    (void)page;
    mach_vm_address_t address = (uintptr_t)ptr;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    kern_return_t result = mach_vm_region(mach_task_self(), &address, &size,
        VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info, &count, &object);
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    assert(result == KERN_INVALID_ADDRESS ||
           (result == KERN_SUCCESS && address > (uintptr_t)ptr));
#else
    unsigned char resident;
    errno = 0;
    assert(mincore((void *)ptr, page, &resident) == -1 && errno == ENOMEM);
#endif
}

static void put32(FILE *fp, uint32_t v) { assert(fwrite(&v, 4, 1, fp) == 1); }
static void put64(FILE *fp, uint64_t v) { assert(fwrite(&v, 8, 1, fp) == 1); }
static void putstr(FILE *fp, const char *s) {
    put64(fp, strlen(s));
    assert(fwrite(s, 1, strlen(s), fp) == strlen(s));
}

static void string_kv(FILE *fp, const char *key, const char *value) {
    putstr(fp, key); put32(fp, GGUF_VALUE_STRING); putstr(fp, value);
}

static void tensor(FILE *fp, const char *name, uint32_t type,
                   uint64_t width, uint64_t rows, uint64_t offset) {
    putstr(fp, name); put32(fp, 2); put64(fp, width); put64(fp, rows);
    put32(fp, type); put64(fp, offset);
}

static int run_fixture(int bad_layout) {
    enum { ALIGN = 16384 };
    const uint32_t rows = (1u << 24) + 3;
    const uint64_t table_bytes = (uint64_t)rows * DS4_ENGRAM_ROW_BYTES;
    const uint64_t first = 2 * ALIGN + (bad_layout ? 32 : 0);
    const uint64_t second = align_up(first + table_bytes, ALIGN);
    const uint64_t file_size = second + table_bytes;
    char path[] = "/tmp/ds41-gguf.XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "w+b");
    assert(fp);
    put32(fp, DS4_GGUF_MAGIC); put32(fp, 3); put64(fp, 3); put64(fp, 3);
    string_kv(fp, "general.architecture", "deepseek41");
    string_kv(fp, "deepseek41.engram.encoding", "e4m3_e8m0_32_row264");
    putstr(fp, "general.alignment"); put32(fp, GGUF_VALUE_UINT32); put32(fp, ALIGN);
    tensor(fp, "test.weight", DS4_TENSOR_F32, 16, 1, 0);
    tensor(fp, "blk.1.engram_embd.weight", 24, 264, rows, first - ALIGN);
    tensor(fp, "blk.14.engram_embd.weight", 24, 264, rows, second - ALIGN);
    assert(ftell(fp) < ALIGN);
    assert(fflush(fp) == 0 && ftruncate(fd, (off_t)file_size) == 0);
    uint8_t row[DS4_ENGRAM_ROW_BYTES];
    memset(row, 56, DS4_ENGRAM_DIM);
    memset(row + DS4_ENGRAM_DIM, 127, DS4_ENGRAM_ROW_BYTES - DS4_ENGRAM_DIM);
    assert(pwrite(fd, row, sizeof(row), (off_t)(second + table_bytes - sizeof(row))) == sizeof(row));
    if (bad_layout) {
        pid_t pid = fork();
        assert(pid >= 0);
        if (pid == 0) {
            ds4_model m;
            model_open(&m, path, false, false);
            model_close(&m);
            _exit(0);
        }
        int status;
        assert(waitpid(pid, &status, 0) == pid);
        assert(WIFEXITED(status) && WEXITSTATUS(status) != 0);
    } else for (int shared = 0; shared < 2; shared++) {
        ds4_model m;
        model_open(&m, path, shared != 0, false);
        assert(m.file_size == file_size && m.size == first);
        assert(m.max_tensor_bytes == 64);
        char resident;
        assert(mincore((void *)m.map, ALIGN, &resident) == 0);
        check_unmapped(m.map + first, ALIGN);
        const ds4_tensor *t = model_find_tensor(&m, "blk.14.engram_embd.weight");
        assert(t && t->abs_offset == second && t->bytes == table_bytes);
        assert(*(const float *)tensor_data(&m, model_find_tensor(&m, "test.weight")) == 0);
        model_warm_weights(&m);
        ds4_engram_table table;
        assert(ds4_engram_table_open(&table, path, t->abs_offset, rows));
        const uint32_t id = rows - 1;
        float values[DS4_ENGRAM_DIM];
        assert(ds4_engram_read(&table, &id, 1, values));
        for (int i = 0; i < DS4_ENGRAM_DIM; i++) assert(values[i] == 1);
        ds4_engram_table_close(&table);
        model_close(&m);
    }
    assert(fclose(fp) == 0 && unlink(path) == 0);
    return 0;
}

static void check_model_layout(const char *path) {
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    assert(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    ds4_weights w;
    weights_bind(&w, &m, false, 0, UINT32_MAX, true, false);
    assert(weights_layers_bound(&w, 0, UINT32_MAX));
    assert(weights_have_output_head(&w) && !w.output_hc_fn);
    check_unmapped(m.map + m.size, 16384);
    ds4_model_map_span_vec all = {0}, dense = {0};
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        assert((w.layer[il].attn_compressor_kv != NULL) == ds41_kv_source(il));
        assert((w.layer[il].indexer_attn_q_b != NULL) == ds41_index_source(il));
        assert((w.layer[il].engram_kv != NULL) == ds41_engram_layer(il));
        model_map_span_vec_include_layer(&all, &w.layer[il]);
        model_map_span_vec_include_layer_decode_static(&dense, &w.layer[il]);
    }
    for (uint32_t i = 0; i < all.len; i++) assert(all.v[i].end <= m.size);
    for (uint32_t i = 0; i < dense.len; i++) assert(dense.v[i].end <= m.size);
    free(all.v);
    free(dense.v);
    model_summary(&m);
    model_close(&m);
    puts("V4.1 complete model layout: PASS");
}

/* Reuse the real loader/vocabulary without binding weights or opening a GPU. */
static int tokenizer_bench(int argc, char **argv) {
    assert(argc >= 5);
    char *end = NULL;
    const unsigned long repeats = strtoul(argv[3], &end, 10);
    assert(end != argv[3] && *end == '\0' && repeats > 0 && repeats <= INT_MAX);
    ds4_model model;
    ds4_vocab vocab;
    const double open_start = now_sec();
    model_open(&model, argv[2], true, false);
    config_validate_model(&model);
    const double vocab_start = now_sec();
    vocab_load(&vocab, &model);
    printf("{\"setup\":true,\"model_open_seconds\":%.9f,\"vocab_load_seconds\":%.9f,\"vocab_size\":%d}\n",
           vocab_start - open_start, now_sec() - vocab_start, vocab.n_vocab);
    for (int file = 4; file < argc; file++) {
        FILE *fp = fopen(argv[file], "rb");
        assert(fp && fseek(fp, 0, SEEK_END) == 0);
        const long size = ftell(fp);
        assert(size >= 0 && (uint64_t)size < SIZE_MAX);
        rewind(fp);
        char *text = xmalloc((size_t)size + 1);
        assert(fread(text, 1, (size_t)size, fp) == (size_t)size);
        assert(fclose(fp) == 0);
        text[size] = '\0';
        assert(strlen(text) == (size_t)size);
        for (int chat = 0; chat < 2; chat++) {
            token_vec expected = {0}, work = {0};
            if (chat) tokenize_rendered_chat_vocab(&vocab, text, &expected);
            else bpe_tokenize_text(&vocab, text, &expected);
            /* Warm the reusable output vector; per-piece allocations remain timed. */
            if (chat) tokenize_rendered_chat_vocab(&vocab, text, &work);
            else bpe_tokenize_text(&vocab, text, &work);
            assert(work.len == expected.len);
            const double start = now_sec();
            for (unsigned long repeat = 0; repeat < repeats; repeat++) {
                work.len = 0;
                if (chat) tokenize_rendered_chat_vocab(&vocab, text, &work);
                else bpe_tokenize_text(&vocab, text, &work);
                assert(work.len == expected.len);
            }
            const double elapsed = now_sec() - start;
            assert(!expected.len || !memcmp(work.v, expected.v,
                       (size_t)expected.len * sizeof(expected.v[0])));
            printf("{\"case\":%d,\"mode\":\"%s\",\"bytes\":%ld,\"repeats\":%lu,\"tokens\":%d,\"seconds\":%.9f,\"ids\":[",
                   file - 4, chat ? "chat" : "raw", size, repeats, expected.len, elapsed);
            for (int i = 0; i < expected.len; i++)
                printf("%s%d", i ? "," : "", expected.v[i]);
            puts("]}");
            token_vec_free(&work);
            token_vec_free(&expected);
        }
        free(text);
    }
    vocab_free(&vocab);
    model_close(&model);
    return 0;
}

/* Exercise the public byte decoder against every real vocabulary entry,
 * including tokens that are arbitrary byte fragments rather than UTF-8 text. */
static int token_text_dump(const char *path) {
    ds4_model model;
    ds4_engine engine = {0};
    model_open(&model, path, true, false);
    config_validate_model(&model);
    vocab_load(&engine.core.vocab, &model);
    for (int id = -1; id <= engine.core.vocab.n_vocab; id++) {
        size_t len = SIZE_MAX;
        char *text = ds4_token_text(&engine, id, &len);
        assert(text && text[len] == '\0');
        printf("{\"id\":%d,\"bytes\":[", id);
        for (size_t i = 0; i < len; i++)
            printf("%s%u", i ? "," : "", (unsigned char)text[i]);
        puts("]}");
        free(text);
    }
    vocab_free(&engine.core.vocab);
    model_close(&model);
    return 0;
}

/* Exercise actual ownership and binding twice without reading weight payloads.
 * Identities contain scalar values and offsets, never process-local addresses. */
static uint64_t lifecycle_hash(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *p = data;
    for (size_t i = 0; i < bytes; i++) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int model_lifecycle(const char *path) {
    for (int pass = 0; pass < 2; pass++) {
        ds4_model model;
        const double start = now_sec();
        model_open(&model, path, pass != 0, false);
        const double opened = now_sec();
        config_validate_model(&model);
        const double configured = now_sec();
        ds4_weights weights;
        weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
        const double bound = now_sec();
        assert(weights_layers_bound(&weights, 0, UINT32_MAX));
        assert(weights_have_output_head(&weights));
        const long page = sysconf(_SC_PAGESIZE);
        assert(page > 0);
        const bool disk_tail = model.size < model.file_size;
        if (disk_tail) check_unmapped(model.map + model.size, (size_t)page);
        if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41) {
            assert(disk_tail);
            const ds4_tensor *first = model_find_tensor(&model, "blk.1.engram_embd.weight");
            const ds4_tensor *second = model_find_tensor(&model, "blk.14.engram_embd.weight");
            assert(first && second && first->abs_offset >= model.size && second->abs_offset >= model.size);
        } else if (ds4_model_is_qwen4()) {
            assert(disk_tail && model.ngram_fd >= 0 && model.ngram_tensor);
            assert(model.ngram_tensor->abs_offset == model.size);
        }

        uint64_t directory_hash = UINT64_C(1469598103934665603);
        for (uint64_t i = 0; i < model.n_tensors; i++) {
            const ds4_tensor *t = &model.tensors[i];
#define HASH_TENSOR(field) directory_hash = lifecycle_hash(directory_hash, &t->field, sizeof(t->field))
            HASH_TENSOR(name.len);
            directory_hash = lifecycle_hash(directory_hash, t->name.ptr, (size_t)t->name.len);
            HASH_TENSOR(ndim);
            HASH_TENSOR(dim);
            HASH_TENSOR(type);
            HASH_TENSOR(rel_offset);
            HASH_TENSOR(abs_offset);
            HASH_TENSOR(elements);
            HASH_TENSOR(bytes);
#undef HASH_TENSOR
        }
        /* Both actual records currently contain only ds4_tensor pointers.
         * memcpy reads their object representation without type-punning. A
         * changed/non-pointer member fails the directory-membership assertion. */
        _Static_assert(sizeof(ds4_weights) % sizeof(ds4_tensor *) == 0, "weight pointer slots");
        _Static_assert(sizeof(ds4_layer_weights) % sizeof(ds4_tensor *) == 0, "layer pointer slots");
        uint64_t weights_hash = UINT64_C(1469598103934665603), bound_count = 0;
        const uintptr_t first_record = (uintptr_t)model.tensors;
        const uintptr_t last_record = first_record + (size_t)model.n_tensors * sizeof(model.tensors[0]);
        for (size_t slot = 0; slot < sizeof(weights) / sizeof(ds4_tensor *); slot++) {
            const ds4_tensor *tensor;
            memcpy(&tensor, (const unsigned char *)&weights + slot * sizeof(tensor), sizeof(tensor));
            uint64_t ordinal = UINT64_MAX;
            if (tensor) {
                const uintptr_t address = (uintptr_t)tensor;
                assert(address >= first_record && address < last_record);
                assert((address - first_record) % sizeof(model.tensors[0]) == 0);
                ordinal = (address - first_record) / sizeof(model.tensors[0]);
                bound_count++;
            }
            weights_hash = lifecycle_hash(weights_hash, &ordinal, sizeof(ordinal));
        }
        printf("{\"kind\":\"identity\",\"pass\":%d,\"shared\":%s,\"file_size\":%" PRIu64
               ",\"mapped_size\":%" PRIu64 ",\"metadata_count\":%" PRIu64 ",\"tensor_count\":%" PRIu64
               ",\"tensor_data_pos\":%" PRIu64 ",\"max_tensor_bytes\":%" PRIu64
               ",\"directory_hash\":\"%016" PRIx64 "\",\"bound_slots\":%" PRIu64
               ",\"weights_hash\":\"%016" PRIx64 "\",\"disk_tail\":%s,\"ngram_fd_open\":%s,\"shape\":{\"name\":",
               pass, pass ? "true" : "false", model.file_size, model.size, model.n_kv,
               model.n_tensors, model.tensor_data_pos, model.max_tensor_bytes, directory_hash,
               bound_count, weights_hash, disk_tail ? "true" : "false", model.ngram_fd >= 0 ? "true" : "false");
        ds4_json_write_string(stdout, g_ds4_shape.name);
#define SHAPE_U(field) printf(",\"" #field "\":%" PRIu64, (uint64_t)g_ds4_shape.field)
#define SHAPE_F(field) printf(",\"" #field "\":%.9g", (double)g_ds4_shape.field)
        SHAPE_U(family);
        SHAPE_U(variant);
        SHAPE_U(n_layer);
        SHAPE_U(n_embd);
        SHAPE_U(n_vocab);
        SHAPE_U(n_head);
        SHAPE_U(n_head_kv);
        SHAPE_U(n_head_dim);
        SHAPE_U(n_value_dim);
        SHAPE_U(n_rot);
        SHAPE_U(n_out_group);
        SHAPE_U(n_lora_q);
        SHAPE_U(n_lora_o);
        SHAPE_U(n_expert);
        SHAPE_U(n_expert_used);
        SHAPE_U(n_expert_shared);
        SHAPE_U(n_ff_exp);
        SHAPE_U(n_ff_dense);
        SHAPE_U(n_hash_layer);
        SHAPE_U(n_swa);
        SHAPE_U(n_indexer_head);
        SHAPE_U(n_indexer_head_dim);
        SHAPE_U(n_indexer_top_k);
        SHAPE_U(n_hc);
        SHAPE_U(n_hc_sinkhorn_iter);
        SHAPE_U(n_nextn_predict);
        SHAPE_U(n_leading_dense);
        SHAPE_U(n_kv_lora);
        SHAPE_U(n_key_mla);
        SHAPE_U(n_value_mla);
        SHAPE_U(n_kda_head);
        SHAPE_U(n_kda_head_dim);
        SHAPE_U(n_kda_conv);
        SHAPE_U(n_hc_lowrank);
        SHAPE_U(n_lin_k_head);
        SHAPE_U(n_lin_v_head);
        SHAPE_U(n_lin_head_dim);
        SHAPE_U(n_lin_conv);
        SHAPE_U(n_full_attn_interval);
        SHAPE_U(n_ple_layer);
        SHAPE_U(n_ple_ngram);
        SHAPE_U(n_ple_heads_per_ngram);
        SHAPE_U(n_ple_conv);
        SHAPE_U(n_ple_head_dim);
        printf(",\"ple_eos_id\":%d", g_ds4_shape.ple_eos_id);
        SHAPE_F(rms_eps);
        SHAPE_F(hc_eps);
        SHAPE_F(expert_weight_scale);
        SHAPE_F(swiglu_clamp_exp);
        SHAPE_F(rope_freq_base);
        SHAPE_F(rope_scale_factor);
        SHAPE_F(rope_yarn_beta_fast);
        SHAPE_F(rope_yarn_beta_slow);
        SHAPE_F(compress_rope_freq_base);
        SHAPE_F(kda_gate_lower_bound);
        SHAPE_U(rope_orig_ctx);
#undef SHAPE_U
#undef SHAPE_F
        printf("},\"compression_ratios\":[");
        for (uint32_t layer = 0; layer < DS4_N_LAYER; layer++)
            printf("%s%u", layer ? "," : "", g_ds4_compress_ratios[layer]);
        puts("]}");

        const void *old_map = model.map;
        const int old_fd = model.fd, old_ngram_fd = model.ngram_fd;
        const double before_close = now_sec();
        model_close(&model);
        const double closed = now_sec();
        assert(model.fd == -1 && model.ngram_fd == -1 && !model.map && !model.kv && !model.tensors);
        assert(!model.size && !model.file_size && !model.n_kv && !model.n_tensors && !model.ngram_tensor);
        errno = 0;
        assert(fcntl(old_fd, F_GETFD) == -1 && errno == EBADF);
        if (old_ngram_fd >= 0) {
            errno = 0;
            assert(fcntl(old_ngram_fd, F_GETFD) == -1 && errno == EBADF);
        }
        check_unmapped(old_map, (size_t)page);
        printf("{\"kind\":\"timing\",\"pass\":%d,\"open_seconds\":%.9f,\"config_seconds\":%.9f,\"bind_seconds\":%.9f,\"close_seconds\":%.9f,\"close_reset\":true}\n",
               pass, opened - start, configured - opened, bound - configured, closed - before_close);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--model-lifecycle")) return model_lifecycle(argv[2]);
    if (argc == 3 && !strcmp(argv[1], "--token-text")) return token_text_dump(argv[2]);
    if (argc >= 2 && !strcmp(argv[1], "--tokenizer-bench")) return tokenizer_bench(argc, argv);
    if (argc == 2) {
        check_model_layout(argv[1]);
        return 0;
    }
    assert(argc == 1);
    run_fixture(0);
    run_fixture(1);
    puts("V4.1 disk-only GGUF extent: PASS");
    return 0;
}
