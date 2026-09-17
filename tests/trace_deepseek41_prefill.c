/* Diagnostic-only instrumented runtime. Normal ds4.c builds have no trace I/O. */
#define DS4_TEST_V41_TRACE 1
#include "../ds4.c"
int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: %s MODEL PROMPT TOKENS scalar|batch OUTDIR\n", argv[0]); return 2; }
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .ssd_streaming = true, .power_percent = 100};
    ds4_engine *e = NULL; ds4_session *s = NULL; ds4_tokens tokens = {0};
    char *text = NULL, err[256] = {0}; size_t bytes;
    int rc = 1; uint32_t count = (uint32_t)atoi(argv[3]);
    if (!count || count > 2048 || !imatrix_read_text_file(argv[2], &text, &bytes) ||
        ds4_engine_open(&e, &opt) || ds4_session_create(&s, e, 4096)) goto done;
    ds4_tokenize_text(e, text, &tokens);
    if (count > (uint32_t)tokens.len) goto done;
    ds41_trace_dir = argv[5]; ds41_trace_pos = count - 1;
    if (!strcmp(argv[4], "scalar")) {
        for (uint32_t i = 0; i < count; i++)
            if (!ds41_graph_step(&s->ds41_graph, &e->model, &e->weights, tokens.v[i], NULL)) goto done;
    } else if (!strcmp(argv[4], "batch")) {
        if (!ds41_graph_prefill(&s->ds41_graph, &e->model, &e->weights,
                               tokens.v, count, NULL, NULL, count, NULL, NULL)) goto done;
    } else goto done;
    rc = 0;
done:
    if (rc) fprintf(stderr, "trace failed: %s\n", err);
    free(text); ds4_tokens_free(&tokens); ds4_session_free(s); ds4_engine_close(e); return rc;
}
