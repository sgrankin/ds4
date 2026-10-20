int ds4_session_capture_suffix_pair_cache(ds4_session *s,
                                          ds4_suffix_cache *cache,
                                          char *err, size_t errlen) {
#ifdef DS4_HAS_DEEPSEEK41_GPU
    if (!cache) return ds4_session_capture_suffix_cache(s, cache, err, errlen);
    cache->engine = NULL;
    cache->pair_tape.len = 0;
    cache->pair_start = cache->pair_end = 0;
    ds41_suffix_pair_tape *t = s && s->ds41_graph_ready ?
                               s->ds41_graph.suffix_pair_tape : NULL;
    if (!t || !ds41_suffix_supported(s) || s->approximate_suffix) goto fail;
    const uint32_t end = s->ds41_graph.pos;
    uint32_t first = 0;
    uint64_t bytes = 0;
    if (!ds41_suffix_pair_tape_coverage(t, s->ds41_graph.ctx, end, &first, &bytes)) goto fail;
    if (ds4_session_capture_suffix_cache(s, cache, err, errlen) ||
        !ds4_gpu_synchronize()) goto fail;
    if (cache->pair_tape.cap < bytes) {
        uint8_t *p = realloc(cache->pair_tape.ptr, (size_t)bytes);
        if (!p) goto fail;
        cache->pair_tape.ptr = p;
        cache->pair_tape.cap = bytes;
    }
    if (!ds41_suffix_pair_tape_copy(t, s->ds41_graph.ctx, first, end,
                                    cache->pair_tape.ptr, cache->pair_tape.cap)) goto fail;
    cache->pair_start = first; cache->pair_end = end;
    cache->pair_tape.len = bytes;
    return 0;
fail:
    cache->engine = NULL;
    cache->pair_tape.len = 0;
    cache->pair_start = cache->pair_end = 0;
    if (!err || !errlen || !err[0])
        payload_set_err(err, errlen, "no complete native pair tape at source endpoint");
    return 1;
#else
    (void)s; (void)cache;
    payload_set_err(err, errlen, "DS4.1 pair cache requires GPU support");
    return 1;
#endif
}

