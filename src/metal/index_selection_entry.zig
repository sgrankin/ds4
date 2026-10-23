//! Sole C facade; reusable native policy is in index_selection.zig.
const c = @import("metal_c");
const Index = @import("index_selection.zig");
const Tensor = c.ds4_gpu_tensor;
export fn ds4_gpu_glm_indexer_score_one_tensor(scores: ?*Tensor, q: ?*const Tensor, weights: ?*const Tensor, keys: ?*const Tensor, rows: u32, heads: u32, width: u32, scale: f32, half: bool) c_int {
    return Index.scoreOne(scores, q, weights, keys, rows, heads, width, scale, half);
}
export fn ds4_gpu_glm_indexer_scores_batch_tensor(scores: ?*Tensor, q: ?*const Tensor, weights: ?*const Tensor, keys: ?*const Tensor, rows: u32, tokens: u32, start: u32, heads: u32, width: u32, scale: f32, half: bool) c_int {
    return Index.scoreBatch(scores, q, weights, keys, rows, tokens, start, 1, heads, width, scale, half, false);
}
export fn ds4_gpu_glm53_indexer_scores_batch_tensor(scores: ?*Tensor, q: ?*const Tensor, weights: ?*const Tensor, keys: ?*const Tensor, rows: u32, tokens: u32, start: u32, pool: u32, heads: u32, width: u32, scale: f32, half: bool) c_int {
    if (pool != 4) return 0;
    return Index.scoreBatch(scores, q, weights, keys, rows, tokens, start, pool, heads, width, scale, half, false);
}
export fn ds4_gpu_dsv41_indexer_scores_batch(scores: ?*Tensor, q: ?*const Tensor, weights: ?*const Tensor, keys: ?*const Tensor, rows: u32, tokens: u32, start: u32, ratio: u32) c_int {
    if (ratio != 1 and ratio != 2) return 0;
    return Index.scoreBatch(scores, q, weights, keys, rows, tokens, start, ratio, 32, 128, 1.0 / 64.0, false, true);
}
export fn ds4_gpu_dsv41_indexer_packed_bytes(source_rows: u32, rows: u32) u64 {
    return Index.packedBytes(source_rows, rows);
}
export fn ds4_gpu_dsv41_indexer_pack(packed_tensor: ?*Tensor, q: ?*const Tensor, keys: ?*const Tensor, source_rows: u32, rows: u32) c_int {
    return Index.pack(packed_tensor, q, keys, source_rows, rows);
}
export fn ds4_gpu_dsv41_indexer_scores_packed(scores: ?*Tensor, q: ?*const Tensor, weights: ?*const Tensor, keys: ?*const Tensor, packed_tensor: ?*const Tensor, source_rows: u32, rows: u32, start: u32, ratio: u32, packed_rows: u32, offset: u32) c_int {
    return Index.scorePacked(scores, q, weights, keys, packed_tensor, source_rows, rows, start, ratio, packed_rows, offset);
}
export fn ds4_gpu_indexer_topk_tensor(selected: ?*Tensor, scores: ?*const Tensor, width: u32, rows: u32, k: u32) c_int {
    return Index.topk(selected, scores, width, rows, k, 0, 0);
}
export fn ds4_gpu_dsv41_indexer_topk_batch(selected: ?*Tensor, scores: ?*const Tensor, width: u32, rows: u32, start: u32, ratio: u32) c_int {
    return Index.causalTopk(selected, scores, width, rows, start, ratio);
}
export fn ds4_gpu_dsv4_topk_mask_tensor(mask: ?*Tensor, selected: ?*const Tensor, width: u32, rows: u32, k: u32) c_int {
    return Index.mask(mask, selected, width, rows, k);
}
export fn ds4_gpu_sort_i32_rows_asc_tensor(out: ?*Tensor, source: ?*const Tensor, width: u32, rows: u32) c_int {
    return Index.sortRows(out, source, width, rows);
}
