//! Sole C facade for complete prefill/batch attention; no body duplication.
const c = @import("metal_c");
const Batch = @import("attention_batch.zig");
const Tensor = c.ds4_gpu_tensor;
export fn ds4_gpu_attention_prefill_raw_heads_range_tensor(heads: ?*Tensor, model_map: ?*const anyopaque, model_size: u64, sinks_offset: u64, q: ?*const Tensor, raw_kv: ?*const Tensor, q_row0: u32, n_q: u32, n_kv: u32, window: u32, n_head: u32, head_dim: u32) c_int {
    return Batch.run(.prefill_raw, .{
        .heads = heads,
        .model = model_map,
        .model_size = model_size,
        .sinks_offset = sinks_offset,
        .query = q,
        .raw = raw_kv,
        .q_row0 = q_row0,
        .rows = n_q,
        .raw_rows = n_kv,
        .window = window,
        .heads_count = n_head,
        .width = head_dim,
    });
}
export fn ds4_gpu_attention_prefill_raw_heads_tensor(heads: ?*Tensor, model_map: ?*const anyopaque, model_size: u64, sinks_offset: u64, q: ?*const Tensor, raw_kv: ?*const Tensor, n_tokens: u32, window: u32, n_head: u32, head_dim: u32) c_int {
    return Batch.run(.prefill_raw, .{
        .heads = heads,
        .model = model_map,
        .model_size = model_size,
        .sinks_offset = sinks_offset,
        .query = q,
        .raw = raw_kv,
        .rows = n_tokens,
        .window = window,
        .heads_count = n_head,
        .width = head_dim,
        .raw_rows = n_tokens,
    });
}
export fn ds4_gpu_glm_attention_dense_compact_lora_causal_tensor(lora_out: ?*Tensor, qk_low: ?*const Tensor, kv_lora_cache: ?*const Tensor, q_row0: u32, n_q: u32, n_kv: u32, cache_cap: u32, cache_f16: bool, n_head: u32, kv_lora_dim: u32, qk_dim: u32) c_int {
    return Batch.run(.glm_compact, .{
        .heads = lora_out,
        .query = qk_low,
        .raw = kv_lora_cache,
        .q_row0 = q_row0,
        .rows = n_q,
        .raw_rows = n_kv,
        .raw_cap = cache_cap,
        .raw_half = cache_f16,
        .heads_count = n_head,
        .width = kv_lora_dim,
        .scale_width = qk_dim,
    });
}
export fn ds4_gpu_attention_decode_raw_batch_heads_tensor(heads: ?*Tensor, model_map: ?*const anyopaque, model_size: u64, sinks_offset: u64, q: ?*const Tensor, raw_kv: ?*const Tensor, n_tokens: u32, pos0: u32, n_raw: u32, raw_cap: u32, raw_start: u32, window: u32, n_head: u32, head_dim: u32) c_int {
    return Batch.run(.batch_raw, .{
        .heads = heads,
        .model = model_map,
        .model_size = model_size,
        .sinks_offset = sinks_offset,
        .query = q,
        .raw = raw_kv,
        .rows = n_tokens,
        .pos0 = pos0,
        .raw_rows = n_raw,
        .raw_cap = raw_cap,
        .raw_start = raw_start,
        .window = window,
        .heads_count = n_head,
        .width = head_dim,
    });
}
export fn ds4_gpu_attention_noncausal_raw_batch_heads_tensor(heads: ?*Tensor, model_map: ?*const anyopaque, model_size: u64, sinks_offset: u64, q: ?*const Tensor, raw_kv: ?*const Tensor, n_tokens: u32, n_raw: u32, raw_cap: u32, raw_start: u32, n_head: u32, head_dim: u32) c_int {
    return Batch.run(.noncausal, .{
        .heads = heads,
        .model = model_map,
        .model_size = model_size,
        .sinks_offset = sinks_offset,
        .query = q,
        .raw = raw_kv,
        .rows = n_tokens,
        .raw_rows = n_raw,
        .raw_cap = raw_cap,
        .raw_start = raw_start,
        .heads_count = n_head,
        .width = head_dim,
    });
}
export fn ds4_gpu_attention_decode_mixed_batch_heads_tensor(heads: ?*Tensor, model_map: ?*const anyopaque, model_size: u64, sinks_offset: u64, q: ?*const Tensor, raw_kv: ?*const Tensor, comp_kv: ?*const Tensor, comp_kv_f16: u32, comp_mask: ?*const Tensor, use_comp_mask: u32, n_tokens: u32, pos0: u32, n_raw: u32, raw_cap: u32, raw_start: u32, n_comp: u32, window: u32, ratio: u32, n_head: u32, head_dim: u32) c_int {
    return Batch.run(.batch_mixed, .{
        .heads = heads,
        .model = model_map,
        .model_size = model_size,
        .sinks_offset = sinks_offset,
        .query = q,
        .raw = raw_kv,
        .comp = comp_kv,
        .comp_half = comp_kv_f16 != 0,
        .mask = comp_mask,
        .use_mask = use_comp_mask != 0,
        .rows = n_tokens,
        .pos0 = pos0,
        .raw_rows = n_raw,
        .raw_cap = raw_cap,
        .raw_start = raw_start,
        .comp_rows = n_comp,
        .window = window,
        .ratio = ratio,
        .heads_count = n_head,
        .width = head_dim,
    });
}
export fn ds4_gpu_attention_visual_mixed_batch_heads_tensor(heads: ?*Tensor, model_map: ?*const anyopaque, model_size: u64, sinks_offset: u64, q: ?*const Tensor, raw_kv: ?*const Tensor, comp_kv: ?*const Tensor, comp_kv_f16: u32, comp_mask: ?*const Tensor, use_comp_mask: u32, tokens: [*c]const i32, vocab_size: u32, n_tokens: u32, pos0: u32, n_raw: u32, raw_cap: u32, raw_start: u32, n_comp: u32, window: u32, ratio: u32, n_head: u32, head_dim: u32) c_int {
    return Batch.run(.batch_visual, .{
        .heads = heads,
        .model = model_map,
        .model_size = model_size,
        .sinks_offset = sinks_offset,
        .query = q,
        .raw = raw_kv,
        .comp = comp_kv,
        .comp_half = comp_kv_f16 != 0,
        .mask = comp_mask,
        .use_mask = use_comp_mask != 0,
        .visual_tokens = tokens,
        .vocab_size = vocab_size,
        .rows = n_tokens,
        .pos0 = pos0,
        .raw_rows = n_raw,
        .raw_cap = raw_cap,
        .raw_start = raw_start,
        .comp_rows = n_comp,
        .window = window,
        .ratio = ratio,
        .heads_count = n_head,
        .width = head_dim,
    });
}
export fn ds4_gpu_attention_indexed_mixed_batch_heads_tensor(heads: ?*Tensor, model_map: ?*const anyopaque, model_size: u64, sinks_offset: u64, q: ?*const Tensor, raw_kv: ?*const Tensor, comp_kv: ?*const Tensor, comp_kv_f16: u32, topk: ?*const Tensor, n_tokens: u32, pos0: u32, n_raw: u32, raw_cap: u32, raw_start: u32, n_comp: u32, top_k: u32, window: u32, ratio: u32, n_head: u32, head_dim: u32) c_int {
    return Batch.run(.indexed, .{
        .heads = heads,
        .model = model_map,
        .model_size = model_size,
        .sinks_offset = sinks_offset,
        .query = q,
        .raw = raw_kv,
        .comp = comp_kv,
        .comp_half = comp_kv_f16 != 0,
        .topk = topk,
        .rows = n_tokens,
        .pos0 = pos0,
        .raw_rows = n_raw,
        .raw_cap = raw_cap,
        .raw_start = raw_start,
        .comp_rows = n_comp,
        .top_k = top_k,
        .window = window,
        .ratio = ratio,
        .heads_count = n_head,
        .width = head_dim,
    });
}
export fn ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(heads: ?*Tensor, model_map: ?*const anyopaque, model_size: u64, sinks_offset: u64, q: ?*const Tensor, raw_kv: ?*const Tensor, comp_kv: ?*const Tensor, comp_kv_f16: u32, q_row0: u32, n_q: u32, n_tokens: u32, n_comp: u32, window: u32, ratio: u32, n_head: u32, head_dim: u32) c_int {
    return Batch.run(.prefill_mixed, .{
        .heads = heads,
        .model = model_map,
        .model_size = model_size,
        .sinks_offset = sinks_offset,
        .query = q,
        .raw = raw_kv,
        .comp = comp_kv,
        .comp_half = comp_kv_f16 != 0,
        .q_row0 = q_row0,
        .rows = n_q,
        .raw_rows = n_tokens,
        .comp_rows = n_comp,
        .window = window,
        .ratio = ratio,
        .heads_count = n_head,
        .width = head_dim,
    });
}
export fn ds4_gpu_attention_prefill_static_mixed_heads_tensor(heads: ?*Tensor, model_map: ?*const anyopaque, model_size: u64, sinks_offset: u64, q: ?*const Tensor, raw_kv: ?*const Tensor, comp_kv: ?*const Tensor, comp_kv_f16: u32, n_tokens: u32, n_comp: u32, window: u32, ratio: u32, n_head: u32, head_dim: u32) c_int {
    return Batch.run(.prefill_mixed, .{
        .heads = heads,
        .model = model_map,
        .model_size = model_size,
        .sinks_offset = sinks_offset,
        .query = q,
        .raw = raw_kv,
        .comp = comp_kv,
        .comp_half = comp_kv_f16 != 0,
        .rows = n_tokens,
        .comp_rows = n_comp,
        .window = window,
        .ratio = ratio,
        .heads_count = n_head,
        .width = head_dim,
        .raw_rows = n_tokens,
    });
}
export fn ds4_gpu_attention_prefill_masked_mixed_heads_tensor(heads: ?*Tensor, model_map: ?*const anyopaque, model_size: u64, sinks_offset: u64, q: ?*const Tensor, raw_kv: ?*const Tensor, comp_kv: ?*const Tensor, comp_kv_f16: u32, comp_mask: ?*const Tensor, n_tokens: u32, n_comp: u32, window: u32, ratio: u32, n_head: u32, head_dim: u32) c_int {
    return Batch.run(.prefill_masked, .{
        .heads = heads,
        .model = model_map,
        .model_size = model_size,
        .sinks_offset = sinks_offset,
        .query = q,
        .raw = raw_kv,
        .comp = comp_kv,
        .comp_half = comp_kv_f16 != 0,
        .mask = comp_mask,
        .rows = n_tokens,
        .comp_rows = n_comp,
        .window = window,
        .ratio = ratio,
        .heads_count = n_head,
        .width = head_dim,
        .raw_rows = n_tokens,
        .use_mask = true,
    });
}
