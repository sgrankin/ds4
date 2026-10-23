//! Sole existing GPU ABI facade; all embedding transactions live in the owner.
const c = @import("metal_c");
const owner = @import("embedding.zig");
const Tensor = c.ds4_gpu_tensor;
export fn ds4_gpu_embed_token_q8_0_tensor(
    out: ?*Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    vocab: u32,
    token: u32,
    width: u32,
) c_int {
    return @intFromBool((owner.Embedding{
        .out = out,
        .model = model,
        .size = size,
        .offset = offset,
        .vocab = vocab,
        .width = width,
        .single = token,
    }).quant(c.DS4_METAL_TENSOR_Q8_0));
}
export fn ds4_gpu_embed_tokens_q8_0_tensor(
    out: ?*Tensor,
    ids: ?*const Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    vocab: u32,
    tokens: u32,
    width: u32,
) c_int {
    return @intFromBool((owner.Embedding{
        .out = out,
        .ids = ids,
        .model = model,
        .size = size,
        .offset = offset,
        .vocab = vocab,
        .tokens = tokens,
        .width = width,
    }).quant(c.DS4_METAL_TENSOR_Q8_0));
}
export fn ds4_gpu_embed_token_quant_tensor(
    out: ?*Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    format: u32,
    vocab: u32,
    token: u32,
    width: u32,
) c_int {
    return @intFromBool((owner.Embedding{
        .out = out,
        .model = model,
        .size = size,
        .offset = offset,
        .vocab = vocab,
        .width = width,
        .single = token,
    }).quant(format));
}
export fn ds4_gpu_embed_tokens_quant_tensor(
    out: ?*Tensor,
    ids: ?*const Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    format: u32,
    vocab: u32,
    tokens: u32,
    width: u32,
) c_int {
    return @intFromBool((owner.Embedding{
        .out = out,
        .ids = ids,
        .model = model,
        .size = size,
        .offset = offset,
        .vocab = vocab,
        .tokens = tokens,
        .width = width,
    }).quant(format));
}
export fn ds4_gpu_embed_token_hc_tensor(
    out: ?*Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    vocab: u32,
    token: u32,
    width: u32,
    copies: u32,
) c_int {
    return @intFromBool((owner.Embedding{
        .out = out,
        .model = model,
        .size = size,
        .offset = offset,
        .vocab = vocab,
        .width = width,
        .single = token,
    }).hc(copies));
}
export fn ds4_gpu_embed_tokens_hc_tensor(
    out: ?*Tensor,
    ids: ?*const Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    vocab: u32,
    tokens: u32,
    width: u32,
    copies: u32,
) c_int {
    return @intFromBool((owner.Embedding{
        .out = out,
        .ids = ids,
        .model = model,
        .size = size,
        .offset = offset,
        .vocab = vocab,
        .tokens = tokens,
        .width = width,
    }).hc(copies));
}
export fn ds4_gpu_repeat_hc_tensor(out: ?*Tensor, row: ?*const Tensor, width: u32, copies: u32) c_int {
    return @intFromBool(owner.repeatRows(out, row, 1, width, copies, true));
}
export fn ds4_gpu_repeat_hc_rows_tensor(
    out: ?*Tensor,
    rows: ?*const Tensor,
    tokens: u32,
    width: u32,
    copies: u32,
) c_int {
    return @intFromBool(owner.repeatRows(out, rows, tokens, width, copies, false));
}
export fn ds4_gpu_glm53_embedding_bf16(
    out: ?*Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    ids: ?*const Tensor,
    tokens: u32,
    width: u32,
    vocab: u32,
) c_int {
    return @intFromBool((owner.Embedding{
        .out = out,
        .ids = ids,
        .model = model,
        .size = size,
        .offset = offset,
        .vocab = vocab,
        .tokens = tokens,
        .width = width,
    }).bf16());
}
export fn ds4_gpu_dsv41_gather_kv(
    out: ?*Tensor,
    source: ?*const Tensor,
    ids: ?*const Tensor,
    rows: u32,
    selected: u32,
) c_int {
    return @intFromBool(owner.gather(out, source, ids, rows, selected, c.DS4_EMBED_F32_GATHER));
}
export fn ds4_gpu_dsv41_gather_kv_f16(
    out: ?*Tensor,
    source: ?*const Tensor,
    ids: ?*const Tensor,
    rows: u32,
    selected: u32,
) c_int {
    return @intFromBool(owner.gather(out, source, ids, rows, selected, c.DS4_EMBED_F16));
}
export fn ds4_gpu_dsv41_gather_kv_f16_packed(
    out: ?*Tensor,
    source: ?*const Tensor,
    ids: ?*const Tensor,
    rows: u32,
    selected: u32,
) c_int {
    return @intFromBool(owner.gather(out, source, ids, rows, selected, c.DS4_EMBED_PACKED_GATHER));
}
