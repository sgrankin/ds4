//! Model-free placement math: keep these exports with the cold model object so
//! include-ds4.c fixtures do not pull in a second production C/GPU owner.
const c = @import("gguf_c");
const memory = @import("memory.zig");

export fn ds4_native_qwen4_prefill_chunk(context: u32) u32 {
    return memory.prefillChunk(context);
}

export fn ds4_native_qwen4_context_memory(config: *const c.ds4_qwen4_config, context: u32, prefill_chunk: u32) c.ds4_context_memory {
    return memory.estimate(config, context, prefill_chunk);
}
