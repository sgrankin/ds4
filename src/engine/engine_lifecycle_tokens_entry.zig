//! CPU-safe vocabulary policy entry: passing the optional immutable tokenizer
//! avoids importing the hot Flint engine/factory into model-free fixtures.
const c = @import("gguf_c");
const tokens = @import("engine_lifecycle_tokens.zig");
export fn ds4_native_engine_overthink_set(core: *c.ds4_engine_core, family: c.ds4_model_family, flint: ?*const c.flint_tokenizer, penalty: f32, think_only: bool) bool {
    return tokens.set(core, family, flint, penalty, think_only);
}
export fn ds4_native_engine_overthink_configure(core: *c.ds4_engine_core, family: c.ds4_model_family, flint: ?*const c.flint_tokenizer) bool {
    return tokens.configure(core, family, flint);
}
