//! Metal V4.1 prefill scheduling over explicit graph/model state. These are
//! the released eligibility decisions, not a new autotuner: environment flags
//! retain their original presence/value semantics and unsupported cases fall
//! back to token-major execution.
const c = @import("gguf_c");
const Config = @import("../config.zig").ModelConfig;
const Graph = c.ds41_gpu_graph;
const Weights = c.ds4_weights;

fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn boundedSetting(name: [*:0]const u8, lower: u32, upper: u32) ?u32 {
    const text = c.getenv(name);
    if (text == null or text[0] == 0) return null;
    var end: [*c]u8 = null;
    const value = c.strtoul(text, &end, 10);
    return if (end != text and end[0] == 0 and value >= lower and value <= upper) @intCast(value) else null;
}

pub const Policy = struct {
    config: *const Config,
    graph: *const Graph,
    weights: *const Weights,

    pub fn tpBatchEnabled(self: *const Policy) bool {
        return self.graph.tp_world == 1 or (self.graph.tp_world == 2 and
            !present("DS4_METAL_DISABLE_V41_BATCH_ATTN") and
            !present("DS4_METAL_DISABLE_V41_BATCH_CORE") and
            !present("DS4_METAL_DISABLE_V41_BATCH_MOE") and
            !present("DS4_METAL_DISABLE_V41_BATCH_HC"));
    }

    pub fn exactShort(self: *const Policy, count: u32) bool {
        const graph = self.graph;
        // Full sweeps lose on shorter tails/smaller buffers. The default is
        // the measured 8K-buffer, long-tail regime; diagnostics may opt in.
        const enabled = !present("DS4_METAL_DISABLE_V41_EXACT_SHORT_PREFILL") and
            (present("DS4_METAL_V41_EXACT_SHORT_PREFILL") or (graph.prefill_cap == 8192 and count >= 768));
        return enabled and graph.streaming and graph.tp_world == 1 and !graph.quality and graph.imatrix == null and
            graph.image_count == 0 and !graph.encoder_resident and graph.pos != 0 and
            count >= 256 and count < 1024 and count <= graph.prefill_cap and
            c.ds4_gpu_stream_expert_cache_configured_count() >= self.config.shape.n_layer * self.config.shape.n_expert / 2;
    }

    pub fn largeExpertCache(self: *const Policy) bool {
        return self.graph.streaming and self.graph.pos != 0 and
            c.ds4_gpu_stream_expert_cache_configured_count() >= self.config.shape.n_layer * self.config.shape.n_expert / 2;
    }

    pub fn selectedTileCap(self: *const Policy) u32 {
        var tile: u32 = if (self.largeExpertCache() and !present("DS4_METAL_DISABLE_V41_WIDER_SELECTED_TILES")) 128 else 8;
        if (boundedSetting("DS4_METAL_V41_SELECTED_TILE", 2, 128)) |value| tile = value;
        return @min(tile, self.graph.prefill_cap);
    }

    // Small layer-major tiles keep the ordinary selected-expert cache. They
    // must never map or seed an entire expert layer for a handful of tokens.
    pub fn selectedSmall(self: *const Policy, count: u32) bool {
        const graph = self.graph;
        const shape = &self.config.shape;
        if (present("DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS") or !graph.streaming or graph.tp_world != 1 or
            graph.quality or graph.imatrix != null or graph.image_count != 0 or graph.encoder_resident or
            graph.pos == 0 or count < 2 or count > self.selectedTileCap() or count > graph.prefill_cap) return false;
        const gate_type = self.weights.layer[0].ffn_gate_exps.*.type;
        const down_type = self.weights.layer[0].ffn_down_exps.*.type;
        if (c.ds4_gpu_stream_expert_batch_supported(count, shape.n_expert, shape.n_expert_used, gate_type, down_type) == 0) return false;
        for (self.weights.layer[1..shape.n_layer]) |layer| {
            if (layer.ffn_gate_exps.*.type != gate_type or layer.ffn_down_exps.*.type != down_type) return false;
        }
        return true;
    }

    pub fn prefillCount(self: *const Policy, remaining: u32) u32 {
        const graph = self.graph;
        // TP batches use the bulk protocol; row-gate ablations stay token-major.
        if (!self.tpBatchEnabled() or graph.imatrix != null or present("DS4_METAL_DISABLE_V41_LAYER_PREFILL")) return 1;
        var minimum: u32 = if (graph.tp_world == 2 and !present("DS4_METAL_DISABLE_V41_TP_SMALL_PREFILL")) 32 else 256;
        // Resident appends avoid SSD sweeps. In particular a server 128-token
        // mixed quantum must not turn into scalar prefill.
        if (!graph.streaming and graph.tp_world == 1) minimum = 8;
        const large_cache = self.largeExpertCache();
        if (large_cache) minimum = 1024;
        const wider_selected = large_cache and !present("DS4_METAL_DISABLE_V41_WIDER_SELECTED_TILES");
        const small: u32 = @min(remaining, self.selectedTileCap());
        var selected_limit: u32 = if (wider_selected) 1024 else 256;
        if (present("DS4_METAL_V41_SELECTED_MEDIUM")) selected_limit = 768;
        if (boundedSetting("DS4_METAL_V41_SELECTED_LIMIT", 256, 1024)) |value| selected_limit = value;
        if (remaining < selected_limit and self.selectedSmall(small)) return small;
        if (remaining < minimum) return if (self.exactShort(remaining)) remaining else 1;
        if (graph.carry_cap != 0 and remaining >= 4096 and !present("DS4_METAL_DISABLE_V41_WIDE_PREFILL")) {
            const rows: u32 = @min(remaining, graph.carry_cap);
            return rows - rows % 2048;
        }
        return @min(remaining, @min(graph.prefill_cap, 2048));
    }

    pub fn encoderChunk(self: *const Policy, rows: u32) u32 {
        if (rows < 8192 and self.graph.prefill_cap > 2048) return 2048;
        // Keep the decoder suffix optimization for 8K prompts.
        if (rows < 16384 and self.graph.prefill_cap > 4096) return 4096;
        return self.graph.prefill_cap;
    }

    pub fn replayRows(self: *const Policy, rows: u32) u32 {
        const replay = replaySetting();
        if (replay == 0 or rows < 8192 or rows <= self.encoderChunk(rows) or
            present("DS4_METAL_DISABLE_V41_DECODER_SUFFIX")) return 0;
        const graph = self.graph;
        if (graph.tp_world != 1 or graph.dspark != null or graph.image_count != 0 or graph.imatrix != null or graph.quality)
            c.ds4_engine_model_die("V4.1 decoder replay requires ordinary single-GPU text prefill without DSpark");
        return replay;
    }
};

// Approximate only the wide decoder's local replay. Encoder rows and complete
// compressed/index publication remain native; attention masks the raw floor.
pub fn replaySetting() u32 {
    const text = c.getenv("DS4_V41_DECODER_REPLAY_ROWS");
    if (text == null or text[0] == 0 or c.strcmp(text, "0") == 0) return 0;
    if (c.strcmp(text, "128") == 0) return 128;
    if (c.strcmp(text, "256") == 0) return 256;
    c.ds4_engine_model_die("DS4_V41_DECODER_REPLAY_ROWS must be 0, 128 or 256");
}
pub fn replaySelected() bool {
    const text = c.getenv("DS4_V41_DECODER_REPLAY_SELECTED");
    return text != null and c.strcmp(text, "1") == 0;
}
