//! Compatibility capture of the actual process policy, never a copied owner.
const c = @import("gguf_c");
const Storage = @import("storage.zig");
const Graph = c.ds4_glm_gpu_graph;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Layer = c.ds4_layer_weights;
const Policy = c.ds4_streaming_policy;

export fn ds4_native_glm_storage_selected(graph: ?*const Graph, layer: ?*const Layer, index: u32, policy: *const Policy) bool {
    return Storage.selectedCache(graph, layer, index, policy);
}
export fn ds4_native_glm_storage_mtp(graph: ?*const Graph, layer: ?*const Layer, index: u32) bool {
    return Storage.mtpCache(graph, layer, index);
}
export fn ds4_native_glm_storage_seed(graph: ?*const Graph, weights: ?*const Weights, layer: ?*const Layer, index: u32, policy: *const Policy) bool {
    return Storage.seedExpected(graph, weights, layer, index, policy);
}
export fn ds4_native_glm_storage_token(graph: ?*Graph, model: *const Model, weights: *const Weights) bool {
    return Storage.mapToken(graph, model, weights);
}
export fn ds4_native_glm_storage_output(graph: ?*Graph, model: *const Model, weights: *const Weights) bool {
    return Storage.mapOutput(graph, model, weights);
}
export fn ds4_native_glm_storage_begin(graph: *Graph, model: *const Model, weights: *const Weights, from_hidden: bool, static_map: *bool, policy: *const Policy) bool {
    return Storage.mapBegin(graph, model, weights, from_hidden, static_map, policy);
}
export fn ds4_native_glm_storage_layer(graph: ?*Graph, model: *const Model, weights: *const Weights, layer: u32, policy: *const Policy) bool {
    return Storage.mapLayer(graph, model, weights, layer, policy);
}
export fn ds4_native_glm_storage_prefill(graph: ?*Graph, model: *const Model, weights: *const Weights, layer: u32, rows: u32, full: bool, policy: *const Policy) bool {
    return Storage.mapPrefill(graph, model, weights, layer, rows, full, policy);
}
