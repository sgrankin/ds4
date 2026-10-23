//! Test-only access to the actual native suffix selectors.
const c = @import("agent_runtime_c");
const Runtime = @import("context_runtime").Runtime;

export fn agent_context_suffix_select(worker: *c.agent_worker, edited: *const c.ds4_tokens, fresh: c_int, plan: *c.agent_context_suffix_plan) bool {
    return (Runtime{ .worker = worker }).selectSuffix(edited, fresh, plan);
}

export fn agent_context_suffix_select_disk(worker: *c.agent_worker, edited: *const c.ds4_tokens, fresh: c_int, plan: *c.agent_context_suffix_plan) bool {
    return (Runtime{ .worker = worker }).selectDisk(edited, fresh, plan);
}
