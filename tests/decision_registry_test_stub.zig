//! Model-free server fixtures never load a decision engine. Materialization
//! still checks the registry through its real ABI and sees no available model.
const c = @import("hosted_test_c");

export fn ds4_decision_registry_models(_: ?*const c.ds4_decision_registry, _: [*c]const u8) [*c]u8 {
    return null;
}
