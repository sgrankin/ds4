//! Launch-directory model opening precedes tool working-directory changes.
//! Public Engine/TP owners and the real signal action retain their C ABI.
const std = @import("std");
const c = @import("agent_runtime_c");
const Options = @import("options.zig");
const Frontend = @import("frontend.zig");
const Ui = @import("ui_text.zig");

pub fn run(argc: c_int, argv: [*c][*c]u8, default_backend: c.ds4_backend, frontend: Frontend.Frontend, handler: ?*const fn (c_int) callconv(.c) void) c_int {
    var config = Options.parse(argc, argv, default_backend);
    if (config.chdir_path != null) {
        var info: c.struct_stat = undefined;
        if (c.stat(config.chdir_path, &info) != 0) {
            _ = c.fprintf(c.__stderrp, "ds4-agent: invalid working directory %s: %s\n", config.chdir_path, c.strerror(c.__error().*));
            return 1;
        }
        if (info.st_mode & c.S_IFMT != c.S_IFDIR) {
            _ = c.fprintf(c.__stderrp, "ds4-agent: %s is not a directory\n", config.chdir_path);
            return 1;
        }
    }
    config.engine.context_size = config.gen.ctx_size;
    config.engine.placement_ctx_hint = config.gen.ctx_size;
    const gpu_arguments = config.gpu_vram_arg != null or config.gpu_devices_arg != null;
    if (gpu_arguments) config.engine.backend = if (config.gpu_vram_arg != null and std.mem.eql(u8, Ui.text(config.gpu_vram_arg), "0")) c.DS4_BACKEND_CPU else c.DS4_BACKEND_CUDA;
    var engine: ?*c.ds4_engine = null;
    if (gpu_arguments) {
        var gpu = std.mem.zeroes(c.ds4_gpu_config);
        var skip_cuda = false;
        var errors: [256]u8 = undefined;
        if (c.parse_gpu_vram_arg(config.gpu_vram_arg, config.gpu_devices_arg, &gpu, &skip_cuda, &errors, errors.len) != 0) {
            _ = c.fprintf(c.__stderrp, "ds4-agent: %s\n", &errors);
            return 2;
        }
        if (skip_cuda) {
            config.engine.backend = c.DS4_BACKEND_CPU;
            if (c.ds4_engine_open(&engine, &config.engine) != 0) return 1;
        } else {
            const automatic = (config.gpu_vram_arg != null and std.mem.eql(u8, Ui.text(config.gpu_vram_arg), "auto")) or (config.gpu_vram_arg == null and config.gpu_devices_arg != null);
            var layout: [256]u8 = undefined;
            if (c.format_gpu_layout_line(&gpu, automatic, &layout, layout.len) > 0) {
                _ = c.fprintf(c.__stdoutp, "%s\n", &layout);
                _ = c.fflush(c.__stdoutp);
            }
            config.engine.backend = c.DS4_BACKEND_CUDA;
            if (c.ds4_engine_create_with_gpu_config(&engine, &config.engine, &gpu) != 0) return 1;
        }
    } else if (c.ds4_engine_open(&engine, &config.engine) != 0) return 1;
    if (c.ds4_think_mode_level(config.gen.think_mode) >= 0 and !c.ds4_engine_is_deepseek41(engine)) {
        _ = c.fprintf(c.__stderrp, "ds4-agent: --think-level requires a DeepSeek V4.1 model\n");
        c.ds4_engine_close(engine);
        return 2;
    }
    var leader: ?*c.ds4_tp = null;
    if (config.engine.tp.role == c.DS4_TP_LEADER) {
        var errors: [256]u8 = @splat(0);
        var identity = std.mem.zeroes(c.ds4_tp_identity);
        identity.gguf_bytes = c.ds4_engine_model_bytes(engine);
        identity.model_id = @intCast(c.ds4_engine_model_id(engine));
        identity.n_layer = @intCast(c.ds4_engine_layer_count(engine));
        identity.n_embd = @intCast(c.ds4_engine_embd_dim(engine));
        identity.n_vocab = @intCast(c.ds4_engine_vocab_size(engine));
        identity.quant_bits = @intCast(c.ds4_engine_routed_quant_bits(engine));
        identity.ctx_size = @intCast(config.gen.ctx_size);
        c.ds4_engine_tp_gate_schedule(engine, &identity.gate_slot_start, &identity.gate_slot_step, &identity.gates_per_token, &identity.gate_slot_mask);
        if (c.ds4_tp_create(&leader, &config.engine.tp, &identity, &errors, errors.len) == 0 or c.ds4_engine_tp_bind(engine, leader, &errors, errors.len) == 0) {
            _ = c.fprintf(c.__stderrp, "ds4-agent: %s\n", &errors);
            c.ds4_tp_free(leader);
            c.ds4_engine_close(engine);
            return 1;
        }
    }
    Options.modelDefaults(engine, &config.gen);
    // Resolve model paths and Metal kernel sources in the launch directory.
    // Tools use --chdir only after the inference engine has finished opening.
    if (config.chdir_path != null and c.chdir(config.chdir_path) != 0) {
        _ = c.fprintf(c.__stderrp, "ds4-agent: failed to chdir to %s: %s\n", config.chdir_path, c.strerror(c.__error().*));
        if (leader != null) _ = c.ds4_tp_send_stop(leader);
        c.ds4_engine_close(engine);
        c.ds4_tp_free(leader);
        return 1;
    }
    var previous: c.struct_sigaction = undefined;
    const installed = !config.non_interactive and c.ds4_agent_frontend_signal(true, handler, &previous);
    const result = if (config.non_interactive) Frontend.headless(engine, &config) else frontend.interactive(engine, &config);
    if (installed) _ = c.ds4_agent_frontend_signal(false, handler, &previous);
    if (leader != null) _ = c.ds4_tp_send_stop(leader);
    c.ds4_engine_close(engine);
    c.ds4_tp_free(leader);
    c.free(config.gen.prompt_owned);
    c.ds4_prompt_prefix_free(&config.gen.prefix);
    return result;
}
