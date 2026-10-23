const std = @import("std");
const builtin = @import("builtin");

const c_flags: []const []const u8 = &.{ "-O3", "-ffast-math", "-g", "-mcpu=native", "-Wall", "-Wextra", "-std=c99", "-UNDEBUG" };
const strict_flags: []const []const u8 = &.{ "-O3", "-fno-fast-math", "-g", "-mcpu=native", "-Wall", "-Wextra", "-std=c99", "-UNDEBUG" };
const debug_flags: []const []const u8 = &.{ "-O0", "-fno-fast-math", "-g", "-mcpu=native", "-Wall", "-Wextra", "-std=c99", "-UNDEBUG" };
const objc_flags: []const []const u8 = &.{ "-O3", "-ffast-math", "-g", "-mcpu=native", "-Wall", "-Wextra", "-fobjc-arc", "-UNDEBUG" };

fn module(b: *std.Build, target: std.Build.ResolvedTarget, sdk: []const u8, optimize: std.builtin.Optimize) *std.Build.Module {
    const m = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
    m.addSystemIncludePath(.{ .cwd_relative = b.fmt("{s}/usr/include", .{sdk}) });
    m.addFrameworkPath(.{ .cwd_relative = b.fmt("{s}/System/Library/Frameworks", .{sdk}) });
    m.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/usr/lib", .{sdk}) });
    return m;
}

fn object(b: *std.Build, target: std.Build.ResolvedTarget, sdk: []const u8, clang: []const u8, source: []const u8, name: []const u8, optimize: std.builtin.Optimize, flags: []const []const u8) std.Build.LazyPath {
    if (std.mem.endsWith(u8, source, ".zig")) {
        const m = module(b, target, sdk, optimize);
        m.root_source_file = b.path(source);
        return b.addObject(.{ .name = name, .root_module = m }).getEmittedBin();
    }
    const compile = b.addSystemCommand(&.{clang});
    compile.setCwd(b.path("."));
    compile.addArgs(flags);
    const version = target.result.os.version_range.semver.min;
    const clang_target = b.fmt("arm64-apple-macos{d}.{d}.{d}", .{ version.major, version.minor, version.patch });
    compile.addArgs(&.{ "-isysroot", sdk, "-target", clang_target, "-I", ".", "-MMD", "-MF" });
    // Run parses Clang's discovered project-header dependencies for cache reuse.
    _ = compile.addDepFileOutputArg2(b.fmt("{s}.d", .{name}), .{});
    compile.addArg("-c");
    compile.addFileArg2(b.path(source), .{});
    compile.addArg("-o");
    return compile.addOutputFileArg2(b.fmt("{s}.o", .{name}), .{});
}

// Generate declarations from real project headers; handwritten modules share
// these Module objects where their C records must have the same Zig identity.
fn generatedApi(b: *std.Build, target: std.Build.ResolvedTarget, sdk: []const u8, optimize: std.builtin.Optimize, header: []const u8) *std.Build.Module {
    const api = b.addTranslateC(.{
        .root_source_file = b.path(header),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    api.addIncludePath(b.path("."));
    api.addSystemIncludePath(.{ .cwd_relative = b.fmt("{s}/usr/include", .{sdk}) });
    api.addFrameworkPath(.{ .cwd_relative = b.fmt("{s}/System/Library/Frameworks", .{sdk}) });
    api.defineCMacro("_FORTIFY_SOURCE", "0");
    return api.createModule();
}

fn nativeObject(b: *std.Build, target: std.Build.ResolvedTarget, sdk: []const u8, optimize: std.builtin.Optimize, source: []const u8, api_name: []const u8, api: *std.Build.Module) std.Build.LazyPath {
    const m = module(b, target, sdk, optimize);
    m.root_source_file = b.path(source);
    m.addImport(api_name, api);
    if (std.mem.startsWith(u8, source, "src/server/")) servingImports(b, m, target, optimize);
    if (std.mem.eql(u8, source, "src/server/serving_entry.zig"))
        m.addImport("server_http_c", generatedApi(b, target, sdk, optimize, "src/server/http_api.h"));
    return b.addObject(.{ .name = std.fs.path.stem(source), .root_module = m }).getEmittedBin();
}

// Protocol bridges and model implementations share native types, rather than
// obtaining their type identity from a server C header. These modules also
// compile in the standalone fixture without the server, engine or Metal.
fn servingImports(b: *std.Build, destination: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.Optimize) void {
    const types = b.createModule(.{ .root_source_file = b.path("src/serving/types.zig"), .target = target, .optimize = optimize });
    const model = b.createModule(.{ .root_source_file = b.path("src/models/ds4.zig"), .target = target, .optimize = optimize });
    model.addImport("serving_types", types);
    const prompt = b.createModule(.{ .root_source_file = b.path("src/models/deepseek41/prompt.zig"), .target = target, .optimize = optimize });
    prompt.addImport("serving_types", types);
    model.addImport("ds41_prompt", prompt);
    destination.addImport("serving_types", types);
    destination.addImport("ds4_model", model);
    destination.addImport("ds41_prompt", prompt);
}

fn executable(b: *std.Build, target: std.Build.ResolvedTarget, sdk: []const u8, core: *std.Build.Step.Compile, cache: ?std.Build.LazyPath, kvstore: ?std.Build.LazyPath, name: []const u8, frontend: std.Build.LazyPath, optimize: std.builtin.Optimize) *std.Build.Step.Compile {
    const m = module(b, target, sdk, optimize);
    m.addObjectFile(frontend);
    if (cache) |cache_object| m.addObjectFile(cache_object);
    if (kvstore) |kv_object| m.addObjectFile(kv_object);
    m.linkLibrary(core);
    m.linkFramework("Foundation", .{});
    m.linkFramework("Metal", .{});
    return b.addExecutable(.{ .name = name, .root_module = m });
}

pub fn build(b: *std.Build) void {
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64)
        @panic("This build supports Apple Silicon macOS Metal only.");
    const sdk = b.option([]const u8, "sdk", "Apple macOS SDK path") orelse
        std.mem.trim(u8, b.run(&.{ "xcrun", "--show-sdk-path" }), "\r\n");
    const clang = std.mem.trim(u8, b.run(&.{ "xcrun", "--find", "clang" }), "\r\n");
    // Leave the OS unspecified so Zig detects the native macOS version.
    const target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .cpu_model = .native });
    const mode = b.option(enum { Debug, ReleaseFast }, "optimize", "Host modules and fixture optimization") orelse .ReleaseFast;
    const optimize: std.builtin.Optimize = if (mode == .Debug) .debug else .fast;

    const test_flags = if (optimize == .debug) debug_flags else strict_flags;
    // Shared public declarations/layouts for the handwritten host modules.
    const c_api = generatedApi(b, target, sdk, optimize, "src/c_api.h");

    const core_module = module(b, target, sdk, .fast);
    const suffix_match_module = matcher: {
        const m = module(b, target, sdk, optimize);
        m.root_source_file = b.path("src/suffix_match.zig");
        break :matcher m;
    };
    const native_web = web: {
        const api = generatedApi(b, target, sdk, optimize, "src/web_api.h");
        break :web nativeObject(b, target, sdk, optimize, "src/web_entry.zig", "web_c", api);
    };
    const web_data = object(b, target, sdk, clang, "ds4_web_data.c", "web-data", .fast, c_flags);
    core_module.addObjectFile(web_data);
    core_module.addObjectFile(native_web);
    for ([_][]const u8{
        "ds4.c",      "ds4_distributed.c", "ds4_tp.c",       "ds4_layer_pack.c",
        "ds4_help.c", "ds4_eval_cases.c",  "ds4_gpu_args.c", "linenoise.c",
        "rax.c",      "ds4_metal.m",
    }) |source| {
        const flags = if (std.mem.endsWith(u8, source, ".m")) objc_flags else c_flags;
        core_module.addObjectFile(object(b, target, sdk, clang, source, b.fmt("core-{s}", .{std.fs.path.stem(source)}), .fast, flags));
    }
    // Reuse the exact image objects for production and the libc-only test library.
    const image_object = object(b, target, sdk, clang, "ds4_image.c", "core-ds4_image", .fast, c_flags);
    core_module.addObjectFile(image_object);
    const image_library_module = module(b, target, sdk, optimize);
    image_library_module.addObjectFile(image_object);
    for ([_][]const u8{ "src/ssd.zig", "src/prompt_prefix.zig", "src/engram.zig" }) |source| {
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, source, "c_api", c_api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/engine/deepseek41/graph_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/gguf.zig", "gguf_c", api));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/engine/input.zig", "gguf_c", api));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/engine/vision_free_entry.zig", "gguf_c", api));
        const route_math = module(b, target, sdk, optimize);
        route_math.root_source_file = b.path("src/resident_route.zig");
        route_math.addImport("gguf_c", api);
        const graph_module = module(b, target, sdk, optimize);
        graph_module.root_source_file = b.path("src/engine/graph.zig");
        graph_module.addImport("gguf_c", api);
        graph_module.addImport("resident_route", route_math);
        core_module.addObjectFile(b.addObject(.{ .name = "graph", .root_module = graph_module }).getEmittedBin());
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "ds4_cli_chat.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/cli_chat.zig", "cli_chat_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "ds4_tokenizer.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/tokenizer.zig", "tokenizer_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "ds4_pretokenizer.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/pretokenizer.zig", "pretokenizer_c", api));
    }
    {
        const vocab_api = generatedApi(b, target, sdk, optimize, "ds4_vocab.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/chat.zig", "chat_c", vocab_api));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/vocab.zig", "vocab_c", vocab_api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "ds4_agent_tools.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/agent_tools.zig", "agent_tools_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/agent_file_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/agent_file.zig", "agent_file_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "ds4_sampling.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/sampling.zig", "sampling_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/engine/session_sampling_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/engine/session_sampling_entry.zig", "gguf_c", api));
    }
    const native_response_store = store: {
        const api = generatedApi(b, target, sdk, optimize, "src/response_store_api.h");
        const selected = nativeObject(b, target, sdk, optimize, "src/response_store.zig", "response_store_c", api);
        core_module.addObjectFile(selected);
        break :store selected;
    };
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/http_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/http.zig", "server_http_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/input_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/input_entry.zig", "server_input_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/runtime_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/coordinator.zig", "server_runtime_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/tool_memory_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/tool_memory_entry.zig", "server_tool_memory_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/output_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/output_entry.zig", "server_output_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/render_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/render.zig", "server_render_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/json_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/json.zig", "server_json_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/reply_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/reply_entry.zig", "server_reply_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/protocol_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/protocol_entry.zig", "server_protocol_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/conversations_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/conversations_entry.zig", "server_conversations_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/materialize_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/materialize_entry.zig", "server_materialize_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/decode_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/decode_entry.zig", "server_decode_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/reuse_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/reuse_entry.zig", "server_reuse_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/persist_api.h");
        const m = module(b, target, sdk, optimize);
        m.root_source_file = b.path("src/server/persist_entry.zig");
        m.addImport("server_persist_c", api);
        m.addImport("suffix_match", suffix_match_module);
        core_module.addObjectFile(b.addObject(.{ .name = "persist_entry", .root_module = m }).getEmittedBin());
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/generation_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/generation_entry.zig", "server_generation_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/serving_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/serving_entry.zig", "server_serving_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/boot_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/boot_entry.zig", "server_boot_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/server/images_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/server/image_resources_entry.zig", "server_images_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/metal/model_views_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/model_views_entry.zig", "model_views_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/metal/exact_cache_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/exact_cache_entry.zig", "exact_cache_c", api));
    }
    const native_resident_route = route: {
        const api = generatedApi(b, target, sdk, optimize, "src/resident_route_api.h");
        const selected = nativeObject(b, target, sdk, optimize, "src/resident_route_entry.zig", "gguf_c", api);
        core_module.addObjectFile(selected);
        break :route selected;
    };
    {
        const api = generatedApi(b, target, sdk, optimize, "src/engine/vision_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/engine/vision_entry.zig", "gguf_c", api));
    }
    {
        const api = generatedApi(b, target, sdk, optimize, "src/metal/resources_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/resources_entry.zig", "resource_c", api));
    }
    {
        const metal_c = generatedApi(b, target, sdk, optimize, "src/metal/bridge.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/f16_publication.zig", "metal_c", metal_c));
        const hc_object = nativeObject(b, target, sdk, optimize, "src/metal/hc_projection.zig", "metal_c", metal_c);
        core_module.addObjectFile(hc_object);
        const norm_object = nativeObject(b, target, sdk, optimize, "src/metal/norm.zig", "metal_c", metal_c);
        core_module.addObjectFile(norm_object);
        const activation_object = nativeObject(b, target, sdk, optimize, "src/metal/activation.zig", "metal_c", metal_c);
        core_module.addObjectFile(activation_object);
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/matmul_f32.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/index_selection_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/dense_matmul_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/attention_decode_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/attention_batch_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/routed_batch_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/routed_single_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/expert_cache_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/expert_tables_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/expert_load_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/attention_output_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/rope_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/compressor_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/scheduling_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/embedding_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/quant_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/hc_control_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/router_shared_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/router_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/primitives_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/supported_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/tp_control_entry.zig", "metal_c", metal_c));
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/metal/bootstrap_entry.zig", "metal_c", metal_c));
        const activation_test_module = module(b, target, sdk, optimize);
        activation_test_module.addObjectFile(object(b, target, sdk, clang, "src/metal/activation_test.c", "activation-test", optimize, test_flags));
        activation_test_module.addObjectFile(activation_object);
        const activation_test = b.addExecutable(.{ .name = "test-activation", .root_module = activation_test_module });
        b.step("native-activation-test", "Build the standalone CPU activation packet and ownership fixture").dependOn(&b.addInstallArtifact(activation_test, .{}).step);
        const norm_test_module = module(b, target, sdk, optimize);
        norm_test_module.addObjectFile(object(b, target, sdk, clang, "src/metal/norm_test.c", "norm-test", optimize, test_flags));
        norm_test_module.addObjectFile(norm_object);
        const norm_test = b.addExecutable(.{ .name = "test-norm-cpu", .root_module = norm_test_module });
        b.step("native-norm-test", "Build the standalone C/native norm descriptor and ownership fixture").dependOn(&b.addInstallArtifact(norm_test, .{}).step);

        const hc_test_module = module(b, target, sdk, optimize);
        hc_test_module.addObjectFile(object(b, target, sdk, clang, "src/metal/hc_projection_test.c", "hc-projection-test", optimize, test_flags));
        hc_test_module.addObjectFile(hc_object);
        const hc_test = b.addExecutable(.{ .name = "test-hc-projection", .root_module = hc_test_module });
        b.step("native-hc-projection-test", "Build the standalone C/native HC descriptor and ownership fixture").dependOn(&b.addInstallArtifact(hc_test, .{}).step);

        // This fixture supplies CPU stubs; it does not link the GPU/core archive.
        const test_module = module(b, target, sdk, optimize);
        test_module.root_source_file = b.path("src/metal/f16_publication_test.zig");
        test_module.addImport("metal_c", metal_c);
        const test_exe = b.addExecutable(.{ .name = "test-f16-publication", .root_module = test_module });
        b.step("native-f16-publication-test", "Build the standalone CPU descriptor/ownership fixture").dependOn(&b.addInstallArtifact(test_exe, .{}).step);
    }
    {
        const image_c = generatedApi(b, target, sdk, optimize, "src/image_api.h");
        const decode_object = nativeObject(b, target, sdk, optimize, "src/image.zig", "image_c", image_c);
        const preprocess_object = nativeObject(b, target, sdk, optimize, "src/image_preprocess.zig", "image_c", image_c);
        core_module.addObjectFile(decode_object);
        core_module.addObjectFile(preprocess_object);
        image_library_module.addObjectFile(decode_object);
        image_library_module.addObjectFile(preprocess_object);
    }
    const image_library = b.addLibrary(.{ .name = "ds4-image", .linkage = .dynamic, .root_module = image_library_module });
    b.step("image-library", "Build a libc-only library from the selected production image objects").dependOn(&b.addInstallArtifact(image_library, .{}).step);
    const core = b.addLibrary(.{ .name = "ds4-metal-c", .root_module = core_module });
    {
        const boundary_module = module(b, target, sdk, optimize);
        boundary_module.root_source_file = b.path("src/serving/boundary_test.zig");
        servingImports(b, boundary_module, target, optimize);
        const boundary_test = b.addExecutable(.{ .name = "test-serving-boundary", .root_module = boundary_module });
        b.step("serving-boundary-test", "Build the native request/model boundary fixture without engine or Metal objects").dependOn(&b.addInstallArtifact(boundary_test, .{}).step);
        const frames_module = module(b, target, sdk, optimize);
        frames_module.root_source_file = b.path("src/server/websocket_frames_test.zig");
        frames_module.addImport("server_http_c", generatedApi(b, target, sdk, optimize, "src/server/http_api.h"));
        const frames_test = b.addExecutable(.{ .name = "test-websocket-frames", .root_module = frames_module });
        b.step("websocket-frames-test", "Build focused WebSocket framing/ownership checks without a model").dependOn(&b.addInstallArtifact(frames_test, .{}).step);
    }
    const cache = cache_object: {
        const api = generatedApi(b, target, sdk, optimize, "src/state_cache_api.h");
        break :cache_object nativeObject(b, target, sdk, optimize, "src/state_cache.zig", "state_cache_c", api);
    };
    {
        const api = generatedApi(b, target, sdk, optimize, "src/cache_runtime_api.h");
        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/cache_runtime_entry.zig", "cache_runtime_c", api));
    }
    const kvstore = nativeObject(b, target, sdk, optimize, "src/kvstore.zig", "c_api", c_api);

    const native_bench = bench: {
        const api = generatedApi(b, target, sdk, optimize, "src/bench_api.h");
        break :bench nativeObject(b, target, sdk, optimize, "src/bench.zig", "bench_c", api);
    };

    const native_eval = eval: {
        const api = generatedApi(b, target, sdk, optimize, "src/eval/eval_api.h");
        break :eval nativeObject(b, target, sdk, optimize, "src/eval/eval_entry.zig", "eval_c", api);
    };
    const eval_data = object(b, target, sdk, clang, "ds4_eval_data.c", "eval-data", .fast, c_flags);
    const eval_native_shim = object(b, target, sdk, clang, "ds4_eval_native.c", "eval-native-os-tests", .fast, c_flags);
    const native_cli = cli: {
        const api = generatedApi(b, target, sdk, optimize, "src/cli_api.h");
        break :cli nativeObject(b, target, sdk, optimize, "src/cli.zig", "cli_c", api);
    };
    // Presentation and the following Stream owner share real record identity.
    const agent_presentation_api = generatedApi(b, target, sdk, optimize, "src/agent/history_api.h");
    const native_agent_presentation = nativeObject(b, target, sdk, optimize, "src/agent/presentation_entry.zig", "agent_presentation_c", agent_presentation_api);
    const native_agent_stream = nativeObject(b, target, sdk, optimize, "src/agent/stream_entry.zig", "agent_presentation_c", agent_presentation_api);
    const native_agent_history = nativeObject(b, target, sdk, optimize, "src/agent/history_entry.zig", "agent_presentation_c", agent_presentation_api);
    const agent_tool_actions_api = generatedApi(b, target, sdk, optimize, "src/agent/executor_api.h");
    const native_agent_tool_actions = nativeObject(b, target, sdk, optimize, "src/agent/tool_actions_entry.zig", "agent_tool_actions_c", agent_tool_actions_api);
    const native_agent_edit = nativeObject(b, target, sdk, optimize, "src/agent/edit_entry.zig", "agent_tool_actions_c", agent_tool_actions_api);
    const native_agent_executor = nativeObject(b, target, sdk, optimize, "src/agent/executor_entry.zig", "agent_tool_actions_c", agent_tool_actions_api);
    const agent_sessions_api = generatedApi(b, target, sdk, optimize, "src/agent/checkpoint_api.h");
    const native_agent_sessions = nativeObject(b, target, sdk, optimize, "src/agent/sessions_entry.zig", "agent_sessions_c", agent_sessions_api);
    const native_agent_checkpoint = nativeObject(b, target, sdk, optimize, "src/agent/checkpoint_entry.zig", "agent_sessions_c", agent_sessions_api);
    const agent_runtime_api = generatedApi(b, target, sdk, optimize, "src/agent/frontend_api.h");
    const prompt_prefix_module = prefix: {
        const m = module(b, target, sdk, optimize);
        m.root_source_file = b.path("src/prompt_prefix_append.zig");
        break :prefix m;
    };
    const agent_runtime_sources = [_][]const u8{ "context_runtime_entry", "result_entry", "communications_entry", "turn_entry", "summary_entry", "lifecycle_entry", "terminal_entry", "frontend_entry" };
    const native_agent_runtime = runtime: {
        var objects: [agent_runtime_sources.len]std.Build.LazyPath = undefined;
        for (agent_runtime_sources, 0..) |name, i| {
            const m = module(b, target, sdk, optimize);
            m.root_source_file = b.path(b.fmt("src/agent/{s}.zig", .{name}));
            m.addImport("agent_runtime_c", agent_runtime_api);
            m.addImport("agent_presentation_c", agent_presentation_api);
            m.addImport("suffix_match", suffix_match_module);
            m.addImport("prompt_prefix_append", prompt_prefix_module);
            objects[i] = b.addObject(.{ .name = name, .root_module = m }).getEmittedBin();
        }
        break :runtime objects;
    };
    // Keep agent context transactions scoped to the agent and its fixtures.
    const native_agent_context = context: {
        const api = generatedApi(b, target, sdk, optimize, "src/agent_context_api.h");
        break :context nativeObject(b, target, sdk, optimize, "src/agent_context_entry.zig", "agent_context_c", api);
    };
    const native_agent_bash = bash: {
        const api = generatedApi(b, target, sdk, optimize, "src/agent_bash_api.h");
        break :bash nativeObject(b, target, sdk, optimize, "src/agent_bash_entry.zig", "agent_bash_c", api);
    };
    const native_agent_compaction = compaction: {
        const api = generatedApi(b, target, sdk, optimize, "src/agent_compaction_api.h");
        break :compaction nativeObject(b, target, sdk, optimize, "src/agent_compaction_entry.zig", "agent_compaction_c", api);
    };
    const server_main = server: {
        const api = generatedApi(b, target, sdk, optimize, "src/server/boot_api.h");
        break :server nativeObject(b, target, sdk, optimize, "src/server/main.zig", "server_boot_c", api);
    };
    const agent_main = nativeObject(b, target, sdk, optimize, "src/agent/main.zig", "agent_runtime_c", agent_runtime_api);
    const server_support = object(b, target, sdk, clang, "ds4_server.c", "server-support", .fast, c_flags);
    const agent_support = object(b, target, sdk, clang, "ds4_agent.c", "agent-support", .fast, c_flags);
    inline for (.{
        .{ "ds4", native_cli },
        .{ "ds4-server", server_main },
        .{ "ds4-agent", agent_main },
        .{ "ds4-bench", native_bench },
        .{ "ds4-eval", native_eval },
    }) |program| {
        const uses_cache = std.mem.eql(u8, program[0], "ds4-server") or std.mem.eql(u8, program[0], "ds4-agent");
        const frontend = program[1];
        const exe = executable(b, target, sdk, core, if (uses_cache) cache else null, if (uses_cache) kvstore else null, program[0], frontend, .fast);
        if (std.mem.eql(u8, program[0], "ds4-server")) exe.root_module.addObjectFile(server_support);
        if (std.mem.eql(u8, program[0], "ds4-agent")) {
            exe.root_module.addObjectFile(agent_support);
            exe.root_module.addObjectFile(native_agent_context);
            exe.root_module.addObjectFile(native_agent_bash);
            exe.root_module.addObjectFile(native_agent_compaction);
            exe.root_module.addObjectFile(native_agent_presentation);
            exe.root_module.addObjectFile(native_agent_stream);
            exe.root_module.addObjectFile(native_agent_history);
            exe.root_module.addObjectFile(native_agent_tool_actions);
            exe.root_module.addObjectFile(native_agent_edit);
            exe.root_module.addObjectFile(native_agent_executor);
            for (native_agent_runtime) |runtime_object| exe.root_module.addObjectFile(runtime_object);
            exe.root_module.addObjectFile(native_agent_sessions);
            exe.root_module.addObjectFile(native_agent_checkpoint);
        }
        if (std.mem.eql(u8, program[0], "ds4-eval")) {
            exe.root_module.addObjectFile(eval_data);
            exe.root_module.addObjectFile(eval_native_shim);
        }
        b.installArtifact(exe);
        b.step(program[0], "Build this Metal program").dependOn(&b.addInstallArtifact(exe, .{}).step);
    }
    // Keep the continuation scorer finite-aware, as in gguf-tools/Makefile.
    // It links the same engine and persistence owners as the serving programs.
    const quality_flags: []const []const u8 = &.{ "-O3", "-Wall", "-Wextra", "-std=c11", "-mcpu=native" };
    const quality_frontend = object(b, target, sdk, clang, "gguf-tools/quality-testing/score_official.c", "quality-score-main", optimize, quality_flags);
    const quality_score = executable(b, target, sdk, core, cache, kvstore, "score-official", quality_frontend, optimize);
    b.step("quality-score", "Build the official-continuation quality scorer against the production engine")
        .dependOn(&b.addInstallArtifact(quality_score, .{}).step);
    const web_test_module = module(b, target, sdk, optimize);
    web_test_module.addObjectFile(object(b, target, sdk, clang, "tests/test_web_recovery.c", "test-web-recovery-main", optimize, test_flags));
    web_test_module.addObjectFile(web_data);
    web_test_module.addObjectFile(native_web);
    const web_test = b.addExecutable(.{ .name = "test-web-recovery", .root_module = web_test_module });
    b.step("web-test", "Build the existing local Web recovery fixture").dependOn(&b.addInstallArtifact(web_test, .{}).step);
    const resident_step = b.step("resident-route-test", "Build the three unchanged standalone resident-route fixtures");
    inline for (.{
        .{ "test-resident-route", "tests/test_resident_route.c" },
        .{ "test-resident-three", "tests/test_resident_three.c" },
        .{ "test-resident-drop", "tests/test_resident_drop.c" },
    }) |fixture| {
        const test_module = module(b, target, sdk, optimize);
        test_module.addObjectFile(object(b, target, sdk, clang, fixture[1], fixture[0], optimize, test_flags));
        test_module.addObjectFile(native_resident_route);
        const test_exe = b.addExecutable(.{ .name = fixture[0], .root_module = test_module });
        resident_step.dependOn(&b.addInstallArtifact(test_exe, .{}).step);
    }
    const response_store_test_module = module(b, target, sdk, optimize);
    response_store_test_module.addObjectFile(object(b, target, sdk, clang, "tests/test_response_store.c", "test-response-store-main", optimize, test_flags));
    response_store_test_module.addObjectFile(native_response_store);
    const response_store_test = b.addExecutable(.{ .name = "test-response-store", .root_module = response_store_test_module });
    b.step("response-store-test", "Build the unchanged standalone raw response-store fixture").dependOn(&b.addInstallArtifact(response_store_test, .{}).step);
    const image_test_frontend = object(b, target, sdk, clang, "tests/test_image_decode.c", "test-image-decode-main", optimize, test_flags);
    const image_test = executable(b, target, sdk, core, null, null, "test-image-decode", image_test_frontend, optimize);
    b.step("image-test", "Build the production image decode/pixel fixture").dependOn(&b.addInstallArtifact(image_test, .{}).step);
    const vision_image_test_frontend = object(b, target, sdk, clang, "tests/test_deepseek4_vision_image.c", "test-vision-image-main", optimize, test_flags);
    const vision_image_test = executable(b, target, sdk, core, null, null, "test-deepseek4-vision-image", vision_image_test_frontend, optimize);
    b.step("vision-image-test", "Build the existing production image preprocessing/layout fixture").dependOn(&b.addInstallArtifact(vision_image_test, .{}).step);
    {
        const vocab_test_frontend = object(b, target, sdk, clang, "tests/test_vocab.c", "test-vocab-main", optimize, test_flags);
        const vocab_test = executable(b, target, sdk, core, null, null, "test-vocab", vocab_test_frontend, optimize);
        b.step("vocab-test", "Build the native production vocabulary lifecycle fixture").dependOn(&b.addInstallArtifact(vocab_test, .{}).step);
    }
    const norm_metal_frontend = object(b, target, sdk, clang, "tests/test_metal_norm_bf16.c", "test-metal-norm-bf16-main", optimize, test_flags);
    const norm_metal_test = executable(b, target, sdk, core, null, null, "test-metal-norm-bf16", norm_metal_frontend, optimize);
    b.step("norm-metal-test", "Build the existing weighted/BF16 and QKV GPU fixture (run separately)").dependOn(&b.addInstallArtifact(norm_metal_test, .{}).step);
    inline for (.{
        .{ "dense-mpp-test", "test-metal-dense-mpp", "tests/test_metal_dense_mpp.c" },
        .{ "shared-bf16-test", "test-metal-shared-bf16", "tests/test_metal_shared_bf16.c" },
        .{ "ssd-experts-test", "test-metal-ssd-experts", "tests/test_metal_ssd_experts.c" },
        .{ "command-memory-test", "test-metal-command-memory", "tests/test_metal_command_memory.c" },
        .{ "qwen-kernels-test", "test-qwen4-kernels", "tests/test_qwen4_kernels.c" },
        .{ "glm53-kda-test", "test-glm53-kda", "tests/test_glm53_kda.c" },
    }) |fixture| {
        const frontend = object(b, target, sdk, clang, fixture[2], fixture[1], optimize, test_flags);
        const test_exe = executable(b, target, sdk, core, null, null, fixture[1], frontend, optimize);
        b.step(fixture[0], "Build the existing production-linked GPU fixture (run separately)").dependOn(&b.addInstallArtifact(test_exe, .{}).step);
    }
    const metal_test_frontend = object(b, target, sdk, clang, "tests/test_deepseek41_metal.c", "test-deepseek41-metal-main", optimize, test_flags);
    const metal_test = executable(b, target, sdk, core, null, null, "test-deepseek41-metal", metal_test_frontend, optimize);
    b.step("metal-test", "Build the production-linked Metal fixture (run separately)").dependOn(&b.addInstallArtifact(metal_test, .{}).step);
    // This fixture includes the selected engine; the archive supplies graph
    // ownership/GPU support without extracting a second engine definition.
    // Match Make's optimized, finite-aware engine fixture. At C -O0 the whole
    // included engine retains unreachable alternate-backend references that
    // Metal does not implement. Native graph objects still honor Debug mode.
    const graph_test_base = std.mem.concat(b.allocator, []const u8, &.{ strict_flags, &.{"-Wno-unused-function"} }) catch @panic("OOM");
    // The same bookkeeping assertions also exercise native frontier actions
    // with GPU support selected, plus the existing pooled-indexer rollback.
    const session_gpu_frontend = object(b, target, sdk, clang, "tests/test_session_state.c", "test-session-state-gpu-main", optimize, graph_test_base);
    const session_gpu_test = executable(b, target, sdk, core, null, null, "test-session-state-gpu", session_gpu_frontend, optimize);
    b.step("session-state-gpu-test", "Build the existing session fixture with GPU/native frontier support (run separately)").dependOn(&b.addInstallArtifact(session_gpu_test, .{}).step);
    // The production native drafter and session cycle, through the real engine.
    const dspark_oracle_frontend = object(b, target, sdk, clang, "tests/test_deepseek41_dspark.c", "dspark-native-oracle-main", optimize, graph_test_base);
    const dspark_oracle = executable(b, target, sdk, core, cache, kvstore, "dspark-native-oracle", dspark_oracle_frontend, optimize);
    b.step("dspark-native-oracle", "Build the trained-DSpark production fixture (run separately)")
        .dependOn(&b.addInstallArtifact(dspark_oracle, .{}).step);
    const graph_test_frontend = object(b, target, sdk, clang, "tests/test_deepseek41_graph.c", "test-deepseek41-graph-main", optimize, graph_test_base);
    const graph_test = executable(b, target, sdk, core, null, null, "test-deepseek41-graph", graph_test_frontend, optimize);
    b.step("graph-test", "Build the existing production-linked V4.1 graph fixture (run separately)").dependOn(&b.addInstallArtifact(graph_test, .{}).step);
    // Reuse the existing production-linked private-replay/SSD/F16 probe.
    const replay_checkpoint_frontend = object(b, target, sdk, clang, "speed-bench/replay_checkpoint_bench.c", "replay-checkpoint-main", optimize, test_flags);
    const replay_checkpoint = executable(b, target, sdk, core, cache, kvstore, "replay-checkpoint-bench", replay_checkpoint_frontend, optimize);
    b.step("replay-checkpoint-bench", "Build the existing private replay SSD/F16 probe (run separately)").dependOn(&b.addInstallArtifact(replay_checkpoint, .{}).step);
    const payload_frontend = object(b, target, sdk, clang, "speed-bench/experiments/zig-engine-payload-20261022/payload_roundtrip.c", "payload-roundtrip-main", optimize, test_flags);
    const payload_test = executable(b, target, sdk, core, cache, kvstore, "payload-roundtrip", payload_frontend, optimize);
    b.step("payload-test", "Build the public canonical-payload exchange fixture (run separately)").dependOn(&b.addInstallArtifact(payload_test, .{}).step);
    const suffix_frontend = object(b, target, sdk, clang, "speed-bench/experiments/recipe-rightshift-hour39-20261008/repeat.c", "suffix-repeat-main", optimize, test_flags);
    const suffix_test = executable(b, target, sdk, core, cache, kvstore, "suffix-repeat", suffix_frontend, optimize);
    b.step("suffix-test", "Build the existing repeated relocation fixture (run separately)").dependOn(&b.addInstallArtifact(suffix_test, .{}).step);
    const dspark_verify_frontend = object(b, target, sdk, clang, "speed-bench/experiments/zig-engine-dspark-20261022/verify_commit.c", "dspark-target-verify-main", optimize, graph_test_base);
    const dspark_verify_test = executable(b, target, sdk, core, cache, kvstore, "dspark-target-verify", dspark_verify_frontend, optimize);
    b.step("dspark-target-test", "Build the capture-only target verification/commit fixture (run separately)").dependOn(&b.addInstallArtifact(dspark_verify_test, .{}).step);
    const suffix_host_step = b.step("suffix-host-test", "Build the existing real suffix format/export fixtures");
    inline for (.{
        .{ "test-suffix-diagnostic-format", "tests/test_suffix_diagnostic_format.c" },
        .{ "test-session-suffix-exports", "tests/test_session_suffix_exports.c" },
    }) |fixture| {
        const frontend = object(b, target, sdk, clang, fixture[1], fixture[0], optimize, graph_test_base);
        const test_exe = executable(b, target, sdk, core, cache, kvstore, fixture[0], frontend, optimize);
        suffix_host_step.dependOn(&b.addInstallArtifact(test_exe, .{}).step);
    }
    const pretokenizer_test_frontend = object(b, target, sdk, clang, "tests/test_pretokenizer.c", "test-pretokenizer-main", optimize, test_flags);
    const pretokenizer_test = executable(b, target, sdk, core, null, null, "test-pretokenizer", pretokenizer_test_frontend, optimize);
    b.step("pretokenizer-test", "Build the production lexer/maps/BPE fixture").dependOn(&b.addInstallArtifact(pretokenizer_test, .{}).step);
    const tokenizer_test_frontend = object(b, target, sdk, clang, "tests/test_tokenizer_piece.c", "test-tokenizer-piece-main", optimize, test_flags);
    const tokenizer_test = executable(b, target, sdk, core, null, null, "test-tokenizer-piece", tokenizer_test_frontend, optimize);
    b.step("tokenizer-test", "Build the production tokenizer map/piece fixture").dependOn(&b.addInstallArtifact(tokenizer_test, .{}).step);
    // This fixture includes ds4_cli.c and opens a model; build only here.
    const cli_test_frontend = object(b, target, sdk, clang, "tests/test_deepseek41_cli.c", "test-deepseek41-cli-main", optimize, test_flags);
    const cli_test = executable(b, target, sdk, core, null, null, "test-deepseek41-cli", cli_test_frontend, optimize);
    b.step("cli-chat-test", "Build the production-linked CLI chat fixture (run separately)").dependOn(&b.addInstallArtifact(cli_test, .{}).step);
    const engram_test_frontend = object(b, target, sdk, clang, "tests/test_engram.c", "test-engram-main", optimize, test_flags);
    const engram_test = executable(b, target, sdk, core, null, null, "test-engram", engram_test_frontend, optimize);
    b.step("engram-test", "Build the public Engram hashing and disk-row fixture").dependOn(&b.addInstallArtifact(engram_test, .{}).step);
    const ssd_test_frontend = object(b, target, sdk, clang, "tests/test_ssd_cache.c", "test-ssd-cache-main", optimize, test_flags);
    const ssd_test = executable(b, target, sdk, core, null, null, "test-ssd-cache", ssd_test_frontend, optimize);
    b.step("ssd-test", "Build the existing SSD sizing and lock fixture").dependOn(&b.addInstallArtifact(ssd_test, .{}).step);
    const prefix_test_frontend = object(b, target, sdk, clang, "tests/test_prompt_prefix.c", "test-prompt-prefix-main", optimize, test_flags);
    const prefix_test = executable(b, target, sdk, core, null, null, "test-prompt-prefix", prefix_test_frontend, optimize);
    b.step("prompt-prefix-test", "Build the existing prompt-prefix parser fixture").dependOn(&b.addInstallArtifact(prefix_test, .{}).step);
    const cache_test_frontend = object(b, target, sdk, clang, "tests/test_state_cache.c", "test-state-cache-main", optimize, test_flags);
    const cache_test = executable(b, target, sdk, core, cache, kvstore, "test-state-cache", cache_test_frontend, optimize);
    b.step("state-cache-test", "Build the existing production state-cache fixture").dependOn(&b.addInstallArtifact(cache_test, .{}).step);
    const agent_test_base = std.mem.concat(b.allocator, []const u8, &.{ test_flags, &.{"-Wno-unused-function"} }) catch @panic("OOM");
    // This fixture includes ds4_agent.c and calls the selected production parser.
    const agent_test_frontend = object(b, target, sdk, clang, "tests/ds4_agent_test.c", "ds4_agent_test-main", optimize, agent_test_base);
    const agent_test = executable(b, target, sdk, core, cache, kvstore, "ds4_agent_test", agent_test_frontend, optimize);
    {
        agent_test.root_module.addObjectFile(native_agent_context);
        agent_test.root_module.addObjectFile(native_agent_bash);
        agent_test.root_module.addObjectFile(native_agent_compaction);
        agent_test.root_module.addObjectFile(native_agent_presentation);
        agent_test.root_module.addObjectFile(native_agent_stream);
        agent_test.root_module.addObjectFile(native_agent_history);
        agent_test.root_module.addObjectFile(native_agent_tool_actions);
        agent_test.root_module.addObjectFile(native_agent_edit);
        agent_test.root_module.addObjectFile(native_agent_executor);
        for (native_agent_runtime) |runtime_object| agent_test.root_module.addObjectFile(runtime_object);
        agent_test.root_module.addObjectFile(native_agent_sessions);
        agent_test.root_module.addObjectFile(native_agent_checkpoint);
    }
    b.step("agent-test", "Build the existing production-linked agent fixture").dependOn(&b.addInstallArtifact(agent_test, .{}).step);
    // These include the actual selected agent; model opening runs separately.
    const live_context_frontend = object(b, target, sdk, clang, "tests/test_agent_live_context.c", "test-agent-live-context-main", optimize, agent_test_base);
    const live_context_test = executable(b, target, sdk, core, cache, kvstore, "test-agent-live-context", live_context_frontend, optimize);
    {
        live_context_test.root_module.addObjectFile(native_agent_context);
        live_context_test.root_module.addObjectFile(native_agent_bash);
        live_context_test.root_module.addObjectFile(native_agent_compaction);
        live_context_test.root_module.addObjectFile(native_agent_presentation);
        live_context_test.root_module.addObjectFile(native_agent_stream);
        live_context_test.root_module.addObjectFile(native_agent_history);
        live_context_test.root_module.addObjectFile(native_agent_tool_actions);
        live_context_test.root_module.addObjectFile(native_agent_edit);
        live_context_test.root_module.addObjectFile(native_agent_executor);
        for (native_agent_runtime) |runtime_object| live_context_test.root_module.addObjectFile(runtime_object);
        live_context_test.root_module.addObjectFile(native_agent_sessions);
        live_context_test.root_module.addObjectFile(native_agent_checkpoint);
    }
    b.step("agent-live-context-test", "Build the existing real-vocabulary Context fixture (run separately)")
        .dependOn(&b.addInstallArtifact(live_context_test, .{}).step);
    // This test owns fake Sessions while exercising the production controller.
    // Keep its mock API mappings for the owners that reach Engine/Session.
    const suffix_mock_frontend = object(b, target, sdk, clang, "tests/test_agent_context_suffix_mock.c", "test-agent-context-suffix-mock-main", optimize, agent_test_base);
    const suffix_mock_test = executable(b, target, sdk, core, cache, kvstore, "test-agent-context-suffix-mock", suffix_mock_frontend, optimize);
    {
        const api = generatedApi(b, target, sdk, optimize, "tests/agent_context_mock_api.h");
        suffix_mock_test.root_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/agent_context_entry.zig", "agent_context_c", api));
        suffix_mock_test.root_module.addObjectFile(native_agent_bash);
        const compaction_api = generatedApi(b, target, sdk, optimize, "tests/agent_compaction_mock_api.h");
        suffix_mock_test.root_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/agent_compaction_entry.zig", "agent_compaction_c", compaction_api));
        suffix_mock_test.root_module.addObjectFile(native_agent_presentation);
        suffix_mock_test.root_module.addObjectFile(native_agent_stream);
        suffix_mock_test.root_module.addObjectFile(native_agent_history);
        suffix_mock_test.root_module.addObjectFile(native_agent_tool_actions);
        suffix_mock_test.root_module.addObjectFile(native_agent_edit);
        suffix_mock_test.root_module.addObjectFile(native_agent_executor);
        const sessions_api = generatedApi(b, target, sdk, optimize, "tests/agent_sessions_mock_api.h");
        suffix_mock_test.root_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/agent/sessions_entry.zig", "agent_sessions_c", sessions_api));
        suffix_mock_test.root_module.addObjectFile(nativeObject(b, target, sdk, optimize, "src/agent/checkpoint_entry.zig", "agent_sessions_c", sessions_api));
        const runtime_api = generatedApi(b, target, sdk, optimize, "tests/agent_runtime_mock_api.h");
        for (agent_runtime_sources) |name| {
            const m = module(b, target, sdk, optimize);
            m.root_source_file = b.path(b.fmt("src/agent/{s}.zig", .{name}));
            m.addImport("agent_runtime_c", runtime_api);
            m.addImport("agent_presentation_c", agent_presentation_api);
            m.addImport("suffix_match", suffix_match_module);
            m.addImport("prompt_prefix_append", prompt_prefix_module);
            suffix_mock_test.root_module.addObjectFile(b.addObject(.{ .name = b.fmt("mock-{s}", .{name}), .root_module = m }).getEmittedBin());
        }
        const context_runtime = module(b, target, sdk, optimize);
        context_runtime.root_source_file = b.path("src/agent/context_runtime.zig");
        context_runtime.addImport("agent_runtime_c", runtime_api);
        context_runtime.addImport("suffix_match", suffix_match_module);
        const selectors = module(b, target, sdk, optimize);
        selectors.root_source_file = b.path("tests/agent_runtime_mock_entry.zig");
        selectors.addImport("agent_runtime_c", runtime_api);
        selectors.addImport("context_runtime", context_runtime);
        suffix_mock_test.root_module.addObjectFile(b.addObject(.{ .name = "mock-context-selectors", .root_module = selectors }).getEmittedBin());
    }
    b.step("agent-context-suffix-test", "Build the existing model-free suffix/Context transaction fixture")
        .dependOn(&b.addInstallArtifact(suffix_mock_test, .{}).step);
    const server_test_base = std.mem.concat(b.allocator, []const u8, &.{ test_flags, &.{"-Wno-unused-function"} }) catch @panic("OOM");
    // This C fixture includes ds4_server.c; do not link a second server object.
    const server_test_frontend = object(b, target, sdk, clang, "tests/ds4_test.c", "ds4_test-main", optimize, server_test_base);
    const server_test = executable(b, target, sdk, core, cache, kvstore, "ds4_test", server_test_frontend, optimize);
    b.step("server-test", "Build the existing C server fixture against selected host modules").dependOn(&b.addInstallArtifact(server_test, .{}).step);
    const cursor_test_frontend = object(b, target, sdk, clang, "tests/test_gguf_cursor.c", "test-gguf-cursor-main", optimize, test_flags);
    const cursor_test = executable(b, target, sdk, core, null, null, "test-gguf-cursor", cursor_test_frontend, optimize);
    b.step("gguf-test", "Build the production GGUF cursor fixture").dependOn(&b.addInstallArtifact(cursor_test, .{}).step);
    const sparse_base = std.mem.concat(b.allocator, []const u8, &.{ test_flags, &.{ "-DDS4_NO_GPU", "-Wno-unused-function" } }) catch @panic("OOM");
    // The existing Qwen fixture defines DS4_NO_GPU itself and includes ds4.c.
    const qwen_ngram_base = std.mem.concat(b.allocator, []const u8, &.{ test_flags, &.{"-Wno-unused-function"} }) catch @panic("OOM");
    const qwen_ngram_frontend = object(b, target, sdk, clang, "tests/test_qwen4_ngrams.c", "test-qwen4-ngrams-main", optimize, qwen_ngram_base);
    const qwen_ngram_test = executable(b, target, sdk, core, null, null, "test-qwen4-ngrams", qwen_ngram_frontend, optimize);
    b.step("qwen-ngram-test", "Build the existing disk-only Qwen mapping fixture (run separately)").dependOn(&b.addInstallArtifact(qwen_ngram_test, .{}).step);
    // Match Make's primary sampling test policy: finite-aware test source,
    // ordinary production C math flags on the actual engine test TU.
    const sampling_math_flags = if (optimize == .debug) debug_flags else c_flags;
    const sampling_engine_base = std.mem.concat(b.allocator, []const u8, &.{ sampling_math_flags, &.{ "-DDS4_NO_GPU=1", "-DDS4_TEST_HOOKS=1", "-Wno-unused-function" } }) catch @panic("OOM");
    const sampling_fixture_flags = std.mem.concat(b.allocator, []const u8, &.{ sampling_math_flags, &.{ "-DDS4_TEST_HOOKS=1", "-fno-finite-math-only" } }) catch @panic("OOM");
    const sampling_engine = object(b, target, sdk, clang, "ds4.c", "sampling-engine", optimize, sampling_engine_base);
    const sampling_frontend = object(b, target, sdk, clang, "tests/test_sampling.c", "test-sampling-main", optimize, sampling_fixture_flags);
    const sampling_test = executable(b, target, sdk, core, null, null, "test-sampling", sampling_frontend, optimize);
    sampling_test.root_module.addObjectFile(sampling_engine);
    b.step("sampling-test", "Build the unchanged sampler fixture with actual Make math policy").dependOn(&b.addInstallArtifact(sampling_test, .{}).step);
    const session_test_frontend = object(b, target, sdk, clang, "tests/test_session_state.c", "test-session-state-main", optimize, sparse_base);
    const session_test = executable(b, target, sdk, core, null, null, "test-session-state", session_test_frontend, optimize);
    b.step("session-state-test", "Build the existing model-free session/chat fixture").dependOn(&b.addInstallArtifact(session_test, .{}).step);
    // This fixture includes ds4.c. Archive extraction supplies support/GGUF only.
    const sparse_test_frontend = object(b, target, sdk, clang, "tests/test_deepseek41_gguf.c", "test-deepseek41-gguf-main", optimize, sparse_base);
    const sparse_test = executable(b, target, sdk, core, null, null, "test-deepseek41-gguf", sparse_test_frontend, optimize);
    b.step("gguf-model-test", "Build the existing sparse model-loader fixture (run separately)").dependOn(&b.addInstallArtifact(sparse_test, .{}).step);
    b.step("tokenizer-model-test", "Build the existing loader fixture with tokenizer benchmark mode (run separately)").dependOn(&b.addInstallArtifact(sparse_test, .{}).step);
}
