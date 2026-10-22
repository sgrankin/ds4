//! GGUF mapping lifetime. Metadata/tensor records and their borrowed strings
//! live until the mapping owner is closed; disk-only tails are never weights.
const std = @import("std");
const c = @import("gguf_c");
const Record = c.ds4_model;
const Tensor = c.ds4_tensor;
const gguf_magic: u32 = 0x46554747; // "GGUF", little endian.
const tensor_bf16: u32 = 30; // The existing GGUF BF16 format tag.
const LoadError = error{LoadFailed};

fn emptyRecord() Record {
    var record = std.mem.zeroes(Record);
    record.fd = -1;
    record.ngram_fd = -1;
    return record;
}

// Capture the error before cleanup changes errno or releases cursor storage.
// The C facade retains the original fatal policy; a native engine can inspect
// this failure after init has already released every partially opened resource.
pub const Failure = struct {
    message: [256]u8 = @splat(0),
    operation: ?[*:0]const u8 = null,
    path: [*c]const u8 = null,
    code: c_int = 0,

    fn invalid(self: *Failure, message: [*c]const u8) LoadError {
        const bytes = std.mem.span(message);
        const count = @min(bytes.len, self.message.len - 1);
        @memcpy(self.message[0..count], bytes[0..count]);
        self.message[count] = 0;
        self.operation = null;
        return error.LoadFailed;
    }

    fn system(self: *Failure, operation: [*:0]const u8, path: [*c]const u8) LoadError {
        self.operation = operation;
        self.path = path;
        self.code = c.__error().*;
        return error.LoadFailed;
    }

    pub fn die(self: *const Failure) noreturn {
        if (self.operation) |operation| {
            c.__error().* = self.code;
            c.ds4_engine_model_die_errno(operation, self.path);
        }
        c.ds4_engine_model_die(&self.message);
    }
};

pub const Model = struct {
    record: Record = emptyRecord(),

    pub fn init(path: [*c]const u8, metal_mapping: bool, prefetch_cpu: bool, failure: *Failure) LoadError!Model {
        var self = Model{};
        errdefer self.deinit();

        self.record.fd = c.open(path, c.O_RDONLY);
        if (self.record.fd == -1) return failure.system("cannot open model", path);
        var status: c.struct_stat = undefined;
        if (c.fstat(self.record.fd, &status) == -1) return failure.system("cannot stat model", path);
        if (status.st_size < 32) return failure.invalid("model file is too small to be GGUF");

        // Metal wraps slices as no-copy MTLBuffers, requiring MAP_SHARED. CPU
        // reads retain MAP_PRIVATE: late faults while streaming a huge shared
        // GGUF have triggered Darwin VM map-count panics, not ordinary errors.
        // Tokenizer inspection uses the private mapping without prefetching.
        const length: usize = @intCast(status.st_size);
        const flags = if (metal_mapping) c.MAP_SHARED else c.MAP_PRIVATE;
        const mapping = c.mmap(null, length, c.PROT_READ, flags, self.record.fd, 0);
        if (mapping == c.MAP_FAILED) return failure.system("cannot mmap model", path);
        self.record.map = @ptrCast(mapping);
        self.record.size = @intCast(length);
        self.record.file_size = self.record.size;

        var cursor = c.ds4_cursor{
            .base = self.record.map,
            .size = self.record.size,
            .pos = 0,
            .@"error" = @splat(0),
        };
        var magic: u32 = undefined;
        if (!c.ds4_gguf_u32(&cursor, &magic)) return failure.invalid(&cursor.@"error");
        if (magic != gguf_magic) return failure.invalid("model is not a GGUF file");
        if (!c.ds4_gguf_u32(&cursor, &self.record.version) or
            !c.ds4_gguf_u64(&cursor, &self.record.n_tensors) or
            !c.ds4_gguf_u64(&cursor, &self.record.n_kv)) return failure.invalid(&cursor.@"error");
        if (self.record.version != 3) return failure.invalid("only GGUF v3 is supported");

        try self.readDirectories(&cursor, failure);
        try self.unmapEngram(failure);
        try self.unmapQwenNgrams(path, failure);
        if (!metal_mapping and prefetch_cpu) self.prefetchCpu();
        return self;
    }

    pub fn deinit(self: *Model) void {
        closeRecord(&self.record);
    }

    // Existing C engine storage takes over the same record without a second
    // allocation or a second owner. Native callers can keep Model directly.
    pub fn take(self: *Model) Record {
        const record = self.record;
        self.record = emptyRecord();
        return record;
    }

    fn tensors(self: *const Model) []const Tensor {
        return if (self.record.n_tensors == 0) &.{} else self.record.tensors[0..@intCast(self.record.n_tensors)];
    }

    fn readDirectories(self: *Model, cursor: *c.ds4_cursor, failure: *Failure) LoadError!void {
        // Every entry needs at least one remaining byte. Admit counts before
        // calloc so a tiny file cannot request an enormous descriptor array.
        if (self.record.n_kv > cursor.size - cursor.pos)
            return failure.invalid("GGUF metadata count exceeds file size");
        const metadata = c.calloc(@intCast(self.record.n_kv), @sizeOf(c.ds4_gguf_kv)) orelse
            return failure.invalid("out of memory while allocating metadata table");
        self.record.kv = @ptrCast(@alignCast(metadata));
        if (!c.ds4_gguf_parse_metadata(cursor, self.record.kv, self.record.n_kv, &self.record.alignment))
            return failure.invalid(&cursor.@"error");

        if (self.record.n_tensors > cursor.size - cursor.pos)
            return failure.invalid("GGUF tensor count exceeds file size");
        const descriptors = c.calloc(@intCast(self.record.n_tensors), @sizeOf(Tensor)) orelse
            return failure.invalid("out of memory while allocating tensor table");
        self.record.tensors = @ptrCast(@alignCast(descriptors));
        const message = c.ds4_gguf_parse_tensors(&self.record, cursor);
        if (message != null) return failure.invalid(message);
    }

    fn hasArchitecture(self: *const Model, name: []const u8) bool {
        var architecture = std.mem.zeroes(c.ds4_str);
        return c.ds4_gguf_get_string(&self.record, "general.architecture", &architecture) and equal(architecture, name);
    }

    fn unmapEngram(self: *Model, failure: *Failure) LoadError!void {
        if (!self.hasArchitecture("deepseek41")) return;
        var encoding = std.mem.zeroes(c.ds4_str);
        if (!c.ds4_gguf_get_string(&self.record, "deepseek41.engram.encoding", &encoding) or
            !equal(encoding, "e4m3_e8m0_32_row264")) return failure.invalid("unsupported V4.1 Engram row encoding");

        // Engram is deliberately outside the weight mapping, not just omitted
        // from residency lists: warming/no-copy views must never fault its
        // 189 GiB random-access table into the model's resident weight range.
        var tables: [2]?*const Tensor = @splat(null);
        var resident_end = self.record.tensor_data_pos;
        self.record.max_tensor_bytes = 0;
        for (self.tensors()) |*tensor| {
            const index: ?usize = if (equal(tensor.name, "blk.1.engram_embd.weight")) 0 else if (equal(tensor.name, "blk.14.engram_embd.weight")) 1 else null;
            if (index) |slot| {
                if (tables[slot] != null or tensor.ndim != 2 or tensor.type != 24 or
                    tensor.dim[0] != 264 or tensor.dim[1] == 0 or tensor.dim[1] > std.math.maxInt(u32))
                    return failure.invalid("invalid or duplicate V4.1 Engram tensor");
                tables[slot] = tensor;
            } else {
                if (tensor.bytes == 0) return failure.invalid("unsupported V4.1 resident tensor type");
                resident_end = @max(resident_end, tensor.abs_offset + tensor.bytes);
                self.record.max_tensor_bytes = @max(self.record.max_tensor_bytes, tensor.bytes);
            }
        }
        const first = tables[0] orelse return failure.invalid("V4.1 GGUF is missing Engram tables");
        const second = tables[1] orelse return failure.invalid("V4.1 GGUF is missing Engram tables");
        const start = first.abs_offset;
        const page = c.sysconf(c._SC_PAGESIZE);
        if (page <= 0 or start % @as(u64, @intCast(page)) != 0 or resident_end > start or
            second.abs_offset < start + first.bytes)
            return failure.invalid("V4.1 Engram tables must follow the page-aligned main weights");
        if (c.munmap(@constCast(self.record.map + @as(usize, @intCast(start))), @intCast(self.record.size - start)) != 0)
            return failure.system("cannot unmap disk-only region", "Engram");
        self.record.size = start;
    }

    fn unmapQwenNgrams(self: *Model, path: [*c]const u8, failure: *Failure) LoadError!void {
        if (!self.hasArchitecture("qwen4exp")) return;
        var table: ?*const Tensor = null;
        var resident_end = self.record.tensor_data_pos;
        self.record.max_tensor_bytes = 0;
        for (self.tensors()) |*tensor| {
            if (equal(tensor.name, "per_layer_token_embd.weight")) {
                if (table != null or tensor.type != tensor_bf16 or tensor.ndim != 2 or
                    tensor.dim[0] == 0 or tensor.dim[0] > 160 or tensor.dim[1] == 0 or tensor.dim[1] > std.math.maxInt(u32))
                    return failure.invalid("Qwen requires original BF16 n-grams; repack with gguf-tools/qwen4_native_ngrams.py");
                table = tensor;
            } else {
                if (tensor.bytes == 0) return failure.invalid("unsupported Qwen resident tensor type");
                resident_end = @max(resident_end, tensor.abs_offset + tensor.bytes);
                self.record.max_tensor_bytes = @max(self.record.max_tensor_bytes, tensor.bytes);
            }
        }
        // Tokenizer inspection still opens old main-only files. Inference
        // rejects them later when binding the required n-gram tensor.
        const ngrams = table orelse return;
        const page = c.sysconf(c._SC_PAGESIZE);
        if (page <= 0 or ngrams.abs_offset % @as(u64, @intCast(page)) != 0 or resident_end > ngrams.abs_offset)
            return failure.invalid("Qwen n-grams must follow page-aligned weights; repack with gguf-tools/qwen4_native_ngrams.py");

        const descriptor = c.open(path, c.O_RDONLY | c.O_CLOEXEC);
        if (descriptor < 0) return failure.system("cannot open n-gram table", path);
        var transferred = false;
        defer {
            if (!transferred) _ = c.close(descriptor);
        }
        var main_status: c.struct_stat = undefined;
        var table_status: c.struct_stat = undefined;
        if (c.fstat(descriptor, &table_status) != 0 or c.fstat(self.record.fd, &main_status) != 0 or
            table_status.st_dev != main_status.st_dev or table_status.st_ino != main_status.st_ino or
            table_status.st_size < 0 or @as(u64, @intCast(table_status.st_size)) != self.record.file_size)
            return failure.invalid("Qwen GGUF changed while opening its n-grams");
        if (c.fcntl(descriptor, c.F_NOCACHE, @as(c_int, 1)) != 0 or c.fcntl(descriptor, c.F_RDAHEAD, @as(c_int, 0)) != 0)
            return failure.system("cannot configure n-gram disk reads", path);
        if (c.munmap(@constCast(self.record.map + @as(usize, @intCast(ngrams.abs_offset))), @intCast(self.record.size - ngrams.abs_offset)) != 0)
            return failure.system("cannot unmap disk-only n-grams", path);
        self.record.size = ngrams.abs_offset;
        self.record.ngram_fd = descriptor;
        self.record.ngram_tensor = ngrams;
        transferred = true;
    }

    fn prefetchCpu(self: *const Model) void {
        if (self.record.map == null or self.record.size == 0) return;
        // This is only a page-cache hint, not a copy or pin. Bringing pages in
        // before CPU generation avoids late Darwin file-backed VM fault stress.
        if (@hasDecl(c, "POSIX_MADV_WILLNEED")) {
            const result = c.posix_madvise(@constCast(self.record.map), @intCast(self.record.size), c.POSIX_MADV_WILLNEED);
            if (result != 0) c.ds4_engine_model_prefetch_warning(result);
        }
    }
};

fn equal(value: c.ds4_str, wanted: []const u8) bool {
    return value.len == wanted.len and (wanted.len == 0 or std.mem.eql(u8, value.ptr[0..wanted.len], wanted));
}

fn closeRecord(record: *Record) void {
    if (record.ngram_tensor != null and record.ngram_fd >= 0) _ = c.close(record.ngram_fd);
    c.free(record.kv);
    c.free(record.tensors);
    if (record.map != null) _ = c.munmap(@constCast(record.map), @intCast(record.size));
    if (record.fd >= 0) _ = c.close(record.fd);
    record.* = emptyRecord();
}

pub fn openRecord(record: *Record, path: [*c]const u8, metal_mapping: bool, prefetch_cpu: bool) void {
    record.* = emptyRecord(); // The old entry point initializes caller storage.
    var failure = Failure{};
    var owner = Model.init(path, metal_mapping, prefetch_cpu, &failure) catch failure.die();
    defer owner.deinit();
    record.* = owner.take();
}

pub fn closeRecordPointer(record: ?*Record) void {
    if (record) |value| closeRecord(value);
}
