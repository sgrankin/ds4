//! Numeric sampling only: caller-owned arrays/RNG, no engine or session state.
//! Preserve filter order, scalar accumulation and RNG draws across fast paths.
const std = @import("std");
const c = @import("sampling_c");
const negative_infinity: f32 = -1.0e30; // Existing finite sentinel, not -inf.
const Candidate = struct { id: c_int, logit: f32, prob: f32 };

fn values(pointer: [*c]const f32, count: usize) []const f32 {
    return if (count == 0) &.{} else pointer[0..count];
}
fn allocate(comptime T: type, count: usize) []T {
    const memory = c.ds4_sampling_alloc(count * @sizeOf(T)) orelse unreachable;
    return @as([*]T, @ptrCast(@alignCast(memory)))[0..count];
}
fn argmaxRange(logits: []const f32, begin: usize, end: usize, best: *c_int, best_value: *f32) void {
    @setFloatMode(.strict);
    var ids: [8]c_int = @splat(best.*);
    var maxima: [8]f32 = @splat(best_value.*);
    var position = begin;
    while (end - position >= 8) : (position += 8) {
        inline for (0..8) |lane| {
            const value = logits[position + lane];
            if (value > maxima[lane]) {
                maxima[lane] = value;
                ids[lane] = @intCast(position + lane);
            }
        }
    }
    // Each lane keeps its first winner; merge equal winners by token ID.
    inline for (0..8) |lane| {
        if (maxima[lane] > best_value.* or (maxima[lane] == best_value.* and ids[lane] < best.*)) {
            best_value.* = maxima[lane];
            best.* = ids[lane];
        }
    }
    for (position..end) |index| {
        if (logits[index] > best_value.*) {
            best_value.* = logits[index];
            best.* = @intCast(index);
        }
    }
}
export fn ds4_sampling_argmax(pointer: [*c]const f32, count: u32) c_int {
    @setFloatMode(.strict);
    const logits = values(pointer, count);
    var best: c_int = 0;
    var maximum = negative_infinity;
    if (c.getenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == null) {
        argmaxRange(logits, 0, logits.len, &best, &maximum);
    } else {
        for (logits, 0..) |value, index| {
            if (value > maximum) {
                maximum = value;
                best = @intCast(index);
            }
        }
    }
    return best;
}
export fn ds4_sampling_argmax_excluding(pointer: [*c]const f32, count: u32, excluded: c_int) c_int {
    @setFloatMode(.strict);
    if (pointer == null) return -1;
    const logits = values(pointer, count);
    if (c.getenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") != null) {
        var best: c_int = -1;
        var maximum = negative_infinity;
        for (logits, 0..) |value, index| {
            if (@as(c_int, @intCast(index)) == excluded) continue;
            if (best < 0 or value > maximum) {
                maximum = value;
                best = @intCast(index);
            }
        }
        return best;
    }
    const first: usize = if (excluded == 0) 1 else 0;
    if (first >= logits.len) return -1;
    var best: c_int = @intCast(first);
    var maximum = logits[first];
    const begin = first + 1;
    if (excluded >= 0 and @as(usize, @intCast(excluded)) >= begin and @as(usize, @intCast(excluded)) < logits.len) {
        const omitted: usize = @intCast(excluded);
        argmaxRange(logits, begin, omitted, &best, &maximum);
        argmaxRange(logits, omitted + 1, logits.len, &best, &maximum);
    } else argmaxRange(logits, begin, logits.len, &best, &maximum);
    return best;
}

export fn ds4_sampling_rng_f32(state: *u64) f32 {
    @setFloatMode(.strict);
    var value = state.*;
    if (value == 0) value = 0x9e3779b97f4a7c15;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    state.* = value;
    const mixed = value *% 0x2545f4914f6cdd1d;
    return @as(f32, @floatFromInt((mixed >> 40) & 0xffffff)) / 16777216.0;
}
fn greater(a: Candidate, b: Candidate) bool {
    @setFloatMode(.strict);
    return if (a.logit != b.logit) a.logit > b.logit else a.id < b.id;
}
fn before(_: void, a: Candidate, b: Candidate) bool {
    return greater(a, b); // All admitted candidates have finite logits.
}
fn sort(candidates: []Candidate) void {
    std.mem.sortUnstable(Candidate, candidates, {}, before);
}
fn siftUp(heap: []Candidate, start: usize) void {
    var index = start;
    while (index > 0) {
        const parent = (index - 1) / 2;
        if (!greater(heap[parent], heap[index])) break;
        std.mem.swap(Candidate, &heap[parent], &heap[index]);
        index = parent;
    }
}
fn siftDown(heap: []Candidate) void {
    var index: usize = 0;
    while (true) {
        const left = index * 2 + 1;
        const right = left + 1;
        var smallest = index;
        if (left < heap.len and greater(heap[smallest], heap[left])) smallest = left;
        if (right < heap.len and greater(heap[smallest], heap[right])) smallest = right;
        if (smallest == index) return;
        std.mem.swap(Candidate, &heap[index], &heap[smallest]);
        index = smallest;
    }
}
fn draw(candidates: []const Candidate, sum: f32, rng: *u64, fallback: c_int) c_int {
    @setFloatMode(.strict);
    var remaining = ds4_sampling_rng_f32(rng) * sum;
    for (candidates) |candidate| {
        remaining -= candidate.prob;
        if (remaining <= 0) return candidate.id;
    }
    return fallback;
}

// Materialize the public distribution for speculative p/q acceptance and the
// corrected rejection sample, rather than only returning a sampled token.
export fn ds4_sampling_build_probabilities(pointer: [*c]const f32, count: u32, temperature: f32, requested_k: c_int, requested_p: f32, requested_min: f32, output: [*c]f32) bool {
    @setFloatMode(.strict);
    if (pointer == null or output == null or count == 0) return false;
    const logits = values(pointer, count);
    const probabilities = output[0..count];
    @memset(probabilities, 0);
    if (temperature <= 0) {
        probabilities[@intCast(ds4_sampling_argmax(pointer, count))] = 1;
        return true;
    }
    const top_p: f32 = if (requested_p <= 0 or requested_p > 1) 1 else requested_p;
    const min_p: f32 = if (requested_min < 0) 0 else requested_min;
    var top_k = requested_k;
    if (top_k > 1024) top_k = 1024;
    if (top_k < 0) top_k = 0;
    if (@as(u32, @intCast(top_k)) > count) top_k = @intCast(count);
    var storage: [1024]Candidate = undefined;
    const capacity: usize = if (top_k > 0) @intCast(top_k) else 512;
    var finite: usize = 0;
    var maximum = negative_infinity;
    for (logits) |value| {
        if (!std.math.isFinite(value)) continue;
        finite += 1;
        if (value > maximum) maximum = value;
    }
    if (finite == 0) {
        probabilities[@intCast(ds4_sampling_argmax(pointer, count))] = 1;
        return true;
    }
    var heap_count: usize = 0;
    var full_sum: f32 = 0;
    var heap_sum: f32 = 0;
    for (logits, 0..) |value, index| {
        if (!std.math.isFinite(value)) continue;
        const raw = c.expf((value - maximum) / temperature);
        full_sum += raw;
        const candidate = Candidate{ .id = @intCast(index), .logit = value, .prob = raw };
        if (heap_count < capacity) {
            storage[heap_count] = candidate;
            heap_sum += raw;
            siftUp(&storage, heap_count);
            heap_count += 1;
        } else if (greater(candidate, storage[0])) {
            heap_sum -= storage[0].prob;
            storage[0] = candidate;
            heap_sum += raw;
            siftDown(storage[0..heap_count]);
        }
    }
    if (full_sum <= 0 or !std.math.isFinite(full_sum) or heap_count == 0) {
        probabilities[@intCast(ds4_sampling_argmax(pointer, count))] = 1;
        return true;
    }
    const heap = storage[0..heap_count];
    sort(heap);
    var base_sum = if (top_k > 0) heap_sum else full_sum;
    var filtered_sum: f32 = 0;
    var filtered: usize = 0;
    var stopped_min = false;
    var reached_top = false;
    const minimum = heap[0].prob * min_p;
    for (heap, 0..) |candidate, index| {
        if (index > 0 and candidate.prob < minimum) {
            stopped_min = true;
            break;
        }
        filtered_sum += candidate.prob;
        filtered += 1;
        if (filtered_sum / base_sum >= top_p) {
            reached_top = true;
            break;
        }
    }
    const complete = top_k > 0 or heap_count == finite or reached_top or
        (stopped_min and heap[heap_count - 1].prob < minimum);
    var candidates = heap;
    defer if (!complete) c.free(candidates.ptr);
    if (!complete) {
        candidates = allocate(Candidate, finite);
        var next: usize = 0;
        for (logits, 0..) |value, index| {
            if (!std.math.isFinite(value)) continue;
            candidates[next] = .{ .id = @intCast(index), .logit = value, .prob = c.expf((value - maximum) / temperature) };
            next += 1;
        }
        sort(candidates);
        base_sum = full_sum;
        filtered_sum = 0;
        filtered = 0;
        const full_minimum = candidates[0].prob * min_p;
        for (candidates, 0..) |candidate, index| {
            if (index > 0 and candidate.prob < full_minimum) break;
            filtered_sum += candidate.prob;
            filtered += 1;
            if (filtered_sum / base_sum >= top_p) break;
        }
    }
    if (filtered == 0 or filtered_sum <= 0 or !std.math.isFinite(filtered_sum)) {
        probabilities[@intCast(candidates[0].id)] = 1;
    } else {
        const inverse: f32 = 1.0 / filtered_sum;
        for (candidates[0..filtered]) |candidate| probabilities[@intCast(candidate.id)] = candidate.prob * inverse;
    }
    return true;
}

export fn ds4_sampling_probabilities(pointer: [*c]const f32, count: u32, maybe_rng: ?*u64) c_int {
    @setFloatMode(.strict);
    if (pointer == null or maybe_rng == null or count == 0) return -1;
    const probabilities = values(pointer, count);
    var sum: f32 = 0;
    var best: c_int = 0;
    var maximum = probabilities[0];
    for (probabilities, 0..) |probability, index| {
        if (probability > 0 and std.math.isFinite(probability)) sum += probability;
        if (probability > maximum) {
            maximum = probability;
            best = @intCast(index);
        }
    }
    if (sum <= 0 or !std.math.isFinite(sum)) return best;
    var remaining = ds4_sampling_rng_f32(maybe_rng.?) * sum;
    for (probabilities, 0..) |probability, index| {
        if (probability <= 0 or !std.math.isFinite(probability)) continue;
        remaining -= probability;
        if (remaining <= 0) return @intCast(index);
    }
    return best;
}

// Rejection correction: the normalized positive part of p-q restores the
// target distribution after rejecting a speculative draft token.
export fn ds4_sampling_residual_probabilities(target: [*c]f32, draft: [*c]const f32, count: u32, maybe_rng: ?*u64) c_int {
    @setFloatMode(.strict);
    if (target == null or draft == null or maybe_rng == null or count == 0) return -1;
    var best: c_int = 0;
    var maximum = target[0];
    var sum: f32 = 0;
    for (0..count) |index| {
        if (target[index] > maximum) {
            maximum = target[index];
            best = @intCast(index);
        }
        var residual = target[index] - draft[index];
        if (!(residual > 0) or !std.math.isFinite(residual)) residual = 0;
        target[index] = residual;
        sum += residual;
    }
    if (sum <= 0 or !std.math.isFinite(sum)) return best;
    return ds4_sampling_probabilities(target, count, maybe_rng);
}

fn fastTopP(logits: []const f32, finite: usize, maximum: f32, best: c_int, temperature: f32, top_p: f32, min_p: f32, rng: *u64) ?c_int {
    @setFloatMode(.strict);
    if (finite == 0 or (finite > 512 and top_p >= 0.999)) return null;
    const capacity = @min(finite, 512);
    var storage: [512]Candidate = undefined;
    var count: usize = 0;
    var sum: f32 = 0;
    var heap_sum: f32 = 0;
    for (logits, 0..) |value, index| {
        if (!std.math.isFinite(value)) continue;
        const probability = c.expf((value - maximum) / temperature);
        sum += probability;
        const candidate = Candidate{ .id = @intCast(index), .logit = value, .prob = probability };
        if (count < capacity) {
            storage[count] = candidate;
            heap_sum += probability;
            siftUp(&storage, count);
            count += 1;
        } else if (greater(candidate, storage[0])) {
            heap_sum -= storage[0].prob;
            storage[0] = candidate;
            heap_sum += probability;
            siftDown(storage[0..count]);
        }
    }
    if (sum <= 0 or !std.math.isFinite(sum)) return best;
    if (count < finite and heap_sum < top_p * sum) return null;
    const heap = storage[0..count];
    sort(heap);
    const relative: f32 = if (min_p > 0) min_p else 0;
    const minimum = (heap[0].prob / sum) * relative;
    const raw_minimum = heap[0].prob * relative;
    var filtered_sum: f32 = 0;
    var filtered: usize = 0;
    var stopped_min = false;
    for (heap, 0..) |candidate, index| {
        const probability = candidate.prob / sum;
        if (index > 0 and probability < minimum) {
            stopped_min = true;
            break;
        }
        filtered_sum += candidate.prob;
        filtered += 1;
        if (filtered_sum / sum >= top_p) break;
    }
    if (count < finite and stopped_min and min_p > 0 and heap[count - 1].prob >= raw_minimum) return null;
    if (filtered == 0) return best;
    return draw(heap[0..filtered], filtered_sum, rng, heap[filtered - 1].id);
}

fn fullVocabulary(logits: []const f32, temperature: f32, top_p: f32, min_p: f32, rng: *u64, scratch: []f32) c_int {
    @setFloatMode(.strict);
    var maximum = negative_infinity;
    var best: c_int = 0;
    var finite: usize = 0;
    for (logits, 0..) |value, index| {
        if (!std.math.isFinite(value)) continue;
        finite += 1;
        if (value > maximum) {
            maximum = value;
            best = @intCast(index);
        }
    }
    if (finite == 0) return ds4_sampling_argmax(logits.ptr, @intCast(logits.len));
    if (top_p < 1) {
        if (fastTopP(logits, finite, maximum, best, temperature, top_p, min_p, rng)) |token| return token;
    }
    if (top_p >= 1) {
        var sum: f32 = 0;
        const relative: f32 = if (min_p > 0) min_p else 0;
        if (relative > 1) return best;
        // Find a conservative rejection boundary using the same expf as the
        // probability path. Near-boundary values still use the expf comparison.
        var reject = negative_infinity;
        var have_reject = false;
        if (relative > 0 and std.math.isFinite(relative)) {
            var cutoff = c.logf(relative);
            var attempt: usize = 0;
            while (attempt < 8 and std.math.isFinite(cutoff)) : (attempt += 1) {
                cutoff = c.nextafterf(cutoff, -std.math.floatMax(f32));
                if (c.expf(cutoff) < relative) {
                    reject = cutoff;
                    have_reject = true;
                    break;
                }
            }
        }
        for (logits, 0..) |value, index| {
            scratch[index] = -1;
            if (!std.math.isFinite(value)) continue;
            const scaled = (value - maximum) / temperature;
            if (have_reject and scaled <= reject) continue;
            const probability = c.expf(scaled);
            if (probability < relative) continue;
            scratch[index] = probability;
            sum += probability;
        }
        if (sum <= 0 or !std.math.isFinite(sum)) return best;
        var remaining = ds4_sampling_rng_f32(rng) * sum;
        for (scratch, 0..) |probability, index| {
            if (probability < 0) continue;
            remaining -= probability;
            if (remaining <= 0) return @intCast(index);
        }
        return best;
    }

    var count: usize = 0;
    var sum: f32 = 0;
    var candidates: []Candidate = undefined;
    if (min_p > 0 and min_p <= 1) {
        // Normalization cancels in min-p, but retain the full sum in original
        // order before sorting only survivors: nucleus mass and RNG stay exact.
        for (logits, 0..) |value, index| {
            scratch[index] = -1;
            if (!std.math.isFinite(value)) continue;
            const probability = c.expf((value - maximum) / temperature);
            scratch[index] = probability;
            sum += probability;
        }
        if (sum <= 0 or !std.math.isFinite(sum)) return best;
        const minimum: f32 = (1.0 / sum) * min_p;
        for (scratch) |probability| {
            if (probability < 0 or probability / sum < minimum) continue;
            count += 1;
        }
        if (count == 0) return best;
        candidates = allocate(Candidate, count);
        var next: usize = 0;
        for (scratch, 0..) |probability, index| {
            if (probability < 0 or probability / sum < minimum) continue;
            candidates[next] = .{ .id = @intCast(index), .logit = logits[index], .prob = probability };
            next += 1;
        }
    } else {
        candidates = allocate(Candidate, finite);
        for (logits, 0..) |value, index| {
            if (!std.math.isFinite(value)) continue;
            const probability = c.expf((value - maximum) / temperature);
            candidates[count] = .{ .id = @intCast(index), .logit = value, .prob = probability };
            count += 1;
            sum += probability;
        }
    }
    defer c.free(candidates.ptr);
    if (sum <= 0 or !std.math.isFinite(sum)) return best;
    sort(candidates);
    const minimum = (candidates[0].prob / sum) * (if (min_p > 0) min_p else @as(f32, 0));
    var filtered_sum: f32 = 0;
    var filtered: usize = 0;
    for (candidates, 0..) |candidate, index| {
        const probability = candidate.prob / sum;
        if (index > 0 and probability < minimum) break;
        filtered_sum += candidate.prob;
        filtered += 1;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) return best;
    return draw(candidates[0..filtered], filtered_sum, rng, candidates[filtered - 1].id);
}

export fn ds4_sampling_pick(pointer: [*c]const f32, count: u32, temperature: f32, requested_k: c_int, requested_p: f32, requested_min: f32, rng: *u64, maybe_scratch: [*c]f32) c_int {
    @setFloatMode(.strict);
    if (temperature <= 0) return ds4_sampling_argmax(pointer, count);
    const logits = values(pointer, count);
    const top_p: f32 = if (requested_p <= 0 or requested_p > 1) 1 else requested_p;
    const min_p: f32 = if (requested_min < 0) 0 else requested_min;
    if (requested_k <= 0) {
        const scratch = if (maybe_scratch == null) allocate(f32, count) else maybe_scratch[0..count];
        defer if (maybe_scratch == null) c.free(scratch.ptr);
        return fullVocabulary(logits, temperature, top_p, min_p, rng, scratch);
    }
    const capacity: usize = @min(@as(u32, @intCast(@min(requested_k, 1024))), count);
    var storage: [1024]Candidate = undefined;
    var length: usize = 0;
    for (logits, 0..) |value, index| {
        if (!std.math.isFinite(value)) continue;
        if (length == capacity and value <= storage[length - 1].logit) continue;
        var position = if (length < capacity) length else length - 1;
        if (length < capacity) length += 1;
        while (position > 0 and storage[position - 1].logit < value) : (position -= 1) storage[position] = storage[position - 1];
        storage[position] = .{ .id = @intCast(index), .logit = value, .prob = 0 };
    }
    if (length == 0) return ds4_sampling_argmax(pointer, count);
    const candidates = storage[0..length];
    const maximum = candidates[0].logit;
    var sum: f32 = 0;
    for (candidates) |*candidate| {
        candidate.prob = c.expf((candidate.logit - maximum) / temperature);
        sum += candidate.prob;
    }
    if (sum <= 0 or !std.math.isFinite(sum)) return candidates[0].id;
    const minimum = (candidates[0].prob / sum) * min_p;
    var filtered_sum: f32 = 0;
    var filtered: usize = 0;
    for (candidates, 0..) |candidate, index| {
        const probability = candidate.prob / sum;
        if (index > 0 and probability < minimum) break;
        filtered_sum += candidate.prob;
        filtered += 1;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) return candidates[0].id;
    return draw(candidates[0..filtered], filtered_sum, rng, candidates[filtered - 1].id);
}
