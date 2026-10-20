/* CPU-only phase snapshot checks: no model open, tensors or Metal commands. */
#include "ds4_gpu.h"
#include "ds4_resident_route.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const bool enabled = argc == 2 && !strcmp(argv[1], "on");
    if (enabled) {
        setenv("DS4_METAL_V41_RESIDENT_ROUTE_PROFILE", "1", 1);
        setenv("DS4_METAL_STREAMING_EXPERT_IO_METRICS", "1", 1);
    } else {
        unsetenv("DS4_METAL_V41_RESIDENT_ROUTE_PROFILE");
        unsetenv("DS4_METAL_STREAMING_EXPERT_IO_METRICS");
    }
    /* Configuring disabled SSD mode only clears empty cache metadata. */
    ds4_gpu_set_ssd_streaming(false);
    ds4_resident_route_metrics a, b;
    ds4_gpu_stream_expert_io_metrics io_a, io_b;
    ds4_resident_route_metrics_read(NULL);
    assert(!ds4_gpu_stream_expert_io_metrics_read(NULL));
    ds4_resident_route_metrics_read(&a);
    ds4_resident_route_metrics_read(&b);
    assert(a.enabled == enabled && !memcmp(&a, &b, sizeof(a)));
    for (unsigned i = 0; i < DS4_RESIDENT_ROUTE_PROFILE_LAYERS; i++)
        assert(a.layer[i].calls == 0 && a.layer[i].plans == 0);
    assert(ds4_gpu_stream_expert_io_metrics_read(&io_a));
    assert(ds4_gpu_stream_expert_io_metrics_read(&io_b));
    assert(io_a.explicit_io_metrics_enabled == enabled);
    assert(io_a.pread_calls == 0 && io_a.pread_requested_bytes == 0 && io_a.pread_returned_bytes == 0);
    assert(io_b.cache_hits == 0 && io_b.cache_misses == 0 && io_b.cache_loader_returned_bytes == 0);
    assert(io_a.process_io_valid && io_b.process_io_valid);
    assert(io_b.process_disk_read_bytes >= io_a.process_disk_read_bytes);
    puts("PASS: CPU-only phase metrics snapshot, no model or GPU work");
    return 0;
}
