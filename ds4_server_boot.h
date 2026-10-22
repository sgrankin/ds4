#ifndef DS4_SERVER_BOOT_H
#define DS4_SERVER_BOOT_H

#include "ds4_server_serving.h"

/* Existing borrowed command-line configuration, moved once from the C TU.
 * Strings remain borrowed from argv/environment through startup and serving. */
typedef struct {
    ds4_engine_options engine;
    const char *gpu_vram_arg;
    const char *gpu_devices_arg;
    const char *host;
    int port;
    int ctx_size;
    int default_tokens;
    const char *chdir_path;
    const char *trace_path;
    const char *kv_disk_dir;
    uint64_t kv_disk_space_mb;
    kv_cache_options kv_cache;
    bool kv_cache_reject_different_quant;
    bool disable_exact_dsml_tool_replay;
    int tool_memory_max_ids;
    bool enable_cors;
    int batched_sessions;
    int mixed_prefill_quantum;
} server_config;

server_config ds4_server_boot_parse(int argc, char **argv,
    ds4_backend default_backend);
void ds4_server_boot_close(server *runtime);
int ds4_server_boot_main(int argc, char **argv,
    volatile sig_atomic_t *stop, volatile sig_atomic_t *listen_fd,
    ds4_backend default_backend);

/* The one existing async-signal installation boundary uses the same C flags
 * and handler as all in-flight HTTP sends. No second signal state exists. */
void ds4_server_boot_install_signals(void);


#endif
