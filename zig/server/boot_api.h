#ifndef DS4_SERVER_BOOT_API_H
#define DS4_SERVER_BOOT_API_H
#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L
#define _FORTIFY_SOURCE 0
#include "ds4_server_boot.h"
#include "ds4_server_images.h"
#include "ds4_state_cache_runtime.h"
#include "ds4_server_tool_memory.h"
#include "ds4_response_store.h"
#include "ds4_gpu_args.h"
#include "ds4_distributed.h"
#include "ds4_help.h"
#include "ds4_ssd.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#ifdef DS4_ROCM_BUILD
#define DS4_SERVER_BOOT_ROCM 1
#else
#define DS4_SERVER_BOOT_ROCM 0
#endif
#endif
