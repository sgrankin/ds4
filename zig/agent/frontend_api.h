#ifndef DS4_AGENT_FRONTEND_API_H
#define DS4_AGENT_FRONTEND_API_H
#include "terminal_api.h"
#include "ds4_agent_frontend.h"
#include "ds4_gpu_args.h"
#include "ds4_help.h"
#include "ds4_distributed.h"
#include "ds4_tp.h"
#include <ctype.h>
#include <limits.h>
#include <signal.h>
#ifdef DS4_ROCM_BUILD
#define DS4_AGENT_FRONTEND_ROCM 1
#else
#define DS4_AGENT_FRONTEND_ROCM 0
#endif
#endif
