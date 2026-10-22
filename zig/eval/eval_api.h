#ifndef DS4_EVAL_FRONTEND_API_H
#define DS4_EVAL_FRONTEND_API_H
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE 1
#endif
#include "ds4_eval_frontend.h"
#include "ds4_help.h"
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#ifdef DS4_ROCM_BUILD
#define DS4_EVAL_ROCM 1
#else
#define DS4_EVAL_ROCM 0
#endif
#ifdef DS4_NO_GPU
#define DS4_EVAL_DEFAULT_BACKEND DS4_BACKEND_CPU
#elif defined(__APPLE__)
#define DS4_EVAL_DEFAULT_BACKEND DS4_BACKEND_METAL
#else
#define DS4_EVAL_DEFAULT_BACKEND DS4_BACKEND_CUDA
#endif
#endif
