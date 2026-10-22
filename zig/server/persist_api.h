#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L
#define _FORTIFY_SOURCE 0
#include "ds4_server_persist.h"
#include "ds4_suffix_match.h"
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
