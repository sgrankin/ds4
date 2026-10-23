#ifndef DS4_ZIG_IMAGE_API_H
#define DS4_ZIG_IMAGE_API_H

/* Declarations only. Iris remains the existing C codec implementation; native
 * image policy imports its real records rather than duplicating their layout. */
#ifndef _FORTIFY_SOURCE
#define _FORTIFY_SOURCE 0
#endif
#include "ds4_image.h"
#include "third_party/iris/jpeg.h"
#include "third_party/iris/png.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#endif
