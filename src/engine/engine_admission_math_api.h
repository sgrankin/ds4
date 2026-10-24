#ifndef DS4_ENGINE_ADMISSION_MATH_API_H
#define DS4_ENGINE_ADMISSION_MATH_API_H
#include <stdbool.h>
#include <stdint.h>
double ds4_native_glm_guard_default_reserve(uint64_t base, uint64_t model, bool glm53);
uint64_t ds4_native_glm_guard_budget(uint64_t base, uint64_t wired,
    double fraction, double reserve_gib);
#endif
