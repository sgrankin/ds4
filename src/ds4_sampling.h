#ifndef DS4_SAMPLING_H
#define DS4_SAMPLING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

/* Private numeric sampling boundary. Arrays and RNG state are caller-owned;
 * no engine/session/model layout crosses it. Logits are never modified.
 * Greedy selection retains the -1e30 sentinel, first-index ties and original
 * NaN behavior. Presence of DS4_CPU_DISABLE_UNROLLED_ARGMAX selects scalar
 * argmax; excluding returns -1 when no candidate remains. */
int ds4_sampling_argmax(const float *logits, uint32_t count);
int ds4_sampling_argmax_excluding(const float *logits, uint32_t count,
                                  int excluded_id);

/* xorshift64*: zero seeds use the original fixed seed. The state stores the
 * pre-multiply value; the result uses its high 24 bits in [0,1). */
float ds4_sampling_rng_f32(uint64_t *state);

/* Materialize the distribution represented by the public knobs for the
 * speculative p/q acceptance and corrected rejection sample. Invalid pointers
 * or zero count return false. Greedy/fallback paths consume no RNG draws. */
bool ds4_sampling_build_probabilities(const float *logits, uint32_t count,
                                      float temperature, int top_k,
                                      float top_p, float min_p, float *probs);
int ds4_sampling_probabilities(const float *probs, uint32_t count,
                                uint64_t *rng);
/* Replace target with max(0,p-q) before sampling its normalized distribution.
 * If there is no residual mass, return the original target's best candidate. */
int ds4_sampling_residual_probabilities(float *target, const float *draft,
                                         uint32_t count, uint64_t *rng);

/* Apply temperature, top-k, min-p and nucleus filtering in the original order.
 * Optional scratch holds count floats and is overwritten; NULL requests a
 * temporary allocation. Candidate and probability temporaries are freed before
 * return. Fast/fallback routes retain their original RNG draw counts. */
int ds4_sampling_pick(const float *logits, uint32_t count, float temperature,
                       int top_k, float top_p, float min_p, uint64_t *rng,
                       float *prob_scratch);

/* The C engine implements this through xmalloc, preserving allocation-guard
 * admission and fatal OOM diagnostics. Results are libc-owned; free normally. */
void *ds4_sampling_alloc(size_t bytes);

#endif
