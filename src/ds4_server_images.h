#ifndef DS4_SERVER_IMAGES_H
#define DS4_SERVER_IMAGES_H
#include "ds4_server_materialize.h"

/* Owns the actual image-input vector and exact-byte cache records already in
 * Input/Runtime. Cache access/encode is serialized by Server.inference_mu.
 * Returned embedding bytes are independent libc-owned copies. */
void ds4_server_images_free(server_image_inputs *images);
void ds4_server_images_cache_clear(server_image_cache *cache);
bool ds4_server_images_cache_get(server_image_cache *cache,
    const server_image_input *input, ds4_vision_embedding *out);
void ds4_server_images_cache_put(server_image_cache *cache,
    const server_image_input *input, const ds4_vision_embedding *embedding,
    uint32_t dimension, size_t budget);
/* ds4_server_image_{base64,data_uri} and ds4_server_materialize_tokenize retain
 * their existing declarations/signatures in the actual Input/Materialize API. */
#endif
