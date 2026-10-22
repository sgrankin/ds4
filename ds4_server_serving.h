#ifndef DS4_SERVER_SERVING_H
#define DS4_SERVER_SERVING_H

#include "ds4_server_generation.h"
#include "ds4_server_materialize.h"
#include "ds4_server_http.h"

/* Existing pthread connection packet, moved once from ds4_server.c. It is
 * consumed before the serving transaction borrows the runtime and socket. */
typedef struct {
    server *srv;
    int fd;
} client_arg;

const char *ds4_server_serving_model_id(ds4_engine *engine);
bool ds4_server_serving_known_model(const char *id);
bool ds4_server_serving_decode(ds4_engine *engine, server *runtime,
    ds4_server_input_endpoint endpoint, const char *body, int default_tokens,
    int context, request *out, char *error, size_t capacity);
bool ds4_server_serving_exceeds_context(const request *req, int context);
int ds4_server_serving_required_slot(server *runtime, const job *pending);
int ds4_server_serving_score(server *runtime, server_slot *slot,
    const job *pending, int required_slot, time_t now);
void ds4_server_serving_dispatch(server *runtime);
bool ds4_server_serving_enqueue(server *runtime, job *pending);
void ds4_server_serving_worker(server *runtime, const volatile sig_atomic_t *stop);
void ds4_server_serving_slot_worker(server_slot *slot,
    const volatile sig_atomic_t *stop);
void ds4_server_serving_cancel(server *runtime, job *pending);
void ds4_server_serving_wait(server *runtime, job *pending);
void ds4_server_serving_stop(server *runtime);
void ds4_server_serving_models_json(buf *output, const char *id,
    const char *name, int context, int default_tokens);
void ds4_server_serving_client(server *runtime, int fd,
    const volatile sig_atomic_t *stop);

#endif
