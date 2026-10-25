#ifndef DS4_CLEF_H
#define DS4_CLEF_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

typedef struct ds4_decision_engine ds4_decision_engine;
typedef struct ds4_decision_registry ds4_decision_registry;
typedef bool (*ds4_decision_cancel_fn)(void *context);

bool ds4_decision_engine_detect(const char *directory);
ds4_decision_engine *ds4_decision_engine_open(const char *directory,
    size_t max_tokens, char *error, size_t error_size);
void ds4_decision_engine_close(ds4_decision_engine *engine);
const char *ds4_decision_engine_name(const ds4_decision_engine *engine);
size_t ds4_decision_engine_context(const ds4_decision_engine *engine);
/* The caller owns the returned libc allocation. Model work retires before
 * returning, including on cancellation; request data can then be released. */
int ds4_decision_engine_request(ds4_decision_engine *engine, const char *body,
    bool openai, char **output, char *error, size_t error_size,
    ds4_decision_cancel_fn cancel, void *cancel_context);
ds4_decision_registry *ds4_decision_registry_create(void);
void ds4_decision_registry_free(ds4_decision_registry *registry);
bool ds4_decision_registry_add(ds4_decision_registry *registry,
    const char *directory, size_t max_tokens, char *error, size_t error_size);
size_t ds4_decision_registry_count(const ds4_decision_registry *registry);
/* id==NULL returns a JSON array; a specific id returns one model object. */
char *ds4_decision_registry_models(const ds4_decision_registry *registry,
    const char *id);
int ds4_decision_registry_request(ds4_decision_registry *registry,
    const char *body, bool openai, char **output, char *error,
    size_t error_size, ds4_decision_cancel_fn cancel, void *cancel_context);
#endif
