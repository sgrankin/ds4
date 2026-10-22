#ifndef DS4_CONVERSATIONS_H
#define DS4_CONVERSATIONS_H

/* Included after the server's JSON/input helpers. These records outlive model
 * slots and response files. Manual item edits deliberately discard private
 * replay; their visible items remain available for a new rendered context. */
#include "ds4_server_input.h"

typedef struct {
    char *object, *items, *replay;
    unsigned long long version;
    bool deleted;
} ds4_conversation;

#ifdef DS4_NATIVE_CONVERSATIONS
/* These are complete storage/route actions, independent of Server/Session.
 * Owned record strings and returned JSON use libc; statuses match HTTP. */
char *ds4_conversation_json_field(const char *object, const char *wanted);
void ds4_conversation_free(ds4_conversation *conversation);
int ds4_conversation_load(const char *root, const char *id,
    ds4_conversation *conversation, char *error, size_t error_capacity);
bool ds4_conversation_input_items(const char *raw, char **normalized,
    char *error, size_t error_capacity);
int ds4_conversation_publish(const char *root, const char *id,
    unsigned long long expected_version, const char *input, const char *output,
    const char *replay, char *error, size_t error_capacity);
int ds4_conversation_route(const char *root, const char *method,
    const char *path, const char *body, char **public_json,
    char *error, size_t error_capacity);
#else
static pthread_mutex_t ds4_conversation_mutex = PTHREAD_MUTEX_INITIALIZER;
static void responses_random_id(char *dst, size_t dstlen, const char *prefix);

static int ds4_conversation_error(char *err, size_t len, int status, const char *text) {
    if (err && len) snprintf(err, len, "%s", text);
    return status;
}

/* A raw owned value, including its JSON quoting/brackets, or NULL if absent. */
static char *ds4_conversation_json_field(const char *object, const char *wanted) {
    if (!object) return NULL;
    const char *p = object;
    json_ws(&p);
    if (*p++ != '{') return NULL;
    char *found = NULL;
    for (;;) {
        json_ws(&p);
        if (*p == '}') { p++; break; }
        char *key = NULL, *value = NULL;
        if (!json_string(&p, &key)) goto fail;
        json_ws(&p);
        if (*p++ != ':' || !json_raw_value(&p, &value)) { free(key); goto fail; }
        if (!strcmp(key, wanted)) { free(found); found = value; value = NULL; }
        free(key);
        free(value);
        json_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') { p++; break; }
        goto fail;
    }
    json_ws(&p);
    if (*p) goto fail;
    return found;
fail:
    free(found);
    return NULL;
}

static bool ds4_conversation_valid_id(const char *id) {
    if (!id || strncmp(id, "conv_", 5) || !id[5] || strlen(id) > 35) return false;
    for (const unsigned char *p = (const unsigned char *)id + 5; *p; p++) {
        if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
              (*p >= '0' && *p <= '9') || *p == '_')) return false;
    }
    return true;
}

static bool ds4_conversation_item_id(const char *id) {
    if (!id || !*id || strlen(id) > 128) return false;
    for (const unsigned char *p = (const unsigned char *)id; *p; p++) {
        if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
              (*p >= '0' && *p <= '9') || *p == '_')) return false;
    }
    return true;
}

static void ds4_conversation_free(ds4_conversation *c) {
    if (!c) return;
    free(c->object); free(c->items); free(c->replay);
    memset(c, 0, sizeof(*c));
}

static int ds4_conversation_location(const char *root, const char *id,
                                     char dir[PATH_MAX], char key[41],
                                     char *err, size_t errlen) {
    if (!root || !*root)
        return ds4_conversation_error(err, errlen, 500, "conversation store is not configured");
    if (!ds4_conversation_valid_id(id))
        return ds4_conversation_error(err, errlen, 404, "conversation not found");
    int n = snprintf(dir, PATH_MAX, "%s/conversations", root);
    if (n < 0 || n >= PATH_MAX)
        return ds4_conversation_error(err, errlen, 500, "conversation store path is too long");
    snprintf(key, 41, "resp_%s", id);
    return 0;
}

static bool ds4_conversation_u64(const char *raw, unsigned long long *value) {
    if (!raw || *raw < '0' || *raw > '9') return false;
    errno = 0;
    char *end = NULL;
    unsigned long long n = strtoull(raw, &end, 10);
    if (errno || !end || *end) return false;
    *value = n;
    return true;
}

static bool ds4_conversation_array(const char *raw, char ***items, size_t *count) {
    *items = NULL; *count = 0;
    if (!raw) return false;
    const char *p = raw;
    json_ws(&p);
    if (*p++ != '[') return false;
    for (;;) {
        json_ws(&p);
        if (*p == ']') { p++; break; }
        char *item = NULL;
        if (*p != '{' || !json_raw_value(&p, &item)) goto fail;
        *items = xrealloc(*items, (*count + 1) * sizeof(**items));
        (*items)[(*count)++] = item;
        json_ws(&p);
        if (*p == ',') { p++; json_ws(&p); if (*p == ']') goto fail; continue; }
        if (*p == ']') { p++; break; }
        goto fail;
    }
    json_ws(&p);
    if (!*p) return true;
fail:
    for (size_t i = 0; i < *count; i++) free((*items)[i]);
    free(*items); *items = NULL; *count = 0;
    return false;
}

static void ds4_conversation_array_free(char **items, size_t n) {
    for (size_t i = 0; i < n; i++) free(items[i]);
    free(items);
}

static char *ds4_conversation_array_json(char **items, size_t n) {
    buf b = {0};
    buf_putc(&b, '[');
    for (size_t i = 0; i < n; i++) {
        if (i) buf_putc(&b, ',');
        buf_puts(&b, items[i]);
    }
    buf_putc(&b, ']');
    return buf_take(&b);
}

static int ds4_conversation_load_unlocked(const char *root, const char *id,
                                         ds4_conversation *c, char *err, size_t errlen) {
    memset(c, 0, sizeof(*c));
    char dir[PATH_MAX], key[41];
    int rc = ds4_conversation_location(root, id, dir, key, err, errlen);
    if (rc) return rc;
    char *record = NULL;
    size_t bytes = 0;
    rc = ds4_response_store_read(dir, key, &record, &bytes, err, errlen);
    if (rc) return rc == 1 ? ds4_conversation_error(err, errlen, 404, "conversation was not found") : 500;
    c->object = ds4_conversation_json_field(record, "conversation");
    c->items = ds4_conversation_json_field(record, "items");
    c->replay = ds4_conversation_json_field(record, "replay");
    char *version = ds4_conversation_json_field(record, "version");
    char *deleted = ds4_conversation_json_field(record, "deleted");
    char *object_id = c->object ? responses_json_string_field(c->object, "id") : NULL;
    char **items = NULL;
    size_t n = 0;
    bool ok = c->object && c->items && c->replay && object_id && !strcmp(object_id, id) &&
              ds4_conversation_u64(version, &c->version) && c->version > 0 &&
              deleted && (!strcmp(deleted, "true") || !strcmp(deleted, "false")) &&
              ds4_conversation_array(c->items, &items, &n);
    c->deleted = deleted && !strcmp(deleted, "true");
    if (c->replay && !strcmp(c->replay, "null")) { free(c->replay); c->replay = NULL; }
    ds4_conversation_array_free(items, n);
    free(version); free(deleted); free(object_id); free(record);
    if (!ok) {
        ds4_conversation_free(c);
        return ds4_conversation_error(err, errlen, 500, "invalid conversation record");
    }
    return 0;
}

static int ds4_conversation_load(const char *root, const char *id,
                                ds4_conversation *c, char *err, size_t errlen) {
    pthread_mutex_lock(&ds4_conversation_mutex);
    int rc = ds4_conversation_load_unlocked(root, id, c, err, errlen);
    if (!rc && c->deleted) {
        ds4_conversation_free(c);
        rc = ds4_conversation_error(err, errlen, 404, "conversation was deleted");
    }
    pthread_mutex_unlock(&ds4_conversation_mutex);
    return rc;
}

static int ds4_conversation_write_unlocked(const char *root, const char *id,
                                          const ds4_conversation *c,
                                          char *err, size_t errlen) {
    char dir[PATH_MAX], key[41];
    int rc = ds4_conversation_location(root, id, dir, key, err, errlen);
    if (rc) return rc;
    if (ds4_response_store_init(dir, err, errlen)) return 500;
    buf b = {0};
    buf_puts(&b, "{\"conversation\":"); buf_puts(&b, c->object);
    buf_puts(&b, ",\"items\":"); buf_puts(&b, c->items);
    buf_puts(&b, ",\"replay\":"); buf_puts(&b, c->replay ? c->replay : "null");
    buf_printf(&b, ",\"version\":%llu,\"deleted\":%s}", c->version, c->deleted ? "true" : "false");
    rc = ds4_response_store_write(dir, key, b.ptr, b.len, err, errlen) ? 500 : 0;
    buf_free(&b);
    return rc;
}

/* Normalize easy input strings and assign public item IDs without changing
 * tool call IDs. The ordinary Responses parser remains the replay validator. */
static bool ds4_conversation_input_items(const char *raw, char **normalized,
                                          char *err, size_t errlen) {
    *normalized = NULL;
    if (!raw) return false;
    const char *p = raw;
    json_ws(&p);
    char *easy = NULL;
    if (*p == '"') {
        char *text = NULL;
        if (!json_string(&p, &text)) goto fail;
        json_ws(&p);
        if (*p) { free(text); goto fail; }
        buf b = {0};
        buf_puts(&b, "[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":");
        json_escape(&b, text); buf_puts(&b, "}]}]");
        free(text); easy = buf_take(&b); raw = easy;
    }
    char **items = NULL;
    size_t n = 0;
    if (!ds4_conversation_array(raw, &items, &n)) { free(easy); goto fail; }
    chat_msgs msgs = {0};
    p = raw;
    bool ok = parse_responses_input(&p, &msgs, NULL, NULL, false);
    json_ws(&p);
    ok = ok && !*p;
    for (int i = 0; i < msgs.len; i++) if (msgs.v[i].images.len) ok = false;
    chat_msgs_free(&msgs);
    if (!ok) goto item_fail;
    for (size_t i = 0; i < n; i++) {
        char *type = responses_json_string_field(items[i], "type");
        char *id = responses_json_string_field(items[i], "id");
        char *role = responses_json_string_field(items[i], "role");
        if (!type) type = xstrdup("message");
        bool message = !strcmp(type, "message");
        bool supported = message || !strcmp(type, "function_call") ||
                         !strcmp(type, "function_call_output") || !strcmp(type, "reasoning");
        if (!supported ||
            (role && strcmp(role, "user") && strcmp(role, "assistant") &&
             strcmp(role, "system") && strcmp(role, "developer"))) {
            free(type); free(id); free(role); goto item_fail;
        }
        if (id && !ds4_conversation_item_id(id)) { free(type); free(id); free(role); goto item_fail; }
        if (!id) {
            char generated[48];
            responses_random_id(generated, sizeof(generated), message ? "msg_" :
                !strcmp(type, "function_call") ? "fc_" :
                !strcmp(type, "function_call_output") ? "fco_" : "rs_");
            id = xstrdup(generated);
        }
        for (size_t j = 0; j < i; j++) {
            char *prior = responses_json_string_field(items[j], "id");
            bool duplicate = prior && !strcmp(prior, id);
            free(prior);
            if (duplicate) { free(type); free(id); free(role); goto item_fail; }
        }
        buf b = {0};
        buf_puts(&b, "{\"id\":"); json_escape(&b, id);
        buf_puts(&b, ",\"type\":"); json_escape(&b, type);
        if (message) { buf_puts(&b, ",\"role\":"); json_escape(&b, role ? role : "user"); }
        char *status = ds4_conversation_json_field(items[i], "status");
        if (message && !status) buf_puts(&b, ",\"status\":\"completed\"");
        free(status);
        const char *q = items[i];
        json_ws(&q); q++;
        for (;;) {
            json_ws(&q);
            if (*q == '}') break;
            char *key = NULL, *value = NULL;
            if (!json_string(&q, &key)) { buf_free(&b); free(type); free(id); free(role); goto item_fail; }
            json_ws(&q); q++;
            if (!json_raw_value(&q, &value)) { free(key); buf_free(&b); free(type); free(id); free(role); goto item_fail; }
            if (strcmp(key, "id") && strcmp(key, "type") && !(message && !strcmp(key, "role"))) {
                buf_putc(&b, ','); json_escape(&b, key); buf_putc(&b, ':');
                if (message && !strcmp(key, "content") && *value == '"') {
                    const char *text_p = value;
                    char *text = NULL;
                    json_string(&text_p, &text);
                    buf_puts(&b, role && !strcmp(role, "assistant") ?
                        "[{\"type\":\"output_text\",\"annotations\":[],\"text\":" :
                        "[{\"type\":\"input_text\",\"text\":");
                    json_escape(&b, text ? text : ""); buf_puts(&b, "}]"); free(text);
                } else buf_puts(&b, value);
            }
            free(key); free(value);
            json_ws(&q); if (*q == ',') { q++; continue; } break;
        }
        buf_putc(&b, '}');
        free(type); free(id); free(role); free(items[i]); items[i] = buf_take(&b);
    }
    *normalized = ds4_conversation_array_json(items, n);
    ds4_conversation_array_free(items, n); free(easy);
    return true;
item_fail:
    ds4_conversation_array_free(items, n); free(easy);
fail:
    ds4_conversation_error(err, errlen, 400, "conversation items require supported text/function items");
    return false;
}

static char *ds4_conversation_concat(const char *a, const char *b, const char *c) {
    const char *arrays[] = {a, b, c};
    buf out = {0};
    char **ids = NULL;
    size_t id_count = 0;
    buf_putc(&out, '[');
    bool any = false;
    for (size_t k = 0; k < 3; k++) {
        char **items = NULL; size_t n = 0;
        if (!ds4_conversation_array(arrays[k] ? arrays[k] : "[]", &items, &n)) goto fail;
        for (size_t i = 0; i < n; i++) {
            char *id = responses_json_string_field(items[i], "id");
            bool duplicate = !id;
            for (size_t j = 0; id && j < id_count; j++)
                if (!strcmp(id, ids[j])) { duplicate = true; break; }
            if (duplicate) {
                free(id); ds4_conversation_array_free(items, n); goto fail;
            }
            ids = xrealloc(ids, (id_count + 1) * sizeof(*ids));
            ids[id_count++] = id;
            if (any) buf_putc(&out, ',');
            buf_puts(&out, items[i]); any = true;
        }
        ds4_conversation_array_free(items, n);
    }
    buf_putc(&out, ']');
    ds4_conversation_array_free(ids, id_count);
    return buf_take(&out);
fail:
    ds4_conversation_array_free(ids, id_count);
    buf_free(&out);
    return NULL;
}

static int ds4_conversation_publish(const char *root, const char *id,
                                    unsigned long long expected_version,
                                    const char *new_items, const char *output_items,
                                    const char *replay, char *err, size_t errlen) {
    char *input = NULL, *output = NULL;
    if (!ds4_conversation_input_items(new_items ? new_items : "[]", &input, err, errlen) ||
        !ds4_conversation_input_items(output_items ? output_items : "[]", &output, err, errlen)) {
        free(input); free(output);
        return ds4_conversation_error(err, errlen, 500, "invalid completed conversation publication");
    }
    pthread_mutex_lock(&ds4_conversation_mutex);
    ds4_conversation c = {0};
    int rc = ds4_conversation_load_unlocked(root, id, &c, err, errlen);
    if (!rc && c.deleted) rc = ds4_conversation_error(err, errlen, 404, "conversation was deleted");
    if (!rc && c.version != expected_version)
        rc = ds4_conversation_error(err, errlen, 409, "conversation changed during response generation");
    if (!rc && c.version == ULLONG_MAX)
        rc = ds4_conversation_error(err, errlen, 500, "conversation version exhausted");
    if (!rc) {
        char *joined = ds4_conversation_concat(c.items, input, output);
        if (!joined) rc = ds4_conversation_error(err, errlen, 500, "invalid conversation items");
        else {
            free(c.items); c.items = joined;
            free(c.replay); c.replay = replay ? xstrdup(replay) : NULL;
            c.version++;
            rc = ds4_conversation_write_unlocked(root, id, &c, err, errlen);
        }
    }
    ds4_conversation_free(&c);
    pthread_mutex_unlock(&ds4_conversation_mutex);
    free(input); free(output);
    return rc;
}

static char *ds4_conversation_list(char **items, size_t n, bool asc, size_t limit,
                                   const char *after, bool *cursor_found) {
    size_t start = 0;
    *cursor_found = !after;
    if (after) {
        for (size_t i = 0; i < n; i++) {
            size_t at = asc ? i : n - 1 - i;
            char *id = responses_json_string_field(items[at], "id");
            bool match = id && !strcmp(id, after); free(id);
            if (match) { start = i + 1; *cursor_found = true; break; }
        }
    }
    size_t take = start <= n ? n - start : 0;
    if (take > limit) take = limit;
    buf b = {0};
    buf_puts(&b, "{\"object\":\"list\",\"data\":[");
    for (size_t i = 0; i < take; i++) {
        if (i) buf_putc(&b, ',');
        buf_puts(&b, items[asc ? start + i : n - 1 - start - i]);
    }
    buf_puts(&b, "],\"first_id\":");
    char *first = take ? responses_json_string_field(items[asc ? start : n - 1 - start], "id") : NULL;
    char *last = take ? responses_json_string_field(items[asc ? start + take - 1 : n - start - take], "id") : NULL;
    if (first) json_escape(&b, first); else buf_puts(&b, "null");
    buf_puts(&b, ",\"last_id\":");
    if (last) json_escape(&b, last); else buf_puts(&b, "null");
    buf_printf(&b, ",\"has_more\":%s}", start + take < n ? "true" : "false");
    free(first); free(last);
    return buf_take(&b);
}

/* 0 means another route; otherwise the returned HTTP status is authoritative.
 * Deletion leaves a tombstone with its item bytes intact. */
static int ds4_conversation_route(const char *root, const char *method,
                                  const char *path, const char *body,
                                  char **public_json, char *err, size_t errlen) {
    *public_json = NULL;
    const char *route = path;
    if (!strncmp(route, "/v1/", 4)) route += 3;
    if (strncmp(route, "/conversations", 14) ||
        (route[14] && route[14] != '/' && route[14] != '?')) return 0;
    if (!root || !*root)
        return ds4_conversation_error(err, errlen, 500, "conversation store is not configured");
    char *owned = xstrdup(route + 14);
    char *query = strchr(owned, '?');
    if (query) *query++ = 0;
    bool asc = false;
    size_t limit = 20;
    char *after = NULL;
    if (query) {
        char *save = NULL;
        for (char *part = strtok_r(query, "&", &save); part; part = strtok_r(NULL, "&", &save)) {
            char *value = strchr(part, '=');
            if (!value) continue;
            *value++ = 0;
            if (!strcmp(part, "order")) {
                if (strcmp(value, "asc") && strcmp(value, "desc")) goto bad_query;
                asc = !strcmp(value, "asc");
            } else if (!strcmp(part, "limit")) {
                unsigned long long parsed = 0;
                if (!ds4_conversation_u64(value, &parsed) || parsed < 1 || parsed > 100) goto bad_query;
                limit = (size_t)parsed;
            } else if (!strcmp(part, "after")) {
                if (!ds4_conversation_item_id(value)) goto bad_query;
                after = value;
            }
        }
    }
    if (!*owned) {
        if (strcmp(method, "POST")) { free(owned); return 405; }
        const char *body_p = body ? body : "{}";
        json_ws(&body_p);
        if (*body_p != '{' || !json_skip_value(&body_p)) { free(owned); return 400; }
        json_ws(&body_p);
        if (*body_p) { free(owned); return 400; }
        char *raw_items = ds4_conversation_json_field(body ? body : "{}", "items");
        char *metadata = ds4_conversation_json_field(body ? body : "{}", "metadata");
        char *normalized = NULL;
        if (!ds4_conversation_input_items(raw_items ? raw_items : "[]", &normalized, err, errlen)) {
            free(raw_items); free(metadata); free(owned); return 400;
        }
        char **items = NULL; size_t n = 0;
        ds4_conversation_array(normalized, &items, &n);
        ds4_conversation_array_free(items, n);
        if (n > 20 || (metadata && *metadata != '{' && strcmp(metadata, "null"))) {
            free(raw_items); free(metadata); free(normalized); free(owned);
            return ds4_conversation_error(err, errlen, 400, "invalid initial items or metadata");
        }
        char id[40]; responses_random_id(id, sizeof(id), "conv_");
        buf b = {0};
        buf_puts(&b, "{\"id\":"); json_escape(&b, id);
        buf_printf(&b, ",\"object\":\"conversation\",\"created_at\":%lld,\"metadata\":", (long long)time(NULL));
        buf_puts(&b, metadata && strcmp(metadata, "null") ? metadata : "{}"); buf_putc(&b, '}');
        ds4_conversation c = {.object = buf_take(&b), .items = normalized, .version = 1};
        pthread_mutex_lock(&ds4_conversation_mutex);
        int rc = ds4_conversation_write_unlocked(root, id, &c, err, errlen);
        pthread_mutex_unlock(&ds4_conversation_mutex);
        if (!rc) *public_json = xstrdup(c.object);
        ds4_conversation_free(&c);
        free(raw_items); free(metadata); free(owned);
        return rc ? rc : 200;
    }
    if (*owned != '/') { free(owned); return 404; }
    char *id = owned + 1;
    char *tail = strchr(id, '/');
    if (tail) *tail++ = 0;
    if (!ds4_conversation_valid_id(id)) { free(owned); return 404; }
    char *item_id = NULL;
    if (tail) {
        if (strncmp(tail, "items", 5) || (tail[5] && tail[5] != '/')) { free(owned); return 404; }
        if (tail[5] == '/') {
            item_id = tail + 6;
            if (!ds4_conversation_item_id(item_id)) { free(owned); return 404; }
        }
    }
    pthread_mutex_lock(&ds4_conversation_mutex);
    ds4_conversation c = {0};
    int rc = ds4_conversation_load_unlocked(root, id, &c, err, errlen);
    char **items = NULL; size_t n = 0;
    if (rc) goto done;
    if (c.deleted && (!tail || strcmp(method, "GET"))) {
        rc = ds4_conversation_error(err, errlen, 404, "conversation was deleted"); goto done;
    }
    if (!tail) {
        if (!strcmp(method, "GET")) *public_json = xstrdup(c.object);
        else if (!strcmp(method, "DELETE")) {
            c.deleted = true; c.version++;
            rc = ds4_conversation_write_unlocked(root, id, &c, err, errlen);
            if (!rc) {
                buf b = {0}; buf_puts(&b, "{\"id\":"); json_escape(&b, id);
                buf_puts(&b, ",\"object\":\"conversation.deleted\",\"deleted\":true}");
                *public_json = buf_take(&b);
            }
        } else rc = 405;
        goto done;
    }
    if (!ds4_conversation_array(c.items, &items, &n)) { rc = 500; goto done; }
    if (item_id) {
        size_t found = n;
        for (size_t i = 0; i < n; i++) {
            char *candidate = responses_json_string_field(items[i], "id");
            bool match = candidate && !strcmp(candidate, item_id); free(candidate);
            if (match) { found = i; break; }
        }
        if (found == n) rc = 404;
        else if (!strcmp(method, "GET")) *public_json = xstrdup(items[found]);
        else if (!strcmp(method, "DELETE")) {
            free(items[found]);
            memmove(items + found, items + found + 1, (n - found - 1) * sizeof(*items)); n--;
            free(c.items); c.items = ds4_conversation_array_json(items, n);
            free(c.replay); c.replay = NULL; c.version++;
            rc = ds4_conversation_write_unlocked(root, id, &c, err, errlen);
            if (!rc) *public_json = xstrdup(c.object);
        } else rc = 405;
    } else if (!strcmp(method, "GET")) {
        bool cursor_found = false;
        *public_json = ds4_conversation_list(items, n, asc, limit, after, &cursor_found);
        if (!cursor_found) { free(*public_json); *public_json = NULL; rc = 404; }
    } else if (!strcmp(method, "POST")) {
        char *raw_items = ds4_conversation_json_field(body ? body : "{}", "items");
        char *normalized = NULL;
        if (!raw_items || !ds4_conversation_input_items(raw_items, &normalized, err, errlen)) rc = 400;
        else {
            char *joined = ds4_conversation_concat(c.items, normalized, NULL);
            char *validated = NULL;
            /* Duplicate public IDs would make retrieval/deletion ambiguous. */
            if (!joined || !ds4_conversation_input_items(joined, &validated, err, errlen)) rc = 400;
            else {
                free(c.items); c.items = validated;
                free(c.replay); c.replay = NULL; c.version++;
                rc = ds4_conversation_write_unlocked(root, id, &c, err, errlen);
                if (!rc) {
                    char **added = NULL; size_t added_n = 0;
                    ds4_conversation_array(normalized, &added, &added_n);
                    bool ignored;
                    *public_json = ds4_conversation_list(added, added_n, true, added_n, NULL, &ignored);
                    ds4_conversation_array_free(added, added_n);
                }
            }
            free(joined);
        }
        free(raw_items); free(normalized);
    } else rc = 405;
done:
    ds4_conversation_array_free(items, n);
    ds4_conversation_free(&c);
    pthread_mutex_unlock(&ds4_conversation_mutex);
    free(owned);
    return rc ? rc : 200;
bad_query:
    free(owned);
    return ds4_conversation_error(err, errlen, 400, "invalid conversation list query");
}

#endif /* DS4_NATIVE_CONVERSATIONS */

#endif
