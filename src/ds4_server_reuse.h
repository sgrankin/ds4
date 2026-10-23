#ifndef DS4_SERVER_REUSE_H
#define DS4_SERVER_REUSE_H

#include "ds4_server_runtime.h"

/* Actual routing/execution result, extracted once from the server. Both paths
 * consume the same tier decision; no session state or address is copied. */
typedef enum {
    REUSE_NONE = 0,
    REUSE_RESPONSES_VISIBLE,
    REUSE_RESPONSES_TOOL_OUTPUT,
    REUSE_ANTHROPIC_TOOL_OUTPUT,
    REUSE_MEMORY_REWIND,
    REUSE_MEMORY_TOKEN,
    REUSE_THINKING_VISIBLE,
    REUSE_MEMORY_TEXT,
} slot_reuse_kind;

typedef struct {
    slot_reuse_kind kind;
    int reuse_tokens;   /* live tokens reusable; 0 iff REUSE_NONE */
    size_t suffix_off;  /* text tiers: byte offset of the new suffix text */
    int matched_ids;    /* tool-output tiers: number of bound call ids */
} slot_reuse;

char *ds4_server_reuse_visible_key(const request *req, const char *text,
    visible_image_key *images);
bool ds4_server_reuse_images_match(const visible_image_key *incoming,
    const visible_image_key *old, size_t boundary);
void ds4_server_reuse_profile(const server *runtime, const request *req,
    char profile[41]);
void ds4_server_reuse_tool_clear(live_tool_state *state);
void ds4_server_reuse_tool_free(live_tool_state *state);
void ds4_server_reuse_visible_clear(visible_live_state *state);
void ds4_server_reuse_visible_free(visible_live_state *state);
/* API_OPENAI selects the tool-less thinking-visible record in these two
 * whole lifecycle actions; Responses/Anthropic select their actual records. */
void ds4_server_reuse_clear(server *runtime, server_slot *slot, api_style api);
void ds4_server_reuse_remember(server *runtime, server_slot *slot, api_style api,
    const char *visible, const tool_calls *calls, const request *req);
bool ds4_server_reuse_matches_locked(server_slot *slot, api_style api,
    const stop_list *ids, int live_tokens);
bool ds4_server_reuse_matches(server *runtime, server_slot *slot, api_style api,
    const stop_list *ids, int live_tokens);
bool ds4_server_reuse_canonical(const server *runtime, const request *req);
slot_reuse ds4_server_reuse_probe_locked(server *runtime, server_slot *slot,
    const request *req);
slot_reuse ds4_server_reuse_probe(server *runtime, server_slot *slot,
    const request *req);
bool ds4_server_reuse_suffix(server *runtime, server_slot *slot,
    const request *req, const char *suffix, ds4_tokens *out);
bool ds4_server_reuse_unavailable(const request *req, bool materialized);
void ds4_server_reuse_refresh(server *runtime, server_slot *slot);
int ds4_server_reuse_rewind_target(bool can_rewind, int old_pos,
    int prompt_len, int common);

#endif
