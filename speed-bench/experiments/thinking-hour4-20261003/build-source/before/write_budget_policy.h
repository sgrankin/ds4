/* Private frontend-only protocol policy. No sampler, tensor or session API. */
#ifndef PT_WRITE_BUDGET_POLICY_H
#define PT_WRITE_BUDGET_POLICY_H
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

enum { PW_NATIVE = 0, PW_FORCE = 1, PW_REFUSE = -1 };
enum { PW_CLOSE_TOKEN = 128822, PW_LIMIT = 1024, PW_TOTAL_CAP = 2048, PW_CONTEXT = 8192 };
typedef struct {
    int limit, prompt_pos, last_pos, last_count, force_pos, force_count;
    bool armed, pending, eval_checked, committed;
} pw_policy;
typedef struct {
    int pos, checkpoint_len, count, global_remaining, round_remaining, context_remaining;
    bool inside, stream_inside, renderer_inside, parser_search;
    bool greedy, dsml_active, pending_text, delimiter_prefix, assistant_after_boundary;
} pw_view;

/* Default is literal 0; empty/whitespace/other spellings are refusals. */
static int pw_parse_limit(const char *text) {
    if (!text || !strcmp(text, "0")) return 0;
    if (!strcmp(text, "1024")) return PW_LIMIT;
    return -1;
}
static bool pw_arm(pw_policy *p, int limit, int prompt_pos) {
    if (!p || (limit != 0 && limit != PW_LIMIT) || prompt_pos < 1 ||
        prompt_pos > PW_CONTEXT) return false;
    memset(p,0,sizeof *p);
    p->limit=limit;p->prompt_pos=prompt_pos;p->last_pos=prompt_pos;p->armed=limit != 0;
    return true;
}

/* Literal native8a47 bodies, only the private function names differ. The
 * native ds4_test_* wrappers are compiled only with DS4_TEST_HOOKS; this
 * caller deliberately does not declare/call those unavailable native APIs. */
static bool pw_tokens_inside_thinking(const int *tokens, int n,
                                  int start_id, int end_id) {
    for (int i = n - 1; i >= 0; i--) {
        if (end_id >= 0 && tokens[i] == end_id) return false;
        if (start_id >= 0 && tokens[i] == start_id) return true;
    }
    return false;
}
static int pw_tokens_generated_thinking(const int *tokens, int n, int prompt_len,
                                     int assistant_id, int start_id, int end_id) {
    if (!tokens || n <= 0 || start_id < 0 || end_id < 0) return 0;
    int turn_start = 0;
    if (assistant_id >= 0) {
        for (int i = n - 1; i >= 0; i--) {
            if (tokens[i] == assistant_id) { turn_start = i + 1; break; }
        }
    }
    if (prompt_len < 0) prompt_len = 0;
    if (prompt_len > n) prompt_len = n;
    bool inside = false;
    int count = 0;
    for (int i = turn_start; i < n; i++) {
        if (tokens[i] == start_id) inside = true;
        else if (tokens[i] == end_id) inside = false;
        else if (inside && i >= prompt_len) count++;
    }
    return count;
}

/* Decision has no side effect. Unsafe eligible boundaries never become a
 * native fallback. The caller reserves/publishes the forced slot separately. */
static int pw_decide(const pw_policy *p, const pw_view *v) {
    if (!p || !v) return PW_REFUSE;
    if (!p->limit) return PW_NATIVE;
    if (p->limit != PW_LIMIT) return PW_REFUSE;
    if (!p->armed || p->pending || v->pos != v->checkpoint_len ||
        v->pos < p->last_pos || v->pos < p->prompt_pos || v->pos > PW_CONTEXT ||
        v->count < p->last_count || v->count > v->pos-p->prompt_pos ||
        v->assistant_after_boundary) return PW_REFUSE;
    if (p->committed && (v->inside || v->stream_inside || v->renderer_inside)) return PW_REFUSE;
    if (!v->inside || v->count < p->limit) return PW_NATIVE;
    if (!v->stream_inside || !v->renderer_inside || !v->parser_search ||
        v->greedy || v->dsml_active || v->pending_text || v->delimiter_prefix ||
        v->global_remaining < 2 || v->round_remaining < 2 ||
        v->context_remaining < 2) return PW_REFUSE;
    return PW_FORCE;
}
static bool pw_observe(pw_policy *p, const pw_view *v) {
    if (pw_decide(p,v) == PW_REFUSE) return false;
    p->last_pos=v->pos; p->last_count=v->count;
    return true;
}
static bool pw_reserve_force(pw_policy *p, const pw_view *v) {
    if (pw_decide(p,v) != PW_FORCE) return false;
    p->last_pos=v->pos; p->last_count=v->count;
    p->force_pos=v->pos; p->force_count=v->count;
    p->pending=true; p->eval_checked=false;
    return true;
}
static bool pw_after_eval(pw_policy *p, int pos, int stored_token, bool inside) {
    if (!p || !p->pending || p->eval_checked || p->committed ||
        pos != p->force_pos+1 || stored_token != PW_CLOSE_TOKEN || inside) return false;
    p->eval_checked=true;
    return true;
}
static bool pw_after_stream(pw_policy *p, int pos, int stored_token, int transcript_token,
                            bool inside, bool stream_inside, bool renderer_inside,
                            bool parser_search, bool dsml_active, bool pending_text,
                            bool delimiter_prefix) {
    if (!p || !p->pending || !p->eval_checked || p->committed ||
        pos != p->force_pos+1 || stored_token != PW_CLOSE_TOKEN ||
        transcript_token != PW_CLOSE_TOKEN || inside || stream_inside || renderer_inside ||
        !parser_search || dsml_active || pending_text || delimiter_prefix) return false;
    p->pending=false; p->committed=true; p->last_pos=pos;
    return true;
}
#endif
