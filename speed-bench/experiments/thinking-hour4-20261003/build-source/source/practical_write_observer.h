/* Included after the unchanged practical_observer.h in the owning worker. */
#include "write_budget_policy.h"
static pw_policy pt_w_policy;
static FILE *pt_w_events;
static int pt_w_limit, pt_w_start=-1, pt_w_end=-1, pt_w_assistant=-1;
static int pt_w_native_calls, pt_w_forced_steps;

static void pt_w_die(const char *why, int rc) {
    if (pt_w_events) {
        fprintf(pt_w_events,"{\"type\":\"refusal\",\"reason\":\"%s\",\"stage\":%d,\"decisions\":%d,\"native_sampler_calls\":%d,\"forced_steps\":%d}\n",
                why,pt_stage,pt_steps,pt_w_native_calls,pt_w_forced_steps);
        fflush(pt_w_events);
    }
    pt_die(why,rc);
}
static void pt_write_init(void) {
    pt_w_limit=pw_parse_limit(getenv("PT_WRITE_REASONING_BUDGET"));
    if (pt_w_limit < 0) pt_die("write-budget-must-be-0-to-2048",1);
    pt_w_events=pt_file("write-budget.jsonl");
    fprintf(pt_w_events,"{\"type\":\"init\",\"policy_type\":\"protocol-boundary-token\",\"write_budget\":%d,\"global_decision_cap\":2048,\"read_final_native\":true}\n",pt_w_limit);
    fflush(pt_w_events);
}
static void pt_write_pre_sync(agent_worker *w,const agent_config *cfg) {
    /* Qualification is scalar for all brackets. No draft forwards/RNG can
     * bypass the full raw/per-decision observer. This only queries metadata. */
    if ((ds4_engine_mtp_draft_tokens(w->engine)>1 && getenv("DS4_MTP_SPEC_DISABLE")==NULL) ||
        cfg->edit_upto || w->image_count) pt_w_die("unsupported-generation-path",1);
}
static int pt_w_control(agent_worker *w,const char *text,int required) {
    ds4_tokens v={0}; ds4_tokenize_rendered_chat(w->engine,text,&v);
    if (v.len != 1) { ds4_tokens_free(&v); pt_w_die("thinking-control-not-one-token",1); }
    int token=v.v[0]; ds4_tokens_free(&v);
    size_t n=0; char *piece=ds4_token_text(w->engine,token,&n);
    bool valid=token>=0 && token<129280 && (required<0 || token==required) &&
        piece && n==strlen(text) && !memcmp(piece,text,n) &&
        ds4_token_is_thinking_control(w->engine,token);
    free(piece);
    if (!valid) pt_w_die("thinking-control-identity",1);
    return token;
}
static void pt_write_after_sync(agent_worker *w,const agent_config *cfg,int round,
                                 const ds4_tokens *prompt) {
    if (!pt_w_limit || pt_stage != 1) return;
    if (round != 1 || pt_w_policy.armed || !ds4_engine_is_deepseek41(w->engine) ||
        !ds4_think_mode_enabled(effective_think_mode(cfg))) pt_w_die("write-phase-binding",1);
    const ds4_tokens *checkpoint=ds4_session_tokens(w->session);
    int pos=ds4_session_pos(w->session);
    if (!ds4_session_checkpoint_valid(w->session) || !checkpoint || !prompt || pos<1 || pos!=checkpoint->len || pos!=prompt->len ||
        memcmp(checkpoint->v,prompt->v,(size_t)pos*sizeof(int))) pt_w_die("post-sync-checkpoint",1);
    pt_w_start=pt_w_control(w,"<think>",-1);
    pt_w_end=pt_w_control(w,"</think>",PW_CLOSE_TOKEN);
    pt_w_assistant=ds4_token_assistant(w->engine);
    if (pt_w_start==pt_w_end || pt_w_assistant<0 || pt_w_assistant>=129280 ||
        ds4_token_is_stop_for_think_mode(w->engine,pt_w_end,effective_think_mode(cfg)) ||
        !pw_tokens_inside_thinking(checkpoint->v,pos,pt_w_start,pt_w_end) ||
        !pw_arm(&pt_w_policy,pt_w_limit,pos)) pt_w_die("write-sync-protocol",1);
    fprintf(pt_w_events,"{\"type\":\"armed-after-successful-sync\",\"round\":%d,\"prompt_pos\":%d,\"count\":0,\"start_id\":%d,\"end_id\":%d,\"assistant_id\":%d}\n",round,pos,pt_w_start,pt_w_end,pt_w_assistant);
    fflush(pt_w_events);
}
static pw_view pt_w_view(agent_worker *w,agent_stream_renderer *stream,bool greedy,
                           int round_remaining,int context_remaining) {
    const ds4_tokens *checkpoint=ds4_session_tokens(w->session);
    int pos=ds4_session_pos(w->session);
    if (!ds4_session_checkpoint_valid(w->session) || !checkpoint || pos<pt_w_policy.prompt_pos || pos!=checkpoint->len ||
        pos>8192 || (pos && !checkpoint->v)) pt_w_die("write-checkpoint-extents",1);
    bool assistant_after=false;
    for (int i=pt_w_policy.prompt_pos;i<pos;i++)
        if (checkpoint->v[i]==pt_w_assistant) assistant_after=true;
    return (pw_view){
        .pos=pos, .checkpoint_len=checkpoint->len,
        .count=pw_tokens_generated_thinking(checkpoint->v,pos,pt_w_policy.prompt_pos,
                                           pt_w_assistant,pt_w_start,pt_w_end),
        .global_remaining=PW_TOTAL_CAP-pt_steps, .round_remaining=round_remaining,
        .context_remaining=context_remaining,
        .inside=pw_tokens_inside_thinking(checkpoint->v,pos,pt_w_start,pt_w_end),
        .stream_inside=stream->in_think, .renderer_inside=stream->renderer->in_think,
        .parser_search=stream->parser->state==AGENT_DSML_SEARCH,
        .greedy=greedy, .dsml_active=stream->dsml_active,
        .pending_text=stream->pending_len != 0,
        .delimiter_prefix=stream->dsml_start_len != 0,
        .assistant_after_boundary=assistant_after,
    };
}
static int pt_write_sample(agent_worker *w,const agent_config *cfg,bool greedy,uint64_t *rng,
                             int round,agent_stream_renderer *stream,
                             int round_remaining,int context_remaining) {
    if (pt_w_policy.pending) pt_w_die("sample-before-force-commit",1);
    if (!pt_w_limit || pt_stage != 1) {
        int token=pt_sample(w,cfg,greedy,rng,round,stream->in_think,(int)stream->parser->state);
        pt_w_native_calls++; return token;
    }
    if (round != 1 || !pt_w_policy.armed) pt_w_die("write-sample-binding",1);
    pw_view view=pt_w_view(w,stream,greedy,round_remaining,context_remaining);
    int decision=pw_decide(&pt_w_policy,&view);
    if (decision==PW_REFUSE) pt_w_die("unsafe-write-budget-boundary",3);
    if (decision==PW_NATIVE) {
        if (!pw_observe(&pt_w_policy,&view)) pt_w_die("write-counter-regression",1);
        int index=pt_steps;
        int token=pt_sample(w,cfg,greedy,rng,round,stream->in_think,(int)stream->parser->state);
        pt_w_native_calls++;
        fprintf(pt_w_events,"{\"type\":\"decision\",\"intervention\":\"native-sample\",\"index\":%d,\"round\":%d,\"pos\":%d,\"count_before\":%d,\"inside_before\":%s}\n",index,round,view.pos,view.count,view.inside?"true":"false");
        fflush(pt_w_events); return token;
    }
    if (worker_should_interrupt(w)) pt_w_die("interrupted-before-write-force",3);
    /* Read the current actual native raw vector BEFORE reserving/publishing
     * the protocol token. The sampler is never called on this path. */
    int n=ds4_engine_vocab_size(w->engine);
    uint32_t slots=ds4_gpu_stream_expert_cache_configured_count();
    if (n!=129280 || slots!=6136 || pt_w_end!=PW_CLOSE_TOKEN ||
        ds4_token_is_stop_for_think_mode(w->engine,pt_w_end,effective_think_mode(cfg)))
        pt_w_die("force-native-protocol-gate",1);
    size_t length=0; char *piece=ds4_token_text(w->engine,pt_w_end,&length);
    if (!piece || length!=8 || memcmp(piece,"</think>",8)) {
        free(piece); pt_w_die("force-piece-identity",1);
    }
    if(!pt_light_audit) {
        float *raw=xmalloc((size_t)n*sizeof(float));
        if (ds4_session_copy_logits(w->session,raw,n)!=n) pt_w_die("force-raw-copy",1);
        if (fwrite(raw,sizeof(float),n,pt_raw)!=(size_t)n) pt_w_die("force-raw-write",1);
        free(raw);
    }
    uint64_t before=*rng;
    if (worker_should_interrupt(w)) { free(piece); pt_w_die("interrupted-before-force-reservation",3); }
    if (pt_w_forced_steps || !pw_reserve_force(&pt_w_policy,&view)) pt_w_die("force-reservation",1);
    int index=pt_steps++; pt_w_forced_steps++;
    /* The forced decision is already charged before the ordinary ONE eval. */
    fprintf(pt_events,"{\"type\":\"sample\",\"index\":%d,\"round\":%d,\"pos\":%d,\"configured_slots\":%u,\"rng_before\":\"%016llx\",\"rng_after\":\"%016llx\",\"greedy\":false,\"inside\":true,\"parser\":%d,\"token\":%d,\"stop\":false,\"piece_hex\":\"",
            index,round,view.pos,slots,(unsigned long long)before,(unsigned long long)*rng,(int)stream->parser->state,pt_w_end);
    pt_hex(pt_events,piece,length); fputs("\"}\n",pt_events); fflush(pt_events); free(piece);
    fprintf(pt_w_events,"{\"type\":\"decision\",\"policy_type\":\"protocol-boundary-token\",\"intervention\":\"forced-thinking-close\",\"index\":%d,\"round\":%d,\"pos\":%d,\"count_before\":%d,\"inside_before\":true,\"token\":128822,\"piece_hex\":\"3c2f7468696e6b3e\",\"stop\":false,\"rng_before\":\"%016llx\",\"rng_after\":\"%016llx\",\"global_remaining_before\":%d,\"round_remaining_before\":%d,\"context_remaining_before\":%d}\n",
            index,round,view.pos,view.count,(unsigned long long)before,(unsigned long long)*rng,view.global_remaining,view.round_remaining,view.context_remaining);
    fflush(pt_w_events); return pt_w_end;
}
static void pt_write_after_eval(agent_worker *w,int token) {
    if (!pt_w_policy.pending) return;
    const ds4_tokens *checkpoint=ds4_session_tokens(w->session);
    int pos=ds4_session_pos(w->session);
    if (!ds4_session_checkpoint_valid(w->session) || !checkpoint || pos<1 || pos!=checkpoint->len || token!=pt_w_end ||
        !pw_after_eval(&pt_w_policy,pos,checkpoint->v[pos-1],
                      pw_tokens_inside_thinking(checkpoint->v,pos,pt_w_start,pt_w_end)))
        pt_w_die("forced-evaluation-postcondition",1);
    fprintf(pt_w_events,"{\"type\":\"after-force-eval\",\"pos\":%d,\"stored_last_token\":%d,\"inside\":false,\"eval_delta\":1}\n",pos,checkpoint->v[pos-1]);
    fflush(pt_w_events);
}
static void pt_write_after_finish(agent_worker *w,int token,agent_stream_renderer *stream) {
    if (!pt_w_policy.pending) return;
    const ds4_tokens *checkpoint=ds4_session_tokens(w->session);
    int pos=ds4_session_pos(w->session);
    if (!ds4_session_checkpoint_valid(w->session) || !checkpoint || pos<1 || pos!=checkpoint->len || token!=pt_w_end ||
        w->transcript.len!=pos || memcmp(checkpoint->v,w->transcript.v,(size_t)pos*sizeof(int)) ||
        stream->dsml_in_think || stream->tool_preflight_error ||
        !pw_after_stream(&pt_w_policy,pos,checkpoint->v[pos-1],w->transcript.v[pos-1],
                         pw_tokens_inside_thinking(checkpoint->v,pos,pt_w_start,pt_w_end),
                         stream->in_think,stream->renderer->in_think,
                         stream->parser->state==AGENT_DSML_SEARCH,stream->dsml_active,
                         stream->pending_len!=0,stream->dsml_start_len!=0))
        pt_w_die("forced-renderer-parser-postcondition",1);
    fprintf(pt_w_events,"{\"type\":\"committed-force\",\"pos\":%d,\"session_transcript_equal\":true,\"stream_inside\":false,\"renderer_inside\":false,\"parser_search\":true}\n",pos);
    fflush(pt_w_events);
}
static int pt_write_finish(int legacy_rc) {
    int rc=legacy_rc;
    if (pt_w_policy.pending || pt_steps!=pt_w_native_calls+pt_w_forced_steps ||
        (pt_w_forced_steps && !pt_w_policy.committed)) rc=rc?rc:1;
    fprintf(pt_w_events,"{\"type\":\"finish\",\"legacy_rc\":%d,\"returncode\":%d,\"decisions\":%d,\"native_sampler_calls\":%d,\"forced_steps\":%d,\"force_committed\":%s,\"force_pending\":%s,\"extra_forwards\":0,\"forced_rng_draws\":0}\n",
            legacy_rc,rc,pt_steps,pt_w_native_calls,pt_w_forced_steps,pt_w_policy.committed?"true":"false",pt_w_policy.pending?"true":"false");
    if (fflush(pt_w_events) || fclose(pt_w_events)) rc=rc?rc:1;
    return rc;
}
