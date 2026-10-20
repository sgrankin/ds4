/* Private native-only helper. No policy/logit mutation or extra model forwards. */
#include "native_effort_final_score.h"

#define HOLDOUT_CAP 2048
#define HOLDOUT_SEED UINT64_C(20261002)

static int level_value(const char *text, int *level) {
    if (!strcmp(text,"0")) {*level=0;return 1;}
    if (!strcmp(text,"25")) {*level=25;return 1;}
    if (!strcmp(text,"75")) {*level=75;return 1;}
    return 0;
}

static int self_test(void) {
    if (score_self_test()) return 1;
    int value=-1;
    const char *good[]={"0","25","75"};
    const char *bad[]={"","00","025","-1","1","50","100","nan","25x"};
    for (size_t i=0;i<sizeof(good)/sizeof(*good);i++) if (!level_value(good[i],&value)) return 1;
    for (size_t i=0;i<sizeof(bad)/sizeof(*bad);i++) if (level_value(bad[i],&value)) return 1;
    if (ds4_think_mode_enabled((ds4_think_mode)DS4_THINK_LEVEL_BASE) ||
        !ds4_think_mode_enabled((ds4_think_mode)(DS4_THINK_LEVEL_BASE+25)) ||
        !ds4_think_mode_enabled((ds4_think_mode)(DS4_THINK_LEVEL_BASE+75)) ||
        ds4_deepseek41_reasoning_effort_text((ds4_think_mode)DS4_THINK_LEVEL_BASE)!=NULL) return 1;
    puts("native holdout levels0/25/75 and no-thinking protocol: pass");
    return 0;
}

int main(int argc,char **argv) {
    if (argc==2 && !strcmp(argv[1],"--helper-self-test")) return self_test();
    int level=0;
    if (argc!=4 || !level_value(argv[3],&level) || !argv[2][0] ||
        strspn(argv[2],"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=strlen(argv[2])) {
        fputs("usage: native-effort-holdout MODEL.gguf FRESH_RUN_NAME 0|25|75\n",stderr);return 2;
    }
    char output[4096],path[4096];
    if (snprintf(output,sizeof(output),"speed-bench/ideas/artifacts/%s",argv[2]) >= (int)sizeof(output)-128) return 2;
    if (mkdir("speed-bench/ideas",0755)!=0 && errno!=EEXIST) return 2;
    if (mkdir("speed-bench/ideas/artifacts",0755)!=0 && errno!=EEXIST) return 2;
    if (mkdir(output,0755)!=0) {perror("fresh holdout output");return 2;}
    snprintf(path,sizeof(path),"%s/results.jsonl",output);
    FILE *results=fopen(path,"wx");if (!results) return 2;
    ds4_engine_options opt={0};
    opt.model_path=argv[1];opt.backend=DS4_BACKEND_METAL;opt.n_threads=2;
    opt.context_size=8192;opt.prefill_chunk=512;opt.power_percent=100;
    opt.ssd_streaming=true;opt.ssd_streaming_cache_bytes=UINT64_C(64)*1024*1024*1024;
    opt.mtp_draft_tokens=1;
    ds4_engine *engine=NULL;ds4_session *session=NULL;
    if (ds4_engine_open(&engine,&opt)!=0 || ds4_session_create(&session,engine,opt.context_size)!=0) return 1;
    if (!ds4_engine_set_overthinking_penalty(engine,0.0f,true)) return 1;
    ds4_session_gpu_warmup(session);
    const ds4_think_mode mode=(ds4_think_mode)(DS4_THINK_LEVEL_BASE+level);
    const int thinking=ds4_think_mode_enabled(mode);
    const struct {const char *id,*prompt,*answer;answer_kind kind;} cases[]={
#include "native_effort_holdout_cases.h"
    };
    int invalid=0;char err[256]={0};
    for (size_t ci=0;ci<sizeof(cases)/sizeof(*cases);ci++) {
        ds4_tokens prompt={0};ds4_chat_begin(engine,&prompt);
        ds4_chat_append_think_prefix(engine,&prompt,mode);
        ds4_chat_append_message(engine,&prompt,"user",cases[ci].prompt);
        ds4_chat_append_assistant_prefix(engine,&prompt,mode);
        snprintf(path,sizeof(path),"%s/%s.prompt.ids",output,cases[ci].id);
        FILE *ids=fopen(path,"wx");if (!ids) return 1;
        for (int i=0;i<prompt.len;i++) fprintf(ids,"%d\n",prompt.v[i]);
        if (fclose(ids)!=0) return 1;
        const int prompt_tokens=prompt.len;
        const double p0=seconds();
        if (ds4_session_sync(session,&prompt,err,sizeof(err))!=0) {fprintf(stderr,"sync: %s\n",err);return 1;}
        const double prefill=seconds()-p0;ds4_tokens_free(&prompt);
        snprintf(path,sizeof(path),"%s/%s.tokens.jsonl",output,cases[ci].id);
        FILE *tokens=fopen(path,"wx");if (!tokens) return 1;
        char *text=calloc((size_t)HOLDOUT_CAP*512+1,1);if (!text) return 1;
        size_t used=0,final_offset=0;
        int generated=0,reasoning=0,closed=!thinking,generated_close=0,stopped=0;
        uint64_t rng=HOLDOUT_SEED;const double d0=seconds();
        while (generated<HOLDOUT_CAP) {
            ds4_overthinking_info info={0};
            if (!ds4_session_overthinking_info(session,&info) || info.penalty!=0 || info.active ||
                (!thinking && info.inside_thinking)) {fputs("native policy/state mismatch\n",stderr);return 1;}
            const uint64_t before=rng;
            const int token=ds4_session_sample(session,PILOT_TEMPERATURE,0,0.95f,0.0f,&rng);
            if (token<0) return 1;
            const int stop=ds4_token_is_stop_for_think_mode(engine,token,mode);
            size_t n=0;char *piece=ds4_token_text(engine,token,&n);if (!piece) return 1;
            fprintf(tokens,"{\"step\":%d,\"token_id\":%d,\"stop\":%s,\"inside_thinking\":%s,\"generated_thinking_tokens\":%d,\"policy_active\":false,\"penalty\":0,\"rng_before\":\"%016llx\",\"rng_after\":\"%016llx\",\"piece_hex\":\"",
                generated,token,stop?"true":"false",info.inside_thinking?"true":"false",info.generated_thinking_tokens,
                (unsigned long long)before,(unsigned long long)rng);
            for (size_t i=0;i<n;i++) fprintf(tokens,"%02x",(unsigned char)piece[i]);
            fputs("\"}\n",tokens);
            if (stop) {free(piece);stopped=1;break;}
            if (used+n>=(size_t)HOLDOUT_CAP*512) return 1;
            memcpy(text+used,piece,n);used+=n;
            if (!closed) reasoning++;
            if (n==8 && !memcmp(piece,"</think>",8)) {closed=1;generated_close=1;final_offset=used;}
            free(piece);generated++;
            if (generated<HOLDOUT_CAP && ds4_session_eval(session,token,err,sizeof(err))!=0) {fprintf(stderr,"eval: %s\n",err);return 1;}
        }
        const double decode=seconds()-d0;if (fclose(tokens)!=0) return 1;
        snprintf(path,sizeof(path),"%s/%s.txt",output,cases[ci].id);
        FILE *raw=fopen(path,"wx");if (!raw) return 1;
        if (fwrite(text,1,used,raw)!=used || fclose(raw)!=0) return 1;
        char *copy=strdup(closed?text+final_offset:"");if (!copy) return 1;
        char parsed[1024]={0};
        const int parse_ok=closed&&parse_answer(copy,cases[ci].kind,parsed,sizeof(parsed));
        const int correct=parse_ok&&exact_answer(parsed,cases[ci].answer);
        const char *status=!parse_ok?"INVALID":correct?"CORRECT":"INCORRECT";
        const char *reason=parse_ok?"none":!closed?"unclosed_reasoning":!*trim(copy)?"empty_final":"unsupported_final_format";
        invalid+=!parse_ok;
        fputs("{\"case\":",results);json_string(results,cases[ci].id);
        fprintf(results,",\"suite\":\"independent-counting-v1\",\"think_level\":%d,\"native_penalty\":0,\"think_enabled\":%s,\"prompt_supplied_think_close\":%s,\"generated_think_close\":%s,\"status\":\"%s\",\"parse_ok\":%s,\"parse_reason\":\"%s\",\"scorer\":\"strict-final-v2\",\"seed\":20261002,\"temperature_float_value\":%.17g,\"top_p_float_value\":%.17g,\"min_p\":0,\"max_tokens\":2048,\"cache_gib_requested\":64,\"context_size\":8192,\"prefill_chunk\":512,\"prompt_tokens\":%d,\"generated_tokens\":%d,\"reasoning_tokens\":%d,\"think_closed\":%s,\"stopped\":%s,\"cap_hit\":%s,\"correct_exact\":%s,\"prefill_seconds\":%.6f,\"decode_seconds\":%.6f,\"parsed_answer\":",
            level,thinking?"true":"false",thinking?"false":"true",generated_close?"true":"false",status,
            parse_ok?"true":"false",reason,(double)PILOT_TEMPERATURE,(double)0.95f,prompt_tokens,generated,reasoning,
            closed?"true":"false",stopped?"true":"false",generated>=HOLDOUT_CAP?"true":"false",correct?"true":"false",prefill,decode);
        json_string(results,parsed);fputs(",\"final\":",results);json_string(results,closed?text+final_offset:"");
        fputs(",\"expected\":",results);json_string(results,cases[ci].answer);
        fputs(",\"prompt\":",results);json_string(results,cases[ci].prompt);
        fprintf(results,",\"raw_trace\":\"%s.txt\",\"token_trace\":\"%s.tokens.jsonl\",\"prompt_trace\":\"%s.prompt.ids\"}\n",cases[ci].id,cases[ci].id,cases[ci].id);
        if (fflush(results)!=0) return 1;
        fprintf(stderr,"native_holdout effort=%d case=%s status=%s generated=%d stopped=%d cap=%d\n",level,cases[ci].id,status,generated,stopped,generated>=HOLDOUT_CAP);
        free(copy);free(text);
    }
    if (fclose(results)!=0) return 1;
    ds4_session_free(session);ds4_engine_close(engine);
    return invalid?3:0;
}
