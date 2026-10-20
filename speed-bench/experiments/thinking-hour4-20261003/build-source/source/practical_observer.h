/* Private, uncompiled frontend observer. No extra forward or RNG draw. */
#include <math.h>
#include "source/ds4_gpu.h"
static FILE *pt_events, *pt_raw;
static bool pt_light_audit;
static char pt_dir[PATH_MAX];
static char *pt_config;
static size_t pt_config_len;
static int pt_stage, pt_steps, pt_rounds, pt_final_stop;
static void pt_hex(FILE *f, const char *s, size_t n) {
    static const char h[]="0123456789abcdef";
    for(size_t i=0;i<n;i++) { unsigned char c=(unsigned char)s[i]; fputc(h[c>>4],f);fputc(h[c&15],f); }
}
static void pt_die(const char *why, int rc) {
    if(pt_events) { fprintf(pt_events,"{\"type\":\"refusal\",\"reason\":\"%s\",\"stage\":%d,\"steps\":%d}\n",why,pt_stage,pt_steps); fflush(pt_events); }
    if(pt_raw)fflush(pt_raw);
    exit(rc);
}
static FILE *pt_file(const char *name) {
    char path[PATH_MAX];int n=snprintf(path,sizeof(path),"%s/%s",pt_dir,name);
    if(n<0 || (size_t)n>=sizeof(path))pt_die("output-path",1);
    int fd=open(path,O_CREAT|O_EXCL|O_WRONLY,0600);
    if(fd<0)pt_die("exclusive-output",1);
    FILE *f=fdopen(fd,"wb");if(!f) {close(fd);pt_die("fdopen",1);}return f;
}
static void pt_init(void) {
    const char *dir=getenv("PT_AUDIT_OUT");
    if(!dir || !dir[0] || strlen(dir)>=sizeof(pt_dir))exit(1);
    strcpy(pt_dir,dir);pt_events=pt_file("events.jsonl");pt_raw=pt_file("raw.fp32");
    pt_light_audit=getenv("PT_LIGHT_AUDIT") && !strcmp(getenv("PT_LIGHT_AUDIT"),"1");
    FILE *f=fopen("config.json","rb");if(!f)pt_die("config-open",1);
    pt_config=xmalloc(16385);pt_config_len=fread(pt_config,1,16385,f);
    if(ferror(f)||!feof(f)||pt_config_len>16384)pt_die("config-bound",1);
    fclose(f);pt_config[pt_config_len]=0;
    fprintf(pt_events,"{\"type\":\"init\",\"cap\":2048,\"max_rounds\":3,\"extra_forwards\":0,\"extra_rng_draws\":0}\n");fflush(pt_events);
}
static void pt_calls_before(const agent_tool_calls *calls) {
    if(!calls || calls->len!=1)pt_die("one-call-per-round",3);
    const agent_tool_call *c=&calls->v[0];
    const char *name=pt_stage==0?"read":pt_stage==1?"write":NULL;
    if(!name || !c->name || strcmp(c->name,name))pt_die("tool-name-order",3);
    int need=pt_stage==0?3:2;
    if(c->argc!=need)pt_die("argument-count",3);
    for(int i=0;i<c->argc;i++) {
        if(!c->args[i].name || !c->args[i].value)pt_die("missing-argument",3);
        for(int j=0;j<i;j++)if(!strcmp(c->args[i].name,c->args[j].name))pt_die("duplicate-argument",3);
        bool allowed=!strcmp(c->args[i].name,"path") || (pt_stage==0 && (!strcmp(c->args[i].name,"whole")||!strcmp(c->args[i].name,"raw"))) || (pt_stage==1 && !strcmp(c->args[i].name,"content"));
        if(!allowed)pt_die("unknown-argument",3);
        bool string=!strcmp(c->args[i].name,"path")||!strcmp(c->args[i].name,"content");
        if(c->args[i].is_string!=string)pt_die("argument-type",3);
    }
    const char *path=agent_tool_arg_value(c,"path");
    if(!path || strcmp(path,pt_stage==0?"config.json":"solution.py"))pt_die("path-allowlist",3);
    struct stat st;
    if(pt_stage==0) {
        if(lstat(path,&st)||!S_ISREG(st.st_mode)||S_ISLNK(st.st_mode))pt_die("input-not-regular",3);
        if(strcmp(agent_tool_arg_value(c,"whole"),"true") || strcmp(agent_tool_arg_value(c,"raw"),"true"))pt_die("read-entire-raw-config",3);
    } else {
        if(lstat(path,&st)==0 || errno!=ENOENT)pt_die("first-write-only",3);
        const char *code=agent_tool_arg_value(c,"content");size_t n=strlen(code);
        if(n>65536)pt_die("code-bound",3);
        FILE *snap=pt_file("first-solution.py");if(fwrite(code,1,n,snap)!=n || fclose(snap))pt_die("snapshot-write",1);
    }
    fprintf(pt_events,"{\"type\":\"tool-before\",\"stage\":%d,\"name\":\"%s\",\"path\":\"%s\"}\n",pt_stage,name,path);fflush(pt_events);
}
static void pt_tool_after(const agent_tool_call *c,const char *result) {
    fprintf(pt_events,"{\"type\":\"tool-after\",\"stage\":%d,\"result_hex\":\"",pt_stage);pt_hex(pt_events,result,strlen(result));fputs("\"}\n",pt_events);fflush(pt_events);
    if(pt_stage==0 && (strlen(result)!=pt_config_len || memcmp(result,pt_config,pt_config_len)))pt_die("read-result-identity",3);
    if(pt_stage==1 && strncmp(result,"Wrote ",6))pt_die("write-failed",3);
    (void)c;pt_stage++;
}
static void pt_prompt(agent_worker *w,int round,const ds4_tokens *tokens) {
    (void)w;
    if(round!=pt_rounds || round>=3)pt_die("round-limit-or-order",3);
    char name[80];snprintf(name,sizeof(name),"round-%d.ids.i32",round);FILE *f=pt_file(name);
    if(tokens->len<1 || tokens->len>8192 || fwrite(tokens->v,sizeof(int),tokens->len,f)!=(size_t)tokens->len || fclose(f))pt_die("prompt-ids",1);
    fprintf(pt_events,"{\"type\":\"prompt\",\"round\":%d,\"count\":%d}\n",round,tokens->len);fflush(pt_events);pt_rounds++;
}
static int pt_sample(agent_worker *w,const agent_config *cfg,bool greedy,uint64_t *rng,int round,bool inside,int parser_state) {
    if(pt_steps>=2048)pt_die("total-output-cap",3);
    int n=ds4_engine_vocab_size(w->engine);
    if(n!=129280)pt_die("vocabulary",1);
    uint32_t slots=ds4_gpu_stream_expert_cache_configured_count();
    if(slots!=6136)pt_die("configured-cache-slots",1);
    if(!pt_light_audit) {
        float *raw=xmalloc((size_t)n*sizeof(float));
        if(ds4_session_copy_logits(w->session,raw,n)!=n)pt_die("raw-copy",1);
        if(fwrite(raw,sizeof(float),n,pt_raw)!=(size_t)n)pt_die("raw-write",1);
        free(raw);
    }
    uint64_t before=*rng;
    int token=worker_sample_with_mode(w,cfg,greedy,rng);
    size_t length=0;char *piece=ds4_token_text(w->engine,token,&length);
    if(token<0 || token>=n || !piece || length>65536)pt_die("token-piece-bound",1);
    bool stop=ds4_token_is_stop_for_think_mode(w->engine,token,effective_think_mode(cfg));
    fprintf(pt_events,"{\"type\":\"sample\",\"index\":%d,\"round\":%d,\"pos\":%d,\"configured_slots\":%u,\"rng_before\":\"%016llx\",\"rng_after\":\"%016llx\",\"greedy\":%s,\"inside\":%s,\"parser\":%d,\"token\":%d,\"stop\":%s,\"piece_hex\":\"",pt_steps,round,ds4_session_pos(w->session),slots,(unsigned long long)before,(unsigned long long)*rng,greedy?"true":"false",inside?"true":"false",parser_state,token,stop?"true":"false");
    pt_hex(pt_events,piece,length);fputs("\"}\n",pt_events);fflush(pt_events);free(piece);pt_steps++;return token;
}
static void pt_round_end(int round,int generated,int maximum,bool limited,bool stopped,bool tool,bool malformed,bool early,bool interrupted) {
    fprintf(pt_events,"{\"type\":\"round-end\",\"round\":%d,\"generated\":%d,\"maximum\":%d,\"context_limited\":%s,\"model_stopped\":%s,\"got_tool\":%s,\"malformed\":%s,\"early_error\":%s,\"interrupted\":%s}\n",round,generated,maximum,limited?"true":"false",stopped?"true":"false",tool?"true":"false",malformed?"true":"false",early?"true":"false",interrupted?"true":"false");
    fflush(pt_events);
    if(limited||malformed||early||interrupted)pt_die("round-not-complete",3);
    if(!tool)pt_final_stop=stopped;
}
static int pt_finish(int old_rc) {
    int rc=old_rc?old_rc:(pt_stage==2 && pt_rounds==3 && pt_final_stop?0:3);
    fprintf(pt_events,"{\"type\":\"finish\",\"legacy_rc\":%d,\"returncode\":%d,\"stage\":%d,\"rounds\":%d,\"samples\":%d,\"actual_stop\":%s}\n",old_rc,rc,pt_stage,pt_rounds,pt_steps,pt_final_stop?"true":"false");
    fflush(pt_events);fflush(pt_raw);fclose(pt_events);fclose(pt_raw);free(pt_config);return rc;
}
