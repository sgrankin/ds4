/* Private synchronous model-worker fixture. The included agent transaction is
 * unchanged. Forwarding observers count real native calls and inject one
 * post-move interruption. No UI worker/thread or model opens in --self-test. */
#define ds4_session_load_snapshot fixture_load
#define ds4_session_save_snapshot fixture_save
#define ds4_session_sync fixture_sync
#define ds4_session_capture_suffix_pair_cache fixture_capture
#define ds4_session_reuse_suffix_repaired_pairs fixture_graft
#define DS4_AGENT_TEST_NO_MAIN
#include "../ds4_agent.c"
#undef ds4_session_load_snapshot
#undef ds4_session_save_snapshot
#undef ds4_session_sync
#undef ds4_session_capture_suffix_pair_cache
#undef ds4_session_reuse_suffix_repaired_pairs

extern int ds4_session_load_snapshot(ds4_session *,const ds4_session_snapshot *,char *,size_t);
extern int ds4_session_save_snapshot(ds4_session *,ds4_session_snapshot *,char *,size_t);
extern int ds4_session_sync(ds4_session *,const ds4_tokens *,char *,size_t);
extern int ds4_session_capture_suffix_pair_cache(ds4_session *,ds4_suffix_cache *,char *,size_t);
extern int ds4_session_reuse_suffix_repaired_pairs(ds4_session *,const ds4_suffix_cache *,const ds4_tokens *,uint32_t,ds4_suffix_relocation_stats *,char *,size_t);

#define APP_BUDGET (UINT64_C(128)<<20)
#define P 1409
#define SHIFT 65
#define SPAN 512
#define TAIL 128
#define DECODE 16
#define SOURCE_END (P+SHIFT+SPAN)
#define OLD_N (SOURCE_END+TAIL)
#define NEW_N (OLD_N-SHIFT)
#define INPUT_MAGIC UINT32_C(0x41535431)
#define RAW_MAGIC UINT32_C(0x41534731)
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"agent-pair:%d: %s: %s\n",__LINE__,#x,err);goto done;}}while(0)
typedef struct { ds4_tokens old,edited,targets; int user,vocab; } fixture_input;
typedef struct {const void *identity;uint64_t bytes;} fixture_allocation;
static fixture_allocation allocations[32];static int allocation_n;
static bool observing,cancel_prefix_load,observer_failed,cancel_triggered;
static bool saving_text_only,persistence_native_call;
static agent_worker *observed_worker;
static int calls_save,calls_load,calls_sync,calls_capture,calls_graft;
static const uint8_t *source_state_ptr,*source_tape_ptr,*prefix_state_ptr;
static const int *source_prefix_ptr;
static uint64_t actual_unique_peak;
static uint8_t ring_key,scratch_key,struct_key,destination_key;

static bool allocation_add(const void *p,uint64_t bytes){
    if(!p||!bytes||allocation_n>=32)return false;
    for(int i=0;i<allocation_n;i++)if(allocations[i].identity==p)return allocations[i].bytes==bytes;
    allocations[allocation_n++]=(fixture_allocation){p,bytes};return true;
}
static uint64_t allocation_sum(void){
    uint64_t n=0;for(int i=0;i<allocation_n;i++){if(allocations[i].bytes>UINT64_MAX-n)return UINT64_MAX;n+=allocations[i].bytes;}return n;
}
static void replace_destination(const ds4_tokens *t){
    for(int i=0;i<allocation_n;i++)if(allocations[i].identity==&destination_key){
        allocations[i]=(fixture_allocation){t->v,(uint64_t)t->cap*sizeof(int)};return;
    }
    observer_failed=true;
}
int fixture_load(ds4_session *s,const ds4_session_snapshot *in,char *err,size_t n){
    if(saving_text_only)persistence_native_call=true;
    if(observing){calls_load++;if(calls_load==1&&in->ptr!=prefix_state_ptr)observer_failed=true;}
    int rc=ds4_session_load_snapshot(s,in,err,n);
    if(observing&&cancel_prefix_load&&calls_load==1&&rc==0){
        pthread_mutex_lock(&observed_worker->mu);observed_worker->interrupt=true;pthread_mutex_unlock(&observed_worker->mu);cancel_triggered=true;
    }
    return rc;
}
int fixture_save(ds4_session *s,ds4_session_snapshot *out,char *err,size_t n){
    if(saving_text_only)persistence_native_call=true;
    if(observing)calls_save++;
    int rc=ds4_session_save_snapshot(s,out,err,n);
    if(observing&&rc==0){
        if(!allocation_add(out->ptr,out->cap)||allocation_sum()>APP_BUDGET)observer_failed=true;
        if(allocation_sum()>actual_unique_peak)actual_unique_peak=allocation_sum();
    }
    return rc;
}
int fixture_sync(ds4_session *s,const ds4_tokens *t,char *err,size_t n){
    if(saving_text_only)persistence_native_call=true;
    if(observing){calls_sync++;replace_destination(t);}
    return ds4_session_sync(s,t,err,n);
}
int fixture_capture(ds4_session *s,ds4_suffix_cache *c,char *err,size_t n){
    if(saving_text_only)persistence_native_call=true;
    if(observing)calls_capture++;
    return ds4_session_capture_suffix_pair_cache(s,c,err,n);
}
int fixture_graft(ds4_session *s,const ds4_suffix_cache *c,const ds4_tokens *t,uint32_t first,ds4_suffix_relocation_stats *stats,char *err,size_t n){
    if(saving_text_only)persistence_native_call=true;
    if(observing){
        calls_graft++;
        if(c->state.ptr!=source_state_ptr||c->pair_tape.ptr!=source_tape_ptr||
           !allocation_add(c->state.ptr,c->state.cap)||!allocation_add(c->pair_tape.ptr,c->pair_tape.cap)||
           memcmp(source_prefix_ptr,observed_worker->transcript.v,SOURCE_END*sizeof(int))||
           allocation_sum()>APP_BUDGET||observer_failed){
            snprintf(err,n,"private native allocation/source ownership check failed");return 1;
        }
        if(allocation_sum()>actual_unique_peak)actual_unique_peak=allocation_sum();
    }
    return ds4_session_reuse_suffix_repaired_pairs(s,c,t,first,stats,err,n);
}

static void free_input(fixture_input *a){ds4_tokens_free(&a->old);ds4_tokens_free(&a->edited);ds4_tokens_free(&a->targets);memset(a,0,sizeof(*a));}
static bool input_valid(const fixture_input *a){
    if(a->vocab!=129280||a->user<0||a->user>=a->vocab||a->old.len!=OLD_N||a->edited.len!=NEW_N||a->targets.len!=DECODE)return false;
    for(int i=0;i<OLD_N;i++)if(a->old.v[i]<0||a->old.v[i]>=a->vocab||(a->old.v[i]==a->user&&i!=SOURCE_END))return false;
    for(int i=0;i<DECODE;i++)if(a->targets.v[i]<0||a->targets.v[i]>=a->vocab)return false;
    if(a->old.v[SOURCE_END]!=a->user||a->edited.v[SOURCE_END-SHIFT]!=a->user||a->old.v[P]==a->edited.v[P])return false;
    return !memcmp(a->old.v,a->edited.v,P*sizeof(int))&&!memcmp(a->old.v+P+SHIFT,a->edited.v+P,(SPAN+TAIL)*sizeof(int));
}
static bool write_input(const char *path,const fixture_input *a){
    FILE *f=fopen(path,"wbx");if(!f)return false;uint32_t h[]={INPUT_MAGIC,1,(uint32_t)a->vocab,(uint32_t)a->user,OLD_N,NEW_N,DECODE,0};
    bool ok=fwrite(h,sizeof(h),1,f)==1&&fwrite(a->old.v,4,OLD_N,f)==OLD_N&&fwrite(a->edited.v,4,NEW_N,f)==NEW_N&&fwrite(a->targets.v,4,DECODE,f)==DECODE;
    if(fclose(f))ok=false;return ok;
}
static bool read_input(const char *path,fixture_input *a){
    FILE *f=fopen(path,"rb");if(!f)return false;uint32_t h[8]={0};bool ok=fread(h,sizeof(h),1,f)==1;
    if(!ok||h[0]!=INPUT_MAGIC||h[1]!=1||h[2]!=129280||h[3]>=h[2]||h[4]!=OLD_N||h[5]!=NEW_N||h[6]!=DECODE||h[7]){fclose(f);return false;}
    a->vocab=(int)h[2];a->user=(int)h[3];a->old.len=a->old.cap=OLD_N;a->edited.len=a->edited.cap=NEW_N;a->targets.len=a->targets.cap=DECODE;
    a->old.v=malloc(OLD_N*4);a->edited.v=malloc(NEW_N*4);a->targets.v=malloc(DECODE*4);
    ok=a->old.v&&a->edited.v&&a->targets.v;
    if(ok)ok=fread(a->old.v,4,OLD_N,f)==OLD_N&&fread(a->edited.v,4,NEW_N,f)==NEW_N&&fread(a->targets.v,4,DECODE,f)==DECODE&&fgetc(f)==EOF&&!ferror(f)&&input_valid(a);
    if(fclose(f))ok=false;if(!ok)free_input(a);return ok;
}
static bool make_input(ds4_engine *engine,const char *corpus,fixture_input *a){
    FILE *f=fopen(corpus,"rb");if(!f)return false;if(fseek(f,0,SEEK_END)){fclose(f);return false;}long n=ftell(f);if(n<=0||n>(16<<20)||fseek(f,0,SEEK_SET)){fclose(f);return false;}
    char *text=malloc((size_t)n+1);if(!text){fclose(f);return false;}bool ok=fread(text,1,(size_t)n,f)==(size_t)n;if(fclose(f))ok=false;if(!ok){free(text);return false;}text[n]=0;
    ds4_tokens tokens={0};ds4_tokenize_text(engine,text,&tokens);free(text);
    a->vocab=ds4_engine_vocab_size(engine);a->user=ds4_token_user(engine);
    if(a->vocab!=129280||a->user<0||a->user>=a->vocab){ds4_tokens_free(&tokens);return false;}
    if(tokens.len<=OLD_N+DECODE){ds4_tokens_free(&tokens);return false;}
    ds4_tokens old={tokens.v,OLD_N,OLD_N};ds4_tokens_copy(&a->old,&old);a->old.v[SOURCE_END]=a->user;
    /* Deterministic pre-outcome boundary sentinel, in the deleted region only.
     * This prevents accidental equal corpus pieces from moving the LCP. */
    if(a->old.v[P]==a->old.v[P+SHIFT]){
        a->old.v[P]=(a->old.v[P+SHIFT]+1)%a->vocab;
        if(a->old.v[P]==a->user)a->old.v[P]=(a->old.v[P]+1)%a->vocab;
    }
    for(int i=0;i<P;i++)ds4_tokens_push(&a->edited,a->old.v[i]);
    for(int i=P+SHIFT;i<OLD_N;i++)ds4_tokens_push(&a->edited,a->old.v[i]);
    for(int i=0;i<DECODE;i++)ds4_tokens_push(&a->targets,tokens.v[OLD_N+i]);
    ds4_tokens_free(&tokens);return input_valid(a);
}
static bool file_path(char *out,size_t n,const char *base,const char *suffix){return snprintf(out,n,"%s%s",base,suffix)>0&&strlen(base)+strlen(suffix)<n;}
static bool save_file(const char *base,const char *suffix,const void *p,size_t n){
    char path[2048];if(!file_path(path,sizeof(path),base,suffix))return false;FILE *f=fopen(path,"wbx");if(!f)return false;bool ok=fwrite(p,1,n,f)==n;if(fclose(f))ok=false;return ok;
}
static bool state_equal(const char *base,const char *suffix,const ds4_session_snapshot *s){
    char path[2048];if(!file_path(path,sizeof(path),base,suffix))return false;FILE *f=fopen(path,"rb");if(!f)return false;uint8_t b[32768];uint64_t offset=0;bool ok=true;
    while(offset<s->len){size_t n=s->len-offset<sizeof(b)?(size_t)(s->len-offset):sizeof(b);if(fread(b,1,n,f)!=n||memcmp(b,s->ptr+offset,n)){ok=false;break;}offset+=n;}
    if(fgetc(f)!=EOF||ferror(f))ok=false;if(fclose(f))ok=false;return ok;
}
static bool files_equal(const char *a,const char *as,const char *b,const char *bs){
    char ap[2048],bp[2048];if(!file_path(ap,sizeof(ap),a,as)||!file_path(bp,sizeof(bp),b,bs))return false;
    FILE *af=fopen(ap,"rb"),*bf=fopen(bp,"rb");if(!af||!bf){if(af)fclose(af);if(bf)fclose(bf);return false;}
    uint8_t aa[32768],bb[32768];bool ok=true;
    for(;;){size_t an=fread(aa,1,sizeof(aa),af),bn=fread(bb,1,sizeof(bb),bf);if(an!=bn||memcmp(aa,bb,an)){ok=false;break;}if(an<sizeof(aa)){if(ferror(af)||ferror(bf))ok=false;break;}}
    if(fclose(af))ok=false;if(fclose(bf))ok=false;return ok;
}
static uint64_t file_bytes(const char *base,const char *suffix){
    char p[2048];if(!file_path(p,sizeof(p),base,suffix))return UINT64_MAX;FILE *f=fopen(p,"rb");if(!f)return UINT64_MAX;
    bool ok=fseek(f,0,SEEK_END)==0;long n=ok?ftell(f):-1;if(fclose(f))n=-1;return n<0?UINT64_MAX:(uint64_t)n;
}
static double logz(const float *x,int n){double max=-INFINITY,sum=0;for(int i=0;i<n;i++){if(!isfinite(x[i]))return NAN;if(x[i]>max)max=x[i];}for(int i=0;i<n;i++)sum+=exp((double)x[i]-max);return max+log(sum);}
static void metrics(const float *p,const float *q,int n,int target,double *kl,double *tv,double *delta){
    double a=logz(p,n),b=logz(q,n);*kl=*tv=0;*delta=target<0?0:(b-q[target])-(a-p[target]);
    for(int i=0;i<n;i++){double pp=exp((double)p[i]-a),qq=exp((double)q[i]-b);*kl+=pp*((double)p[i]-a-(double)q[i]+b);*tv+=.5*fabs(pp-qq);}if(*kl<0&&*kl>-1e-10)*kl=0;
}
static int argmax(const float *p,int n){int j=0;for(int i=1;i<n;i++)if(p[i]>p[j])j=i;return j;}
static uint64_t extra_owned(agent_worker *w,const fixture_input *a,uint64_t vectors){
    return (uint64_t)(a->old.cap+a->edited.cap+a->targets.cap+w->transcript.cap)*sizeof(int)+vectors+w->out_cap+sizeof(*w)+sizeof(*w->cfg)+sizeof(*a)+(UINT64_C(64)<<10);
}
/* Independent identity/cap accounting. Model/engine/session native graph and
 * expert cache are monitored separately; this ledger charges all extra app
 * buffers, pair ring, retained/moved frontiers and transaction staging. */
static uint64_t start_ledger(agent_worker *w,const fixture_input *a,uint64_t vectors,bool transaction){
    allocation_n=0;actual_unique_peak=0;observer_failed=false;
    for(int i=0;i<w->context_frontier_count;i++){agent_context_frontier *f=&w->context_frontiers[i];
        if(!allocation_add(f->state.ptr,f->state.cap)||!allocation_add(f->prefix.v,(uint64_t)f->prefix.cap*4)||(f->pair_tape.ptr&&!allocation_add(f->pair_tape.ptr,f->pair_tape.cap)))return UINT64_MAX;
    }
    uint64_t ring=ds4_session_suffix_pair_tape_bytes(w->session);if(ring&&!allocation_add(&ring_key,ring))return UINT64_MAX;
    uint64_t scratch=(UINT64_C(8)<<20)+(UINT64_C(64)<<10)+2u*4096u*sizeof(int)+sizeof(ds4_tokens)+sizeof(ds4_session_snapshot)+sizeof(ds4_suffix_cache)+sizeof(ds4_suffix_relocation_stats);
    if(!allocation_add(&scratch_key,scratch)||!allocation_add(&struct_key,extra_owned(w,a,vectors)))return UINT64_MAX;
    if(transaction&&!allocation_add(&destination_key,P*sizeof(int)))return UINT64_MAX;
    return allocation_sum();
}
static bool save_current(agent_worker *w,const fixture_input *a,uint64_t vectors,const char *base,const char *suffix,const char *reference,char *err,size_t n){
    ds4_session_snapshot s={0};uint64_t owned=start_ledger(w,a,vectors,false),payload=ds4_session_payload_bytes(w->session);
    if(!payload||owned>APP_BUDGET||payload+1>APP_BUDGET-owned){snprintf(err,n,"private snapshot admission over app budget");return false;}
    bool ok=ds4_session_save_snapshot(w->session,&s,err,n)==0&&s.cap<=APP_BUDGET-owned&&save_file(base,suffix,s.ptr,(size_t)s.len);
    if(ok&&reference)ok=state_equal(reference,suffix,&s);ds4_session_snapshot_free(&s);return ok;
}
static bool open_worker(agent_worker *w,agent_config *cfg,ds4_engine *e,const fixture_input *a,const char *trace,bool pair,char *err,size_t n){
    memset(w,0,sizeof(*w));memset(cfg,0,sizeof(*cfg));cfg->non_interactive=true;cfg->gen.ctx_size=4096;cfg->gen.live_context_path="private-unused-context";
    w->engine=e;w->cfg=cfg;w->wake_fd[0]=w->wake_fd[1]=-1;w->initialized=true;w->session_dirty=w->user_activity=true;
    if(pthread_mutex_init(&w->mu,NULL))return false;
    if(pthread_cond_init(&w->cond,NULL)){pthread_mutex_destroy(&w->mu);return false;}
    if(ds4_session_create(&w->session,e,4096))goto fail;
    if(setenv("DS4_AGENT_CONTEXT_SUFFIX_REUSE",pair?"1":"0",1)||setenv("DS4_AGENT_CONTEXT_SUFFIX_PAIR_TAPE",pair?"1":"0",1)||setenv("DS4_AGENT_CONTEXT_FRONTIERS","2",1)||setenv("DS4_AGENT_CONTEXT_FRONTIER_MB","128",1)||setenv("DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL","128",1))goto fail;
    if(pair){uint64_t budget;if(!agent_context_frontier_slots(w,&budget)||!agent_worker_context_pair_ring(w,budget)){snprintf(err,n,"early native pair ring enable failed");goto fail;}}
    ds4_tokens_copy(&w->transcript,&a->old);w->trace=fopen(trace,"wx");if(w->trace)return true;
fail:
    ds4_session_free(w->session);w->session=NULL;ds4_tokens_free(&w->transcript);pthread_cond_destroy(&w->cond);pthread_mutex_destroy(&w->mu);return false;
}
static void close_worker(agent_worker *w){
    observing=false;agent_context_frontiers_clear(w);ds4_session_free(w->session);w->session=NULL;ds4_tokens_free(&w->transcript);if(w->trace)fclose(w->trace);free(w->out);pthread_cond_destroy(&w->cond);pthread_mutex_destroy(&w->mu);memset(w,0,sizeof(*w));
}
static bool capture_admitted(agent_worker *w,const fixture_input *a,uint64_t vectors){
    uint64_t before=start_ledger(w,a,vectors,false),payload=ds4_session_payload_bytes(w->session),pos=(uint64_t)ds4_session_pos(w->session);
    uint64_t forecast=payload+1+pos*sizeof(int)+(agent_worker_context_pair_mode(w)?UINT64_C(1024)*12288u:0);
    if(!payload||before>APP_BUDGET||forecast>APP_BUDGET-before)return false;
    return agent_worker_capture_context_frontier(w)&&start_ledger(w,a,vectors,false)<=APP_BUDGET;
}
static bool build_source(agent_worker *w,const fixture_input *a,uint64_t vectors,const char *base,const char *reference,FILE *raw,float *row,float *ref,char *err,size_t n){
    ds4_tokens view={a->old.v,P,P};if(agent_worker_sync_tokens(w,&view,false,err,n)||!capture_admitted(w,a,vectors)||w->context_frontier_count!=1)return false;
    if(!save_file(base,".prefix.state.bin",w->context_frontiers[0].state.ptr,(size_t)w->context_frontiers[0].state.len)||(reference&&!state_equal(reference,".prefix.state.bin",&w->context_frontiers[0].state)))return false;
    view.len=view.cap=P+SHIFT;if(agent_worker_sync_tokens(w,&view,false,err,n))return false;
    view.len=view.cap=SOURCE_END;if(agent_worker_sync_tokens(w,&view,false,err,n)||!capture_admitted(w,a,vectors)||w->context_frontier_count!=2)return false;
    agent_context_frontier *source=&w->context_frontiers[1];if(!save_file(base,".source.endpoint.state.bin",source->state.ptr,(size_t)source->state.len)||(reference&&!state_equal(reference,".source.endpoint.state.bin",&source->state)))return false;
    if(ds4_session_copy_logits(w->session,row,a->vocab)!=a->vocab||!isfinite(logz(row,a->vocab)))return false;
    if(raw&&fwrite(row,4,(size_t)a->vocab,raw)!=(size_t)a->vocab)return false;
    if(ref&&memcmp(row,ref,(size_t)a->vocab*4))return false;
    if(agent_worker_sync_tokens(w,&a->old,false,err,n))return false;
    if(!save_current(w,a,vectors,base,".source.full.state.bin",reference,err,n))return false;
    if(ds4_session_copy_logits(w->session,row,a->vocab)!=a->vocab||!isfinite(logz(row,a->vocab)))return false;
    return !raw||fwrite(row,4,(size_t)a->vocab,raw)==(size_t)a->vocab;
}

int main(int argc,char **argv){
    if(argc==2&&!strcmp(argv[1],"--self-test")){
        float p[]={logf(.6f),logf(.3f),logf(.1f)},q[]={logf(.3f),logf(.6f),logf(.1f)};double k,t,d;metrics(p,q,3,0,&k,&t,&d);if(fabs(k-.3*log(2.0))>1e-7||fabs(t-.3)>1e-7||fabs(d-log(2.0))>1e-7)return 1;
        fixture_input a={.user=100,.vocab=129280};for(int i=0;i<OLD_N;i++)ds4_tokens_push(&a.old,i==SOURCE_END?100:i==P?2:1);for(int i=0;i<P;i++)ds4_tokens_push(&a.edited,a.old.v[i]);for(int i=P+SHIFT;i<OLD_N;i++)ds4_tokens_push(&a.edited,a.old.v[i]);for(int i=0;i<DECODE;i++)ds4_tokens_push(&a.targets,3);
        bool ok=input_valid(&a);a.old.v[0]=4;ok=ok&&!input_valid(&a);a.old.v[0]=1;a.edited.v[SOURCE_END-SHIFT]=101;ok=ok&&!input_valid(&a);a.edited.v[SOURCE_END-SHIFT]=100;a.targets.v[0]=-1;ok=ok&&!input_valid(&a);a.targets.v[0]=3;a.old.v[1]=100;ok=ok&&!input_valid(&a);free_input(&a);
        allocation_n=0;int identity;ok=ok&&allocation_add(&identity,17)&&allocation_add(&identity,17)&&allocation_sum()==17&&!allocation_add(&identity,18);if(!ok)return 1;
        puts("native caller input/parity/ownership/arithmetic selftest PASS; no engine/model opened");return 0;
    }
    const char *model=NULL,*base=NULL,*input_path=NULL,*corpus=NULL,*reference=NULL,*mode=NULL;bool create=false;
    for(int i=1;i<argc;i++){if(!strcmp(argv[i],"--create-input")){create=true;continue;}if(i+1==argc)return 2;const char *o=argv[i],*v=argv[++i];if(!strcmp(o,"--model"))model=v;else if(!strcmp(o,"--base"))base=v;else if(!strcmp(o,"--input"))input_path=v;else if(!strcmp(o,"--corpus"))corpus=v;else if(!strcmp(o,"--reference-base"))reference=v;else if(!strcmp(o,"--mode"))mode=v;else return 2;}
    bool reuse=mode&&!strcmp(mode,"reuse"),pair=reuse||(mode&&!strcmp(mode,"tape-native"));if(!model||!base||!input_path||!mode||(!pair&&strcmp(mode,"native"))||(create&&(!corpus||reuse))||(reuse&&!reference))return 2;
    int rc=1;char err[256]={0},path[2048];fixture_input a={0};ds4_engine *engine=NULL;agent_worker worker={0};agent_config cfg={0};bool opened=false;FILE *raw=NULL,*refs=NULL,*summary=NULL;float *row=NULL,*ref=NULL;uint64_t vectors=0,peak=0;double sumk=0,sumt=0,sumd=0,reuse_ms=0;int flips=0,bits=0;bool precancel=false,postcancel=false,exports=false,persistence=false,source_move=false;ds4_session_snapshot scratch={0};
    ds4_engine_options options={.model_path=model,.backend=DS4_BACKEND_METAL,.context_size=4096,.prefill_chunk=512,.ssd_streaming=true,.ssd_streaming_cache_bytes=UINT64_C(64)<<30,.power_percent=100};
    REQUIRE(ds4_engine_open(&engine,&options)==0);
    REQUIRE(create?make_input(engine,corpus,&a)&&write_input(input_path,&a):read_input(input_path,&a));REQUIRE(a.user==ds4_token_user(engine)&&a.vocab==ds4_engine_vocab_size(engine));
    /* Complete token arrays are written before prefill or quality outcomes. */
    printf("FROZEN_INPUT_READY user_id=%d old=%d edited=%d teacher=%d\n",a.user,a.old.len,a.edited.len,a.targets.len);fflush(stdout);
    vectors=(uint64_t)a.vocab*2u*sizeof(float);row=malloc((size_t)a.vocab*4);ref=malloc((size_t)a.vocab*4);REQUIRE(row&&ref);
    REQUIRE(file_path(path,sizeof(path),base,".trace"));REQUIRE(open_worker(&worker,&cfg,engine,&a,path,pair,err,sizeof(err)));opened=true;
    REQUIRE(file_path(path,sizeof(path),base,".raw.bin"));raw=fopen(path,"wbx");REQUIRE(raw);uint32_t h[]={RAW_MAGIC,(uint32_t)a.vocab,DECODE,P,SHIFT,SPAN,TAIL,(uint32_t)reuse};REQUIRE(fwrite(h,sizeof(h),1,raw)==1&&fwrite(a.targets.v,4,DECODE,raw)==DECODE);
    REQUIRE(build_source(&worker,&a,vectors,base,reference,raw,row,NULL,err,sizeof(err)));
    if(reuse){
        REQUIRE(file_path(path,sizeof(path),reference,".raw.bin"));refs=fopen(path,"rb");REQUIRE(refs);uint32_t rh[8];int rt[DECODE];REQUIRE(fread(rh,sizeof(rh),1,refs)==1&&fread(rt,sizeof(rt),1,refs)==1&&rh[7]==0);rh[7]=1;REQUIRE(!memcmp(rh,h,sizeof(h))&&!memcmp(rt,a.targets.v,sizeof(rt)));
        REQUIRE(file_path(path,sizeof(path),base,".raw.bin"));REQUIRE(fflush(raw)==0);FILE *own=fopen(path,"rb");REQUIRE(own);bool equal=fseek(own,32+DECODE*4,SEEK_SET)==0;
        for(int i=0;i<2&&equal;i++)equal=fread(row,4,(size_t)a.vocab,own)==(size_t)a.vocab&&fread(ref,4,(size_t)a.vocab,refs)==(size_t)a.vocab&&!memcmp(row,ref,(size_t)a.vocab*4);if(fclose(own))equal=false;REQUIRE(equal);
        ds4_tokens sys={a.old.v,P,P};agent_context_suffix_plan plan;REQUIRE(agent_context_suffix_select(&worker,&a.edited,128,&plan)&&plan.source_start==P+SHIFT&&plan.destination_start==P&&plan.source_end==SOURCE_END&&plan.destination_end==P+SPAN&&plan.fresh_tokens==128);
        worker.interrupt=true;REQUIRE(agent_worker_try_context_suffix(&worker,&a.edited,&sys,err,sizeof(err))==-1);worker_clear_interrupt(&worker);REQUIRE(worker.context_frontier_count==2&&!ds4_session_is_approximate(worker.session)&&agent_tokens_equal(ds4_session_tokens(worker.session),&a.old));err[0]=0;REQUIRE(save_current(&worker,&a,vectors,base,".precancel.state.bin",NULL,err,sizeof(err))&&files_equal(base,".precancel.state.bin",reference,".source.full.state.bin"));precancel=true;
        peak=start_ledger(&worker,&a,vectors,true);uint64_t undo=ds4_session_payload_bytes(worker.session);REQUIRE(undo&&peak<=APP_BUDGET&&undo+1<=APP_BUDGET-peak);
        source_state_ptr=worker.context_frontiers[plan.source_frontier].state.ptr;source_tape_ptr=worker.context_frontiers[plan.source_frontier].pair_tape.ptr;source_prefix_ptr=worker.context_frontiers[plan.source_frontier].prefix.v;prefix_state_ptr=worker.context_frontiers[0].state.ptr;REQUIRE(source_tape_ptr);
        observing=true;observed_worker=&worker;calls_save=calls_load=calls_sync=calls_capture=calls_graft=0;double tick=now_sec();int attempted=agent_worker_try_context_suffix(&worker,&a.edited,&sys,err,sizeof(err));reuse_ms=(now_sec()-tick)*1000;observing=false;
        REQUIRE(attempted==1&&!observer_failed&&calls_save==1&&calls_load==1&&calls_sync==1&&calls_capture==0&&calls_graft==1&&worker.context_frontier_count==1);peak=actual_unique_peak;source_move=true;
        REQUIRE(peak<=APP_BUDGET&&ds4_session_is_approximate(worker.session)&&agent_tokens_equal(ds4_session_tokens(worker.session),&a.edited)&&ds4_session_payload_bytes(worker.session)==0);
        REQUIRE(ds4_session_save_snapshot(worker.session,&scratch,err,sizeof(err))!=0&&scratch.cap==0&&!agent_worker_capture_context_frontier(&worker));err[0]=0;
        uint64_t owned=start_ledger(&worker,&a,vectors,false),expected=file_bytes(reference,".fresh.full.state.bin");REQUIRE(owned<=APP_BUDGET&&expected<APP_BUDGET-owned&&expected!=UINT64_MAX);REQUIRE(ds4_session_suffix_diagnostic_snapshot(worker.session,&scratch,err,sizeof(err))==0&&scratch.cap<=APP_BUDGET-owned&&scratch.len==expected&&scratch.len>=4&&!memcmp(scratch.ptr,"DSSD",4));REQUIRE(save_file(base,".private.state.bin",scratch.ptr,(size_t)scratch.len));REQUIRE(ds4_session_load_snapshot(worker.session,&scratch,err,sizeof(err))!=0&&ds4_session_is_approximate(worker.session));ds4_session_snapshot_free(&scratch);err[0]=0;
        ds4_suffix_cache rejected={0};REQUIRE(ds4_session_capture_suffix_pair_cache(worker.session,&rejected,err,sizeof(err))!=0);ds4_suffix_cache_free(&rejected);exports=true;err[0]=0;
        ds4_tokens_copy(&worker.transcript,&a.edited);REQUIRE(ds4_session_copy_logits(worker.session,row,a.vocab)==a.vocab);char saved_sha[41];REQUIRE(file_path(path,sizeof(path),base,".saved.kv"));
        saving_text_only=true;persistence_native_call=false;bool saved=agent_kv_save_path(&worker,path,&worker.transcript,"agent-session",saved_sha,"private-pair-fixture",1,err,sizeof(err));saving_text_only=false;
        REQUIRE(saved&&!persistence_native_call&&ds4_session_is_approximate(worker.session)&&ds4_session_copy_logits(worker.session,ref,a.vocab)==a.vocab&&!memcmp(row,ref,(size_t)a.vocab*4)&&agent_tokens_equal(ds4_session_tokens(worker.session),&a.edited));persistence=true;
    }else{
        ds4_tokens prefix={a.edited.v,P,P},endpoint={a.edited.v,P+SPAN,P+SPAN};REQUIRE(agent_worker_restore_context_frontier(&worker,&prefix));REQUIRE(agent_worker_sync_tokens(&worker,&endpoint,false,err,sizeof(err))==0);REQUIRE(save_current(&worker,&a,vectors,base,".fresh.endpoint.state.bin",NULL,err,sizeof(err)));REQUIRE(ds4_session_copy_logits(worker.session,row,a.vocab)==a.vocab&&save_file(base,".fresh.endpoint.logits.bin",row,(size_t)a.vocab*4));REQUIRE(agent_worker_sync_tokens(&worker,&a.edited,false,err,sizeof(err))==0);REQUIRE(save_current(&worker,&a,vectors,base,".fresh.full.state.bin",NULL,err,sizeof(err)));
    }
    if(reuse)puts("step,target,delta_nll,kl,tv,argmax_flip,bit_equal");
    for(int step=0;step<=DECODE;step++){
        REQUIRE(ds4_session_copy_logits(worker.session,row,a.vocab)==a.vocab&&isfinite(logz(row,a.vocab)));REQUIRE(fwrite(row,4,(size_t)a.vocab,raw)==(size_t)a.vocab);
        if(reuse){REQUIRE(fread(ref,4,(size_t)a.vocab,refs)==(size_t)a.vocab);double k,t,d;metrics(ref,row,a.vocab,step<DECODE?a.targets.v[step]:-1,&k,&t,&d);REQUIRE(isfinite(k)&&isfinite(t)&&isfinite(d)&&k>=0&&t>=0);int flip=argmax(row,a.vocab)!=argmax(ref,a.vocab),bit=!memcmp(row,ref,(size_t)a.vocab*4);sumk+=k;sumt+=t;sumd+=d;flips+=flip;bits+=bit;printf("%d,%d,%.12g,%.12g,%.12g,%d,%d\n",step,step<DECODE?a.targets.v[step]:-1,d,k,t,flip,bit);}
        if(step<DECODE)REQUIRE(ds4_session_eval(worker.session,a.targets.v[step],err,sizeof(err))==0);
    }
    if(refs){REQUIRE(fgetc(refs)==EOF&&!ferror(refs));REQUIRE(fclose(refs)==0);refs=NULL;}REQUIRE(fclose(raw)==0);raw=NULL;
    if(reuse){
        close_worker(&worker);opened=false;REQUIRE(file_path(path,sizeof(path),base,".cancel.trace"));REQUIRE(open_worker(&worker,&cfg,engine,&a,path,true,err,sizeof(err)));opened=true;
        char cancel_base[2048];REQUIRE(file_path(cancel_base,sizeof(cancel_base),base,".cancel"));REQUIRE(build_source(&worker,&a,vectors,cancel_base,NULL,NULL,row,NULL,err,sizeof(err)));
        agent_context_suffix_plan plan;REQUIRE(agent_context_suffix_select(&worker,&a.edited,128,&plan));uint64_t owned=start_ledger(&worker,&a,vectors,true),undo=ds4_session_payload_bytes(worker.session);REQUIRE(undo&&owned<=APP_BUDGET&&undo+1<=APP_BUDGET-owned);
        prefix_state_ptr=worker.context_frontiers[0].state.ptr;calls_save=calls_load=calls_sync=calls_capture=calls_graft=0;observing=cancel_prefix_load=true;cancel_triggered=false;observed_worker=&worker;ds4_tokens sys={a.old.v,P,P};int attempted=agent_worker_try_context_suffix(&worker,&a.edited,&sys,err,sizeof(err));observing=cancel_prefix_load=false;
        REQUIRE(attempted==-1&&cancel_triggered&&!observer_failed&&calls_save==1&&calls_load==2&&calls_sync==0&&calls_graft==0&&calls_capture==0&&worker.context_frontier_count==1&&!ds4_session_is_approximate(worker.session)&&agent_tokens_equal(ds4_session_tokens(worker.session),&a.old));worker_clear_interrupt(&worker);err[0]=0;
        REQUIRE(save_current(&worker,&a,vectors,base,".rollback.state.bin",NULL,err,sizeof(err))&&files_equal(base,".rollback.state.bin",reference,".source.full.state.bin"));postcancel=true;
    }
    REQUIRE(file_path(path,sizeof(path),base,".summary.json"));summary=fopen(path,"wx");REQUIRE(summary);
    fprintf(summary,"{\"mode\":\"%s\",\"prefix\":%d,\"shift\":%d,\"span\":%d,\"tail\":%d,\"decode\":%d,\"user_id\":%d,\"latest_user_start\":%d,\"fresh_latest_user\":true,\"source_move_no_reload\":%s,\"precancel_preserves_state\":%s,\"postmove_cancel_rolls_back\":%s,\"canonical_and_diagnostic_gates\":%s,\"text_only_save_state_unchanged\":%s,\"transaction_peak_bytes_including_reserves\":%llu,\"app_budget_bytes\":%llu,\"global_peak_observed\":false,\"performed_phase_admissions_checked\":true,\"mean_kl\":%.12g,\"mean_tv\":%.12g,\"sum_delta_nll\":%.12g,\"argmax_flips\":%d,\"bit_equal_rows\":%d,\"reuse_ms_diagnostic\":%.6f}\n",mode,P,SHIFT,SPAN,TAIL,DECODE,a.user,SOURCE_END-SHIFT,source_move?"true":"false",precancel?"true":"false",postcancel?"true":"false",exports?"true":"false",persistence?"true":"false",(unsigned long long)peak,(unsigned long long)APP_BUDGET,sumk/(DECODE+1),sumt/(DECODE+1),sumd,flips,bits,reuse_ms);rc=0;
done:
    observing=cancel_prefix_load=saving_text_only=false;if(raw)fclose(raw);if(refs)fclose(refs);if(summary&&fclose(summary))rc=1;ds4_session_snapshot_free(&scratch);if(opened)close_worker(&worker);free(row);free(ref);free_input(&a);ds4_engine_close(engine);return rc;
}
