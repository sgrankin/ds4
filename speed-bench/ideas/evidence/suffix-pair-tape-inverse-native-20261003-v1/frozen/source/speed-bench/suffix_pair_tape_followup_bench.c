/* Private endpoint-aligned three-geometry follow-up oracle. A baseline build links the untouched native core.
 * Native raw vectors and canonical graph bytes must match before any speed or
 * approximation claim. No model is opened by --self-test. */
#include "ds4.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"pair-oracle:%d %s: %s\n",__LINE__,#x,err); goto done; } } while (0)
#define BUDGET (UINT64_C(128) << 20)
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1000.0+t.tv_nsec/1e6; }
static char *read_text(const char *path) {
    FILE *f=fopen(path,"rb"); if(!f)return NULL;
    if(fseek(f,0,SEEK_END)){fclose(f);return NULL;} long n=ftell(f);
    if(n<0||n>16*1024*1024||fseek(f,0,SEEK_SET)){fclose(f);return NULL;}
    char *p=malloc((size_t)n+1);if(!p){fclose(f);return NULL;}
    bool ok=fread(p,1,(size_t)n,f)==(size_t)n;if(fclose(f))ok=false;
    if(!ok){free(p);return NULL;}p[n]=0;return p;
}
static int integer(const char *s) { char *e;long v=strtol(s,&e,10);return e==s||*e||v<0||v>131072?-1:(int)v; }
static void append(ds4_tokens *a,const ds4_tokens *b,int first,int count) {for(int i=0;i<count;i++)ds4_tokens_push(a,b->v[first+i]);}
static double log_z(const float *x,int n) {
    double peak=-INFINITY,sum=0;for(int i=0;i<n;i++){if(!isfinite(x[i]))return NAN;if(x[i]>peak)peak=x[i];}
    for(int i=0;i<n;i++)sum+=exp((double)x[i]-peak);return peak+log(sum);
}
static int top(const float *x,int n) {int j=0;for(int i=1;i<n;i++)if(x[i]>x[j])j=i;return j;}
static void quality(const float *p,const float *q,int n,int target,double *kl,double *tv,double *dnll) {
    double lp=log_z(p,n),lq=log_z(q,n);*kl=*tv=0;*dnll=target<0?0:(lq-q[target])-(lp-p[target]);
    for(int i=0;i<n;i++){double a=exp((double)p[i]-lp),b=exp((double)q[i]-lq);*kl+=a*((double)p[i]-lp-(double)q[i]+lq);*tv+=.5*fabs(a-b);}
    if(*kl<0&&*kl>-1e-10)*kl=0;
}
static bool save_bytes(const char *base,const char *suffix,const ds4_session_snapshot *snap) {
    char path[2048];if(snprintf(path,sizeof(path),"%s%s",base,suffix)>=(int)sizeof(path))return false;
    FILE *f=fopen(path,"wbx");if(!f)return false;bool ok=fwrite(snap->ptr,1,(size_t)snap->len,f)==(size_t)snap->len;if(fclose(f))ok=false;return ok;
}
/* Compare a saved canonical state with independently produced bytes using
 * bounded stack staging charged by the existing64KiB reserve. A diagnostic
 * comparison may normalize ONLY the validated4-byte magic; all remaining
 * bytes and the complete file extent must match. */
static bool state_file_equal(const char *base,const char *suffix,
                             const ds4_session_snapshot *snap,bool diagnostic) {
    char path[2048];if(snprintf(path,sizeof(path),"%s%s",base,suffix)>=(int)sizeof(path))return false;
    FILE *f=fopen(path,"rb");if(!f)return false;
    uint8_t block[32768];uint64_t offset=0;bool ok=true;
    while(offset<snap->len){
        size_t n=snap->len-offset<sizeof(block)?(size_t)(snap->len-offset):sizeof(block);
        if(fread(block,1,n,f)!=n){ok=false;break;}
        size_t first=0;
        if(diagnostic&&offset==0){
            if(n<4||memcmp(block,"DSV4",4)||memcmp(snap->ptr,"DSSD",4)){ok=false;break;}
            first=4;
        }
        if(memcmp(block+first,snap->ptr+offset+first,n-first)){ok=false;break;}
        offset+=n;
    }
    if(fgetc(f)!=EOF||ferror(f))ok=false;
    if(fclose(f))ok=false;
    return ok;
}

static uint64_t tape_bytes(ds4_session *s) {
#ifdef DS4_PAIR_TAPE_BASELINE
    (void)s;return 0;
#else
    return ds4_session_suffix_pair_tape_bytes(s);
#endif
}
static int enable(ds4_session *s,uint32_t n,char *err,size_t len) {
#ifdef DS4_PAIR_TAPE_BASELINE
    (void)s;(void)n;(void)err;(void)len;return 1;
#else
    return ds4_session_enable_suffix_pair_tape(s,n,err,len);
#endif
}
static int capture(ds4_session *s,ds4_suffix_cache *c,bool taped,char *err,size_t len) {
#ifdef DS4_PAIR_TAPE_BASELINE
    (void)taped;
#else
    if(taped)return ds4_session_capture_suffix_pair_cache(s,c,err,len);
#endif
    return ds4_session_capture_suffix_cache(s,c,err,len);
}
static int reuse(ds4_session *s,const ds4_suffix_cache *c,const ds4_tokens *p,uint32_t start,ds4_suffix_relocation_stats *stats,char *err,size_t len) {
#ifdef DS4_PAIR_TAPE_BASELINE
    (void)s;(void)c;(void)p;(void)start;(void)stats;(void)err;(void)len;return 1;
#else
    return ds4_session_reuse_suffix_repaired_pairs(s,c,p,start,stats,err,len);
#endif
}
typedef struct {int calls,cancel_after;} cancel_state;
static bool cancel(void *p) {cancel_state *s=p;return ++s->calls>s->cancel_after;}
/* App-owned tape/snapshot/vector buffers plus conservative staging allowance.
 * Engine/cache/model memory belongs to the independently monitored runtime.
 * Reserve native payload's8MiB I/O chunk and64KiB stdio/opaque GPU wrappers. */
static uint64_t owned(ds4_session *s,const ds4_suffix_cache *cache,
                      const ds4_session_snapshot *prefix,const ds4_session_snapshot *scratch,
                      uint64_t vectors,uint64_t tokens) {
    return tape_bytes(s)+cache->state.cap+cache->pair_tape.cap+prefix->cap+scratch->cap+
           vectors+tokens+(UINT64_C(8)<<20)+(UINT64_C(64)<<10)+sizeof(*cache)+sizeof(*prefix)+sizeof(*scratch);
}

int main(int argc,char **argv) {
    if(argc==2&&!strcmp(argv[1],"--self-test")){
        float p[]={logf(.6f),logf(.3f),logf(.1f)},q[]={logf(.3f),logf(.6f),logf(.1f)};double k,t,d;
        quality(p,q,3,0,&k,&t,&d);if(fabs(k-.3*log(2.0))>1e-7||fabs(t-.3)>1e-7||fabs(d-log(2.0))>1e-7)return 1;
        puts("pair-oracle arithmetic PASS; no model opened");return 0;
    }
    const char *model=NULL,*corpus_path="speed-bench/promessi_sposi.txt",*base=NULL,*mode="native";
    int prefix_n=1408,shift=65,span=512,tail=128,decode=16,ctx=4096,chunk=512,cache_gb=64,capacity=1024;
    for(int i=1;i<argc;i++){
        if(i+1==argc)return 2;const char *o=argv[i],*v=argv[++i];
        if(!strcmp(o,"--model"))model=v;else if(!strcmp(o,"--corpus"))corpus_path=v;else if(!strcmp(o,"--base"))base=v;else if(!strcmp(o,"--mode"))mode=v;
        else {int n=integer(v);if(n<0)return 2;if(!strcmp(o,"--prefix"))prefix_n=n;else if(!strcmp(o,"--shift"))shift=n;else if(!strcmp(o,"--span"))span=n;else if(!strcmp(o,"--tail"))tail=n;else if(!strcmp(o,"--decode"))decode=n;else if(!strcmp(o,"--ctx"))ctx=n;else if(!strcmp(o,"--chunk"))chunk=n;else if(!strcmp(o,"--cache-gb"))cache_gb=n;else if(!strcmp(o,"--capacity"))capacity=n;else return 2;}
    }
    bool approximate=!strcmp(mode,"reuse"),taped=approximate||!strcmp(mode,"tape-native");
    if(!model||!base||(!approximate&&!taped&&strcmp(mode,"native"))||prefix_n<64||span<64||span>capacity||tail<1||decode<1||decode>64||capacity<64||capacity>4096||chunk<1||chunk>8192||cache_gb<1||cache_gb>96)return 2;
#ifdef DS4_PAIR_TAPE_BASELINE
    if(taped)return 2;
#endif
    int source_start=prefix_n+shift,source_end=source_start+span,old_total=source_end+tail;
    if(old_total+decode>=ctx||capacity>ctx)return 2;
    int rc=1;char err[256]={0},path[2048];char *text=NULL;FILE *raw=NULL,*summary=NULL;
    ds4_engine *engine=NULL;ds4_session *session=NULL;ds4_tokens corpus={0},old={0},edited={0},prefix={0},source_prefix={0},endpoint={0},edited_endpoint={0},targets={0};
    ds4_session_snapshot prefix_state={0},scratch={0};ds4_suffix_cache cache={0};ds4_suffix_relocation_stats stats={0};
    float *references=NULL,*row=NULL;uint64_t peak_owned=0;double fresh_ms=0,reuse_ms=0,sum_kl=0,sum_tv=0,sum_dnll=0;int flips=0,equal=0;bool identity_native_checked=shift==0,identity_native_state_equal=false,identity_native_logits_equal=false,identity_reuse_state_equal=false;
    ds4_engine_options options={.model_path=model,.backend=DS4_BACKEND_METAL,.context_size=ctx,.prefill_chunk=(uint32_t)chunk,.ssd_streaming=true,.ssd_streaming_cache_bytes=(uint64_t)cache_gb<<30,.power_percent=100};
    CHECK(ds4_engine_open(&engine,&options)==0);CHECK(ds4_session_create(&session,engine,ctx)==0);
    int vocab=ds4_engine_vocab_size(engine);size_t bytes=(size_t)vocab*sizeof(float);
    text=read_text(corpus_path);CHECK(text);ds4_tokenize_text(engine,text,&corpus);free(text);text=NULL;CHECK(corpus.len>old_total+decode);
    append(&old,&corpus,0,old_total);append(&prefix,&old,0,prefix_n);append(&source_prefix,&old,0,source_start);append(&endpoint,&old,0,source_end);
    append(&edited,&old,0,prefix_n);append(&edited,&old,source_start,span+tail);append(&edited_endpoint,&edited,0,prefix_n+span);append(&targets,&corpus,old_total,decode);
    uint64_t token_bytes=(uint64_t)(corpus.cap+old.cap+edited.cap+prefix.cap+source_prefix.cap+endpoint.cap+edited_endpoint.cap+targets.cap)*sizeof(int);
    uint64_t vector_bytes=(uint64_t)bytes*(decode+2);
    CHECK(owned(session,&cache,&prefix_state,&scratch,vector_bytes,token_bytes)+
          (taped?(uint64_t)capacity*3u*2u*512u*sizeof(float)+(UINT64_C(64)<<10):0)<BUDGET);
    references=malloc(bytes*(size_t)(decode+1));row=malloc(bytes);CHECK(references&&row);
    if(taped)CHECK(enable(session,(uint32_t)capacity,err,sizeof(err))==0);
    CHECK(ds4_session_sync(session,&prefix,err,sizeof(err))==0);
    CHECK(owned(session,&cache,&prefix_state,&scratch,vector_bytes,token_bytes)+ds4_session_payload_bytes(session)+1<BUDGET);
    CHECK(ds4_session_save_snapshot(session,&prefix_state,err,sizeof(err))==0);
    CHECK(ds4_session_sync(session,&source_prefix,err,sizeof(err))==0);CHECK(ds4_session_sync(session,&endpoint,err,sizeof(err))==0);
    CHECK(owned(session,&cache,&prefix_state,&scratch,vector_bytes,token_bytes)+ds4_session_payload_bytes(session)+1+
          (taped?(uint64_t)capacity*3u*2u*512u*sizeof(float):0)<BUDGET);
    CHECK(capture(session,&cache,taped,err,sizeof(err))==0);
    peak_owned=owned(session,&cache,&prefix_state,&scratch,vector_bytes,token_bytes);CHECK(peak_owned<BUDGET);
    CHECK(ds4_session_sync(session,&old,err,sizeof(err))==0);CHECK(ds4_session_copy_logits(session,row,vocab)==vocab);
    CHECK(isfinite(log_z(row,vocab)));
    CHECK(snprintf(path,sizeof(path),"%s.raw.bin",base)<(int)sizeof(path));raw=fopen(path,"wbx");CHECK(raw);
    uint32_t header[]={UINT32_C(0x53505431),(uint32_t)vocab,(uint32_t)decode,(uint32_t)prefix_n,(uint32_t)shift,(uint32_t)span,(uint32_t)tail,(uint32_t)approximate};
    CHECK(fwrite(header,sizeof(header),1,raw)==1);CHECK(fwrite(targets.v,sizeof(int),(size_t)decode,raw)==(size_t)decode);CHECK(fwrite(row,bytes,1,raw)==1);
    CHECK(owned(session,&cache,&prefix_state,&scratch,vector_bytes,token_bytes)+ds4_session_payload_bytes(session)+1<BUDGET);
    CHECK(ds4_session_save_snapshot(session,&scratch,err,sizeof(err))==0);CHECK(save_bytes(base,".source.state.bin",&scratch));
    CHECK(ds4_session_load_snapshot(session,&prefix_state,err,sizeof(err))==0);double tick=now_ms();CHECK(ds4_session_sync(session,&edited_endpoint,err,sizeof(err))==0);CHECK(ds4_session_sync(session,&edited,err,sizeof(err))==0);fresh_ms=now_ms()-tick;
    uint64_t fresh_payload_bytes=ds4_session_payload_bytes(session);
    CHECK(owned(session,&cache,&prefix_state,&scratch,vector_bytes,token_bytes)+
          (fresh_payload_bytes+1>scratch.cap?fresh_payload_bytes+1-scratch.cap:0)<BUDGET);
    CHECK(ds4_session_save_snapshot(session,&scratch,err,sizeof(err))==0);CHECK(save_bytes(base,".fresh.state.bin",&scratch));
    if(identity_native_checked){
        identity_native_state_equal=state_file_equal(base,".source.state.bin",&scratch,false);
        CHECK(ds4_session_copy_logits(session,references,vocab)==vocab);
        identity_native_logits_equal=!memcmp(row,references,bytes);
        CHECK(identity_native_state_equal&&identity_native_logits_equal);
    }
    for(int j=0;j<=decode;j++){CHECK(ds4_session_copy_logits(session,references+(size_t)j*vocab,vocab)==vocab);CHECK(isfinite(log_z(references+(size_t)j*vocab,vocab)));if(j<decode)CHECK(ds4_session_eval(session,targets.v[j],err,sizeof(err))==0);}
    CHECK(fwrite(references,bytes,(size_t)(decode+1),raw)==(size_t)(decode+1));
    uint64_t active_owned=owned(session,&cache,&prefix_state,&scratch,vector_bytes,token_bytes);if(active_owned>peak_owned)peak_owned=active_owned;CHECK(peak_owned<BUDGET);
    bool rejected_coverage=false,rejected_precancel=false,partial_cancel_invalidated=false;
    if(approximate){
        CHECK(ds4_session_load_snapshot(session,&prefix_state,err,sizeof(err))==0);
        CHECK(ds4_session_suffix_diagnostic_snapshot(session,&scratch,err,sizeof(err))==0);
        CHECK(scratch.len>=4&&!memcmp(scratch.ptr,"DSSD",4));
        uint8_t source_magic[4];memcpy(source_magic,cache.state.ptr,4);
        memcpy(cache.state.ptr,scratch.ptr,4);
        CHECK(reuse(session,&cache,&edited,(uint32_t)source_start,&stats,err,sizeof(err))!=0);
        memcpy(cache.state.ptr,source_magic,4);
        CHECK(ds4_session_load_snapshot(session,&scratch,err,sizeof(err))!=0);
        CHECK(!ds4_session_is_approximate(session));
        CHECK(ds4_session_save_snapshot(session,&scratch,err,sizeof(err))==0);
        CHECK(scratch.len==prefix_state.len&&!memcmp(scratch.ptr,prefix_state.ptr,(size_t)scratch.len));
        if((source_start|prefix_n)&1){
            uint32_t saved=cache.pair_start;cache.pair_start=(uint32_t)source_start+1;CHECK(reuse(session,&cache,&edited,(uint32_t)source_start,&stats,err,sizeof(err))!=0);cache.pair_start=saved;
            uint64_t saved_len=cache.pair_tape.len;cache.pair_tape.len--;CHECK(reuse(session,&cache,&edited,(uint32_t)source_start,&stats,err,sizeof(err))!=0);cache.pair_tape.len=saved_len;
            CHECK(ds4_session_save_snapshot(session,&scratch,err,sizeof(err))==0);CHECK(scratch.len==prefix_state.len&&!memcmp(scratch.ptr,prefix_state.ptr,(size_t)scratch.len));rejected_coverage=true;
        }
        cancel_state stop={.cancel_after=0};ds4_session_set_cancel(session,cancel,&stop);CHECK(reuse(session,&cache,&edited,(uint32_t)source_start,&stats,err,sizeof(err))!=0);ds4_session_set_cancel(session,NULL,NULL);
        CHECK(ds4_session_save_snapshot(session,&scratch,err,sizeof(err))==0);CHECK(scratch.len==prefix_state.len&&!memcmp(scratch.ptr,prefix_state.ptr,(size_t)scratch.len));rejected_precancel=true;
        if((source_start|prefix_n)&1){
            stop=(cancel_state){.cancel_after=2};ds4_session_set_cancel(session,cancel,&stop);CHECK(reuse(session,&cache,&edited,(uint32_t)source_start,&stats,err,sizeof(err))!=0);ds4_session_set_cancel(session,NULL,NULL);
            CHECK(ds4_session_payload_bytes(session)==0);CHECK(ds4_session_save_snapshot(session,&scratch,err,sizeof(err))!=0);partial_cancel_invalidated=true;
            CHECK(ds4_session_load_snapshot(session,&prefix_state,err,sizeof(err))==0);
        }
        err[0]=0;tick=now_ms();CHECK(reuse(session,&cache,&edited,(uint32_t)source_start,&stats,err,sizeof(err))==0);reuse_ms=now_ms()-tick;
        CHECK(ds4_session_is_approximate(session)&&ds4_session_payload_bytes(session)==0);CHECK(ds4_session_save_snapshot(session,&scratch,err,sizeof(err))!=0);
        CHECK(owned(session,&cache,&prefix_state,&scratch,vector_bytes,token_bytes)+
              (fresh_payload_bytes+1>scratch.cap?fresh_payload_bytes+1-scratch.cap:0)<BUDGET);
        CHECK(ds4_session_suffix_diagnostic_snapshot(session,&scratch,err,sizeof(err))==0);CHECK(save_bytes(base,".private.state.bin",&scratch));
        CHECK(scratch.len>=4&&!memcmp(scratch.ptr,"DSSD",4));
        if(identity_native_checked){identity_reuse_state_equal=state_file_equal(base,".source.state.bin",&scratch,true);CHECK(identity_reuse_state_equal);}
        CHECK(ds4_session_load_snapshot(session,&scratch,err,sizeof(err))!=0);CHECK(ds4_session_is_approximate(session));
        ds4_suffix_cache rejected={0};CHECK(capture(session,&rejected,false,err,sizeof(err))!=0);ds4_suffix_cache_free(&rejected);
        puts("step,target,delta_nll,kl,tv,argmax_flip,bit_equal");
        for(int j=0;j<=decode;j++){
            CHECK(ds4_session_copy_logits(session,row,vocab)==vocab);double k,t,d;quality(references+(size_t)j*vocab,row,vocab,j<decode?targets.v[j]:-1,&k,&t,&d);CHECK(isfinite(k)&&isfinite(t)&&isfinite(d)&&k>=-1e-10);
            int flip=top(row,vocab)!=top(references+(size_t)j*vocab,vocab),bit=!memcmp(row,references+(size_t)j*vocab,bytes);sum_kl+=k;sum_tv+=t;sum_dnll+=d;flips+=flip;equal+=bit;
            printf("%d,%d,%.12g,%.12g,%.12g,%d,%d\n",j,j<decode?targets.v[j]:-1,d,k,t,flip,bit);CHECK(fwrite(row,bytes,1,raw)==1);if(j<decode)CHECK(ds4_session_eval(session,targets.v[j],err,sizeof(err))==0);
        }
    }
    if(approximate&&identity_native_checked)CHECK(equal==decode+1&&flips==0);
    CHECK(snprintf(path,sizeof(path),"%s.summary.json",base)<(int)sizeof(path));summary=fopen(path,"wx");CHECK(summary);
    fprintf(summary,"{\"mode\":\"%s\",\"endpoint_partition_aligned\":true,\"identity_native_checked\":%s,\"identity_native_state_equal\":%s,\"identity_native_logits_equal\":%s,\"identity_reuse_state_equal\":%s,\"prefix\":%d,\"shift\":%d,\"span\":%d,\"tail\":%d,\"decode\":%d,\"raw_reference_rows\":%d,\"raw_candidate_rows\":%d,\"tape_buffer_and_metadata_bytes\":%llu,\"immutable_tape_bytes\":%llu,\"peak_staging_bytes_including_reserve\":%llu,\"staging_budget_bytes\":%llu,\"coverage_rejections_preserve_state\":%s,\"precancel_preserves_state\":%s,\"partial_cancel_invalidates\":%s,\"approximate_export_blocked\":%s,\"pair_inputs_reused\":%u,\"pair_rows_rebuilt\":%u,\"pair_boundary_rows\":%u,\"reused_tokens\":%u,\"fresh_tail_tokens\":%u,\"fresh_ms_diagnostic\":%.6f,\"reuse_ms_diagnostic\":%.6f,\"mean_kl\":%.12g,\"mean_tv\":%.12g,\"sum_delta_nll\":%.12g,\"argmax_flips\":%d,\"bit_equal_rows\":%d}\n",mode,identity_native_checked?"true":"false",identity_native_state_equal?"true":"false",identity_native_logits_equal?"true":"false",identity_reuse_state_equal?"true":"false",prefix_n,shift,span,tail,decode,decode+2,approximate?decode+1:0,(unsigned long long)tape_bytes(session),(unsigned long long)cache.pair_tape.len,(unsigned long long)peak_owned,(unsigned long long)BUDGET,rejected_coverage?"true":"false",rejected_precancel?"true":"false",partial_cancel_invalidated?"true":"false",approximate?"true":"false",stats.pair_inputs_reused,stats.pair_rows_rebuilt,stats.pair_boundary_rows,stats.reuse.reused_tokens,stats.reuse.fresh_tail_tokens,fresh_ms,reuse_ms,sum_kl/(decode+1),sum_tv/(decode+1),sum_dnll,flips,equal);
    rc=0;
done:
    if(raw&&fclose(raw))rc=1;if(summary&&fclose(summary))rc=1;free(text);free(references);free(row);ds4_tokens_free(&corpus);ds4_tokens_free(&old);ds4_tokens_free(&edited);ds4_tokens_free(&prefix);ds4_tokens_free(&source_prefix);ds4_tokens_free(&endpoint);ds4_tokens_free(&edited_endpoint);ds4_tokens_free(&targets);ds4_session_snapshot_free(&prefix_state);ds4_session_snapshot_free(&scratch);ds4_suffix_cache_free(&cache);ds4_session_free(session);ds4_engine_close(engine);return rc;
}
