/* SOURCE-ONLY fixed native oracle, UNCOMPILED/UNRUN. No tools/sampling/RNG.
 * Future runner must establish literal ROOT grant/MAIN guard before invoking
 * this helper. One engine, at most one graph; opaque cache remains engine-bound.
 * No source diagnostic is imported as canonical. */
#include "ds4_clm_block_private.h"
#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <sys/stat.h>
#define CTX 1024u
#define A_END 128u
#define C_START 254u
#define C_END 319u
#define F_COUNT 128u
#define VOCAB 129280u
#define CAP (UINT64_C(128)<<20)
#define ROW_BYTES ((uint64_t)VOCAB*4u)
#define RESERVE ((UINT64_C(8)<<20)+32768u+4096u+65536u+65536u)
#define TRY(x) do { if(!(x)){fprintf(stderr,"CLM oracle line%d %s: %s\n",__LINE__,#x,r.err);goto done;} }while(0)
static const char *A_TEXT="Keep a current record. Later questions ask for its current access code. Reply only with CODE= followed by the four digits. Use the record stated in the conversation.";
static const char *B_OLD="The current access code for Record X is 3141. Earlier planning notes proposed checking the desk lamp, listing the shelf boxes, and reviewing the weather sensor. Those checks are complete; their details are irrelevant to the access code.";
static const char *B_NEW="The current access code for Record X is 3141. The earlier lamp, shelf, and weather checks are complete.";
static const char *C_TEXT="Tool notes: the weather sensor reported a mild afternoon. The desk lamp is off. Three unlabelled boxes remain on a shelf. These observations do not specify an access code.";
static const char *F_TEXT="What is the current access code for Record X? Reply with only CODE= followed by the four digits, without spaces or explanation.";
static const int GOLD[4]={83552,31,18321,19};
static const char *ARMS[5]={"source-old-isolation","identity-graft","fresh-changed-isolation","changed-graft","changed-unarmed-native"};
typedef struct {
    ds4_engine *e;ds4_session *s;ds4_clm_block_cache *cache;
    ds4_tokens a,b_old,b_new,c,f,gold,old,changed;
    ds4_session_snapshot exact_a,diagnostic;
    float *raw;
    FILE *ledger,*io;
    uint64_t peak_observed,peak_forecast;
    char err[256];
} run;
static uint32_t u32(const unsigned char *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static void p32(unsigned char *p,uint32_t x){for(unsigned i=0;i<4;i++)p[i]=(unsigned char)(x>>(8u*i));}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
static bool join(char *o,size_t n,const char *a,const char *b){int k=snprintf(o,n,"%s/%s",a,b);return k>=0&&(size_t)k<n;}
static bool write_bytes(const char *p,const void *v,uint64_t n){FILE *f=fopen(p,"wbx");if(!f)return false;bool ok=fwrite(v,1,(size_t)n,f)==n;if(fclose(f))ok=false;return ok;}
static uint64_t caller(run *r,bool omit_edited,const ds4_tokens *edited){
    /* Owned opaque cache/new policy are counted by the API where documented.
     * Input/token owners are distinct except borrowed edited->v, omitted from
     * this declaration only because graft already charges that exact owner. */
    const ds4_tokens *t[]={&r->a,&r->b_old,&r->b_new,&r->c,&r->f,&r->gold,&r->old,&r->changed};
    uint64_t total=r->exact_a.cap+r->diagnostic.cap+ROW_BYTES;
    for(unsigned i=0;i<8;i++)if(!omit_edited||t[i]->v!=edited->v)total+=(uint64_t)t[i]->cap*4u;
    return total;
}
static bool budget(run *r,const char *stage,uint64_t forecast){
    const ds4_tokens *checkpoint=r->s?ds4_session_tokens(r->s):NULL;
    uint64_t terms[]={caller(r,false,NULL),r->cache?ds4_clm_block_cache_bytes(r->cache):0,
        r->s?ds4_clm_block_owned_bytes(r->s):0,checkpoint?(uint64_t)checkpoint->cap*4u:0,RESERVE,forecast};
    uint64_t total=0;bool admitted=true;
    for(unsigned i=0;i<sizeof(terms)/sizeof(terms[0]);i++)if(terms[i]>CAP-total){admitted=false;break;}else total+=terms[i];
    if(admitted){if(forecast&&total>r->peak_forecast)r->peak_forecast=total;else if(!forecast&&total>r->peak_observed)r->peak_observed=total;}
    if(r->ledger){fprintf(r->ledger,"{\"stage\":\"%s\",\"admitted\":%s,\"forecast_bytes\":%llu,\"scoped_total\":%llu,\"budget\":%llu}\n",stage,admitted?"true":"false",(unsigned long long)forecast,(unsigned long long)total,(unsigned long long)CAP);fflush(r->ledger);}
    return admitted;
}
static uint64_t state_bytes(uint32_t end){
    uint64_t n=52u+((uint64_t)end+VOCAB)*4u+(uint64_t)40u*(end<128u?end:128u)*512u*4u;
    for(unsigned o=0;o<4;o++){n+=(uint64_t)(end/(o<3?2u:1u))*(512u+128u)*4u;if(o<3&&(end&1u))n+=4096u;}return n;
}
static bool ids_write(const char *p,const ds4_tokens *t){
    FILE *f=fopen(p,"wbx");if(!f)return false;unsigned char h[16];p32(h,0x43494431u);p32(h+4,1u);p32(h+8,(uint32_t)t->len);p32(h+12,VOCAB);
    bool ok=fwrite(h,1,16,f)==16;for(int i=0;ok&&i<t->len;i++){p32(h,(uint32_t)t->v[i]);ok=fwrite(h,1,4,f)==4;}if(fclose(f))ok=false;return ok;
}
static bool append(ds4_tokens *dst,const ds4_tokens *src){for(int i=0;i<src->len;i++)ds4_tokens_push(dst,src->v[i]);return true;}
static bool block(ds4_engine *e,const char *role,const char *text,int target,int pad,ds4_tokens *out){
    ds4_tokens temp={0};ds4_chat_append_message(e,&temp,role,text);
    if(temp.len<1||temp.len>target){ds4_tokens_free(&temp);return false;}
    /* Padding before the actual role marker keeps the entire newest user
     * message/assistant prefix in F; no gold is present in padding. */
    for(int i=temp.len;i<target;i++)ds4_tokens_push(out,pad);
    append(out,&temp);ds4_tokens_free(&temp);return out->len==target;
}
static bool build(run *r,const char *out){
    ds4_tokens pad={0},a_message={0},f_message={0};ds4_tokenize_text(r->e," x",&pad);
    bool ok=pad.len==1&&!ds4_token_is_stop(r->e,pad.v[0])&&!ds4_token_is_thinking_control(r->e,pad.v[0]);
    if(ok){ds4_chat_begin(r->e,&r->a);ds4_chat_append_message(r->e,&a_message,"system",A_TEXT);ok=r->a.len+a_message.len<=128;}
    if(ok){for(int i=r->a.len+a_message.len;i<128;i++)ds4_tokens_push(&r->a,pad.v[0]);append(&r->a,&a_message);}
    if(ok)ok=block(r->e,"user",B_OLD,126,pad.v[0],&r->b_old)&&block(r->e,"user",B_NEW,126,pad.v[0],&r->b_new)&&block(r->e,"tool",C_TEXT,65,pad.v[0],&r->c);
    if(ok){ds4_chat_append_message(r->e,&f_message,"user",F_TEXT);ds4_chat_append_assistant_prefix(r->e,&f_message,DS4_THINK_NONE);ok=f_message.len<=128;}
    if(ok){for(int i=f_message.len;i<128;i++)ds4_tokens_push(&r->f,pad.v[0]);append(&r->f,&f_message);ds4_tokenize_text(r->e,"CODE=3141",&r->gold);ok=r->gold.len==4&&!memcmp(r->gold.v,GOLD,sizeof GOLD);}
    if(ok){append(&r->old,&r->a);append(&r->old,&r->b_old);append(&r->old,&r->c);append(&r->old,&r->f);append(&r->changed,&r->a);append(&r->changed,&r->b_new);append(&r->changed,&r->c);append(&r->changed,&r->f);
        ok=r->a.len==128&&r->old.len==447&&r->changed.len==447&&memcmp(r->b_old.v,r->b_new.v,126u*4u)!=0;}
    ds4_tokens_free(&pad);ds4_tokens_free(&a_message);ds4_tokens_free(&f_message);
    const ds4_tokens *all[]={&r->a,&r->b_old,&r->b_new,&r->c,&r->f,&r->gold,&r->old,&r->changed};const char *names[]={"a.ids","b-old.ids","b-changed.ids","c.ids","f.ids","teacher.ids","old-full.ids","changed-full.ids"};
    for(unsigned i=0;ok&&i<8;i++){char p[2048];ok=join(p,sizeof p,out,names[i])&&ids_write(p,all[i]);}
    return ok&&budget(r,"fixture_built_before_any_evaluation",0);
}
static bool checkpoint(run *r,const ds4_tokens *all,uint32_t n,unsigned teachers){
    const ds4_tokens *t=ds4_session_tokens(r->s);return t&&t->len==(int)(n+teachers)&&ds4_session_pos(r->s)==(int)(n+teachers)&&ds4_session_checkpoint_valid(r->s)&&!memcmp(t->v,all->v,(size_t)n*4u)&&(!teachers||!memcmp(t->v+n,GOLD,teachers*4u));
}
static bool sync_to(run *r,const ds4_tokens *all,uint32_t n){ds4_tokens t={all->v,(int)n,all->cap};return ds4_session_sync(r->s,&t,r->err,sizeof r->err)==0&&checkpoint(r,all,n,0);}
static bool export_state(run *r,const char *path,uint32_t expected,bool diagnostic){
    if(!budget(r,"state_export_forecast",state_bytes(expected)+1u))return false;
    if(diagnostic){if(!ds4_session_is_approximate(r->s)||ds4_session_payload_bytes(r->s))return false;
        if(ds4_session_suffix_diagnostic_snapshot(r->s,&r->diagnostic,r->err,sizeof r->err))return false;
    }else if(ds4_session_is_approximate(r->s)||ds4_session_save_snapshot(r->s,&r->diagnostic,r->err,sizeof r->err))return false;
    const uint32_t header[13]={diagnostic?DS4_SESSION_SUFFIX_DIAGNOSTIC_MAGIC:DS4_SESSION_PAYLOAD_MAGIC,2u,CTX,1u,128u,128u,CTX+1u,expected,40u,512u,128u,VOCAB,0x413431u};
    bool ok=r->diagnostic.len==state_bytes(expected)&&budget(r,"state_export_actual",0);
    for(unsigned i=0;ok&&i<13;i++)ok=u32(r->diagnostic.ptr+4u*i)==header[i];
    if(ok)ok=write_bytes(path,r->diagnostic.ptr,r->diagnostic.len);
    ds4_session_snapshot_free(&r->diagnostic);return ok;
}
static bool row(run *r,FILE *f,uint32_t position,int target){
    unsigned char h[8];p32(h,position);p32(h+4,(uint32_t)target);
    if(ds4_session_copy_logits(r->s,r->raw,VOCAB)!=(int)VOCAB)return false;
    for(unsigned i=0;i<VOCAB;i++)if(!isfinite(r->raw[i]))return false;
    return fwrite(h,1,8,f)==8&&fwrite(r->raw,1,(size_t)ROW_BYTES,f)==ROW_BYTES&&fflush(f)==0;
}
static bool snapshot_io(run *r,const char *stage){ds4_gpu_stream_expert_io_metrics m={0};if(!ds4_gpu_stream_expert_io_metrics_read(&m)||!m.explicit_io_metrics_enabled)return false;
    fprintf(r->io,"%s,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",stage,(unsigned long long)m.pread_calls,(unsigned long long)m.pread_requested_bytes,(unsigned long long)m.pread_returned_bytes,(unsigned long long)m.cache_loader_returned_bytes,(unsigned long long)m.cache_hits,(unsigned long long)m.cache_misses,(unsigned long long)m.cache_evictions);return fflush(r->io)==0;
}
static bool compare_artifacts(const char *out,const char *left,const char *right){
    const char *files[]={"c-end.state","f-end.state","teacher1.state","teacher2.state","teacher3.state","teacher4.state","rows.raw"};unsigned char scratch[8192],other[8192];
    for(unsigned i=0;i<7;i++){char a[2048],b[2048],p[2048];if(!join(p,sizeof p,out,left)||!join(a,sizeof a,p,files[i])||!join(p,sizeof p,out,right)||!join(b,sizeof b,p,files[i]))return false;
        FILE *fa=fopen(a,"rb"),*fb=fopen(b,"rb");bool ok=fa&&fb;while(ok){size_t n=fread(scratch,1,sizeof scratch,fa),m=fread(other,1,sizeof other,fb);if(n!=m||memcmp(scratch,other,n)){ok=false;break;}if(!n){ok=!ferror(fa)&&!ferror(fb);break;}}if(fa&&fclose(fa))ok=false;if(fb&&fclose(fb))ok=false;if(!ok)return false;}
    return true;
}
int main(int argc,char **argv){
    if(argc!=4||strcmp(argv[1],"--guarded-run")){fprintf(stderr,"future ROOT guarded run required\n");return 2;}
    run r={0};int rc=1;const char *model=argv[2],*out=argv[3];FILE *raw_file=NULL;char path[2048],arm_path[2048];
    ds4_engine_options cfg={.model_path=model,.backend=DS4_BACKEND_METAL,.context_size=CTX,.prefill_chunk=128,.ssd_streaming=true,.ssd_streaming_cache_bytes=UINT64_C(64)<<30,.quality=false};
    TRY(ds4_engine_open(&r.e,&cfg)==0);TRY(ds4_engine_is_deepseek41(r.e)&&ds4_engine_vocab_size(r.e)==(int)VOCAB&&ds4_engine_layer_count(r.e)==40);
    r.raw=malloc((size_t)ROW_BYTES);TRY(r.raw);
    TRY(join(path,sizeof path,out,"budgets.jsonl"));r.ledger=fopen(path,"wx");TRY(r.ledger);
    TRY(join(path,sizeof path,out,"io.csv"));r.io=fopen(path,"wx");TRY(r.io);fprintf(r.io,"stage,pread_calls,requested,returned,loader_returned,hits,misses,evictions\n");
    TRY(build(&r,out));r.cache=ds4_clm_block_cache_create();TRY(r.cache);
    for(unsigned arm=0;arm<5;arm++){
        if(r.s){ds4_session_free(r.s);r.s=NULL;}TRY(ds4_session_create(&r.s,r.e,CTX)==0);
        const ds4_tokens *full=arm<2?&r.old:&r.changed;
        TRY(join(arm_path,sizeof arm_path,out,ARMS[arm])&&mkdir(arm_path,0700)==0);
        TRY(sync_to(&r,full,A_END));TRY(budget(&r,"A_canonical_forecast",state_bytes(A_END)+1u));
        TRY(ds4_session_save_snapshot(r.s,&r.diagnostic,r.err,sizeof r.err)==0);
        if(arm==0){r.exact_a=r.diagnostic;r.diagnostic=(ds4_session_snapshot){0};TRY(join(path,sizeof path,out,"immutable-a.state")&&write_bytes(path,r.exact_a.ptr,r.exact_a.len));}
        else{TRY(r.diagnostic.len==r.exact_a.len&&!memcmp(r.diagnostic.ptr,r.exact_a.ptr,(size_t)r.exact_a.len));ds4_session_snapshot_free(&r.diagnostic);}
        TRY(sync_to(&r,full,C_START));TRY(snapshot_io(&r,ARMS[arm]));double start=now();
        ds4_clm_block_layout layout={CTX,A_END,C_START,C_END};
        if(arm==1||arm==3){ds4_tokens endpoint={full->v,(int)C_END,full->cap};
            TRY(ds4_clm_block_graft(r.s,r.cache,&endpoint,caller(&r,true,full),r.err,sizeof r.err)==0);
        }else{if(arm!=4){const ds4_tokens *t=ds4_session_tokens(r.s);TRY(ds4_clm_block_begin(r.s,&r.a,&r.exact_a,layout,caller(&r,false,NULL)-r.exact_a.cap-(uint64_t)r.a.cap*4u+ds4_clm_block_cache_bytes(r.cache)+(uint64_t)t->cap*4u,r.err,sizeof r.err)==0);}
            for(unsigned i=C_START;i<C_END;i++)TRY(ds4_session_eval(r.s,full->v[i],r.err,sizeof r.err)==0);
        }
        TRY(checkpoint(&r,full,C_END,0));double seconds=now()-start;
        if(arm==0){const ds4_tokens *t=ds4_session_tokens(r.s);TRY(ds4_clm_block_capture(r.s,r.cache,caller(&r,false,NULL)+(uint64_t)t->cap*4u,r.err,sizeof r.err)==0);}
        TRY(budget(&r,"C_endpoint",0));TRY(join(path,sizeof path,arm_path,"c-end.state")&&export_state(&r,path,C_END,arm!=4));
        TRY(join(path,sizeof path,arm_path,"rows.raw"));raw_file=fopen(path,"wbx");TRY(raw_file);unsigned char h[16];p32(h,0x434c5231u);p32(h+4,1u);p32(h+8,VOCAB);p32(h+12,6u);TRY(fwrite(h,1,16,raw_file)==16);
        TRY(row(&r,raw_file,C_END,-1));
        for(unsigned i=C_END;i<C_END+F_COUNT;i++)TRY(ds4_session_eval(r.s,full->v[i],r.err,sizeof r.err)==0);
        TRY(checkpoint(&r,full,C_END+F_COUNT,0));TRY(join(path,sizeof path,arm_path,"f-end.state")&&export_state(&r,path,C_END+F_COUNT,arm!=4));TRY(row(&r,raw_file,C_END+F_COUNT,GOLD[0]));
        for(unsigned i=0;i<4;i++){TRY(ds4_session_eval(r.s,GOLD[i],r.err,sizeof r.err)==0);TRY(checkpoint(&r,full,C_END+F_COUNT,i+1));char name[64];snprintf(name,sizeof name,"teacher%u.state",i+1);TRY(join(path,sizeof path,arm_path,name)&&export_state(&r,path,C_END+F_COUNT+i+1,arm!=4));TRY(row(&r,raw_file,C_END+F_COUNT+i+1,i<3?GOLD[i+1]:-1));}
        int close_rc=fclose(raw_file);raw_file=NULL;TRY(close_rc==0);TRY(snapshot_io(&r,ARMS[arm]));
        if(arm==1)TRY(compare_artifacts(out,ARMS[0],ARMS[1]));
        if(arm==3)TRY(compare_artifacts(out,ARMS[2],ARMS[3]));
        printf("{\"arm\":\"%s\",\"complete\":true,\"C_seconds_diagnostic\":%.9g,\"positions\":[319,447,448,449,450,451],\"evaluated_C\":%u,\"reused_C\":%u,\"evaluated_F\":128,\"teachers\":4,\"sampler_calls\":0,\"RNG_calls\":0,\"approximate_export_block\":%s}\n",ARMS[arm],seconds,arm==1||arm==3?0u:65u,arm==1||arm==3?65u:0u,arm!=4?"true":"false");fflush(stdout);
    }
    printf("{\"status\":\"FIVE_ARMS_NATIVE_HELPER_COMPLETE\",\"same_policy_identity_pairs_byte_exact\":true,\"scoped_budget\":%llu,\"peak_scoped_observed\":%llu,\"peak_scoped_forecast\":%llu,\"global_peak_claim\":false}\n",(unsigned long long)CAP,(unsigned long long)r.peak_observed,(unsigned long long)r.peak_forecast);fflush(stdout);rc=0;
done:
    if(raw_file)fclose(raw_file);if(r.ledger)fclose(r.ledger);if(r.io)fclose(r.io);if(r.s)ds4_session_free(r.s);
    ds4_clm_block_cache_free(r.cache);ds4_session_snapshot_free(&r.exact_a);ds4_session_snapshot_free(&r.diagnostic);
    ds4_tokens *all[]={&r.a,&r.b_old,&r.b_new,&r.c,&r.f,&r.gold,&r.old,&r.changed};for(unsigned i=0;i<8;i++)ds4_tokens_free(all[i]);
    free(r.raw);if(r.e)ds4_engine_close(r.e);return rc;
}
