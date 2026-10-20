/* Genuine stale-suffix graft oracle: prefix edit is recomputed, C is skipped.
 * Metrics use untouched model logits and double-precision normalization. */
#include "ds4.h"
#include "suffix_snapshot_compare.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"suffix-oracle:%d: %s: %s\n",__LINE__,#x,err); goto done; } } while (0)
static double ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1000.0+t.tv_nsec/1e6; }
static char *read_text(const char *p) {
    FILE *f=fopen(p,"rb"); if(!f)return NULL;
    if(fseek(f,0,SEEK_END)){fclose(f);return NULL;} long n=ftell(f);
    if(n<0||n>16*1024*1024||fseek(f,0,SEEK_SET)){fclose(f);return NULL;}
    char *s=malloc((size_t)n+1); if(!s){fclose(f);return NULL;}
    bool ok=fread(s,1,(size_t)n,f)==(size_t)n; if(fclose(f))ok=false;
    if(!ok){free(s);return NULL;}s[n]=0;return s;
}
static int number(const char *s) {char *e;long n=strtol(s,&e,10);return e==s||*e||n<1||n>131072?-1:(int)n;}
static void append(ds4_tokens *a,const ds4_tokens *b,int at,int n) {for(int i=0;i<n;i++)ds4_tokens_push(a,b->v[at+i]);}
static double log_z(const float *z,int n) {
    double mx=-INFINITY,total=0;for(int i=0;i<n;i++){if(!isfinite(z[i]))return NAN;if(z[i]>mx)mx=z[i];}
    for(int i=0;i<n;i++)total+=exp((double)z[i]-mx);return mx+log(total);
}
static int argmax(const float *z,int n) {int b=0;for(int i=1;i<n;i++)if(z[i]>z[b])b=i;return b;}
static void metrics(const float *p,const float *q,int n,int target,double *kl,double *tv,double *dnll,double *max_abs) {
    double lp=log_z(p,n),lq=log_z(q,n);*kl=*tv=*max_abs=0;*dnll=target>=0?(lq-q[target])-(lp-p[target]):0;
    for(int i=0;i<n;i++){double a=exp((double)p[i]-lp),b=exp((double)q[i]-lq);*kl+=a*(((double)p[i]-lp)-((double)q[i]-lq));*tv+=fabs(a-b)*.5;double d=fabs((double)p[i]-q[i]);if(d>*max_abs)*max_abs=d;}
    if(*kl<0&&*kl>-1e-10)*kl=0;
}
int main(int argc,char **argv) {
    if(argc==2&&!strcmp(argv[1],"--self-test")){
        const float p[3]={logf(.6f),logf(.3f),logf(.1f)},q[3]={logf(.3f),logf(.6f),logf(.1f)},shift[3]={p[0]+4,p[1]+4,p[2]+4};double kl,tv,dnll,mx;
        metrics(p,q,3,0,&kl,&tv,&dnll,&mx);if(fabs(kl-.3*log(2.0))>1e-7||fabs(tv-.3)>1e-7||fabs(dnll-log(2.0))>1e-7)return 1;
        metrics(p,shift,3,-1,&kl,&tv,&dnll,&mx);if(fabs(kl)>1e-10||tv>1e-7||dnll!=0)return 1;
        puts("suffix-oracle metric arithmetic: PASS; no model opened");return 0;
    }
    const char *model=NULL,*source="speed-bench/promessi_sposi.txt",*old_path=NULL,*new_path=NULL,*target_path=NULL,*raw_path=NULL,*summary_path=NULL;
    int prefix_n=1024,edit_n=128,unchanged_n=512,tail_n=1,decode=8,ctx=4096,chunk=512,cache_gb=64;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--help")){puts("usage: suffix_reuse_bench --model MODEL [--prompt-file CORPUS | --old-tokens FILE --new-tokens FILE] [--target-file TEXT] [--prefix N] [--edit N] [--unchanged N] [--fresh-tail N] [--decode N] [--ctx N] [--chunk N] [--cache-gb N] [--raw FILE] [--summary FILE]");return 0;}
        if(i+1==argc)return 2;const char *o=argv[i],*v=argv[++i];
        if(!strcmp(o,"--model"))model=v;else if(!strcmp(o,"--prompt-file"))source=v;
        else if(!strcmp(o,"--old-tokens"))old_path=v;else if(!strcmp(o,"--new-tokens"))new_path=v;
        else if(!strcmp(o,"--target-file"))target_path=v;else if(!strcmp(o,"--raw"))raw_path=v;else if(!strcmp(o,"--summary"))summary_path=v;
        else {int n=number(v);if(n<0)return 2;if(!strcmp(o,"--prefix"))prefix_n=n;else if(!strcmp(o,"--edit"))edit_n=n;else if(!strcmp(o,"--unchanged"))unchanged_n=n;else if(!strcmp(o,"--fresh-tail"))tail_n=n;else if(!strcmp(o,"--decode"))decode=n;else if(!strcmp(o,"--ctx"))ctx=n;else if(!strcmp(o,"--chunk"))chunk=n;else if(!strcmp(o,"--cache-gb"))cache_gb=n;else return 2;}
    }
    int end=prefix_n+edit_n+unchanged_n-tail_n,total=prefix_n+edit_n+unchanged_n,start=prefix_n+edit_n;
    if(!model||!!old_path!=!!new_path||(start&1)||tail_n>=unchanged_n||total+decode>=ctx||decode>128||chunk>8192||cache_gb>96)return 2;
    int rc=1;char err[256]={0};char *text=NULL;ds4_engine *e=NULL;ds4_session *s=NULL;
    ds4_tokens corpus={0},old={0},edited={0},a={0},ab={0},fresh_end={0},target={0};
    ds4_session_snapshot a_state={0},ab_state={0},old_final={0},observed={0};ds4_suffix_cache cache={0};
    float *reference=NULL,*row=NULL;FILE *raw=NULL,*summary=NULL;double old_ms=0,identity_ms=0,reference_ms=0,candidate_ms=0;
    ds4_suffix_reuse_stats stats={0};int identity_logit_equal=0,identity_state_equal=0,flips=0,bit_rows=0;double sum_kl=0,sum_tv=0,sum_dnll=0;
    ds4_engine_options opt={.model_path=model,.backend=DS4_BACKEND_METAL,.context_size=ctx,.prefill_chunk=(uint32_t)chunk,.ssd_streaming=true,.ssd_streaming_cache_bytes=(uint64_t)cache_gb<<30,.power_percent=100};
    CHECK(ds4_engine_open(&e,&opt)==0);CHECK(ds4_session_create(&s,e,ctx)==0);int vocab=ds4_engine_vocab_size(e);size_t bytes=(size_t)vocab*sizeof(float);
    if(old_path){
        const char *paths[2]={old_path,new_path};ds4_tokens *dest[2]={&old,&edited};
        for(int k=0;k<2;k++){FILE *f=fopen(paths[k],"r");CHECK(f);int tok,got;bool valid=true;while((got=fscanf(f," %d",&tok))==1){if(tok<0||tok>=vocab||dest[k]->len>=ctx){valid=false;break;}ds4_tokens_push(dest[k],tok);}valid=valid&&got==EOF&&!ferror(f);if(fclose(f))valid=false;CHECK(valid);}
        CHECK(old.len==total&&edited.len==total);
        for(int i=0;i<prefix_n;i++)CHECK(old.v[i]==edited.v[i]);for(int i=start;i<total;i++)CHECK(old.v[i]==edited.v[i]);
    }else{
        text=read_text(source);CHECK(text);ds4_tokenize_text(e,text,&corpus);free(text);text=NULL;CHECK(corpus.len>total+edit_n+decode);
        append(&old,&corpus,0,total);append(&edited,&corpus,0,prefix_n);append(&edited,&corpus,total,edit_n);append(&edited,&corpus,start,unchanged_n);
    }
    append(&a,&old,0,prefix_n);append(&ab,&old,0,start);append(&fresh_end,&old,0,end);
    if(target_path){text=read_text(target_path);CHECK(text);ds4_tokenize_text(e,text,&target);free(text);text=NULL;CHECK(target.len>0);if(decode>target.len)decode=target.len;}
    else {if(!corpus.len){text=read_text(source);CHECK(text);ds4_tokenize_text(e,text,&corpus);free(text);text=NULL;}CHECK(corpus.len>total+decode);append(&target,&corpus,total,decode);}
    reference=malloc(bytes*(size_t)(decode+1));row=malloc(bytes);CHECK(reference&&row);
    if(raw_path){raw=fopen(raw_path,"wbx");CHECK(raw);uint32_t h[4]={UINT32_C(0x53554631),(uint32_t)vocab,(uint32_t)(decode+1),(uint32_t)target.len};CHECK(fwrite(h,sizeof(h),1,raw)==1);CHECK(fwrite(target.v,sizeof(int),(size_t)target.len,raw)==(size_t)target.len);}
    double t=ms();CHECK(ds4_session_sync(s,&a,err,sizeof(err))==0);CHECK(ds4_session_save_snapshot(s,&a_state,err,sizeof(err))==0);CHECK(ds4_session_sync(s,&ab,err,sizeof(err))==0);CHECK(ds4_session_save_snapshot(s,&ab_state,err,sizeof(err))==0);CHECK(ds4_session_sync(s,&fresh_end,err,sizeof(err))==0);CHECK(ds4_session_capture_suffix_cache(s,&cache,err,sizeof(err))==0);CHECK(ds4_session_sync(s,&old,err,sizeof(err))==0);old_ms=ms()-t;
    CHECK(ds4_session_copy_logits(s,row,vocab)==vocab);memcpy(reference,row,bytes);CHECK(ds4_session_save_snapshot(s,&old_final,err,sizeof(err))==0);
    CHECK(ds4_session_load_snapshot(s,&ab_state,err,sizeof(err))==0);
    /* Candidate rejection must not silently damage the still-exact prefix. */
    CHECK(ds4_session_reuse_suffix(s,&cache,&old,(uint32_t)start+1,&stats,err,sizeof(err))!=0);
    int full_len=old.len;old.len=end;CHECK(ds4_session_reuse_suffix(s,&cache,&old,(uint32_t)start,&stats,err,sizeof(err))!=0);old.len=full_len;
    int saved_token=old.v[start];old.v[start]=(saved_token+1)%vocab;CHECK(ds4_session_reuse_suffix(s,&cache,&old,(uint32_t)start,&stats,err,sizeof(err))!=0);old.v[start]=saved_token;
    cache.state.ptr[0]^=1;CHECK(ds4_session_reuse_suffix(s,&cache,&old,(uint32_t)start,&stats,err,sizeof(err))!=0);cache.state.ptr[0]^=1;
    cache.ctx_size++;CHECK(ds4_session_reuse_suffix(s,&cache,&old,(uint32_t)start,&stats,err,sizeof(err))!=0);cache.ctx_size--;
    CHECK(ds4_session_save_snapshot(s,&observed,err,sizeof(err))==0);CHECK(observed.len==ab_state.len&&!memcmp(observed.ptr,ab_state.ptr,(size_t)ab_state.len));
    err[0]=0;t=ms();CHECK(ds4_session_reuse_suffix(s,&cache,&old,(uint32_t)start,&stats,err,sizeof(err))==0);identity_ms=ms()-t;CHECK(ds4_session_copy_logits(s,row,vocab)==vocab);identity_logit_equal=!memcmp(reference,row,bytes);CHECK(ds4_session_suffix_diagnostic_snapshot(s,&observed,err,sizeof(err))==0);identity_state_equal=suffix_diagnostic_equal_canonical(&observed,&old_final);CHECK(identity_logit_equal&&identity_state_equal);CHECK(ds4_session_is_approximate(s));CHECK(ds4_session_payload_bytes(s)==0);CHECK(ds4_session_save_snapshot(s,&observed,err,sizeof(err))!=0);
    CHECK(ds4_session_load_snapshot(s,&a_state,err,sizeof(err))==0);CHECK(!ds4_session_is_approximate(s));
    ds4_tokens_free(&ab);append(&ab,&edited,0,start);ds4_tokens_free(&fresh_end);append(&fresh_end,&edited,0,end);
    t=ms();CHECK(ds4_session_sync(s,&ab,err,sizeof(err))==0);CHECK(ds4_session_sync(s,&fresh_end,err,sizeof(err))==0);CHECK(ds4_session_sync(s,&edited,err,sizeof(err))==0);reference_ms=ms()-t;
    for(int j=0;j<=decode;j++){CHECK(ds4_session_copy_logits(s,reference+(size_t)j*vocab,vocab)==vocab);if(j<decode)CHECK(ds4_session_eval(s,target.v[j],err,sizeof(err))==0);}if(raw)CHECK(fwrite(reference,bytes,(size_t)(decode+1),raw)==(size_t)(decode+1));
    CHECK(ds4_session_load_snapshot(s,&a_state,err,sizeof(err))==0);t=ms();CHECK(ds4_session_sync(s,&ab,err,sizeof(err))==0);CHECK(ds4_session_reuse_suffix(s,&cache,&edited,(uint32_t)start,&stats,err,sizeof(err))==0);candidate_ms=ms()-t;
    puts("step,target,base_nll,reuse_nll,delta_nll,kl,tv,argmax_flip,bit_equal,max_abs");
    for(int j=0;j<=decode;j++){
        CHECK(ds4_session_copy_logits(s,row,vocab)==vocab);const float *ref=reference+(size_t)j*vocab;int tok=j<decode?target.v[j]:-1;double kl,tv,dnll,max_abs;metrics(ref,row,vocab,tok,&kl,&tv,&dnll,&max_abs);CHECK(isfinite(kl)&&isfinite(tv)&&isfinite(dnll));int flip=argmax(ref,vocab)!=argmax(row,vocab),equal=!memcmp(ref,row,bytes);flips+=flip;bit_rows+=equal;sum_kl+=kl;sum_tv+=tv;if(j<decode)sum_dnll+=dnll;
        printf("%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%d,%.9g\n",j,tok,tok>=0?log_z(ref,vocab)-ref[tok]:NAN,tok>=0?log_z(row,vocab)-row[tok]:NAN,dnll,kl,tv,flip,equal,max_abs);
        if(raw)CHECK(fwrite(row,bytes,1,raw)==1);if(j<decode)CHECK(ds4_session_eval(s,target.v[j],err,sizeof(err))==0);
    }
    CHECK(ds4_session_is_approximate(s)&&ds4_session_payload_bytes(s)==0);ds4_session_invalidate(s);CHECK(!ds4_session_is_approximate(s));
    if(summary_path){summary=fopen(summary_path,"wx");CHECK(summary);fprintf(summary,"{\n  \"identity_logit_bytes_equal\": true,\n  \"identity_graph_bytes_equal\": true,\n  \"canonical_export_blocked\": true,\n  \"prefix_tokens\": %d, \"edit_tokens\": %d, \"unchanged_tokens\": %d,\n  \"reused_tokens\": %u, \"fresh_tail_tokens\": %u, \"continuation_tokens\": %d,\n  \"copied_bytes\": %llu, \"pending_pairs\": %u,\n  \"old_setup_ms\": %.6f, \"identity_reuse_ms\": %.6f,\n  \"fresh_edit_ms\": %.6f, \"approx_edit_ms\": %.6f,\n  \"mean_kl\": %.12g, \"mean_tv\": %.12g, \"sum_delta_nll\": %.12g,\n  \"argmax_flips\": %d, \"bit_equal_rows\": %d, \"rows\": %d\n}\n",prefix_n,edit_n,unchanged_n,stats.reused_tokens,stats.fresh_tail_tokens,decode,(unsigned long long)stats.copied_bytes,stats.pending_pairs,old_ms,identity_ms,reference_ms,candidate_ms,sum_kl/(decode+1),sum_tv/(decode+1),sum_dnll,flips,bit_rows,decode+1);}
    fprintf(stderr,"SUFFIX prefix=%d edit=%d reused=%u fresh_tail=%u snapshot=%llu copied=%llu identity_logits=%d identity_state=%d fresh_ms=%.3f approximate_ms=%.3f mean_kl=%.9g mean_tv=%.9g delta_nll=%.9g\n",prefix_n,edit_n,stats.reused_tokens,stats.fresh_tail_tokens,(unsigned long long)cache.state.len,(unsigned long long)stats.copied_bytes,identity_logit_equal,identity_state_equal,reference_ms,candidate_ms,sum_kl/(decode+1),sum_tv/(decode+1),sum_dnll);rc=0;
done:
    if(raw&&fclose(raw))rc=1;if(summary&&fclose(summary))rc=1;free(text);free(reference);free(row);ds4_tokens_free(&corpus);ds4_tokens_free(&old);ds4_tokens_free(&edited);ds4_tokens_free(&a);ds4_tokens_free(&ab);ds4_tokens_free(&fresh_end);ds4_tokens_free(&target);ds4_session_snapshot_free(&a_state);ds4_session_snapshot_free(&ab_state);ds4_session_snapshot_free(&old_final);ds4_session_snapshot_free(&observed);ds4_suffix_cache_free(&cache);ds4_session_free(s);ds4_engine_close(e);return rc;
}
