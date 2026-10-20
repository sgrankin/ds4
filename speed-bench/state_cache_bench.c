/* Real DS4.1 state-cache qualification. Reuses context_frontier_bench.c's
 * Promessi token source and snapshot equality checks. One engine, two mutable
 * sessions; stored native state/pre-RoPE originals never alias either graph.
 * Run through the existing pressure guard. No sampling or approximate source.
 */
#include "ds4.h"
#include "ds4_state_cache.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"state-cache-bench:%d: %s: %s\n", \
    __LINE__,#x,err); goto done; } } while (0)
static double ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1000.0+t.tv_nsec/1e6;
}
static uint64_t hash(const void *data,uint64_t n) {
    const uint8_t *p=data; uint64_t h=UINT64_C(14695981039346656037);
    for(uint64_t i=0;i<n;i++) h=(h^p[i])*UINT64_C(1099511628211);
    return h;
}
static char *read_text(const char *path) {
    FILE *f=fopen(path,"rb"); if(!f)return NULL;
    if(fseek(f,0,SEEK_END)){fclose(f);return NULL;} long n=ftell(f);
    if(n<0||n>64*1024*1024||fseek(f,0,SEEK_SET)){fclose(f);return NULL;}
    char *p=malloc((size_t)n+1); bool ok=p&&fread(p,1,(size_t)n,f)==(size_t)n;
    if(fclose(f))ok=false; if(!ok){free(p);return NULL;} p[n]=0; return p;
}
static int positive(const char *s) {
    char *end; long n=strtol(s,&end,10);
    return end==s||*end||n<1||n>131072?-1:(int)n;
}
int main(int argc,char **argv) {
    const char *model=NULL,*directory=NULL,*input="speed-bench/promessi_sposi.txt";
    const char *profile="ds41-native-prerope-state-cache-bench-v1";
    int ctx=8192,chunk=512,cache_gb=64;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--help")) {
            puts("state_cache_bench --model MODEL --cache-dir EXISTING_PARENT/new-cache "
                 "[--prompt-file FILE] [--profile IDENTIFIER] [--ctx 8192] [--chunk 512] [--cache-gb 64]");
            return 0;
        }
        if(i+1==argc)return 2;
        const char *key=argv[i],*value=argv[++i];
        if(!strcmp(key,"--model"))model=value;
        else if(!strcmp(key,"--cache-dir"))directory=value;
        else if(!strcmp(key,"--prompt-file"))input=value;
        else if(!strcmp(key,"--profile"))profile=value;
        else {
            int n=positive(value); if(n<0)return 2;
            if(!strcmp(key,"--ctx"))ctx=n;
            else if(!strcmp(key,"--chunk"))chunk=n;
            else if(!strcmp(key,"--cache-gb"))cache_gb=n;
            else return 2;
        }
    }
    if(!model||!directory||ctx<=4120||chunk>8192||cache_gb>96||!*profile||strlen(profile)>255)return 2;
    int rc=1; char err[256]={0},path[4096],keys[2][41];
    char *text=read_text(input); ds4_engine *engine=NULL; ds4_session *a=NULL,*b=NULL;
    ds4_tokens corpus={0},prefix={0},lookup={0}; ds4_suffix_cache source[2]={{0}},b_capture={0};
    ds4_session_snapshot observed={0}; ds4_state_cache *store=NULL;
    ds4_state_cache_ref *ref=NULL; ds4_state_cache_blob blob={0};
    static const uint32_t endpoints[2]={2048,4097};
    static const uint64_t hot_budgets[3]={0,UINT64_C(1)<<20,UINT64_C(64)<<20};
    uint64_t state_hash[2]={0},pre_hash[2]={0},full_bytes=0;
    CHECK(text);
    ds4_engine_options opt={.model_path=model,.backend=DS4_BACKEND_METAL,
        .context_size=(uint32_t)ctx,.prefill_chunk=(uint32_t)chunk,.ssd_streaming=true,
        .ssd_streaming_cache_bytes=(uint64_t)cache_gb<<30,.power_percent=100};
    CHECK(ds4_engine_open(&engine,&opt)==0&&ds4_engine_is_deepseek41(engine));
    ds4_tokenize_text(engine,text,&corpus); free(text); text=NULL; CHECK(corpus.len>4120);
    CHECK(ds4_session_create(&a,engine,ctx)==0&&ds4_session_create(&b,engine,ctx)==0);
    CHECK(ds4_session_enable_suffix_prerope(a,err,sizeof(err))==0);
    for(int endpoint=0;endpoint<2;endpoint++) {
        while(prefix.len<(int)endpoints[endpoint])ds4_tokens_push(&prefix,corpus.v[prefix.len]);
        double t=ms(); CHECK(ds4_session_sync(a,&prefix,err,sizeof(err))==0);
        CHECK(ds4_session_capture_suffix_cache(a,&source[endpoint],err,sizeof(err))==0);
        CHECK(!source[endpoint].approximate&&source[endpoint].state.len&&source[endpoint].prerope_bytes);
        state_hash[endpoint]=hash(source[endpoint].state.ptr,source[endpoint].state.len);
        pre_hash[endpoint]=hash(source[endpoint].prerope,source[endpoint].prerope_bytes);
        full_bytes+=source[endpoint].state.len+source[endpoint].prerope_bytes;
        fprintf(stderr,"CAPTURE tokens=%u state_bytes=%llu prerope_bytes=%llu state_fnv=%016llx prerope_fnv=%016llx ms=%.3f\n",
            endpoints[endpoint],(unsigned long long)source[endpoint].state.len,
            (unsigned long long)source[endpoint].prerope_bytes,(unsigned long long)state_hash[endpoint],
            (unsigned long long)pre_hash[endpoint],ms()-t);
    }
    for(int i=0;i<4101;i++)ds4_tokens_push(&lookup,corpus.v[i]);
    puts("hot_limit,budget,endpoint,pass,put_ms,materialize_ms,load_recapture_ms,hot_bytes,ssd_bytes,metadata_bytes,endpoints,chunks,pinned_endpoints,hot_reads,disk_reads,evictions,caller_blob_bytes,state_fnv,prerope_fnv,blob_equal,B_recapture_state_equal,B_recapture_prerope_equal,B_after_A_mutation_equal");
    for(int budget=0;budget<3;budget++) {
        CHECK(snprintf(path,sizeof(path),"%s-hot-%d",directory,budget)<(int)sizeof(path));
        store=ds4_state_cache_open(path,hot_budgets[budget],UINT64_C(512)<<20,err,sizeof(err)); CHECK(store);
        double t=ms();
        for(int endpoint=0;endpoint<2;endpoint++)CHECK(ds4_state_cache_put(store,profile,
            source[endpoint].state.ptr,source[endpoint].state.len,source[endpoint].prerope,
            source[endpoint].prerope_bytes,false,keys[endpoint],err,sizeof(err))==0);
        double put_ms=ms()-t; ds4_state_cache_stats stats={0}; ds4_state_cache_get_stats(store,&stats);
        CHECK(stats.endpoints==2&&stats.ssd_bytes<full_bytes&&stats.hot_bytes<=hot_budgets[budget]);
        fprintf(stderr,"DEDUP budget=%d full_blob_bytes=%llu physical_ssd_bytes=%llu metadata_bytes=%llu\n",
            budget,(unsigned long long)full_bytes,(unsigned long long)stats.ssd_bytes,
            (unsigned long long)stats.metadata_bytes);
        CHECK(ds4_state_cache_close(store,err,sizeof(err))==0); store=NULL;
        store=ds4_state_cache_open(path,hot_budgets[budget],UINT64_C(512)<<20,err,sizeof(err)); CHECK(store);
        uint32_t matched=0;
        ref=ds4_state_cache_find_prefix(store,profile,(uint32_t)ctx,lookup.v,(uint32_t)lookup.len,&matched);
        CHECK(ref&&matched==4097); ds4_state_cache_release(ref); ref=NULL;
        /* First pass starts after reopen, second reuses the same hot LRU. */
        for(int pass=0;pass<2;pass++)for(int endpoint=0;endpoint<2;endpoint++) {
            ref=ds4_state_cache_get(store,keys[endpoint]); CHECK(ref); t=ms();
            CHECK(ds4_state_cache_materialize(ref,&blob,err,sizeof(err))==0); double materialize_ms=ms()-t;
            ds4_state_cache_release(ref); ref=NULL;
            CHECK(!blob.approximate&&blob.tokens==endpoints[endpoint]&&blob.ctx_size==(uint32_t)ctx&&
                blob.state_bytes==source[endpoint].state.len&&blob.prerope_bytes==source[endpoint].prerope_bytes);
            CHECK(!memcmp(blob.state,source[endpoint].state.ptr,(size_t)blob.state_bytes)&&
                  !memcmp(blob.prerope,source[endpoint].prerope,(size_t)blob.prerope_bytes));
            ds4_suffix_cache view={.state={.ptr=blob.state,.len=blob.state_bytes,.cap=blob.state_bytes},
                .engine=engine,.ctx_size=ctx,.prerope=blob.prerope,.prerope_bytes=blob.prerope_bytes}; t=ms();
            CHECK(ds4_session_load_suffix_prefix(b,&view,err,sizeof(err))==0);
            CHECK(ds4_session_capture_suffix_cache(b,&b_capture,err,sizeof(err))==0);
            CHECK(!b_capture.approximate&&b_capture.state.len==source[endpoint].state.len&&
                b_capture.prerope_bytes==source[endpoint].prerope_bytes&&
                !memcmp(b_capture.state.ptr,source[endpoint].state.ptr,(size_t)b_capture.state.len)&&
                !memcmp(b_capture.prerope,source[endpoint].prerope,(size_t)b_capture.prerope_bytes));
            double restore_ms=ms()-t;
            ds4_suffix_cache_free(&b_capture); /* Borrowed view still belongs to blob. */
            /* B's loaded native graph and owned materialization survive A advancing. */
            int position=ds4_session_pos(a); CHECK(position<corpus.len);
            CHECK(ds4_session_eval(a,corpus.v[position],err,sizeof(err))==0);
            CHECK(ds4_session_save_snapshot(b,&observed,err,sizeof(err))==0);
            CHECK(observed.len==blob.state_bytes&&!memcmp(observed.ptr,blob.state,(size_t)blob.state_bytes));
            CHECK(hash(blob.state,blob.state_bytes)==state_hash[endpoint]&&
                  hash(blob.prerope,blob.prerope_bytes)==pre_hash[endpoint]);
            ds4_state_cache_get_stats(store,&stats);
            CHECK(!stats.pinned_endpoints&&stats.hot_bytes<=hot_budgets[budget]);
            printf("%llu,%d,%u,%d,%.6f,%.6f,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%016llx,%016llx,1,1,1,1\n",
                (unsigned long long)hot_budgets[budget],budget,endpoints[endpoint],pass,put_ms,materialize_ms,restore_ms,
                (unsigned long long)stats.hot_bytes,(unsigned long long)stats.ssd_bytes,(unsigned long long)stats.metadata_bytes,
                (unsigned long long)stats.endpoints,(unsigned long long)stats.chunks,(unsigned long long)stats.pinned_endpoints,
                (unsigned long long)stats.hot_reads,(unsigned long long)stats.disk_reads,(unsigned long long)stats.endpoint_evictions,
                (unsigned long long)(blob.state_bytes+blob.prerope_bytes),(unsigned long long)state_hash[endpoint],
                (unsigned long long)pre_hash[endpoint]); fflush(stdout); ds4_state_cache_blob_free(&blob);
        }
        CHECK(ds4_state_cache_close(store,err,sizeof(err))==0); store=NULL;
    }
    rc=0;
done:
    ds4_state_cache_release(ref); ds4_state_cache_blob_free(&blob);
    if(store&&ds4_state_cache_close(store,err,sizeof(err)))rc=1;
    ds4_session_snapshot_free(&observed);
    ds4_suffix_cache_free(&b_capture);
    for(int i=0;i<2;i++)ds4_suffix_cache_free(&source[i]);
    ds4_tokens_free(&corpus); ds4_tokens_free(&prefix); ds4_tokens_free(&lookup); free(text);
    ds4_session_free(a); ds4_session_free(b); ds4_engine_close(engine); return rc;
}
