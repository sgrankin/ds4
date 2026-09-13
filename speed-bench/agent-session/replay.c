/* Replay the exact prefill/decode boundaries recorded by a real agent session.
 * Tool execution and sampling are excluded; live.py measures those separately. */
#include "ds4.h"
#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double milliseconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"replay: line %d: %s: %s\n",__LINE__,#x,err); goto done; } } while (0)

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr,"usage: %s MODEL REPLAY LOGITS_OUT SNAPSHOT_OUT\n",argv[0]);
        return 2;
    }
    char err[256] = {0};
    int rc=1, phase=0;
    ds4_engine *engine=NULL;
    ds4_session *session=NULL;
    ds4_tokens history={0};
    ds4_session_snapshot snapshot={0};
    float *logits=NULL;
    FILE *input=fopen(argv[2],"r"), *output=NULL, *state=NULL;
    REQUIRE(input);
    ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_METAL,
        .context_size=100000,.ssd_streaming=true,.power_percent=100};
    const char *cache=getenv("DS4_REPLAY_CACHE_GB");
    if (cache) {
        char *end=NULL;
        unsigned long gb=strtoul(cache,&end,10);
        REQUIRE(end!=cache && !*end && gb>=4 && gb<=96);
        opt.ssd_streaming_cache_bytes=(uint64_t)gb<<30;
    }
    const char *chunk=getenv("DS4_REPLAY_PREFILL_CHUNK");
    if (chunk) {
        char *end=NULL;
        unsigned long rows=strtoul(chunk,&end,10);
        REQUIRE(end!=chunk && !*end && rows>=1 && rows<=8192);
        opt.prefill_chunk=(uint32_t)rows;
    }
    REQUIRE(ds4_engine_open(&engine,&opt)==0);
    REQUIRE(ds4_session_create(&session,engine,100000)==0);
    const int vocab=ds4_engine_vocab_size(engine);
    logits=malloc((size_t)vocab*sizeof(float));
    output=fopen(argv[3],"wb"); state=fopen(argv[4],"wb");
    REQUIRE(logits && output && state);
    puts("phase,kind,context,tokens,ms,first_decode_ms");
    char kind;
    int cached,count;
    while (fscanf(input," %c",&kind)==1) {
        REQUIRE(fscanf(input," %d %d",&cached,&count)==2);
        REQUIRE((kind=='P' || kind=='D') && cached>=0 && count>=0 && count<=100000);
        if (kind=='P') {
            REQUIRE(cached<=history.len && cached+count<=100000);
            history.len=cached;
        } else REQUIRE(cached==0 && history.len+count<=100000);
        const int start=history.len;
        for (int i=0;i<count;i++) {
            int token;
            REQUIRE(fscanf(input," %d",&token)==1 && token>=0 && token<vocab);
            ds4_tokens_push(&history,token);
        }
        fprintf(stderr,"REPLAY phase=%d kind=%c context=%d tokens=%d start\n",phase,kind,history.len,count);
        const double t0=milliseconds();
        double first=0;
        if (kind=='P') {
            REQUIRE(ds4_session_sync(session,&history,err,sizeof(err))==0);
        } else {
            for (int i=0;i<count;i++) {
                REQUIRE(ds4_session_eval(session,history.v[start+i],err,sizeof(err))==0);
                if (!i) {
                    first=milliseconds()-t0;
                    if (getenv("DS4_REPLAY_PROFILE_FIRST_DECODE"))
                        fprintf(stderr,"REPLAY phase=%d first_decode_done %.3f ms\n",phase,first);
                }
            }
        }
        const double elapsed=milliseconds()-t0;
        fprintf(stderr,"REPLAY phase=%d done %.3f ms\n",phase,elapsed);
        REQUIRE(ds4_session_pos(session)==history.len);
        REQUIRE(ds4_session_copy_logits(session,logits,vocab)==vocab);
        REQUIRE(fwrite(logits,sizeof(float),(size_t)vocab,output)==(size_t)vocab);
        printf("%d,%c,%d,%d,%.3f,%.3f\n",phase,kind,history.len,count,elapsed,first);
        if (getenv("DS4_REPLAY_PROFILE_MEMORY")) {
            char label[64];
            snprintf(label,sizeof(label),"replay phase %d",phase);
            ds4_gpu_print_memory_report(label);
        }
        phase++;
        fflush(stdout);
    }
    REQUIRE(feof(input) && phase>1);
    REQUIRE(ds4_session_save_snapshot(session,&snapshot,err,sizeof(err))==0);
    REQUIRE(fwrite(snapshot.ptr,1,snapshot.len,state)==snapshot.len);
    rc=0;
done:
    if (input) fclose(input);
    if (output && fclose(output)!=0) rc=1;
    if (state && fclose(state)!=0) rc=1;
    free(logits); ds4_tokens_free(&history);
    ds4_session_snapshot_free(&snapshot);
    ds4_session_free(session); ds4_engine_close(engine);
    return rc;
}
