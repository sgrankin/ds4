/* Replay the exact prefill/decode boundaries recorded by a real agent session.
 * Tool execution and sampling are excluded; live.py measures those separately. */
#include "ds4.h"
#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef DS4_REPLAY_ROUTE_METRICS
#include "ds4_resident_route.h"

static void phase_metrics(int phase, ds4_resident_route_metrics before,
                          ds4_gpu_stream_expert_io_metrics io_before) {
    ds4_resident_route_metrics after;
    ds4_gpu_stream_expert_io_metrics io_after;
    ds4_resident_route_metrics_read(&after);
    ds4_gpu_stream_expert_io_metrics_read(&io_after);
    for (unsigned layer=0;layer<DS4_RESIDENT_ROUTE_PROFILE_LAYERS;layer++) {
        ds4_resident_route_layer_metrics a=after.layer[layer], b=before.layer[layer];
        if (a.calls==b.calls && a.router_wait_ms==b.router_wait_ms) continue;
        fprintf(stderr,"REPLAY_ROUTE phase=%d layer=%u calls=%llu eligible=%llu snapshots=%llu plans=%llu planned_substitutions=%llu substitutions=%llu router_wait_ms=%.6f selected_read_ms=%.6f lookup_ms=%.6f guard_ms=%.6f snapshot_ms=%.6f plan_ms=%.6f commit_ms=%.6f begin_load_ms=%.6f\n",
            phase,layer,(unsigned long long)(a.calls-b.calls),
            (unsigned long long)(a.eligible-b.eligible),(unsigned long long)(a.snapshots-b.snapshots),
            (unsigned long long)(a.plans-b.plans),(unsigned long long)(a.planned_substitutions-b.planned_substitutions),
            (unsigned long long)(a.substitutions-b.substitutions),
            a.router_wait_ms-b.router_wait_ms,a.selected_read_ms-b.selected_read_ms,
            a.lookup_ms-b.lookup_ms,a.guard_ms-b.guard_ms,a.snapshot_ms-b.snapshot_ms,
            a.plan_ms-b.plan_ms,a.commit_ms-b.commit_ms,a.begin_load_ms-b.begin_load_ms);
    }
    fprintf(stderr,"REPLAY_IO phase=%d explicit_enabled=%d process_valid=%d pread_calls=%llu pread_requested_bytes=%llu pread_returned_bytes=%llu cache_loader_returned_bytes=%llu cache_hits=%llu cache_misses=%llu cache_evictions=%llu process_disk_read_bytes=%llu\n",
        phase,io_after.explicit_io_metrics_enabled,io_before.process_io_valid && io_after.process_io_valid,
        (unsigned long long)(io_after.pread_calls-io_before.pread_calls),
        (unsigned long long)(io_after.pread_requested_bytes-io_before.pread_requested_bytes),
        (unsigned long long)(io_after.pread_returned_bytes-io_before.pread_returned_bytes),
        (unsigned long long)(io_after.cache_loader_returned_bytes-io_before.cache_loader_returned_bytes),
        (unsigned long long)(io_after.cache_hits-io_before.cache_hits),
        (unsigned long long)(io_after.cache_misses-io_before.cache_misses),
        (unsigned long long)(io_after.cache_evictions-io_before.cache_evictions),
        (unsigned long long)(io_after.process_disk_read_bytes-io_before.process_disk_read_bytes));
}
#endif

static double milliseconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"replay: line %d: %s: %s\n",__LINE__,#x,err); goto done; } } while (0)

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr,"usage: %s MODEL REPLAY LOGITS_OUT SNAPSHOT_OUT\n",argv[0]);
        fprintf(stderr,"DS4_REPLAY_TEACHER_DIR: write per-D teacher logits.bin and rows.csv in an existing directory\n"
            "DS4_REPLAY_SKIP_FINAL_SNAPSHOT=1: omit SNAPSHOT_OUT in teacher mode\n");
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
    FILE *teacher_logits=NULL, *teacher_rows=NULL;
    unsigned long long teacher_row=0;
    const char *teacher_dir=getenv("DS4_REPLAY_TEACHER_DIR");
    const char *skip_env=getenv("DS4_REPLAY_SKIP_FINAL_SNAPSHOT");
    const int skip_snapshot=skip_env && !strcmp(skip_env,"1");
    REQUIRE(input);
    REQUIRE(!skip_snapshot || teacher_dir);
    if (teacher_dir) {
        char path[4096];
        REQUIRE(*teacher_dir);
        int length=snprintf(path,sizeof(path),"%s/logits.bin",teacher_dir);
        REQUIRE(length>=0 && (size_t)length<sizeof(path));
        teacher_logits=fopen(path,"wbx");
        REQUIRE(teacher_logits);
        length=snprintf(path,sizeof(path),"%s/rows.csv",teacher_dir);
        REQUIRE(length>=0 && (size_t)length<sizeof(path));
        teacher_rows=fopen(path,"wx");
        REQUIRE(teacher_rows);
        REQUIRE(fputs("index,phase,position,target,vocab\n",teacher_rows)>=0);
        fprintf(stderr,"REPLAY teacher logits enabled; phase times include logit copy and file I/O and are diagnostic only\n");
    }
    uint32_t context=100000;
    const char *context_env=getenv("DS4_REPLAY_CONTEXT");
    if (context_env) {
        char *end=NULL;
        unsigned long rows=strtoul(context_env,&end,10);
        REQUIRE(end!=context_env && !*end && rows>=128 && rows<=100000);
        context=(uint32_t)rows;
    }
    ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_METAL,
        .context_size=context,.ssd_streaming=true,.power_percent=100};
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
    REQUIRE(ds4_session_create(&session,engine,context)==0);
    const int vocab=ds4_engine_vocab_size(engine);
    logits=malloc((size_t)vocab*sizeof(float));
    output=fopen(argv[3],"wb");
    if (!skip_snapshot) state=fopen(argv[4],"wb");
    REQUIRE(logits && output && (skip_snapshot || state));
    puts("phase,kind,context,tokens,ms,first_decode_ms");
    char kind;
    int cached,count;
    while (fscanf(input," %c",&kind)==1) {
        REQUIRE(fscanf(input," %d %d",&cached,&count)==2);
        REQUIRE((kind=='P' || kind=='D') && cached>=0 && count>=0 && (uint32_t)count<=context);
        if (kind=='P') {
            REQUIRE(cached<=history.len && (uint32_t)(cached+count)<=context);
            history.len=cached;
        } else REQUIRE(cached==0 && (uint32_t)(history.len+count)<=context);
        const int start=history.len;
        for (int i=0;i<count;i++) {
            int token;
            REQUIRE(fscanf(input," %d",&token)==1 && token>=0 && token<vocab);
            ds4_tokens_push(&history,token);
        }
        fprintf(stderr,"REPLAY phase=%d kind=%c context=%d tokens=%d start\n",phase,kind,history.len,count);
#ifdef DS4_REPLAY_ROUTE_METRICS
        ds4_resident_route_metrics route_before={0};
        ds4_gpu_stream_expert_io_metrics io_before={0};
        const int metrics=getenv("DS4_REPLAY_ROUTE_METRICS")!=NULL;
        if (metrics) {
            ds4_resident_route_metrics_read(&route_before);
            ds4_gpu_stream_expert_io_metrics_read(&io_before);
        }
#endif
        const double t0=milliseconds();
        double first=0;
        if (kind=='P') {
            REQUIRE(ds4_session_sync(session,&history,err,sizeof(err))==0);
        } else {
            for (int i=0;i<count;i++) {
                if (teacher_logits) {
                    /* These logits predict the recorded token before it enters
                     * KV. Keep original P/D boundaries and all earlier teacher
                     * tokens, so compared policies see identical histories.
                     * Each CSV row indexes one native float32 vocabulary row
                     * (little-endian on the supported Metal host). */
                    REQUIRE(ds4_session_pos(session)==start+i);
                    REQUIRE(ds4_session_copy_logits(session,logits,vocab)==vocab);
                    REQUIRE(fwrite(logits,sizeof(float),(size_t)vocab,teacher_logits)==(size_t)vocab);
                    REQUIRE(fprintf(teacher_rows,"%llu,%d,%d,%d,%d\n",
                        teacher_row,phase,start+i,history.v[start+i],vocab)>0);
                    teacher_row++;
                }
                REQUIRE(ds4_session_eval(session,history.v[start+i],err,sizeof(err))==0);
                if (!i) {
                    first=milliseconds()-t0;
                    if (getenv("DS4_REPLAY_PROFILE_FIRST_DECODE"))
                        fprintf(stderr,"REPLAY phase=%d first_decode_done %.3f ms\n",phase,first);
                }
            }
        }
        const double elapsed=milliseconds()-t0;
#ifdef DS4_REPLAY_ROUTE_METRICS
        if (metrics) phase_metrics(phase,route_before,io_before);
#endif
        fprintf(stderr,"REPLAY phase=%d done %.3f ms\n",phase,elapsed);
        REQUIRE(ds4_session_pos(session)==history.len);
        REQUIRE(ds4_session_copy_logits(session,logits,vocab)==vocab);
        REQUIRE(fwrite(logits,sizeof(float),(size_t)vocab,output)==(size_t)vocab);
        printf("%d,%c,%d,%d,%.3f,%.3f\n",phase,kind,history.len,count,elapsed,first);
        if (getenv("DS4_REPLAY_CACHE_REGIME"))
            fprintf(stderr,"REPLAY_CACHE phase=%d configured_slots=%u resident_entries=%u\n",
                phase,ds4_gpu_stream_expert_cache_configured_count(),
                ds4_gpu_stream_expert_cache_current_count());
        if (getenv("DS4_REPLAY_PROFILE_MEMORY")) {
            char label[64];
            snprintf(label,sizeof(label),"replay phase %d",phase);
            ds4_gpu_print_memory_report(label);
        }
        phase++;
        fflush(stdout);
    }
    REQUIRE(feof(input) && phase>1);
    /* Approximate routing may not publish a canonical snapshot. Quality-only
     * runs can omit that artifact explicitly after recording all teacher rows. */
    if (!skip_snapshot) {
        REQUIRE(ds4_session_save_snapshot(session,&snapshot,err,sizeof(err))==0);
        REQUIRE(fwrite(snapshot.ptr,1,snapshot.len,state)==snapshot.len);
    }
    /* Cumulative cache counters, outside every timed phase. */
    ds4_gpu_print_memory_report("replay final");
    rc=0;
done:
    if (input) fclose(input);
    if (output && fclose(output)!=0) rc=1;
    if (state && fclose(state)!=0) rc=1;
    if (teacher_logits && fclose(teacher_logits)!=0) rc=1;
    if (teacher_rows && fclose(teacher_rows)!=0) rc=1;
    free(logits); ds4_tokens_free(&history);
    ds4_session_snapshot_free(&snapshot);
    ds4_session_free(session); ds4_engine_close(engine);
    return rc;
}
