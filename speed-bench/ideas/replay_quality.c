/* The same paired screen on evolving real-agent tool boundaries. */
#define main unused_manifest_probe_main
#include "quality_probe.c"
#undef main

int main(int argc, char **argv) {
    if (argc < 6 || (argc - 6) % 2) {
        fprintf(stderr, "usage: %s MODEL REPLAY SCORES.csv REFERENCE.bin record|compare [--ctx N] [--cache-gb N] [--chunk N]\n", argv[0]);
        return 2;
    }
    uint32_t ctx=8192, cache_gb=64, chunk=512;
    for (int i=6;i<argc;i+=2) {
        long value=positive(argv[i+1]);
        if (!strcmp(argv[i],"--ctx") && value<=131072) ctx=(uint32_t)value;
        else if (!strcmp(argv[i],"--cache-gb") && value<=96) cache_gb=(uint32_t)value;
        else if (!strcmp(argv[i],"--chunk") && value<=8192) chunk=(uint32_t)value;
        else fail("unknown or out-of-range replay option");
    }
    int record = !strcmp(argv[5], "record");
    if (!record && strcmp(argv[5], "compare")) fail("record or compare required");
    FILE *input = fopen(argv[2], "r"), *out = fopen(argv[3], "wx"),
         *ref = fopen(argv[4], record ? "wbx" : "rb");
    if (!input || !out || !ref) { perror("open replay probe"); return 1; }
    ds4_engine_options opt = {.model_path=argv[1], .backend=DS4_BACKEND_METAL,
        .context_size=ctx, .ssd_streaming=true, .power_percent=100,
        .ssd_streaming_cache_bytes=(uint64_t)cache_gb<<30, .prefill_chunk=chunk};
    ds4_engine *engine = NULL; ds4_session *session = NULL; char err[256]={0};
    if (ds4_engine_open(&engine,&opt) || ds4_session_create(&session,engine,ctx))
        fail("replay engine/session initialization failed");
    const int vocab=ds4_engine_vocab_size(engine);
    uint32_t header[2]={UINT32_C(0x44535231),(uint32_t)vocab};
    if (record) { if (fwrite(header,sizeof(header),1,ref)!=1) fail("write header"); }
    else { uint32_t got[2]; if(fread(got,sizeof(got),1,ref)!=1 || memcmp(header,got,sizeof(got))) fail("reference header mismatch"); }
    float *raw=malloc((size_t)vocab*sizeof(float)), *base=malloc((size_t)vocab*sizeof(float));
    if (!raw || !base) fail("replay allocation failed");
    fprintf(out,"case,step,target,prompt_tokens,base_nll,raw_nll,delta_raw_nll,raw_kl,raw_tv,argmax_flip,decode_ms\n");
    ds4_tokens history={0}, target={0};
    char kind; int cached, n, phase=0, count=0, flips=0, equal=0;
    double nll=0,delta=0,kl_total=0,tv_total=0,prefill_ms=0,decode_ms=0;
    while (fscanf(input," %c",&kind)==1) {
        if(fscanf(input," %d %d",&cached,&n)!=2 || n<0 || (kind=='D' && !n) || (uint32_t)n>ctx || cached<0 || (kind!='P' && kind!='D')) fail("bad replay phase");
        target.len=0;
        for(int i=0;i<n;i++){int tok; if(fscanf(input," %d",&tok)!=1 || tok<0 || tok>=vocab)fail("bad replay token");ds4_tokens_push(&target,tok);}
        if(kind=='P') {
            if(cached>history.len || (uint64_t)cached+(uint32_t)n>=ctx)fail("bad replay prefix");
            history.len=cached;
            for(int i=0;i<n;i++)ds4_tokens_push(&history,target.v[i]);
            double t=now_ms();
            if(ds4_session_sync(session,&history,err,sizeof(err)))fail(err);
            double elapsed=now_ms()-t;prefill_ms+=elapsed;
            fprintf(stderr,"REPLAY_QUALITY phase=%d kind=P context=%d ms=%.3f\n",phase,history.len,elapsed);
        } else {
            if(cached || (uint64_t)history.len+(uint32_t)n>=ctx)fail("bad replay decode phase");
            double phase_nll=0,phase_delta=0,phase_kl=0,phase_ms=0;
            for(int i=0;i<n;i++) {
                if(ds4_session_copy_logits(session,raw,vocab)!=vocab)fail("replay logits copy");
                uint32_t key[4]={(uint32_t)phase,(uint32_t)i,(uint32_t)target.v[i],(uint32_t)history.len};
                uint64_t hash=token_hash(&history,&target);
                if(record) {
                    if(fwrite(key,sizeof(key),1,ref)!=1 || fwrite(&hash,sizeof(hash),1,ref)!=1 || fwrite(raw,sizeof(float),vocab,ref)!=(size_t)vocab)fail("write replay record");
                    memcpy(base,raw,(size_t)vocab*sizeof(float));
                } else {
                    uint32_t got[4];uint64_t h;
                    if(fread(got,sizeof(got),1,ref)!=1 || memcmp(got,key,sizeof(key)) || fread(&h,sizeof(h),1,ref)!=1 || h!=hash || fread(base,sizeof(float),vocab,ref)!=(size_t)vocab)fail("replay reference alignment mismatch");
                }
                double bz=log_z(base,vocab),rz=log_z(raw,vocab),bn=bz-base[target.v[i]],rn=rz-raw[target.v[i]],kl,tv;
                distance(base,bz,raw,rz,vocab,&kl,&tv);
                int flip=argmax(base,vocab)!=argmax(raw,vocab);
                equal+=!memcmp(base,raw,(size_t)vocab*sizeof(float));flips+=flip;
                double t=now_ms();if(ds4_session_eval(session,target.v[i],err,sizeof(err)))fail(err);double elapsed=now_ms()-t;
                fprintf(out,"phase_%03d,%d,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%.6f\n",phase,i,target.v[i],history.len,bn,rn,rn-bn,kl,tv,flip,elapsed);
                ds4_tokens_push(&history,target.v[i]);count++;nll+=rn;delta+=rn-bn;kl_total+=kl;tv_total+=tv;decode_ms+=elapsed;
                phase_nll+=rn;phase_delta+=rn-bn;phase_kl+=kl;phase_ms+=elapsed;
            }
            fprintf(stderr,"REPLAY_QUALITY phase=%d kind=D tokens=%d nll=%.6f delta=%.6f kl=%.6g ms=%.3f\n",phase,n,phase_nll/n,phase_delta/n,phase_kl/n,phase_ms);
        }
        phase++;fflush(out);fflush(ref);
    }
    if(!feof(input) || !count || (!record && fgetc(ref)!=EOF))fail("incomplete or mismatched replay");
    printf("{\"phases\":%d,\"tokens\":%d,\"avg_nll\":%.12g,\"delta_nll\":%.12g,\"kl_base_to_raw\":%.12g,\"tv\":%.12g,\"argmax_flips\":%d,\"bit_equal_rows\":%d,\"prefill_ms\":%.3f,\"decode_ms\":%.3f}\n",phase,count,nll/count,delta/count,kl_total/count,tv_total/count,flips,equal,prefill_ms,decode_ms);
    ds4_gpu_print_memory_report("replay quality final");
    free(raw);free(base);ds4_tokens_free(&history);ds4_tokens_free(&target);
    ds4_session_free(session);ds4_engine_close(engine);
    if(fclose(input)||fclose(out)||fclose(ref))fail("close replay outputs");return 0;
}
