/* Private external-edit controls. Reads immutable native reference bytes;
 * performs no sampling and no additional token evaluation. */
static bool pair_external_reuse;
static uint64_t pair_external_splice_forecast;
static bool pair_external_ref_path(char *out,size_t cap,const char *name){
    const char *root=getenv("DS4_AGENT_PAIR_EXTERNAL_REFERENCE");
    if(!root||!root[0])return false;
    int n=snprintf(out,cap,"%s/%s",root,name);return n>0&&(size_t)n<cap;
}
static bool pair_external_compare_file(const char *path,const void *a,size_t an,const void *b,size_t bn){
    FILE *f=fopen(path,"rb");if(!f)return false;unsigned char chunk[8192];bool ok=true;
    const unsigned char *parts[2]={a,b};size_t sizes[2]={an,bn};
    for(int i=0;i<2&&ok;i++)for(size_t at=0;at<sizes[i];){
        size_t n=sizes[i]-at;if(n>sizeof(chunk))n=sizeof(chunk);
        if(fread(chunk,1,n,f)!=n||memcmp(chunk,parts[i]+at,n)){ok=false;break;}at+=n;
    }
    if(ok)ok=fgetc(f)==EOF&&!ferror(f);if(fclose(f))ok=false;return ok;
}
static bool pair_external_same_ref(const char *name,const void *a,size_t an,const void *b,size_t bn){
    char path[2048];return pair_external_ref_path(path,sizeof(path),name)&&pair_external_compare_file(path,a,an,b,bn);
}
static bool pair_external_ids_equal(const char *path,const ds4_tokens *t,int user){
    uint32_t h[]={UINT32_C(0x41535031),1,(uint32_t)t->len,(uint32_t)user};
    return t->len>=0&&pair_external_compare_file(path,h,sizeof(h),t->v,(size_t)t->len*sizeof(int));
}
static bool pair_external_stage_record(agent_worker *w,const char *stage,bool first,uint64_t owned,uint64_t staged,bool actual){
    char path[2048];if(!pair_actual_path(path,sizeof(path),"control-admission-ledger.jsonl"))return false;
    FILE *f=fopen(path,first?"wx":"a");if(!f)return false;uint64_t reserve=pair_actual_reserve(w);
    bool admitted=pair_actual_admit(owned,staged,reserve);
    bool ok=fprintf(f,"{\"stage\":\"%s\",\"worker_owned_bytes\":%llu,\"staged_bytes\":%llu,\"allocated_caps\":%s,\"reserve_bytes\":%llu,\"budget_bytes\":%llu,\"admitted\":%s,\"observed_global_peak\":false}\n",stage,(unsigned long long)owned,(unsigned long long)staged,actual?"true":"false",(unsigned long long)reserve,(unsigned long long)PAIR_ACTUAL_BUDGET,admitted?"true":"false")>=0;
    if(fclose(f))ok=false;return ok&&admitted;
}
static bool pair_external_preedit_controls(agent_worker *w,char *err,size_t errlen){
    uint64_t owned=pair_actual_owned(w),forecast=ds4_session_payload_bytes(w->session);
    ds4_session_snapshot state={0};float *logits=NULL;bool ok=false;
    if(!forecast||!pair_external_stage_record(w,"canonical_forecast",true,owned,forecast+1,false))goto done;
    if(ds4_session_save_snapshot(w->session,&state,err,errlen)!=0||
       !pair_external_stage_record(w,"canonical_allocated",false,pair_actual_owned(w),state.cap,true))goto done;
    if(!pair_external_reuse&&!pair_external_same_ref("preedit-state.bin",state.ptr,state.len,NULL,0))goto done;
    if(!pair_actual_write("preedit-state.bin",state.ptr,state.len))goto done;
    ds4_session_snapshot_free(&state); /* Never retain this control across the transaction undo. */
    int vocab=ds4_engine_vocab_size(w->engine);if(vocab!=129280)goto done;
    uint64_t bytes=(uint64_t)vocab*sizeof(float);
    if(!pair_external_stage_record(w,"raw_logits_forecast",false,pair_actual_owned(w),bytes,false))goto done;
    logits=malloc((size_t)bytes);if(!logits||ds4_session_copy_logits(w->session,logits,vocab)!=vocab)goto done;
    if(!pair_external_stage_record(w,"raw_logits_allocated",false,pair_actual_owned(w),bytes,true))goto done;
    for(int i=0;i<vocab;i++){uint32_t bits;memcpy(&bits,&logits[i],sizeof(bits));if((bits&UINT32_C(0x7f800000))==UINT32_C(0x7f800000))goto done;}
    uint32_t h[]={UINT32_C(0x45585231),1,(uint32_t)vocab,(uint32_t)ds4_session_pos(w->session)};
    if(!pair_external_reuse&&!pair_external_same_ref("preedit-logits.bin",h,sizeof(h),logits,(size_t)bytes))goto done;
    char path[2048];if(!pair_actual_path(path,sizeof(path),"preedit-logits.bin"))goto done;
    FILE *f=fopen(path,"wbx");if(!f)goto done;
    bool written=fwrite(h,sizeof(h),1,f)==1&&fwrite(logits,1,(size_t)bytes,f)==bytes;
    if(fclose(f))written=false;if(!written)goto done;ok=true;
done:
    free(logits);ds4_session_snapshot_free(&state);
    if(!ok)snprintf(err,errlen,"private canonical/raw preedit parity or admission failed");return ok;
}
static bool pair_external_commit(agent_worker *w,char *err,size_t errlen){
    char before_path[2048],after_path[2048];char *before=NULL,*after=NULL;size_t before_n=0,after_n=0;bool ok=false;
    uint64_t owned=pair_actual_owned(w),undo=ds4_session_payload_bytes(w->session),reserve=pair_actual_reserve(w);
    uint64_t temp=2u*PAIR_ACTUAL_TEXT_CAP+pair_external_splice_forecast+undo+1+(uint64_t)w->transcript.len*sizeof(int);
    if(!undo||!pair_external_stage_record(w,"before_import_forecast",false,owned,temp,false)||!pair_actual_admit(owned,temp,reserve))goto done;
    if(!pair_actual_path(before_path,sizeof(before_path),"selected-before.txt")||
       !pair_actual_path(after_path,sizeof(after_path),"selected-after.txt"))goto done;
    if(agent_read_file_bytes(before_path,&before,&before_n,err,errlen)||
       agent_read_file_bytes(after_path,&after,&after_n,err,errlen)||before_n>=PAIR_ACTUAL_TEXT_CAP||after_n>=PAIR_ACTUAL_TEXT_CAP)goto done;
    /* append_user has not yet mirrored its bytes. Export once, then use the
     * actual protected-prefix import path, with all selector buffers freed. */
    if(!agent_worker_export_live_context(w,err,errlen)||
       agent_replace_file(w->cfg->gen.live_context_path,after,after_n,before,before_n,err,errlen))goto done;
    free(before);before=NULL;free(after);after=NULL;
    int rc=agent_worker_import_live_context(w,err,errlen);
    bool approximate=ds4_session_is_approximate(w->session);
    if(rc!=(pair_external_reuse?2:1)||approximate!=pair_external_reuse)goto done;
    char ids[2048];if(!pair_actual_path(ids,sizeof(ids),"selected-edited.ids")||
       !pair_external_ids_equal(ids,&w->transcript,ds4_token_user(w->engine)))goto done;
    char path[2048];if(!pair_actual_path(path,sizeof(path),"commit.json"))goto done;FILE *f=fopen(path,"wx");if(!f)goto done;
    ok=fprintf(f,"{\"status\":\"EXTERNAL_EDIT_COMMITTED_BEFORE_THIRD_FORWARD\",\"arm\":\"%s\",\"import_returncode\":%d,\"approximate\":%s,\"selector_scratch_freed_before_import\":true,\"before_third_RNG\":true,\"before_third_sampling\":true,\"postimport_ids_exact\":true}\n",pair_external_reuse?"reuse":"native",rc,approximate?"true":"false")>=0;
    if(fclose(f))ok=false;
done:
    free(before);free(after);if(!ok&&!err[0])snprintf(err,errlen,"private external commit parity/reuse gate failed");return ok;
}
static bool pair_external_native_edit(agent_worker *w,char *err,size_t errlen){
    char path[2048];uint32_t h[4];if(!pair_external_ref_path(path,sizeof(path),"choice.bin"))goto failed;
    FILE *f=fopen(path,"rb");if(!f)goto failed;bool read=fread(h,sizeof(h),1,f)==1&&fgetc(f)==EOF&&!ferror(f);if(fclose(f))read=false;
    if(!read||h[0]!=UINT32_C(0x45584331)||h[1]!=1||h[2]>=16||h[3]!=(uint32_t)w->transcript.len)goto failed;
    uint64_t owned=pair_actual_owned(w),reserve=pair_actual_reserve(w);
    if(!pair_actual_admission_record("before_text",true,owned,2u*PAIR_ACTUAL_TEXT_CAP,0,0,0,reserve)||!pair_actual_admit(owned,2u*PAIR_ACTUAL_TEXT_CAP,reserve))goto failed;
    char *old=malloc(PAIR_ACTUAL_TEXT_CAP),*changed=malloc(PAIR_ACTUAL_TEXT_CAP);size_t old_n=0,n=0;ds4_tokens edited={0};bool ok=false;
    if(!old||!changed||!pair_actual_render(w,old,&old_n)||!pair_external_same_ref("selected-before.txt",old,old_n,NULL,0))goto done;
    int user=ds4_token_user(w->engine),first=-1;for(int i=0;i<w->transcript.len;i++)if(w->transcript.v[i]==user){first=i;break;}if(first<0)goto done;
    uint32_t ids_h[]={UINT32_C(0x41535031),1,(uint32_t)w->transcript.len,(uint32_t)user};
    if(!pair_external_same_ref("selected-old.ids",ids_h,sizeof(ids_h),w->transcript.v,(size_t)w->transcript.len*sizeof(int)))goto done;
    /* Reserve the same conservative tokenizer envelope as the A selector,
     * before allocating offsets/middle/token vectors. No native selection. */
    char *start=strstr(old,pair_actual_fact_old),*padding=strstr(old,pair_actual_delete[15]);
    if(!start||!padding||padding<start)goto done;
    uint64_t middle=(uint64_t)(padding+strlen(pair_actual_delete[15])-start)+2u*PAIR_ACTUAL_PIECE_CAP+strlen(pair_actual_req_new)+strlen(pair_actual_fact_new);
    pair_external_splice_forecast=(uint64_t)(w->transcript.len+1)*sizeof(size_t)+middle+2u*middle*sizeof(int)+2u*(middle+(uint64_t)w->transcript.len)*sizeof(int);
    uint64_t undo=ds4_session_payload_bytes(w->session);
    if(!undo||!pair_actual_admission_record("before_splice",false,owned,2u*PAIR_ACTUAL_TEXT_CAP,pair_external_splice_forecast,undo,(uint64_t)w->transcript.len*sizeof(int),reserve)||
       !pair_actual_admit(owned,2u*PAIR_ACTUAL_TEXT_CAP+pair_external_splice_forecast+undo+1+(uint64_t)w->transcript.len*sizeof(int),reserve))goto done;
    memcpy(changed,old,old_n+1);n=old_n;
    if(!pair_actual_replace(changed,&n,pair_actual_fact_old,pair_actual_fact_new)||!pair_actual_replace(changed,&n,pair_actual_req_old,pair_actual_req_new)||!pair_actual_replace(changed,&n,pair_actual_delete[h[2]],"")||
       !agent_live_context_splice_tokens(w,old,old_n,changed,n,first+1,&edited,err,errlen)||!pair_external_same_ref("selected-after.txt",changed,n,NULL,0))goto done;
    if(w->transcript.len-edited.len<=0||!((w->transcript.len-edited.len)&1))goto done;
    uint32_t edited_h[]={UINT32_C(0x41535031),1,(uint32_t)edited.len,(uint32_t)user};
    if(!pair_external_same_ref("selected-edited.ids",edited_h,sizeof(edited_h),edited.v,(size_t)edited.len*sizeof(int)))goto done;
    if(!pair_actual_write("selected-before.txt",old,old_n)||!pair_actual_write("selected-after.txt",changed,n)||!pair_actual_ids("selected-old.ids",&w->transcript,user)||!pair_actual_ids("selected-edited.ids",&edited,user))goto done;
    ok=true;
done:
    ds4_tokens_free(&edited);free(old);free(changed);
    if(ok)return pair_external_commit(w,err,errlen);
failed:
    snprintf(err,errlen,"private native reference/choice/opaque source differs; no alternate edit");return false;
}
