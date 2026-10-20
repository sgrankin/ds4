/* Private preregistered task observation. No model forward/sampler calls. */
#include "agent_suffix_pair_actual_literals.h"
#define PAIR_ACTUAL_BUDGET (UINT64_C(128)<<20)
#define PAIR_ACTUAL_TEXT_CAP (UINT64_C(1)<<20)
#define PAIR_ACTUAL_PIECE_CAP (UINT64_C(64)<<10)
static unsigned pair_actual_turn;
static const char *pair_actual_output;
#include "agent_suffix_pair_actual_ownership.h"
static uint64_t pair_actual_reserve(agent_worker *w){
    uint64_t ctx=(uint64_t)agent_worker_effective_ctx_size(w);
    return (UINT64_C(8)<<20)+PAIR_ACTUAL_PIECE_CAP+ctx*2u*sizeof(int)+
        sizeof(ds4_tokens)+sizeof(ds4_session_snapshot)+sizeof(ds4_suffix_cache)+sizeof(ds4_suffix_relocation_stats);
}
static bool pair_actual_admit(uint64_t owned,uint64_t temporary,uint64_t reserve){
    return owned<=PAIR_ACTUAL_BUDGET&&temporary<=PAIR_ACTUAL_BUDGET-owned&&
        reserve<=PAIR_ACTUAL_BUDGET-owned-temporary;
}
static bool pair_actual_path(char *p,size_t cap,const char *name){
    if(!pair_actual_output||!pair_actual_output[0])return false;
    int n=snprintf(p,cap,"%s/%s",pair_actual_output,name);return n>0&&(size_t)n<cap;
}
static bool pair_actual_write(const char *name,const void *data,size_t bytes){
    char p[2048];if(!pair_actual_path(p,sizeof(p),name))return false;
    FILE *f=fopen(p,"wbx");if(!f)return false;bool ok=fwrite(data,1,bytes,f)==bytes;if(fclose(f))ok=false;return ok;
}
static bool pair_actual_render(agent_worker *w,char *text,size_t *len){
    *len=0;
    for(int i=0;i<w->transcript.len;i++){
        size_t n=0;char *piece=ds4_token_text(w->engine,w->transcript.v[i],&n);
        bool ok=piece&&n<=PAIR_ACTUAL_PIECE_CAP&&n<PAIR_ACTUAL_TEXT_CAP-*len;
        if(ok){memcpy(text+*len,piece,n);*len+=n;}free(piece);if(!ok)return false;
    }
    text[*len]=0;return true;
}
static bool pair_actual_replace(char *text,size_t *len,const char *old,const char *new_text){
    char *at=strstr(text,old);size_t on=strlen(old),nn=strlen(new_text);
    if(!at||strstr(at+on,old)||nn>PAIR_ACTUAL_TEXT_CAP-*len+on-1)return false;
    size_t tail=*len-(size_t)(at-text)-on;memmove(at+nn,at+on,tail+1);memcpy(at,new_text,nn);*len=*len-on+nn;return true;
}
static bool pair_actual_ids(const char *name,const ds4_tokens *t,int user){
    char p[2048];if(!pair_actual_path(p,sizeof(p),name))return false;FILE *f=fopen(p,"wbx");if(!f)return false;
    uint32_t h[]={UINT32_C(0x41535031),1,(uint32_t)t->len,(uint32_t)user};
    bool ok=fwrite(h,sizeof(h),1,f)==1&&fwrite(t->v,sizeof(int),(size_t)t->len,f)==(size_t)t->len;if(fclose(f))ok=false;return ok;
}
static bool pair_actual_select_after_user(agent_worker *w,const char *user_text,char *err,size_t errlen){
    pair_actual_turn++;pair_actual_output=getenv("DS4_AGENT_PAIR_ACTUAL_OUTPUT");
    if(pair_actual_turn==1){
        if(!pair_actual_write("rounds.jsonl","",0)){snprintf(err,errlen,"private round output unavailable");return false;}
    }
    if(pair_actual_turn<3)return true;
    if(pair_actual_turn!=3||strcmp(user_text,pair_actual_third_user)||ds4_session_is_approximate(w->session)||
       !agent_worker_context_pair_mode(w)||agent_worker_effective_ctx_size(w)!=8192){snprintf(err,errlen,"private preregistered third turn/source differs");return false;}
    uint64_t owned=pair_actual_owned(w),reserve=pair_actual_reserve(w),text_alloc=2u*PAIR_ACTUAL_TEXT_CAP;
    if(!pair_actual_admit(owned,text_alloc,reserve)){snprintf(err,errlen,"private text pre-allocation admission failed");return false;}
    char *old=malloc((size_t)PAIR_ACTUAL_TEXT_CAP),*changed=malloc((size_t)PAIR_ACTUAL_TEXT_CAP);ds4_tokens edited={0};bool ok=false;size_t old_len=0;
    if(!old||!changed||!pair_actual_render(w,old,&old_len))goto done;
    int user=ds4_token_user(w->engine),first_user=-1,latest_user=-1;
    for(int i=0;i<w->transcript.len;i++)if(w->transcript.v[i]==user){if(first_user<0)first_user=i;latest_user=i;}
    if(user<0||first_user<0||latest_user<=first_user||w->transcript.len-latest_user<128)goto done;
    char *first=strstr(old,pair_actual_fact_old),*padding=strstr(old,pair_actual_delete[15]);
    if(!first||!padding||padding<first)goto done;
    uint64_t middle=(uint64_t)(padding+strlen(pair_actual_delete[15])-first)+2u*PAIR_ACTUAL_PIECE_CAP+strlen(pair_actual_req_new)+strlen(pair_actual_fact_new);
    /* At most one token per input byte, doubling vector capacity. Includes
     * offsets, middle text and both retokenized/output vectors. */
    uint64_t splice=(uint64_t)(w->transcript.len+1)*sizeof(size_t)+middle+
        2u*middle*sizeof(int)+2u*(middle+(uint64_t)w->transcript.len)*sizeof(int);
    uint64_t undo=ds4_session_payload_bytes(w->session);
    if(!undo||!pair_actual_admit(owned,text_alloc+splice+undo+1+(uint64_t)w->transcript.len*sizeof(int),reserve))goto done;
    for(int candidate=0;candidate<PAIR_ACTUAL_CANDIDATES;candidate++){
        memcpy(changed,old,old_len+1);size_t len=old_len;char attempt[160]={0};agent_context_suffix_plan plan;
        bool replaced=pair_actual_replace(changed,&len,pair_actual_fact_old,pair_actual_fact_new)&&
            pair_actual_replace(changed,&len,pair_actual_req_old,pair_actual_req_new)&&
            pair_actual_replace(changed,&len,pair_actual_delete[candidate],"");
        bool spliced=replaced&&agent_live_context_splice_tokens(w,old,old_len,changed,len,first_user+1,&edited,attempt,sizeof(attempt));
        int displacement=spliced?w->transcript.len-edited.len:0;
        int selected_latest_user=-1;bool selected=spliced&&edited.cap>=0&&displacement>0&&(displacement&1)&&
            agent_context_suffix_select(w,&edited,128,&plan)&&plan.fresh_tokens>=128;
        uint64_t actual=owned+text_alloc+(uint64_t)edited.cap*sizeof(int)+undo+1;
        if(selected){
            for(int i=edited.len-1;i>=0;i--)if(edited.v[i]==user){selected_latest_user=i;break;}
            selected=selected_latest_user>=0&&plan.destination_end<=(uint32_t)selected_latest_user&&
                agent_context_frontier_pair_covers(&w->context_frontiers[plan.source_frontier],plan.source_start)&&
                pair_actual_admit(actual,(uint64_t)plan.destination_start*sizeof(int),reserve);
        }
        char name[64],path[2048];snprintf(name,sizeof(name),"candidate-%02d.json",candidate);
        if(!pair_actual_path(path,sizeof(path),name))goto done;FILE *f=fopen(path,"wx");if(!f)goto done;
        fprintf(f,"{\"candidate\":%d,\"replaced\":%s,\"spliced\":%s,\"displacement\":%d,\"eligible\":%s}\n",candidate,replaced?"true":"false",spliced?"true":"false",displacement,selected?"true":"false");if(fclose(f))goto done;
        if(selected){
            if(!pair_actual_write("selected-before.txt",old,old_len)||!pair_actual_write("selected-after.txt",changed,len)||
               !pair_actual_ids("selected-old.ids",&w->transcript,user)||!pair_actual_ids("selected-edited.ids",&edited,user)||
               !pair_actual_path(path,sizeof(path),"selection.json"))goto done;
            f=fopen(path,"wx");if(!f)goto done;
            fprintf(f,"{\"status\":\"SELECTED_BEFORE_THIRD_EVAL\",\"candidate\":%d,\"displacement\":%d,\"source_start\":%u,\"source_end\":%u,\"destination_start\":%u,\"destination_end\":%u,\"reused\":%u,\"fresh\":%u,\"latest_user_start\":%d,\"user_id\":%d,\"old_tokens\":%d,\"edited_tokens\":%d,\"budget\":%llu,\"preallocation_peak_with_reserves\":%llu,\"current_payload_forecast\":%llu,\"actual_commit_rechecks_admission\":true,\"old_text_bytes\":%zu,\"edited_text_bytes\":%zu}\n",candidate,displacement,plan.source_start,plan.source_end,plan.destination_start,plan.destination_end,plan.reused_tokens,plan.fresh_tokens,selected_latest_user,user,w->transcript.len,edited.len,(unsigned long long)PAIR_ACTUAL_BUDGET,(unsigned long long)(owned+text_alloc+splice+undo+1+(uint64_t)w->transcript.len*sizeof(int)+reserve),(unsigned long long)undo,old_len,len);
            if(fclose(f))goto done;agent_trace(w,"private task selected candidate=%d displacement=%d before_third_eval=1 latest_user_fresh=1",candidate,displacement);ok=true;break;
        }
        ds4_tokens_free(&edited);
    }
done:
    ds4_tokens_free(&edited);free(old);free(changed);
    if(!ok){const char *failure="No frozen candidate was eligible; no third evaluation or repair.\n";(void)pair_actual_write("selection-failure.txt",failure,strlen(failure));snprintf(err,errlen,"private odd-deletion eligibility failed");}
    return ok;
}
static void pair_actual_json_string(FILE *f,const char *s){
    fputc('"',f);for(const unsigned char *p=(const unsigned char *)(s?s:"");*p;p++){
        if(*p=='"'||*p=='\\'){fputc('\\',f);fputc(*p,f);}else if(*p<32)fprintf(f,"\\u%04x",*p);else fputc(*p,f);
    }fputc('"',f);
}
static void pair_actual_round_observation(agent_worker *w,int tool_round,int generated,int carried,int limit,bool context_limited,bool stopped,bool got_tool,bool malformed,bool early,bool interrupted,const agent_tool_calls *calls){
    char path[2048];if(!pair_actual_path(path,sizeof(path),"rounds.jsonl")){fprintf(stderr,"PAIR_ACTUAL_STATUS_ERROR\n");return;}FILE *f=fopen(path,"a");if(!f){fprintf(stderr,"PAIR_ACTUAL_STATUS_ERROR\n");return;}
    fprintf(f,"{\"turn\":%u,\"tool_round\":%d,\"generated\":%d,\"carried\":%d,\"limit\":%d,\"configured_limit\":%d,\"context_limited\":%s,\"model_stopped\":%s,\"got_tool\":%s,\"malformed_tool\":%s,\"early_tool_error\":%s,\"interrupted\":%s,\"at_limit\":%s,\"calls\":[",pair_actual_turn,tool_round,generated,carried,limit,w->cfg->gen.n_predict,context_limited?"true":"false",stopped?"true":"false",got_tool?"true":"false",malformed?"true":"false",early?"true":"false",interrupted?"true":"false",generated==limit?"true":"false");
    for(int i=0;calls&&i<calls->len;i++){
        if(i)fputc(',',f);fputs("{\"name\":",f);pair_actual_json_string(f,calls->v[i].name);fputs(",\"args\":[",f);
        for(int j=0;j<calls->v[i].argc;j++){if(j)fputc(',',f);agent_tool_arg *a=&calls->v[i].args[j];fputs("{\"name\":",f);pair_actual_json_string(f,a->name);fputs(",\"value\":",f);pair_actual_json_string(f,a->value);fprintf(f,",\"is_string\":%s}",a->is_string?"true":"false");}fputs("]}",f);
    }
    fputs("]}\n",f);if(fclose(f)){fprintf(stderr,"PAIR_ACTUAL_STATUS_ERROR\n");return;}
    fprintf(stderr,"PAIR_ACTUAL_ROUND turn=%u tool_round=%d generated=%d stopped=%d tool=%d at_limit=%d\n",pair_actual_turn,tool_round,generated,stopped,got_tool,generated==limit);fflush(stderr);
}
