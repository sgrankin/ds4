/* Standalone libc fixture includes the actual frozen private controls.
 * API bodies below are deterministic engine-less mocks, never native symbols.
 * It proves serialization/admission/temporary ownership and dispatch gates,
 * not real tokenization/graph/RoPE/quality. */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#define PAIR_ACTUAL_BUDGET (UINT64_C(160)<<20)
#define PAIR_ACTUAL_TEXT_CAP (UINT64_C(1)<<20)
#define PAIR_ACTUAL_PIECE_CAP (UINT64_C(64)<<10)
typedef struct {int *v;int len,cap;} ds4_tokens;
typedef struct {unsigned char *ptr;size_t len,cap;} ds4_session_snapshot;
typedef struct {struct {const char *live_context_path;} gen;} mock_cfg;
typedef struct {mock_cfg *cfg;void *engine,*session;ds4_tokens transcript;} agent_worker;
static const char *output;
static unsigned active_allocs,allocations,frees,saves,raw_copies,exports,replaces,imports;
static uint64_t owned_bytes=100;
static bool raw_nan,state_changed,raw_changed,export_fails,replace_fails,import_wrong,approximate;
static uint64_t reported_cap=65;
static int imported_ids[]={3,4};
static void *addresses[32];
static void *track_malloc(size_t bytes){
    void *p=malloc(bytes);if(!p)return NULL;
    for(unsigned i=0;i<32;i++)if(!addresses[i]){addresses[i]=p;active_allocs++;allocations++;return p;}
    abort();
}
static void track_free(void *p){
    if(!p)return;
    for(unsigned i=0;i<32;i++)if(addresses[i]==p){addresses[i]=NULL;assert(active_allocs);active_allocs--;frees++;free(p);return;}
    assert(!"unowned/double free");
}
static uint64_t pair_actual_owned(agent_worker *w){(void)w;return owned_bytes;}
static uint64_t pair_actual_reserve(agent_worker *w){(void)w;return (UINT64_C(8)<<20)+(UINT64_C(128)<<10);}
static bool pair_actual_admit(uint64_t a,uint64_t b,uint64_t c){return a<=PAIR_ACTUAL_BUDGET&&b<=PAIR_ACTUAL_BUDGET-a&&c<=PAIR_ACTUAL_BUDGET-a-b;}
static bool pair_actual_path(char *p,size_t cap,const char *name){int n=snprintf(p,cap,"%s/%s",output,name);return n>0&&(size_t)n<cap;}
static bool pair_actual_write(const char *name,const void *data,size_t n){char p[2048];if(!pair_actual_path(p,sizeof(p),name))return false;FILE *f=fopen(p,"wbx");if(!f)return false;bool ok=fwrite(data,1,n,f)==n;if(fclose(f))ok=false;return ok;}
static bool pair_actual_admission_record(const char *s,bool first,uint64_t a,uint64_t text,uint64_t splice,uint64_t undo,uint64_t checkpoint,uint64_t reserve){(void)s;(void)first;return pair_actual_admit(a,text+splice+undo+checkpoint+1,reserve);}
static uint64_t ds4_session_payload_bytes(void *session){(void)session;return 64;}
static int ds4_session_save_snapshot(void *session,ds4_session_snapshot *s,char *err,size_t errlen){(void)session;(void)err;(void)errlen;saves++;s->ptr=track_malloc(65);if(!s->ptr)return -1;s->len=64;s->cap=(size_t)reported_cap;memset(s->ptr,state_changed?9:7,64);return 0;}
static void ds4_session_snapshot_free(ds4_session_snapshot *s){track_free(s->ptr);memset(s,0,sizeof(*s));}
static int ds4_engine_vocab_size(void *engine){(void)engine;return 129280;}
static int ds4_session_copy_logits(void *session,float *out,int n){(void)session;raw_copies++;assert(active_allocs==1);for(int i=0;i<n;i++)out[i]=(float)(i%13)*.125f;if(raw_changed)out[1]+=1;if(raw_nan){uint32_t bits=UINT32_C(0x7fc00000);memcpy(out+3,&bits,4);}return n;}
static int ds4_session_pos(void *session){(void)session;return 12;}
static int ds4_token_user(void *engine){(void)engine;return 128803;}
static bool ds4_session_is_approximate(void *session){(void)session;return approximate;}
static int agent_read_file_bytes(const char *p,char **out,size_t *len,char *err,size_t errlen){(void)err;(void)errlen;FILE *f=fopen(p,"rb");if(!f)return 1;if(fseek(f,0,SEEK_END)){fclose(f);return 1;}long n=ftell(f);if(n<0||n>1024){fclose(f);return 1;}rewind(f);*out=track_malloc((size_t)n+1);if(!*out){fclose(f);return 1;}*len=(size_t)n;bool ok=fread(*out,1,*len,f)==*len;(*out)[*len]=0;if(fclose(f))ok=false;return ok?0:1;}
static bool agent_worker_export_live_context(agent_worker *w,char *err,size_t errlen){(void)err;(void)errlen;exports++;if(export_fails)return false;FILE *f=fopen(w->cfg->gen.live_context_path,"wb");if(!f)return false;bool ok=fwrite("BEFORE",1,6,f)==6;if(fclose(f))ok=false;return ok;}
static bool agent_replace_file(const char *p,const char *after,size_t n,const char *before,size_t bn,char *err,size_t errlen){(void)err;(void)errlen;replaces++;if(replace_fails)return false;FILE *f=fopen(p,"rb");if(!f)return false;char b[32];size_t got=fread(b,1,sizeof(b),f);fclose(f);if(got!=bn||memcmp(b,before,bn))return false;f=fopen(p,"wb");if(!f)return false;bool ok=fwrite(after,1,n,f)==n;if(fclose(f))ok=false;return ok;}
static int agent_worker_import_live_context(agent_worker *w,char *err,size_t errlen){(void)err;(void)errlen;imports++;assert(active_allocs==0);w->transcript=(ds4_tokens){imported_ids,2,2};const char *arm=getenv("MOCK_ARM");approximate=arm&&!strcmp(arm,"reuse");return import_wrong?0:approximate?2:1;}
/* Native-edit selector dependencies exist solely for type checking; this
 * fixture never exercises tokenizer eligibility and cannot qualify it. */
static bool pair_actual_render(agent_worker *w,char *p,size_t *n){(void)w;(void)p;(void)n;return false;}
static const char *pair_actual_fact_old="fact",*pair_actual_fact_new="new",*pair_actual_req_old="req",*pair_actual_req_new="now";
static const char *pair_actual_delete[16]={"0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f"};
static bool pair_actual_replace(char *p,size_t *n,const char *a,const char *b){(void)p;(void)n;(void)a;(void)b;return false;}
static bool agent_live_context_splice_tokens(agent_worker *w,const char *a,size_t an,const char *b,size_t bn,int first,ds4_tokens *t,char *err,size_t errlen){(void)w;(void)a;(void)an;(void)b;(void)bn;(void)first;(void)t;(void)err;(void)errlen;return false;}
static bool pair_actual_ids(const char *name,const ds4_tokens *t,int user){uint32_t h[]={UINT32_C(0x41535031),1,(uint32_t)t->len,(uint32_t)user};char p[2048];if(!pair_actual_path(p,sizeof(p),name))return false;FILE *f=fopen(p,"wbx");if(!f)return false;bool ok=fwrite(h,sizeof(h),1,f)==1&&fwrite(t->v,sizeof(int),(size_t)t->len,f)==(size_t)t->len;if(fclose(f))ok=false;return ok;}
static void ds4_tokens_free(ds4_tokens *t){track_free(t->v);memset(t,0,sizeof(*t));}
#define malloc track_malloc
#define free track_free
#include "../speed-bench/agent_suffix_pair_external_controls.inc"
#undef malloc
#undef free

static unsigned tests;
static void reset(const char *root,const char *name,char path[2048]){
    assert(!active_allocs);snprintf(path,2048,"%s/%s",root,name);assert(mkdir(path,0700)==0);output=path;
    owned_bytes=100;reported_cap=65;raw_nan=state_changed=raw_changed=export_fails=replace_fails=import_wrong=approximate=false;
}
static void controls(agent_worker *w,const char *root){
    char a[2048],b[2048],p[2048],err[256]={0};reset(root,"preedit-A",a);pair_external_reuse=true;
    assert(pair_external_preedit_controls(w,err,sizeof(err)));assert(!active_allocs);tests++;
    assert(setenv("DS4_AGENT_PAIR_EXTERNAL_REFERENCE",a,1)==0);
    reset(root,"preedit-B",b);pair_external_reuse=false;assert(pair_external_preedit_controls(w,err,sizeof(err)));assert(!active_allocs);tests++;
    reset(root,"raw-mismatch",b);raw_changed=true;assert(!pair_external_preedit_controls(w,err,sizeof(err)));assert(!active_allocs);tests++;
    reset(root,"canonical-mismatch",b);state_changed=true;unsigned before=raw_copies;assert(!pair_external_preedit_controls(w,err,sizeof(err)));assert(raw_copies==before&&!active_allocs);tests++;
    reset(root,"finite-rejection",b);pair_external_reuse=true;raw_nan=true;assert(!pair_external_preedit_controls(w,err,sizeof(err)));assert(!active_allocs);tests++;
    reset(root,"forecast-rejection",b);owned_bytes=PAIR_ACTUAL_BUDGET;before=saves;assert(!pair_external_preedit_controls(w,err,sizeof(err)));assert(saves==before&&!active_allocs);tests++;
    reset(root,"actualcap-rejection",b);reported_cap=PAIR_ACTUAL_BUDGET;assert(!pair_external_preedit_controls(w,err,sizeof(err)));assert(!active_allocs);tests++;
    snprintf(p,sizeof(p),"%s/preedit-state.bin",a);unsigned char state[64];memset(state,7,sizeof(state));assert(pair_external_compare_file(p,state,sizeof(state),NULL,0));assert(!pair_external_compare_file(p,state,63,NULL,0));state[1]=8;assert(!pair_external_compare_file(p,state,sizeof(state),NULL,0));tests+=3;
}
static void commit_setup(agent_worker *w,const char *root,const char *name,char path[2048],char live[2048]){
    reset(root,name,path);assert(pair_actual_write("selected-before.txt","BEFORE",6));assert(pair_actual_write("selected-after.txt","AFTER",5));
    ds4_tokens after={imported_ids,2,2};assert(pair_actual_ids("selected-edited.ids",&after,128803));
    snprintf(live,2048,"%s/live",path);w->cfg->gen.live_context_path=live;pair_external_splice_forecast=512;
}
static void commits(agent_worker *w,const char *root){
    char p[2048],live[2048],err[256]={0};commit_setup(w,root,"commit-A",p,live);pair_external_reuse=true;setenv("MOCK_ARM","reuse",1);unsigned before=imports;
    assert(pair_external_commit(w,err,sizeof(err)));assert(imports==before+1&&approximate&&!active_allocs);tests++;
    commit_setup(w,root,"commit-B",p,live);pair_external_reuse=false;setenv("MOCK_ARM","native",1);before=imports;assert(pair_external_commit(w,err,sizeof(err)));assert(imports==before+1&&!approximate&&!active_allocs);tests++;
    commit_setup(w,root,"export-failure",p,live);export_fails=true;before=imports;assert(!pair_external_commit(w,err,sizeof(err)));assert(imports==before&&!active_allocs);tests++;
    commit_setup(w,root,"replace-failure",p,live);replace_fails=true;before=imports;assert(!pair_external_commit(w,err,sizeof(err)));assert(imports==before&&!active_allocs);tests++;
    commit_setup(w,root,"import-return-failure",p,live);import_wrong=true;assert(!pair_external_commit(w,err,sizeof(err)));assert(!active_allocs);tests++;
    commit_setup(w,root,"import-forecast-failure",p,live);owned_bytes=PAIR_ACTUAL_BUDGET;before=exports;assert(!pair_external_commit(w,err,sizeof(err)));assert(exports==before&&!active_allocs);tests++;
    commit_setup(w,root,"second-read-failure",p,live);snprintf(live,sizeof(live),"%s/selected-after.txt",p);assert(unlink(live)==0);before=imports;assert(!pair_external_commit(w,err,sizeof(err)));assert(imports==before&&!active_allocs);tests++;
}
int main(int argc,char **argv){
    assert(argc==2);assert(mkdir(argv[1],0700)==0);mock_cfg cfg={0};int ids[]={1,2};
    agent_worker w={0};/* Compatible anonymous configuration layout, no native headers. */w.cfg=&cfg;w.transcript=(ds4_tokens){ids,2,2};
    controls(&w,argv[1]);commits(&w,argv[1]);assert(!active_allocs&&allocations==frees);
    printf("external frozen-control mock: %u checks, allocations=%u frees=%u active=0; NO_NATIVE_ENGINE\n",tests,allocations,frees);return 0;
}
