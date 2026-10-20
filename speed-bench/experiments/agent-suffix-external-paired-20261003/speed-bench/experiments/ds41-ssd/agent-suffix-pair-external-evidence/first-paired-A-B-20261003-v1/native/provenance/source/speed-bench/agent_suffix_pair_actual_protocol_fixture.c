/* Model-free production-protocol extraction. No native objects or model. */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include "agent_suffix_pair_actual_literals.h"
enum {AGENT_WORKER_IDLE=1,AGENT_WORKER_PREFILL=2};
typedef struct {int state;int prefill_done,prefill_total;unsigned prefill_label;double prefill_tps;int generated;double gen_tps;bool greedy_sampling;} fixture_status;
typedef struct {pthread_mutex_t mu;pthread_cond_t cond;bool initialized,think_requested,progress_direct;char *cmd_text;fixture_status status;bool *session;bool pair_mode;int ctx;} agent_worker;
static unsigned agent_next_prefill_label(void){return 7;}
static bool ds4_session_is_approximate(bool *s){return *s;}
static bool agent_worker_context_pair_mode(agent_worker *w){return w->pair_mode;}
static int agent_worker_effective_ctx_size(agent_worker *w){return w->ctx;}
static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        perror("ds4-agent: malloc");
        exit(1);
    }
    return p;
}
static char *xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}
static void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n ? n : 1);
    if (!p) {
        perror("ds4-agent: realloc");
        exit(1);
    }
    return p;
}
typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} agent_input_buf;
static void agent_input_buf_append(agent_input_buf *b, const char *s, size_t n) {
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}
static char *agent_input_buf_take(agent_input_buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}
static void agent_input_buf_free(agent_input_buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}
static int agent_read_stdin_available(agent_input_buf *in, bool *eof) {
    char buf[4096];
    for (;;) {
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n > 0) {
            agent_input_buf_append(in, buf, (size_t)n);
            continue;
        }
        if (n == 0) {
            *eof = true;
            return 0;
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
        perror("ds4-agent: read stdin");
        return -1;
    }
}
static bool worker_submit(agent_worker *w, const char *text) {
    pthread_mutex_lock(&w->mu);
    bool ok = w->initialized && w->status.state == AGENT_WORKER_IDLE &&
              !w->cmd_text && !w->think_requested;
    if (ok) {
        w->cmd_text = xstrdup(text);
        /* A submitted turn is no longer idle, even if the worker thread has
         * not yet reached its real prefill accounting.  Non-interactive mode
         * depends on this to avoid exiting in the small handoff window between
         * accepting stdin and starting generation. */
        w->status.state = AGENT_WORKER_PREFILL;
        w->progress_direct = false;
        w->status.prefill_done = 0;
        w->status.prefill_total = 0;
        w->status.prefill_label = agent_next_prefill_label();
        w->status.prefill_tps = 0.0;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        w->status.greedy_sampling = false;
        pthread_cond_signal(&w->cond);
    }
    pthread_mutex_unlock(&w->mu);
    return ok;
}
static bool exact_gate(agent_worker *w,const char *user_text,unsigned pair_actual_turn){return !(pair_actual_turn!=3||strcmp(user_text,pair_actual_third_user)||ds4_session_is_approximate(w->session)||
       !agent_worker_context_pair_mode(w)||agent_worker_effective_ctx_size(w)!=8192);}
#define PAIR_ACTUAL_BUDGET (UINT64_C(128)<<20)
static const char *pair_actual_output;
static bool pair_actual_admit(uint64_t owned,uint64_t temporary,uint64_t reserve){
    return owned<=PAIR_ACTUAL_BUDGET&&temporary<=PAIR_ACTUAL_BUDGET-owned&&
        reserve<=PAIR_ACTUAL_BUDGET-owned-temporary;
}
static bool pair_actual_path(char *p,size_t cap,const char *name){
    if(!pair_actual_output||!pair_actual_output[0])return false;
    int n=snprintf(p,cap,"%s/%s",pair_actual_output,name);return n>0&&(size_t)n<cap;
}
#include "agent_suffix_pair_actual_admission.h"
static double seconds(void){struct timeval t;gettimeofday(&t,NULL);return t.tv_sec+t.tv_usec/1000000.;}
static void save(const char *root,const char *label,int turn,const void *bytes,size_t count){char path[4096];assert(snprintf(path,sizeof(path),"%s/%s-%d.bin",root,label,turn)>0);FILE *f=fopen(path,"wbx");assert(f);assert(fwrite(bytes,1,count,f)==count);assert(!fclose(f));}
static void marker(void){assert(write(1,"+DWARFSTAR_WAITING\n",19)==19);}
static void gate_test(void){bool approx=false;agent_worker w={0};w.session=&approx;w.pair_mode=true;w.ctx=8192;
 assert(exact_gate(&w,pair_actual_third_user,3));size_t n=strlen(pair_actual_third_user);assert(n==1799&&pair_actual_third_user[n-1]=='\n');
 char *s=malloc(n+2);assert(s);memcpy(s,pair_actual_third_user,n+1);s[n-1]=0;assert(!exact_gate(&w,s,3));
 memcpy(s,pair_actual_third_user,n);s[n]='\n';s[n+1]=0;assert(!exact_gate(&w,s,3));
 memcpy(s,pair_actual_third_user,n+1);s[17]=0;assert(!exact_gate(&w,s,3));
 memcpy(s,pair_actual_third_user,n+1);s[17]='\n';assert(!exact_gate(&w,s,3));free(s);
 assert(!exact_gate(&w,pair_actual_third_user,2));assert(!exact_gate(&w,pair_actual_third_user,4));
 approx=true;assert(!exact_gate(&w,pair_actual_third_user,3));approx=false;w.pair_mode=false;assert(!exact_gate(&w,pair_actual_third_user,3));w.pair_mode=true;w.ctx=4096;assert(!exact_gate(&w,pair_actual_third_user,3));
 puts("production protocol exact gate: PASS10");}
static void admission_test(const char *path){pair_actual_output=path;
 assert(pair_actual_admission_record("before_text",true,94000000,2097152,0,0,0,8500000));
 assert(pair_actual_admission_record("before_splice",false,94000000,2097152,2000000,30000000,32768,8500000));
 assert(pair_actual_admission_record("inclusive_boundary",false,PAIR_ACTUAL_BUDGET-40,10,0,0,0,30));
 assert(pair_actual_admission_record("one_over",false,PAIR_ACTUAL_BUDGET-39,10,0,0,0,30));
 assert(!pair_actual_admission_record("overflow",false,1,UINT64_MAX,1,0,0,0));puts("selector forecast ledger: PASS5");}
int main(int argc,char **argv){if(argc==3&&!strcmp(argv[1],"--admission-selftest")){admission_test(argv[2]);return 0;}if(argc==2&&!strcmp(argv[1],"--gate-selftest")){gate_test();return 0;}assert(argc==3&&!strcmp(argv[1],"--protocol"));
 int flags=fcntl(0,F_GETFL);assert(flags>=0&&!fcntl(0,F_SETFL,flags|O_NONBLOCK));
 bool approximate=false;agent_worker w={0};pthread_mutex_init(&w.mu,NULL);pthread_cond_init(&w.cond,NULL);w.initialized=true;w.status.state=AGENT_WORKER_IDLE;w.session=&approximate;w.pair_mode=true;w.ctx=8192;
 agent_input_buf input={0};bool eof=false;double deadline=0;int turn=0;marker();
 while(!eof||input.len){struct pollfd p={.fd=0,.events=POLLIN};int ms=input.len?((deadline-seconds())<=0?0:(int)((deadline-seconds())*1000)+1):-1;
 int rc=poll(&p,1,ms);assert(rc>=0);if(p.revents&(POLLIN|POLLHUP)){size_t previous=input.len;assert(!agent_read_stdin_available(&input,&eof));if(input.len!=previous)deadline=seconds()+.200;}
 if(input.len&&(eof||seconds()>=deadline)){turn++;save(argv[2],"received",turn,input.ptr,input.len);char *prompt=agent_input_buf_take(&input);assert(worker_submit(&w,prompt));free(prompt);assert(w.cmd_text&&w.status.state==AGENT_WORKER_PREFILL);save(argv[2],"submitted",turn,w.cmd_text,strlen(w.cmd_text));
 if(turn==3){const char *value=exact_gate(&w,w.cmd_text,3)?"PASS":"FAIL";save(argv[2],"gate",turn,value,strlen(value));}
 free(w.cmd_text);w.cmd_text=NULL;w.status.state=AGENT_WORKER_IDLE;marker();}}
 agent_input_buf_free(&input);pthread_cond_destroy(&w.cond);pthread_mutex_destroy(&w.mu);assert(turn==3);return 0;}
