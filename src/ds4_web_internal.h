#ifndef DS4_WEB_INTERNAL_H
#define DS4_WEB_INTERNAL_H
#include "ds4_web.h"
#include <stdint.h>
#include <sys/types.h>
#include <limits.h>
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
#define DS4_WEB_DEFAULT_PORT 9333
#define DS4_WEB_CONNECT_TIMEOUT_MS 3000
#define DS4_WEB_CDP_TIMEOUT_MS 20000
#define DS4_WEB_MAX_RESULT_BYTES (1024*1024)
/* Actual private browser/connection/tab records, defined once for the C
 * reference and native owner. Browser profile survives owner retirement. */
struct ds4_web {
    char home[PATH_MAX];
    char profile_dir[PATH_MAX];
    int port;
    pid_t chrome_pid;
    bool browser_allowed;
    ds4_web_confirm_fn confirm;
    void *confirm_privdata;
    ds4_web_log_fn log;
    void *log_privdata;
    ds4_web_cancel_fn cancel;
    void *cancel_privdata;
    int next_cdp_id;
};

typedef struct {
    int fd;
    int next_id;
    ds4_web *web;
} cdp_ws;

typedef struct {
    char *id;
    char *ws_url;
} web_tab;

extern const char *const ds4_web_click_google_consent_js;
extern const char *const ds4_web_extract_search_js;
extern const char *const ds4_web_extract_page_js;
extern const char *const ds4_web_probe_js;
extern const char *const ds4_web_scroll_js;
/* The existing local recovery fixture uses these same selected actions. */
bool ds4_web_open_tab(ds4_web *, const char *, web_tab *, char *, size_t);
void ds4_web_close_tab(ds4_web *, const web_tab *);
void ds4_web_tab_free(web_tab *);
#endif
