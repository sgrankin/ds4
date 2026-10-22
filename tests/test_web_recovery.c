/* Exercise the real CDP/HTTP transport against the local fixture server. */
#ifdef DS4_NATIVE_WEB
#include "../ds4_web_internal.h"
#define web_open_tab ds4_web_open_tab
#define web_close_tab ds4_web_close_tab
#define web_tab_free ds4_web_tab_free
#else
#include "../ds4_web.c"
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    assert(argc == 2);
    ds4_web web = {.port = atoi(argv[1])};
    char err[256] = {0};
    web_tab tab;
    assert(web_open_tab(&web, "about:blank", &tab, err, sizeof(err)));
    assert(!strcmp(tab.id, "http-target"));
    web_close_tab(&web, &tab); /* fixture has only one page: keep it */
    web_tab_free(&tab);
    assert(web_open_tab(&web, "about:blank", &tab, err, sizeof(err)));
    assert(!strcmp(tab.id, "cdp-target"));
    web_close_tab(&web, &tab); /* two pages: close the disposable one */
    web_tab_free(&tab);
    assert(!web_open_tab(&web, "about:blank", &tab, err, sizeof(err)));
    assert(strstr(err, "no target id"));
    web_tab_free(&tab);
    puts("Chrome target recovery: ok");
    return 0;
}
