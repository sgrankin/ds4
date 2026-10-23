#ifndef DS4_SERVER_HTTP_H
#define DS4_SERVER_HTTP_H

#include <stdbool.h>
#include <stddef.h>
#include <signal.h>
#include <sys/types.h>

/* The one-request connection transfers a libc-owned, NUL-terminated body to
 * its caller. body_len counts wire bytes, including any embedded NULs. Method
 * and path retain the server's bounded sscanf grammar; request_free releases
 * the body and clears the whole record. No endpoint or inference state. */
typedef struct {
    char method[8];
    char path[256];
    char *body;
    size_t body_len;
    char *headers;
    size_t headers_len;
    char *prefetched;
    size_t prefetched_len;
} ds4_server_http_request;

/* Sockets are borrowed. The C accept loop retains close/signal/thread ownership.
 * stop_requested points at its existing signal-safe flag for the duration of
 * a send. A stalled client must not block the resident inference coordinator. */
bool ds4_server_http_send_all(int fd, const void *bytes, size_t length,
    const volatile sig_atomic_t *stop_requested);
bool ds4_server_http_response(int fd, bool enable_cors, int status,
    const char *content_type, const char *body,
    const volatile sig_atomic_t *stop_requested);
bool ds4_server_http_sse_headers(int fd, bool enable_cors,
    const volatile sig_atomic_t *stop_requested);
bool ds4_server_http_sse_event(int fd, const char *event, const char *data,
    const volatile sig_atomic_t *stop_requested);

bool ds4_server_http_read_request(int fd, ds4_server_http_request *request);
void ds4_server_http_request_free(ds4_server_http_request *request);
bool ds4_server_http_poll_disconnected(short revents);
bool ds4_server_http_recv_disconnected(int error);
bool ds4_server_http_disconnected(int fd);
int ds4_server_http_listen(const char *host, int port);
void ds4_server_http_configure(int fd);
void ds4_server_http_nonblocking(int fd);

#endif
