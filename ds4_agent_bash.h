#ifndef DS4_AGENT_BASH_H
#define DS4_AGENT_BASH_H

#include <stdbool.h>
#include <stddef.h>
#include <limits.h>
#include <sys/types.h>

typedef struct ds4_agent_bash ds4_agent_bash;
typedef void (*ds4_agent_bash_terminal_fn)(void *userdata);

enum {
    DS4_AGENT_BASH_HEAD_BYTES = 8 * 1024,
    DS4_AGENT_BASH_HEAD_LINES = 100,
    DS4_AGENT_BASH_TAIL_BYTES = 32 * 1024,
    DS4_AGENT_BASH_PROGRESS_TAIL_LINES = 4,
    DS4_AGENT_BASH_FINAL_TAIL_LINES = 20,
};

/* Immutable observation fields copied under the job mutex. Disk reads use this
 * value after unlocking, so slow files cannot delay timeout/reaping checks.
 * This is the actual shared observation record, not a job/worker layout. */
typedef struct {
    int id;
    pid_t pid;
    char path[PATH_MAX];
    double start_time, end_time, timeout_sec;
    size_t bytes;
    int newline_count;
    char last_byte;
    bool observed_once;
    int exit_status;
    bool running, timed_out;
    int output_error;
} ds4_agent_bash_snapshot;

/* Job owns its monitor, mutex, process group and descriptors. C owns its list
 * node/IDs/UI; the sole completion action asks the UI to restore terminal state
 * at a safe point. next_id is incremented after spawn, before monitor creation,
 * preserving the old failure timing. Returned output strings are libc-owned. */
ds4_agent_bash *ds4_agent_bash_start(const char *command, int timeout_sec,
    int *next_id, ds4_agent_bash_terminal_fn terminal_changed, void *userdata,
    char *err, size_t errlen);
void ds4_agent_bash_free(ds4_agent_bash *job);
bool ds4_agent_bash_running(ds4_agent_bash *job);
void ds4_agent_bash_signal(ds4_agent_bash *job, int signal);
void ds4_agent_bash_get_snapshot(ds4_agent_bash *job, bool mark_observed,
    ds4_agent_bash_snapshot *out);
char *ds4_agent_bash_observation(ds4_agent_bash *job, bool mark_observed,
    bool *done);

#endif
