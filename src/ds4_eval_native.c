/* Existing C tests and the real signal-disposition leaves only. */
#include "eval/eval_api.h"
#define find_case_answer ds4_eval_find_case_answer
#define find_case_answer_strict ds4_eval_find_case_answer_strict
#define answer_matches ds4_eval_answer_matches
#define trace_find_next_case ds4_eval_trace_find_next_case
#define trace_copy_model_output ds4_eval_trace_copy_model_output
#define regrade_case_outcome ds4_eval_regrade_case_outcome
#define validate_eval_cases ds4_eval_validate_cases
#include "ds4_eval_selftests.inc"
int ds4_eval_run_extractor_self_tests(void) { return run_extractor_self_tests(); }
void ds4_eval_install_signals(void (*handler)(int)) {
    signal(SIGINT, handler);
    signal(SIGTERM, handler);
#ifdef SIGHUP
    signal(SIGHUP, handler);
#endif
#ifdef SIGQUIT
    signal(SIGQUIT, handler);
#endif
}
void ds4_eval_reraise(int signal_number) {
    signal(signal_number, SIG_DFL);
    raise(signal_number);
}
