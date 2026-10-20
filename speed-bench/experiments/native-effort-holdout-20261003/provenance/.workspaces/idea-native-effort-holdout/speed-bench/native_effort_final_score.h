#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include "overthink_diagnostics.h"
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define PILOT_TEMPERATURE 0.6f

static double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void json_string(FILE *fp, const char *s) {
    fputc('"', fp);
    for (; *s; s++) {
        const unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { fputc('\\', fp); fputc(c, fp); }
        else if (c < 32) fprintf(fp, "\\u%04x", c);
        else fputc(c, fp);
    }
    fputc('"', fp);
}

static int exact_answer(const char *answer, const char *expected) {
    return !strcmp(answer, expected);
}

typedef enum { ANSWER_INTEGER, ANSWER_FRACTION, ANSWER_LIST, ANSWER_INT_PAIR, ANSWER_TEXT } answer_kind;

static char *trim(char *s) {
    while (isspace((unsigned char)*s)) s++;
    size_t n = strlen(s);
    while (n && isspace((unsigned char)s[n - 1])) s[--n] = '\0';
    return s;
}

/* Recognize a complete final answer, never a matching substring. A narrow
 * enclosing wrapper is allowed; extra prose/multiple proposed answers are
 * INVALID and remain available in the raw trace for manual review. */
static char *answer_body(char *s) {
    s = trim(s);
    size_t n = strlen(s);
    if (n >= 6 && !strncmp(s, "```", 3) && !strcmp(s + n - 3, "```")) {
        s[n - 3] = '\0';
        s += 3;
        if (!strncmp(s, "python\n", 7)) s += 7;
        else if (!strncmp(s, "text\n", 5)) s += 5;
        s = trim(s); n = strlen(s);
    }
    if (n >= 4 && !strncmp(s, "**", 2) && !strcmp(s + n - 2, "**")) {
        s[n - 2] = '\0'; s = trim(s + 2); n = strlen(s);
    }
    if (n >= 2 && s[0] == '`' && s[n - 1] == '`') {
        s[n - 1] = '\0'; s = trim(s + 1); n = strlen(s);
    }
    if (n >= 9 && !strncmp(s, "\\boxed{", 7) && s[n - 1] == '}') {
        s[n - 1] = '\0'; s = trim(s + 7);
    }
    return s;
}

static int integer_part(const char **p, long long *value) {
    while (isspace((unsigned char)**p)) (*p)++;
    if (!(isdigit((unsigned char)**p) || **p == '+' || **p == '-')) return 0;
    char *end = NULL;
    errno = 0;
    *value = strtoll(*p, &end, 10);
    if (errno || end == *p) return 0;
    *p = end;
    return 1;
}

static int parse_answer(char *raw, answer_kind kind, char *canonical, size_t cap) {
    const char *s = answer_body(raw);
    if (!*s) return 0;
    long long a = 0, b = 0;
    if (kind == ANSWER_INTEGER || kind == ANSWER_FRACTION || kind == ANSWER_INT_PAIR) {
        if (!integer_part(&s, &a)) return 0;
        if (kind == ANSWER_FRACTION) {
            while (isspace((unsigned char)*s)) s++;
            if (*s++ != '/' || !integer_part(&s, &b) || b == 0) return 0;
        } else if (kind == ANSWER_INT_PAIR) {
            if (!isspace((unsigned char)*s) || !integer_part(&s, &b)) return 0;
        }
        while (isspace((unsigned char)*s)) s++;
        if (*s) return 0;
        if (kind == ANSWER_INTEGER) snprintf(canonical, cap, "%lld", a);
        else if (kind == ANSWER_FRACTION) snprintf(canonical, cap, "%lld/%lld", a, b);
        else snprintf(canonical, cap, "%lld %lld", a, b);
        return 1;
    }
    if (kind == ANSWER_LIST) {
        if (*s++ != '[') return 0;
        size_t used = 1; canonical[0] = '[';
        while (isspace((unsigned char)*s)) s++;
        if (*s != ']') {
            for (;;) {
                if (!integer_part(&s, &a)) return 0;
                char number[32]; snprintf(number, sizeof(number), "%lld", a);
                const size_t n = strlen(number);
                if (used + n + 2 >= cap) return 0;
                memcpy(canonical + used, number, n); used += n;
                while (isspace((unsigned char)*s)) s++;
                if (*s != ',') break;
                s++; canonical[used++] = ',';
            }
        }
        if (*s++ != ']') return 0;
        while (isspace((unsigned char)*s)) s++;
        if (*s || used + 2 > cap) return 0;
        canonical[used++] = ']'; canonical[used] = '\0';
        return 1;
    }
    /* The text task requests one exact printed line. Quotes around that line
     * are accepted, but a multiline explanation is not a parsed answer. */
    size_t n = strlen(s);
    if (n >= 2 && ((s[0] == '"' && s[n - 1] == '"') || (s[0] == '\'' && s[n - 1] == '\''))) { s++; n -= 2; }
    if (n >= cap || memchr(s, '\n', n) || memchr(s, '\r', n)) return 0;
    memcpy(canonical, s, n); canonical[n] = '\0';
    return n != 0;
}

static int score_self_test(void) {
    const struct { const char *raw, *expected; answer_kind kind; int valid; } checks[] = {
        {" 255 \n", "255", ANSWER_INTEGER, 1},
        {"**-4**", "-4", ANSWER_INTEGER, 1},
        {"\\boxed{2/11}", "2/11", ANSWER_FRACTION, 1},
        {"```python\n[-1, -1, 0, 4]\n```", "[-1,-1,0,4]", ANSWER_LIST, 1},
        {"-3 2", "-3 2", ANSWER_INT_PAIR, 1},
        {"`could but error`", "could but error", ANSWER_TEXT, 1},
        {"1255", "1255", ANSWER_INTEGER, 1},
        {"255 or 256", "", ANSWER_INTEGER, 0},
        {"answer: 255", "", ANSWER_INTEGER, 0},
        {"255 trailing", "", ANSWER_INTEGER, 0},
        {"[1,9,25] or [1,9]", "", ANSWER_LIST, 0},
        {"-32", "", ANSWER_INT_PAIR, 0},
        {"2/0", "", ANSWER_FRACTION, 0},
        {"", "", ANSWER_INTEGER, 0}
    };
    for (size_t i = 0; i < sizeof(checks) / sizeof(checks[0]); i++) {
        char raw[256], parsed[256];
        snprintf(raw, sizeof(raw), "%s", checks[i].raw);
        const int valid = parse_answer(raw, checks[i].kind, parsed, sizeof(parsed));
        if (valid != checks[i].valid || (valid && !exact_answer(parsed, checks[i].expected))) {
            fprintf(stderr, "score self-test failed at %zu\n", i); return 1;
        }
    }
    if (exact_answer("couldbuterror", "could but error") ||
        exact_answer("could  but error", "could but error") ||
        exact_answer("-32", "-3 2")) {
        fputs("score self-test failed: exact canonical comparison\n", stderr);
        return 1;
    }
    puts("strict final-answer scorer: 17 checks passed");
    return 0;
}

