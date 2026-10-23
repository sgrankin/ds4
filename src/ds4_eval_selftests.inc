/* Original production extractor/trace tests, shared by C and native frontends. */
static int extractor_self_test_case(const char *name, const eval_case *tc,
                                    const char *generated,
                                    const char *expected_extract) {
    char got[EVAL_ANSWER_MAX];
    find_case_answer(tc, generated, got, sizeof(got));
    if (strcmp(got, expected_extract) == 0 && answer_matches(tc, got)) return 0;

    fprintf(stderr,
            "ds4-eval: extractor self-test failed: %s (got %s, expected %s, key %s)\n",
            name, got, expected_extract, tc->answer ? tc->answer : "?");
    return 1;
}

static int extractor_strict_self_test_case(const char *name, const eval_case *tc,
                                           const char *generated,
                                           bool expected_complete,
                                           const char *expected_extract) {
    char got[EVAL_ANSWER_MAX];
    bool complete = find_case_answer_strict(tc, generated, got, sizeof(got));
    bool ok = complete == expected_complete;
    if (expected_complete)
        ok = ok && !strcmp(got, expected_extract) && answer_matches(tc, got);
    if (ok) return 0;

    fprintf(stderr,
            "ds4-eval: strict extractor self-test failed: %s "
            "(complete %d/%d, got %s, expected %s)\n",
            name, complete, expected_complete, got,
            expected_extract ? expected_extract : "-");
    return 1;
}

static int trace_copy_self_test_case(void) {
    const char *prompt_output =
        "Prompt text may mention\n"
        "MODEL_OUTPUT_BEGIN without being the model block.\n";
    const char *model_output =
        "</think>Trace payload may mention\n"
        "MODEL_OUTPUT_END before the real marker.\n"
        "===== CASE not a real case =====\n"
        "Answer: F";
    char trace_case[1024];
    int n = snprintf(trace_case, sizeof(trace_case),
                     "===== CASE 1/2 SuperGPQA/trace-copy-self-test =====\n"
                     "source: SuperGPQA\n"
                     "id: trace-copy-self-test\n"
                     "QUESTION_PROMPT_BEGIN bytes=%zu\n"
                     "%s\n"
                     "QUESTION_PROMPT_END\n"
                     "MODEL_OUTPUT_BEGIN bytes=%zu\n"
                     "%s\n"
                     "MODEL_OUTPUT_END\n"
                     "\n"
                     "===== CASE 2/2 SuperGPQA/trace-copy-self-test-2 =====\n",
                     strlen(prompt_output), prompt_output,
                     strlen(model_output), model_output);
    if (n < 0 || (size_t)n >= sizeof(trace_case)) {
        fprintf(stderr, "ds4-eval: trace self-test setup failed\n");
        return 1;
    }

    const char *trace_start = trace_case;
    const char *trace_end = trace_case + strlen(trace_case);
    const char *first = trace_find_next_case(trace_start, trace_end);
    const char *after_header = first ? memchr(first, '\n', (size_t)(trace_end - first)) : NULL;
    after_header = after_header ? after_header + 1 : trace_end;
    const char *second = first ? trace_find_next_case(after_header, trace_end) : NULL;
    if (!first || !second || !strstr(second, "===== CASE 2/2")) {
        fprintf(stderr, "ds4-eval: trace self-test failed: embedded case marker skip\n");
        return 1;
    }

    char *copied = trace_copy_model_output(first, second);
    if (!copied || strcmp(copied, model_output) != 0) {
        fprintf(stderr, "ds4-eval: trace self-test failed: MODEL_OUTPUT byte copy\n");
        free(copied);
        return 1;
    }
    free(copied);
    return 0;
}

static int regrade_status_self_test_case(void) {
    int failed = 0;
    const eval_case integer = {.source = "AIME2025", .answer = "82"};
    char got[EVAL_ANSWER_MAX];

    if (regrade_case_outcome(&integer, "PASSED", "</think>Answer: 82",
                             got, sizeof(got)) != REGRADE_PASSED) {
        fprintf(stderr, "ds4-eval: regrade self-test failed: PASSED trace not regraded\n");
        failed++;
    }
    /* An interrupted run whose partial output happens to look correct must not
     * be counted or flagged as drift. */
    if (regrade_case_outcome(&integer, "STOPPED", "</think>Answer: 82",
                             got, sizeof(got)) != REGRADE_NOT_GRADED) {
        fprintf(stderr, "ds4-eval: regrade self-test failed: STOPPED trace was graded\n");
        failed++;
    }
    /* Legacy traces without a status line stay gradeable. */
    if (regrade_case_outcome(&integer, "", "</think>Answer: 82",
                             got, sizeof(got)) != REGRADE_PASSED) {
        fprintf(stderr, "ds4-eval: regrade self-test failed: legacy empty status not graded\n");
        failed++;
    }
    return failed;
}

static int run_extractor_self_tests(void) {
    int failed = 0;

    failed += trace_copy_self_test_case();
    failed += regrade_status_self_test_case();

    const eval_case mc = {
        .source = "SuperGPQA",
        .choice[0] = "A",
        .choice[1] = "B",
        .choice[2] = "C",
        .choice[3] = "D",
        .choice[4] = "E",
        .choice[5] = "F",
        .choice[6] = "G",
        .choice[7] = "H",
        .choice[8] = "I",
        .choice[9] = "J",
        .answer = "F",
    };
    failed += extractor_self_test_case(
        "multiple-choice prefers final answer marker",
        &mc,
        "</think>So answer is 0.716 H+/O2. That corresponds to option F.\n"
        "Thus final answer: F.</think>The visible explanation repeats the calculation.\n"
        "Answer: F",
        "F");
    failed += extractor_self_test_case(
        "multiple-choice prefers Answer-colon over later prose",
        &mc,
        "</think>Answer: F\nThis answer is final; option H is a tempting distractor.",
        "F");
    failed += extractor_self_test_case(
        "multiple-choice preserves loose-answer fallback",
        &mc,
        "</think>The answer is F. This answer is final; option H is tempting.",
        "F");
    failed += extractor_strict_self_test_case(
        "multiple-choice strict final line",
        &mc, "</think>Option H is tempting.\nAnswer: F", true, "F");

    const eval_case integer = {
        .source = "AIME2025",
        .answer = "82",
    };
    failed += extractor_self_test_case(
        "integer prefers final answer marker",
        &integer,
        "</think>I first thought the answer was 80.\nFinal answer: 082",
        "82");
    failed += extractor_self_test_case(
        "integer preserves loose-answer fallback",
        &integer,
        "</think>The answer is 082. This answer comes from AIME 2025.",
        "82");

    failed += extractor_strict_self_test_case(
        "strict grading accepts a final answer line",
        &integer,
        "</think>The result is 82.\n**Answer:** 082\n",
        true, "82");
    failed += extractor_strict_self_test_case(
        "strict grading rejects incidental answer prose",
        &integer,
        "</think>The answer is 82, but the generation was truncated.",
        false, NULL);
    failed += extractor_strict_self_test_case(
        "strict grading rejects text after the answer line",
        &integer,
        "</think>Answer: 82\nA later unfinished sentence",
        false, NULL);

    const eval_case rational = {
        .answer = "25/2",
        .alias[0] = "12.5",
        .answer_kind = EVAL_ANSWER_RATIONAL,
    };
    failed += extractor_strict_self_test_case(
        "rational accepts LaTeX fraction",
        &rational, "Reasoning.\nAnswer: \\frac{25}{2}", true, "25/2");
    failed += extractor_strict_self_test_case(
        "rational accepts declared decimal alias",
        &rational, "Reasoning.\nAnswer: 12.5", true, "12.5");

    const eval_case exact = {
        .answer = "2n-2",
        .alias[0] = "2(n-1)",
        .answer_kind = EVAL_ANSWER_EXACT_TEXT,
    };
    failed += extractor_strict_self_test_case(
        "exact text accepts a declared algebraic alias",
        &exact, "Reasoning.\nAnswer: $2 (n - 1)$", true, "2(n-1)");

    const eval_case ordered = {
        .answer = "(1,8,19),(2,7,13),(4,5,7)",
        .alias[0] = "(4,5,7),(2,7,13),(1,8,19)",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
    };
    failed += extractor_strict_self_test_case(
        "ordered sequence ignores presentation brackets",
        &ordered,
        "Reasoning.\nAnswer: [(1, 8, 19), (2, 7, 13), (4, 5, 7)]",
        true, "1,8,19,2,7,13,4,5,7");
    failed += extractor_strict_self_test_case(
        "ordered sequence accepts a declared ordering alias",
        &ordered,
        "Reasoning.\nAnswer: (4,5,7), (2,7,13), (1,8,19)",
        true, "4,5,7,2,7,13,1,8,19");

    const eval_case compsec = {
        .source = "COMPSEC",
        .answer = "9-10",
    };
    failed += extractor_self_test_case(
        "COMPSEC prefers final answer marker",
        &compsec,
        "</think>I think the answer should be line 10, because CWE-122 may apply.\n"
        "**Answer:** 10</think>The primary write is at line 10.\n"
        "Answer: 10",
        "10");
    failed += extractor_strict_self_test_case(
        "security line set strict final line",
        &compsec, "</think>The fault is nearby.\nAnswer: lines 9 and 10",
        true, "9,10");

    /* --- Regression cases for answer-extractor false negatives. These guard
     *     against the grader under-reporting model accuracy on well-formed
     *     final-answer lines. --- */

    /* Multiple choice: a standalone in-range capital that merely begins an
     * English word ("I think", "A careful") or a contraction ("I'll") must
     * not shadow the choice the model actually states later on the line.
     * Only reachable when the stray letter is itself a valid option, i.e.
     * 10-choice (A-J) cases, of which the embedded set has 24. */
    const eval_case mc_c = {
        .source = "SuperGPQA",
        .choice = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"},
        .answer = "C",
    };
    failed += extractor_self_test_case(
        "MC: leading pronoun 'I' does not shadow the chosen letter",
        &mc_c, "</think>Answer: I think it is C", "C");
    failed += extractor_self_test_case(
        "MC: contraction I'll does not shadow the chosen letter",
        &mc_c, "</think>Answer: I'll go with C.", "C");
    failed += extractor_self_test_case(
        "MC: leading article 'A' does not shadow the chosen letter",
        &mc_c, "</think>Answer: A careful reading shows C.", "C");

    const eval_case mc_i = {
        .source = "SuperGPQA",
        .choice = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"},
        .answer = "I",
    };
    failed += extractor_self_test_case(
        "MC: a genuine standalone 'I' answer is still picked",
        &mc_i, "</think>Answer: I.", "I");

    const eval_case mc4_d = {
        .source = "SuperGPQA",
        .choice = {"A", "B", "C", "D"},
        .answer = "D",
    };
    failed += extractor_self_test_case(
        "MC: out-of-range pronoun is harmless on 4-choice cases",
        &mc4_d, "</think>Answer: I think it is D", "D");

    /* A distractor the model explicitly rejects *before* stating its pick on
     * the same line must not shadow the real choice ("not B, ... D" /
     * "rules out C, leaving D"). The bare first-valid-letter rule grabbed the
     * rejected letter; a small, high-precision negation-cue skip fixes it.
     * "D, not B" is the guard against the naive "take the last letter" fix:
     * the pick is reached and accepted before the rejected distractor. #321 */
    const eval_case mc_d = {
        .source = "SuperGPQA",
        .choice = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"},
        .answer = "D",
    };
    failed += extractor_self_test_case(
        "MC: 'not B' distractor before the pick does not shadow it",
        &mc_d, "</think>Answer: It is not B, the answer is D", "D");
    failed += extractor_self_test_case(
        "MC: 'rules out C, leaving D' grades the surviving choice",
        &mc_d, "</think>Answer: rules out C, leaving D", "D");
    failed += extractor_self_test_case(
        "MC: contraction negation (isn't B) before the pick is skipped",
        &mc_d, "</think>Answer: It isn't B, so D", "D");
    failed += extractor_self_test_case(
        "MC: a leading pick followed by 'not B' is still graded as the pick",
        &mc_d, "</think>Answer: D, not B", "D");

    /* Integer: when the answer line shows the arithmetic, the graded value
     * must be the stated result (right of the last '='), not the first
     * summand/factor; digits on later lines must not leak in either. Many
     * embedded AIME2025 cases ask for a derived sum (m+n, a+b+c, ...). */
    const eval_case int_293 = {.source = "AIME2025", .answer = "293"};
    failed += extractor_self_test_case(
        "integer: sum line grades the total, not the first summand",
        &int_293, "</think>Answer: m+n = 256+37 = 293", "293");
    const eval_case int_62 = {.source = "AIME2025", .answer = "62"};
    failed += extractor_self_test_case(
        "integer: three-term sum line grades the total",
        &int_62, "</think>Answer: a+b+c = 12+25+25 = 62", "62");
    const eval_case int_81 = {.source = "AIME2025", .answer = "81"};
    failed += extractor_self_test_case(
        "integer: product-sum line grades the result, not the first factor",
        &int_81, "</think>Answer: 2*7 + 3*6 + 5*4 + 7*5 = 81", "81");
    const eval_case int_82 = {.source = "AIME2025", .answer = "82"};
    failed += extractor_self_test_case(
        "integer: digits on a later line do not override the answer line",
        &int_82, "</think>Answer: 82\nThe value 2025 is just the year.", "82");

    failed += validate_eval_cases(false);
    if (failed) return 1;
    printf("ds4-eval: answer extractor self-tests passed\n");
    return 0;
}

