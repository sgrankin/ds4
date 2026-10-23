#include "../src/ds4_suffix_match.h"
#include <assert.h>
#include <stdio.h>

static void unique_source(int *source, unsigned n) {
    for (unsigned i = 0; i < n; i++) source[i] = (int)i;
}

static unsigned append(int *edited, unsigned at, const int *source,
                       unsigned first, unsigned end) {
    memcpy(edited + at, source + first, (end - first) * sizeof(*source));
    return at + end - first;
}

static void prefix(int *edited, unsigned n) {
    for (unsigned i = 0; i < n; i++) edited[i] = -1000 - (int)i;
}

int main(void) {
    int source[200], source_before[200], edited[320], edited_before[320];
    ds4_suffix_match_plan plan;
    unique_source(source, 200);
    memcpy(source_before, source, sizeof(source));

    /* Ordinary even shortening; the complete latest question is fresh. */
    prefix(edited, 20);
    unsigned end = append(edited, 20, source, 80, 200);
    for (unsigned i = end; i < end + 32; i++) edited[i] = -2000 - (int)i;
    memcpy(edited_before, edited, (end + 32) * sizeof(*edited));
    assert(ds4_suffix_match(source, 200, edited, end + 32, end, &plan));
    assert(plan.source_start == 80 && plan.destination_start == 20);
    assert(plan.source_end == 200 && plan.destination_end == 140);
    assert(plan.reused_tokens == 120 && plan.fresh_tokens == 32);
    assert(!memcmp(source, source_before, sizeof(source)));
    assert(!memcmp(edited, edited_before, (end + 32) * sizeof(*edited)));
    assert(!ds4_suffix_match(source, 200, edited, end + 32, end - 1, &plan));

    /* Odd displacement stays unsupported. */
    prefix(edited, 21); end = append(edited, 21, source, 80, 200);
    assert(!ds4_suffix_match(source, 200, edited, end + 32, end, &plan));
    /* One fresh prefix separator can fix parity without relocating odd pairs
     * or altering any token of the fresh question. */
    for (unsigned i = end; i < end + 32; i++) edited[i] = -2000 - (int)i;
    int padded[321];
    memcpy(padded, edited, (end + 32) * sizeof(*edited));
    memmove(padded + 11, padded + 10, (end + 32 - 10) * sizeof(*padded));
    padded[10] = -9999;
    assert(ds4_suffix_match(source, 200, padded, end + 33, end + 1, &plan));
    assert(plan.source_start == 80 && plan.destination_start == 22);
    assert(plan.reused_tokens == 120 && plan.fresh_tokens == 32);
    assert(!memcmp(padded + end + 1, edited + end, 32 * sizeof(*edited)));
    assert(!memcmp(source, source_before, sizeof(source)));
    prefix(edited, 100); end = append(edited, 100, source, 80, 200);
    for (unsigned i = end; i < end + 32; i++) edited[i] = -2000 - (int)i;
    memcpy(edited_before, edited, (end + 32) * sizeof(*edited));
    /* Rightward insertion is explicitly admitted; the old entry point remains
     * left-only, and the immutable source and complete latest question stay intact. */
    assert(!ds4_suffix_match(source, 200, edited, end + 32, end, &plan));
    assert(!ds4_suffix_match_shift(source, 200, edited, end + 32, end, false, &plan));
    assert(ds4_suffix_match_shift(source, 200, edited, end + 32, end, true, &plan));
    assert(plan.source_start == 80 && plan.destination_start == 100);
    assert(plan.source_end == 200 && plan.destination_end == 220);
    assert(plan.reused_tokens == 120 && plan.fresh_tokens == 32);
    assert(!memcmp(source, source_before, sizeof(source)));
    assert(!memcmp(edited, edited_before, (end + 32) * sizeof(*edited)));
    assert(!ds4_suffix_match_shift(source, 200, edited, end + 32, end - 1, true, &plan));
    assert(!ds4_suffix_match_shift(source, 200, edited, end + 32, 32, true, &plan));
    prefix(edited, 101); end = append(edited, 101, source, 80, 200);
    assert(!ds4_suffix_match_shift(source, 200, edited, end + 32, end, true, &plan));

    /* A longer replacement may trim an odd start, retaining even pair boundaries. */
    prefix(edited, 99); end = append(edited, 99, source, 79, 200);
    assert(ds4_suffix_match_shift(source, 200, edited, end + 32, end, true, &plan));
    assert(plan.source_start == 80 && plan.destination_start == 100);
    assert(plan.destination_end == 220 && plan.reused_tokens == 120);

    /* An odd matching start trims one token, preserving even pair boundaries. */
    prefix(edited, 19); end = append(edited, 19, source, 79, 200);
    assert(ds4_suffix_match(source, 200, edited, end + 32, end, &plan));
    assert(plan.source_start == 80 && plan.destination_start == 20);
    assert(plan.reused_tokens == 120);

    /* An exact token-zero prefix must never displace a genuine suffix graft. */
    end = append(edited, 0, source, 0, 200);
    assert(!ds4_suffix_match(source, 200, edited, end + 32, end, &plan));

    /* Matching tokens inside the incoming query cannot be spliced into cache. */
    prefix(edited, 32); end = append(edited, 32, source, 80, 200);
    assert(ds4_suffix_match(source, 200, edited, end + 32, end, &plan));
    assert(!ds4_suffix_match(source, 200, edited, end + 32, 32, &plan));

    /* Choose the most reused candidate rather than the rightmost match. */
    prefix(edited, 10); end = append(edited, 10, source, 100, 200);
    for (unsigned i = end; i < end + 20; i++) edited[i] = -3000 - (int)i;
    end = append(edited, end + 20, source, 130, 200);
    assert(ds4_suffix_match(source, 200, edited, end + 32, end, &plan));
    assert(plan.source_start == 100 && plan.destination_start == 10);
    assert(plan.destination_end == 110 && plan.reused_tokens == 100);
    assert(plan.fresh_tokens == end + 32 - 110);

    /* A corrupted anchor fails; pair trimming cannot leave fewer than64 rows. */
    prefix(edited, 20); end = append(edited, 20, source, 80, 200);
    edited[end - 1] = -4000;
    assert(!ds4_suffix_match(source, 200, edited, end + 32, end, &plan));
    prefix(edited, 5); end = append(edited, 5, source, 55, 119);
    assert(!ds4_suffix_match(source, 119, edited, end + 32, end, &plan));

    puts("suffix_match PASS: left/right opt-in geometry, prefix exclusion, fresh-query boundary, longest match, immutable inputs");
    return 0;
}
