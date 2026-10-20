#ifndef DS4_OVERTHINK_MARKERS_H
#define DS4_OVERTHINK_MARKERS_H
#include <stdbool.h>

/* Paper Table 8 order is retained for the original full profile. The narrow
 * flag is an explicit experimental allowlist, not a classifier of token use. */
typedef struct { const char *word; bool hesitation; } ds4_overthink_marker;
static const ds4_overthink_marker ds4_overthink_markers[] = {
    {"perhaps", true}, {"maybe", true}, {"wait", true}, {"Wait", true},
    {"actually", false}, {"hold", false}, {"Hmm", true}, {"hmm", true},
    {"Alternatively", false}, {"alternatively", false},
    {"However", false}, {"however", false}, {"instead", false}, {"Instead", false},
    {"But", false}, {"but", false}, {"though", false}, {"although", false},
    {"yet", false}, {"rather", false}, {"unless", false}, {"otherwise", false},
    {"nonetheless", false}, {"nevertheless", false}, {"regardless", false},
    {"still", false}, {"anyway", false}, {"Or", false}, {"or", false},
    {"either", false}, {"whether", false}, {"uncertain", false}, {"unsure", false},
    {"possibly", false}, {"might", false}, {"could", false}, {"another", false},
    {"different", false}, {"reconsider", true}, {"rethink", true},
    {"backtrack", true}, {"retry", true}, {"recheck", true}, {"revisit", true},
    {"doubt", false}, {"confused", false}, {"wrong", false}, {"mistake", false},
    {"error", false}, {"incorrect", false}
};
#define DS4_OVERTHINK_MARKER_COUNT (sizeof(ds4_overthink_markers) / sizeof(ds4_overthink_markers[0]))
#endif
