#ifndef DS4_CLEF_TOKENIZER_API_H
#define DS4_CLEF_TOKENIZER_API_H
#include "ds4_tokenizer.h"
#include <unicode/uregex.h>
#include <unicode/utext.h>
#include <stdio.h>
#include <stdlib.h>
/* Apple SDK exposes NFC through CoreFoundation. The returned UTF-8 is malloc-owned. */
char *ds4_clef_nfc(const char *text, size_t length, size_t *out_length);
#endif
