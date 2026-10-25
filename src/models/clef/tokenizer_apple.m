/* NFC is the only platform operation in the native Clef tokenizer. */
#include <CoreFoundation/CoreFoundation.h>
#include <stdlib.h>

char *ds4_clef_nfc(const char *text, size_t length, size_t *out_length) {
    CFStringRef source = CFStringCreateWithBytes(kCFAllocatorDefault,
        (const UInt8 *)text, (CFIndex)length, kCFStringEncodingUTF8, false);
    if (!source) return NULL;
    CFMutableStringRef normalized = CFStringCreateMutableCopy(kCFAllocatorDefault, 0, source);
    CFRelease(source);
    if (!normalized) return NULL;
    CFStringNormalize(normalized, kCFStringNormalizationFormC);
    CFIndex count = CFStringGetLength(normalized);
    CFIndex capacity = CFStringGetMaximumSizeForEncoding(count, kCFStringEncodingUTF8) + 1;
    char *result = malloc((size_t)capacity);
    if (result) {
        CFIndex written = 0;
        CFIndex consumed = CFStringGetBytes(normalized, CFRangeMake(0, count),
            kCFStringEncodingUTF8, 0, false, (UInt8 *)result, capacity - 1, &written);
        if (consumed != count) { free(result); result = NULL; }
        else { result[written] = 0; *out_length = (size_t)written; }
    }
    CFRelease(normalized);
    return result;
}
