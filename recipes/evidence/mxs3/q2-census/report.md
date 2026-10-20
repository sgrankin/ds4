# Actual Q2 packing eligibility: bounded census

2026-10-20. Used immutable `antirez/deepseek-v4.1-flash-gguf` revision `9dcee51692e6540f697141fc98b097e3c96492a3`, file `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`. Downloaded **7,520,256 bytes total**: one 6 MiB header plus the first 4096 blocks of expert 0 in gate/down tensors at layers 0 and 20. Every request required HTTP 206 and exact Content-Range/Content-Length before reading its bounded body. No local sparse weight payload was inspected, and no model was evaluated.

`q2_census.py` and the retained `*.bin.json` range records give exact offsets, types, lengths and SHA256. Parsed directory and numerical counts are in `directory.json` and `results.json`.

| Actual field, 4096 blocks per tensor | Layer 0 distinct values | Layer 20 distinct values |
|---|---:|---:|
| IQ2_XXS `d` (16-bit FP16 bits) | 863 | 884 |
| Q2_K `d` (16-bit FP16 bits) | 1104 | 986 |
| Q2_K `dmin` (16-bit FP16 bits) | 920 | 910 |
| Q2_K packed scale/min byte (65536 values) | 209 | 206 |

Even separate Q2_K scale and min nibbles span **14 and 16** symbols respectively in both samples (`nibble-census.json`). All proposed 4096-value dictionary groups exceed eight entries: the exact three-bit dictionary scheme would fall back, rather than reproduce the native MXFP4 result.

Current `ds4.c:1056–1108` defines IQ2_XXS as 66 bytes/256 weights (`uint16_t d` plus 64 packed code bytes), Q2_K as 84 bytes/256 (16 packed affine scale/min bytes, 64 code bytes, FP16 `d,dmin`), and native MXFP4 as 17 bytes/32 (`uint8_t e` plus 16 code bytes). These are different quantization layouts. IQ2_XXS/Q2_K contain no equivalent one-byte E8M0 scale stream that the tested MXS3 codec can replace unchanged.

This is evidence against applying **this exact E8M0/three-bit scheme** to current Q2 weights, not a claim that Q2 data cannot be compressed by any other method. No general-purpose compressor, requantization or runtime change was attempted.
