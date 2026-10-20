# Native Q2 Engram scale census

All eight fixed windows fit a lossless 3-bit scale dictionary. This corrects an
otherwise overbroad statement that the Q2 file cannot benefit from this idea:
its routed experts use other quantization formats, but its two Engram tensors
retain native E4M3 codes and E8M0 scales. It does not establish a serving speedup.

`ds4_engram.h` defines each I8 row as 256 code bytes plus eight scale bytes
(264 total), and `ds4_engram.c` indexes scale `raw[256 + j/32]`.
The existing Q2 directory identifies `blk.1.engram_embd.weight` at byte
162955640832 and `blk.14.engram_embd.weight` at byte264333279232, both shape
`[264, rows]`, type24. The supplied ROOT-verified 1017 Q2 identity is recorded
in `sample-plan.json`; this experiment did not rehash the full model.

Before reading payloads, `sample-plan.json` fixed four 512-row windows per table:
row0, floor(rows/3), floor(2*rows/3), and rows−512. There were exactly eight
`os.pread` calls totaling1,081,344 bytes, below the2MiB limit.

| Tensor | Window start row | Unique scale bytes / 4,096 | Byte-exact |
| --- | ---: | ---: | --- |
| blk.1.engram_embd.weight | 0 | 4 | yes |
| blk.1.engram_embd.weight | 128,002,056 | 4 | yes |
| blk.1.engram_embd.weight | 256,004,112 | 3 | yes |
| blk.1.engram_embd.weight | 384,005,656 | 3 | yes |
| blk.14.engram_embd.weight | 0 | 4 | yes |
| blk.14.engram_embd.weight | 128,005,560 | 4 | yes |
| blk.14.engram_embd.weight | 256,011,121 | 4 | yes |
| blk.14.engram_embd.weight | 384,016,170 | 3 | yes |

`census.py` sorts each window's scale dictionary, preserves the original256 code
bytes per row, and packs its eight scale indices into three little-endian bytes.
A seven-byte block header plus the dictionary records the format; more than
8 distinct scales selects an uncompressed raw fallback. All eight real windows
used the dictionary path, so no real-window fallback is claimed. The retained
`.packed` files are proof artifacts, not a modified GGUF.

Every restored original byte and SHA256 matched, including all code bytes.
Raw sample bytes: 1,081,344; packed: 1,060,949;
saving including block headers/dictionaries: 20,395 bytes
(1.886079%). `results.json` records histograms,
offsets and original/restored/packed hashes for each window.

The ideal payload-only maximum is5/264 =1.893939% of Engram. The two native
Engram payloads total202,758,032,400 bytes, so that maximum is
3,840,114,250 bytes, or
1.050033% of the365,713,686,528-byte model,
before metadata/alignment. The eight windows do not prove eligibility for every
block of either table. No203GB table scan or new full model was written.

This is primarily a disk-format opportunity. Native Engram is already streamed;
its expanded row/graph memory stays the same, so no RAM benefit is established.
A usable format needs per-block dictionary/offset metadata and an adapted reader.
Random264-byte row reads may need an extra dictionary access or wider reads;
page/block alignment, buffering and filesystem read amplification can erase the
small logical byte saving. No decode, model-quality or timing result was measured.

Only this `engram-scale/` namespace was written. No runtime source changes,
compiler, GPU, model forward pass, jj operation or serving benchmark occurred.
