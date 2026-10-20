# Native MXFP4 scale-codebook pilot

2026-10-20. **The lossless compression claim works on the real samples: 3.6704% smaller native expert payloads with4096-block groups, including all pilot overhead. It does not apply directly to our IQ2_XXS/Q2_K model.** No runtime code, GGUF type, Metal kernel or jj state was changed, and no model was evaluated.

## Real input and bounded download

Source: [antirez/deepseek-v4.1-flash-gguf](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf/tree/9dcee51692e6540f697141fc98b097e3c96492a3), snapshot `9dcee51692e6540f697141fc98b097e3c96492a3`, file `DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf`, metadata size49,353,784,544B. Its directory identifies source model revision `1bae0e1088993b9b82de08f59de5be5a34e42170` and native MXFP4 gate/up/down, Q8_0 dense/attention/shared/head.

`range_sample.py` requires HTTP206 and the **exact** `Content-Range` offset/end/whole-file size before reading a response. Each body read is bounded to requested bytes+1, and wrong extents fail. No whole shard was downloaded or hashed. The first4MiB was insufficient for the tokenizer metadata, so a second12MiB range completed the directory. Directory end=5,774,541; tensor data begins=5,774,560. The retained16MiB prefix includes some unused post-header data.

Six samples were selected before inspecting scales: complete expert matrices across layers0/5, gate/up/down, and expert IDs0/191/383. Each is type39,6,266,880B,368,640 blocks; gate/up tensors have GGUF shape[5120,2304,384], down[2304,5120,384]. Total actual expert sample=37,601,280B. Total HTTP bytes including the16MiB directory buffer=54,378,496B (51.86MiB), in eight verified range responses. Every original byte range and individual SHA256 proof is retained. See `sample-plan.json`, `directory.json`, `samples.csv`, and each `.mxfp4.json` for exact source tensor name, expert ID, absolute file offset and byte count. Sampling covers only two early layers in shard1; it does not establish the whole-model scale alphabet.

Every sampled expert used exactly the four scale bytes `{119,120,121,122}`. Scales120/121 dominate; rare119/122 are retained without clipping. This confirms a three-bit dictionary index fits these real matrices. We intentionally measured the requested fixed three-bit scheme, rather than retuning to two bits for this small sample.

## Codec and actual measurement

Pilot format `MXS3P001` is an experimental standalone byte container, **not** an official or registered GGUF format. Native MXFP4 blocks are17B: unchanged one-byte E8M0 exponent and16 packed E2M1 code bytes for32 weights. The pilot keeps every code byte unchanged, stores a sorted scale codebook, and replaces each scale byte with a three-bit codebook index. A group with more than8 distinct scales uses an explicit tag plus its original native bytes. No floating-point arithmetic or requantization is involved.

`codec_pilot.py` measured one in-memory Python encode/decode pass per real sample and grouping. Exact full-byte comparison and SHA256 matched for every reconstruction. These timings exclude network, disk and model execution.

| Codebook group | Original bytes | Packed bytes | Net saving | Raw fallback | Encode MiB/s | Decode MiB/s |
|---|---:|---:|---:|---:|---:|---:|
|256 native blocks|37,601,280|36,246,102|3.60407%|0/8,640 groups|116.75|83.39|
|4096 native blocks|37,601,280|36,221,161|**3.67040%**|0/540 groups|110.57|82.55|
|Whole expert|37,601,280|36,219,030|3.67607%|0/6 experts|110.24|82.34|

For4096-block grouping, original scales occupy2,211,840B; three-bit indices occupy829,440B. All16-byte code payloads remain35,389,440B. Additional overhead is1,621B of dictionaries,540B group tags and120B file headers:2,281B total. Net saved=1,380,119B. The zero-overhead theoretical ceiling for three-bit scales is `(8-3)/(17*8) = 3.6764706%` of a native expert payload.

There were no naturally overflowing groups in these samples. `sanity.py` separately checked a partial final group, eight dictionary symbols, and a nine-symbol raw fallback; all reconstructed the original bytes. Fallback costs one tag byte per uncompressed group, plus the once-per-file20B header. With4096 blocks that tag is1/69,632 of the native group size (~0.00144%), rather than dropping or approximating uncommon scales. Actual boundary check output is `sanity-checks.json`; full real-data results/timings are `results.json` and `pilot.stdout.txt`.

Commands actually run:

```sh
python3 -B .workspaces/upgrade-20261020/mxs3/range_sample.py fetch 0 4194304 header-first4MiB.bin
python3 -B .workspaces/upgrade-20261020/mxs3/range_sample.py fetch 4194304 12582912 header-next12MiB.bin
# directory.json parsed from the concatenated bounded prefix; original chunks retained
python3 -B .workspaces/upgrade-20261020/mxs3/range_sample.py samples
python3 -B .workspaces/upgrade-20261020/mxs3/codec_pilot.py > .workspaces/upgrade-20261020/mxs3/pilot.stdout.txt 2> .workspaces/upgrade-20261020/mxs3/pilot.stderr.txt
python3 -B .workspaces/upgrade-20261020/mxs3/sanity.py
```

The first network approval review timed out without a download; the authorized single retry succeeded. The first directory parse failed visibly because4MiB was insufficient, then the additional bounded range supplied the rest. Neither event caused a full-file read or changed the fixed expert sample selection.

## Why this is not a shrink operation on our Q2 download

Current source `ds4.c:1056–1108` and `gguf-tools/quants.c:48–71` define distinct layouts:

| Layout | Block size | Meaning of scale storage |
|---|---:|---|
|Native MXFP4, type39|17B/32 weights=4.25bits/weight|One E8M0 exponent byte plus16 code bytes. This pilot's target.|
|IQ2_XXS, type16, gate/up|66B/256=2.0625bits/weight|One F16 base scale plus packed grid/sign indices and embedded per32-group scale nibbles. Not a stream of one-byte E8M0 values.|
|Q2_K, type10, down|84B/256=2.625bits/weight|16 bytes of scale/min nibbles,64 bytes two-bit codes, F16 `d` and `dmin`. Different affine reconstruction.|

The native repacker explicitly preserves the released E8M0 bytes and E2M1 codes (`gguf-tools/deepseek4-quantize.c:925–985`). The Q2 recipe explicitly selects IQ2_XXS gate/up and Q2_K down (`gguf-tools/deepseek41_quantize.py:126–128`). Reinterpreting those formats as MXFP4 would corrupt weights. Compressing their scale/grid fields would be a separate experiment.

## Practical value and stopping point

If these sample savings generalized to all40×3×384 trunk expert matrices, native expert bytes would shrink by about10.60GB (~9.88GiB). The ideal three-bit ceiling is10,616,832,000B. All nine native shards total501,602,535,360B, so even that ideal ceiling is only **2.12% of the complete checkpoint**, because Engram/dense and other payloads are unchanged. This is an extrapolation from known shapes, not a full-checkpoint measurement. The current IQ2_XXS/Q2_K file is365,713,686,528B; scale packing would not make the native MXFP4 checkpoint smaller than that Q2 file.

Python's~82.5MiB/s reconstruction is far too slow for a high-throughput SSD expert path. A native C implementation might be much faster, but none was built here. If a loader inflates back into existing17B blocks, cache capacity stays unchanged; only on-disk bytes and miss-transfer traffic shrink. If Metal consumes codebooks/indices directly and the expert cache remains packed, the theoretical capacity gain is about3.8%, while extra unpack instructions can offset bandwidth savings. Only actual miss bytes benefit, so whole-task speedup can be much smaller or negative.

**Decision:** feasible lossless packing for native MXFP4; retain as a small candidate for ROOT review. Do not port a kernel yet. The next bounded step would compare existing MXFP4 versus a codebook-aware read/dequant path on the same real expert bytes, checking exact dequant values/output and measuring expert-miss time plus cache occupancy. No new quantization/task-quality improvement is claimed: reconstructed bytes are exact, but an eventual integrated kernel still needs its own correctness and serving measurement. Stop here as requested.
