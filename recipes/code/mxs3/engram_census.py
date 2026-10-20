"""Fixed-window Engram scale census and lossless 3-bit dictionary proof."""
import collections
import hashlib
import json
import os
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
plan_bytes = (HERE / "sample-plan.json").read_bytes()
plan = json.loads(plan_bytes)
assert sum(w["bytes"] for w in plan["windows"]) == plan["total_pread_bytes"] <= 2 * 1024**2


def pack(raw):
    rows = len(raw) // 264
    dictionary = sorted({s for i in range(rows) for s in raw[i * 264 + 256:(i + 1) * 264]})
    if len(dictionary) > 8:
        return b"ERW1" + struct.pack("<HB", rows, 0) + raw
    index = {s: i for i, s in enumerate(dictionary)}
    out = bytearray(b"ES31" + struct.pack("<HB", rows, len(dictionary)) + bytes(dictionary))
    for i in range(rows):
        row = raw[i * 264:(i + 1) * 264]
        out += row[:256]
        bits = sum(index[s] << (3 * j) for j, s in enumerate(row[256:]))
        out += bits.to_bytes(3, "little")
    return bytes(out)


def unpack(blob):
    rows, count = struct.unpack_from("<HB", blob, 4)
    if blob[:4] == b"ERW1":
        assert count == 0 and len(blob) == 7 + rows * 264
        return blob[7:]
    assert blob[:4] == b"ES31" and 1 <= count <= 8
    dictionary = blob[7:7 + count]
    assert len(blob) == 7 + count + rows * 259
    out = bytearray()
    for i in range(rows):
        offset = 7 + count + i * 259
        out += blob[offset:offset + 256]
        bits = int.from_bytes(blob[offset + 256:offset + 259], "little")
        out += bytes(dictionary[(bits >> (3 * j)) & 7] for j in range(8))
    return bytes(out)


results = []
read_bytes = 0
fd = os.open(plan["model_path"], os.O_RDONLY)
try:
    assert os.fstat(fd).st_size == plan["model_bytes"]
    for number, w in enumerate(plan["windows"]):
        raw = os.pread(fd, w["bytes"], w["offset"])
        read_bytes += len(raw)
        assert len(raw) == w["bytes"] and read_bytes <= plan["maximum_pread_bytes"]
        counts = collections.Counter(s for i in range(w["rows"])
                                     for s in raw[i * 264 + 256:(i + 1) * 264])
        assert sum(counts.values()) == 4096
        encoded = pack(raw)
        restored = unpack(encoded)
        assert restored == raw
        assert all(restored[i * 264:i * 264 + 256] == raw[i * 264:i * 264 + 256]
                   for i in range(w["rows"]))
        path = HERE / f"window-{number:02d}.packed"
        path.write_bytes(encoded)
        results.append({**w, "scale_values": 4096, "unique_scales": len(counts),
                        "scale_counts": dict(sorted(counts.items())), "eligible": len(counts) <= 8,
                        "mode": "dictionary3" if encoded[:4] == b"ES31" else "raw-fallback",
                        "packed_bytes": len(encoded), "saved_bytes_including_metadata": len(raw) - len(encoded),
                        "original_sha256": hashlib.sha256(raw).hexdigest(),
                        "restored_sha256": hashlib.sha256(restored).hexdigest(),
                        "packed_sha256": hashlib.sha256(encoded).hexdigest(),
                        "packed_path": path.name, "byte_exact": True, "code_bytes_unchanged": True})
finally:
    os.close(fd)

result = {"sample_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
          "pread_calls": len(results), "pread_bytes": read_bytes,
          "windows": results, "eligible_windows": sum(r["eligible"] for r in results),
          "sample_raw_bytes": read_bytes, "sample_packed_bytes": sum(r["packed_bytes"] for r in results),
          "ideal_engram_saving_fraction": 5 / 264,
          "ideal_whole_model_saving_fraction": plan["engram_payload_bytes"] * 5 / 264 / plan["model_bytes"],
          "scope": "Eight predeclared windows only. Lossless byte proof; no CPU timing or serving claim."}
(HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "windows"}, indent=2))
for r in results:
    print(r["tensor"], r["label"], r["unique_scales"], r["mode"], r["saved_bytes_including_metadata"])
