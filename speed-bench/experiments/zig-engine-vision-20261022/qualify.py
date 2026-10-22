#!/usr/bin/env python3
"""Compare the existing full vision-encoder fixture in serial C/native arms."""
from pathlib import Path
import argparse
import array
import hashlib
import json
import math
import os
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ["tests/test_deepseek41_graph.c", "ds4.c", "ds4_metal.m", "ds4_gpu.h", "ds4_model_weights.h", "zig/engine/vision.zig", "zig/engine/vision_entry.zig", "zig/engine/vision_api.h", "zig/metal/vision_bridge.inc.m", "build.zig"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-bin-dir", type=Path, required=True)
    parser.add_argument("--native-bin-dir", type=Path, required=True)
    parser.add_argument("--debug-bin-dir", type=Path)
    parser.add_argument("--sidecar", type=Path, default=ROOT / "gguf/DeepSeek-V4.1-Flash-1017-vision-BF16.gguf")
    parser.add_argument("--image", type=Path, default=ROOT / ".workspaces/zig-combined-20261022/vision/red.png")
    parser.add_argument("--out", type=Path, default=ROOT / ".workspaces/zig-engine-vision-rerun")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("DS4", "PT_"))}
    result = {"scope": "One 1017 BF16 sidecar/image; two encodes per fixture arm, complete F32 embeddings compared. Ordered functionality screen, not a latency ablation; legacy4096 sidecar untested.", "source_sha256": {name: sha256(ROOT / name) for name in SOURCES}, "sidecar": {"path": str(args.sidecar.resolve()), "bytes": args.sidecar.stat().st_size}, "image": {"path": str(args.image.resolve()), "sha256": sha256(args.image)}, "runs": [], "comparisons": []}
    arms = [("control", args.control_bin_dir), ("native", args.native_bin_dir)]
    if args.debug_bin_dir:
        arms.append(("native-debug", args.debug_bin_dir))
    reference_bytes = reference_values = reference_geometry = None
    for label, directory in arms:
        folder = out / label
        folder.mkdir()
        binary = directory.resolve() / "test-deepseek41-graph"
        output = folder / "embedding.f32"
        command = list(map(str, [binary, "--vision-encoder", args.sidecar.resolve(), args.image.resolve(), output]))
        start = time.monotonic()
        with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=300)
        row = {"label": label, "command": command, "returncode": process.returncode, "seconds": time.monotonic() - start, "binary_sha256": sha256(binary)}
        result["runs"].append(row)
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        assert process.returncode == 0, label
        match = re.search(r"V4\.1 vision: (\d+)x(\d+) grid, (\d+) rows of (\d+), finite and repeatable", (folder / "stdout.log").read_text())
        assert match, label
        geometry = list(map(int, match.groups()))
        raw = output.read_bytes()
        assert len(raw) == geometry[2] * geometry[3] * 4, label
        values = array.array("f", raw)
        assert all(math.isfinite(value) for value in values), label
        row.update(geometry=geometry, float_count=len(values), embedding_bytes=len(raw), embedding_sha256=hashlib.sha256(raw).hexdigest(), finite=True, internally_byte_repeatable=True)
        if reference_bytes is None:
            reference_bytes, reference_values, reference_geometry = raw, values, geometry
        else:
            assert geometry == reference_geometry, label
            result["comparisons"].append({"candidate": label, "geometry_equal": True, "all_embedding_bytes_equal": raw == reference_bytes, "max_abs": max(abs(a - b) for a, b in zip(reference_values, values)), "different_values": sum(a != b for a, b in zip(reference_values, values))})
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    result["all_valid_execution"] = True
    result["all_embedding_bytes_equal"] = all(row["all_embedding_bytes_equal"] for row in result["comparisons"])
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Vision encoder comparison complete; full-vector differences are reported in results.json.", flush=True)


if __name__ == "__main__":
    main()
