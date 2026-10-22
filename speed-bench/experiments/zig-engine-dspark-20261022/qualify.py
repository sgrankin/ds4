#!/usr/bin/env python3
"""Compare capture-only DSpark verification/commit in selected engine binaries."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-bin-dir", type=Path, required=True)
    parser.add_argument("--native-bin-dir", type=Path, required=True)
    parser.add_argument("--debug-bin-dir", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf")
    parser.add_argument("--out", type=Path, default=ROOT / ".workspaces/zig-engine-dspark-rerun")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("DS4", "PT_"))}
    cases = [(f"{role}-f{32 if f16 == 0 else 16}", role, directory, f16) for f16 in [0, 1] for role, directory in [("control", args.control_bin_dir), ("native", args.native_bin_dir)]]
    if args.debug_bin_dir:
        cases.append(("debug-f16", "debug", args.debug_bin_dir, 1))
    result = {"scope": "Capture-only target verification/commit, not trained drafting or speculative throughput. Cross-implementation identity and existing batch/scalar drift reported separately.", "fixture_sha256": sha256(HERE / "verify_commit.c"), "runs": [], "comparisons": []}
    controls = {}
    for label, role, directory, f16 in cases:
        folder = out / label
        folder.mkdir()
        binary = directory.resolve() / "dspark-target-verify"
        command = list(map(str, [binary, "--model", args.model.resolve(), "--out", folder, "--f16", f16]))
        start = time.monotonic()
        with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=600)
        row = {"label": label, "role": role, "f16": f16, "command": command, "binary_sha256": sha256(binary), "seconds": time.monotonic() - start, "returncode": process.returncode}
        result["runs"].append(row)
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        assert process.returncode == 0, label
        records = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
        assert records[-1] == {"kind": "result", "pass": True, "stage": "fresh_target_continuation"}, label
        frontiers = [item for item in records if item["kind"] == "frontier"]
        logits = [item for item in records if item["kind"] == "logits"]
        assert [(item["prefix"], item["keep"]) for item in frontiers] == [(prefix, keep) for prefix in [127, 129] for keep in [0, 3, 6]], label
        assert len(logits) == 30 and all(item["finite"] for item in logits), label
        raw = folder / "logits.bin"
        with raw.open("rb") as stream:
            header = struct.unpack("<6I", stream.read(24))
        assert header[0] == 0x31565344 and header[2:] == (2, 3, 5, 2), label
        assert raw.stat().st_size == 24 + 60 * header[1] * 4, label
        row.update({"records": records, "raw_header": header, "raw_vector_sha256": sha256(raw), "record_sha256": sha256(folder / "results.jsonl"), "batch_scalar_max_abs": max(item["max_abs"] for item in logits), "batch_scalar_equal_observations": sum(item["byte_equal"] for item in logits), "fresh_batch_scalar_max_abs": max(item["max_abs"] for item in logits if item["freshly_computed"]), "canonical_equal_frontiers": sum(item["payload_equal"] for item in frontiers), "cache_equal_frontiers": sum(item["cache_equal"] for item in frontiers)})
        if role == "control":
            controls[f16] = row
        else:
            reference = controls[f16]
            result["comparisons"].append({"label": label, "f16": f16, "full_vectors_equal": row["raw_vector_sha256"] == reference["raw_vector_sha256"], "all_non_timing_records_equal": records == reference["records"]})
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    result["all_valid_execution"] = True
    result["all_cross_implementation_vectors_equal"] = all(row["full_vectors_equal"] for row in result["comparisons"])
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DSpark target verification/commit complete; identity and batch/scalar drift in results.json.", flush=True)


if __name__ == "__main__":
    main()
