#!/usr/bin/env python3
"""Compare the existing trained DSpark runtime with C/native cycle owners."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import time

ROOT = Path.cwd().resolve()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-bin-dir", type=Path, required=True)
    parser.add_argument("--native-bin-dir", type=Path, required=True)
    parser.add_argument("--debug-bin-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=ROOT / "gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf")
    parser.add_argument("--draft", type=Path, default=ROOT / "gguf/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf")
    parser.add_argument("--out", type=Path, default=ROOT / ".workspaces/zig-engine-20261022/cycle-model")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("DS4", "PT_"))}
    for key, source in re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]', (ROOT / "ds4_metal.m").read_text()):
        env[key] = str(ROOT / source)
    env.update(DS4_V41_COMPRESSED_KV_F16="0", DS4_V41_DECODER_REPLAY_ROWS="0",
               DS4_METAL_V41_RESIDENT_ROUTE_GAP="0", DS4_METAL_V41_RESIDENT_PREFILL_GAP="0",
               DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS="0")
    result = {"scope": "Final trained DSpark Session runtime: C drafter/cycle fixture against native and Debug, otherwise selected native core; no throughput claim.",
              "fixture_sha256": digest(ROOT / "tests/test_deepseek41_dspark.c"),
              "target": str(args.target.resolve()), "draft": str(args.draft.resolve()),
              "runs": [], "comparisons": [], "pass": False}
    expected = [f"cycle exact={exact} index={index} " for exact in range(2) for index in range(8)]
    for label, directory in [("control", args.control_bin_dir), ("native", args.native_bin_dir), ("debug", args.debug_bin_dir)]:
        folder = out / label
        folder.mkdir()
        binary = directory.resolve() / ("dspark-c-oracle" if label == "control" else "dspark-native-oracle")
        command = [str(binary), str(args.target.resolve()), str(args.draft.resolve()), "--runtime"]
        row = {"label": label, "command": command, "binary_sha256": digest(binary),
               "cwd": str(ROOT), "environment": {key: value for key, value in env.items() if key.startswith("DS4")}}
        started = time.monotonic()
        with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
            try:
                row["returncode"] = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=1200).returncode
            except subprocess.TimeoutExpired:
                row["returncode"] = "timeout"
        row["seconds"] = time.monotonic() - started
        lines = (folder / "stderr.log").read_text(errors="replace").splitlines()
        row["cycle_records"] = [line for line in lines if line.startswith("cycle exact=")]
        row["runtime_markers"] = [line for line in lines if line.startswith("verify prefix=") or line.startswith("V4.1 DSpark ")]
        row["valid_runtime"] = row["returncode"] == 0 and len(row["cycle_records"]) == 16 and all(line.startswith(prefix) for line, prefix in zip(row["cycle_records"], expected)) and "V4.1 DSpark cancellation and context-limit clamping: PASS" in lines
        result["runs"].append(row)
        if label != "control":
            result["comparisons"].append({"label": label,
                "all_16_cycle_records_exact": row["cycle_records"] == result["runs"][0]["cycle_records"] and len(row["cycle_records"]) == 16,
                "runtime_markers_exact": row["runtime_markers"] == result["runs"][0]["runtime_markers"]})
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: row[key] for key in ["label", "returncode", "seconds", "valid_runtime"]}), flush=True)
    result["pass"] = all(row["valid_runtime"] for row in result["runs"]) and all(row["all_16_cycle_records_exact"] for row in result["comparisons"])
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["pass"] else 3)


if __name__ == "__main__":
    main()
