#!/usr/bin/env python3
"""Run the existing suffix relocation probe in three bounded C/native pairs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "speed-bench/experiments/recipe-rightshift-hour39-20261008/repeat.c"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-bin-dir", type=Path, required=True)
    parser.add_argument("--native-bin-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf")
    parser.add_argument("--out", type=Path, default=ROOT / ".workspaces/zig-engine-suffix-rerun")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binaries = {"control": args.control_bin_dir.resolve() / "suffix-repeat", "native": args.native_bin_dir.resolve() / "suffix-repeat"}
    env = {key: value for key, value in os.environ.items() if not key.startswith(("DS4", "PT_"))}
    overrides = {"DS4_V41_DECODER_REPLAY_ROWS": "0", "DS4_METAL_V41_RESIDENT_ROUTE_GAP": "0", "DS4_METAL_V41_RESIDENT_PREFILL_GAP": "0", "DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS": "0"}
    env.update(overrides)
    result = {"scope": "Mechanical suffix-owner comparison. Three ordered C/native pairs; native/reuse inside the existing helper means full recomputation/graft, not implementation identity. No speed claim.", "fixture": str(FIXTURE.relative_to(ROOT)), "fixture_sha256": sha256(FIXTURE), "environment_overrides": overrides, "runs": [], "comparisons": []}
    for case, f16, moves, prerope in [("prerope-f32", 0, 4, 1), ("prerope-f16", 1, 4, 1), ("legacy-left-f16", 1, 1, 0)]:
        pair = []
        for role in ["control", "native"]:
            label = f"{case}-{role}"
            folder = out / label
            folder.mkdir()
            command = list(map(str, [binaries[role], "--model", args.model.resolve(), "--out", folder, "--ctx", 4096, "--chunk", 512, "--cache-gb", 32, "--moves", moves, "--prerope", prerope, "--source-f16", f16, "--destination-f16", f16]))
            start = time.monotonic()
            with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=600)
            run = {"label": label, "case": case, "role": role, "command": command, "seconds": time.monotonic() - start, "returncode": process.returncode, "binary_sha256": sha256(binaries[role])}
            result["runs"].append(run)
            (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(run), flush=True)
            assert process.returncode == 0, label
            with (folder / "rows.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            with (folder / "moves.csv").open() as stream:
                timings = list(csv.DictReader(stream))
            assert len(rows) == moves * 4 and len(timings) == moves * 2, label
            assert [int(row["shift"]) for row in timings[::2]] == ([-66, 128, -514, 0] if moves == 4 else [128]), label
            run["teacher_rows"] = len(rows)
            run["quality"] = {"mean_delta_nll": sum(float(row["delta_nll"]) for row in rows) / len(rows), "mean_kl": sum(float(row["kl"]) for row in rows) / len(rows), "argmax_flips": sum(int(row["argmax_flip"]) for row in rows), "recompute_graft_equal_rows": sum(int(row["byte_equal"]) for row in rows)}
            run["artifacts_sha256"] = {name: sha256(folder / name) for name in ["logits.bin", "tokens.json", "rows.csv", "moves.csv"]}
            run["geometry_and_source_hashes"] = [{key: value for key, value in row.items() if not key.endswith("_ms")} for row in timings]
            run["edit_time_ms"] = {arm: sum(float(row["total_edit_ms"]) for row in timings if row["arm"] == arm) for arm in ["native", "reuse"]}
            pair.append(run)
        result["comparisons"].append({"case": case, "all_teacher_vectors_equal": pair[0]["artifacts_sha256"]["logits.bin"] == pair[1]["artifacts_sha256"]["logits.bin"], "tokens_equal": pair[0]["artifacts_sha256"]["tokens.json"] == pair[1]["artifacts_sha256"]["tokens.json"], "geometry_and_source_hashes_equal": pair[0]["geometry_and_source_hashes"] == pair[1]["geometry_and_source_hashes"], "quality_rows_equal": pair[0]["artifacts_sha256"]["rows.csv"] == pair[1]["artifacts_sha256"]["rows.csv"]})
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    result["all_valid_execution"] = True
    result["all_pair_teacher_vectors_equal"] = all(row["all_teacher_vectors_equal"] for row in result["comparisons"])
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Suffix owner execution complete; compare vectors and quality in results.json.", flush=True)


if __name__ == "__main__":
    main()
