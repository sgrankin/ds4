#!/usr/bin/env python3
"""Check same-storage canonical exchanges, then report cross-storage conversions."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-bin-dir", type=Path, required=True)
    parser.add_argument("--native-bin-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf")
    parser.add_argument("--out", type=Path, default=ROOT / ".workspaces/zig-engine-payload-rerun")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binaries = {"control": args.control_bin_dir.resolve() / "payload-roundtrip", "native": args.native_bin_dir.resolve() / "payload-roundtrip"}
    env = {key: value for key, value in os.environ.items() if not key.startswith(("DS4", "PT_"))}
    result = {"scope": "Canonical DSV4v2 wire exchange at1,2,3,127,128,129 with fresh987 continuation. Strict C/native/C cycles per storage mode; two cross-storage imports report rounding differences. No latency claim.", "fixture_sha256": hashlib.sha256((HERE / "payload_roundtrip.c").read_bytes()).hexdigest(), "runs": []}
    references = {}
    folders = {}
    cases = [
        ("control-f32", "control", 0, None, 0),
        ("native-f32", "native", 0, "control-f32", 0),
        ("control-return-f32", "control", 0, "native-f32", 0),
        ("control-f16", "control", 1, None, 1),
        ("native-f16", "native", 1, "control-f16", 1),
        ("control-return-f16", "control", 1, "native-f16", 1),
        ("conversion-f32-to-f16", "native", 1, "control-f32", 0),
        ("conversion-f16-to-f32", "native", 0, "control-f16", 1),
    ]
    for label, arm, f16, source, source_f16 in cases:
        folder = out / label
        folder.mkdir()
        folders[label] = folder
        conversion = source is not None and f16 != source_f16
        command = [binaries[arm], "--model", args.model.resolve(), "--out", folder, "--f16", str(f16)]
        if source is not None:
            command += ["--source", folders[source], "--source-f16", str(source_f16)]
        command = list(map(str, command))
        start = time.monotonic()
        with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=600)
        row = {"label": label, "command": command, "f16": f16, "source_f16": source_f16, "cross_storage_report_only": conversion, "seconds": time.monotonic() - start, "returncode": process.returncode, "binary_sha256": hashlib.sha256(binaries[arm].read_bytes()).hexdigest()}
        result["runs"].append(row)
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        assert process.returncode == 0, label
        records = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
        assert records[-1] == {"kind": "result", "pass": True, "stage": "recover_after_failed_restore"}, label
        assert [r["position"] for r in records if r["kind"] == "frontier"] == [1, 2, 3, 127, 128, 129], label
        row["records"] = records
        row["artifacts_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("frontier-*"))}
        if source is None:
            references[f16] = row["artifacts_sha256"]
        row["all_canonical_and_fresh_artifacts_match_same_storage_control"] = row["artifacts_sha256"] == references[f16]
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        if not conversion:
            assert row["all_canonical_and_fresh_artifacts_match_same_storage_control"], label
    result["all_pass"] = True
    result["cross_storage_differences_are_reported_not_rejected"] = True
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Canonical payload exchange and restore failure boundaries: PASS", flush=True)


if __name__ == "__main__":
    main()
