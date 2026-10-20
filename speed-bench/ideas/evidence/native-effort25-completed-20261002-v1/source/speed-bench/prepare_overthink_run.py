#!/usr/bin/env python3
"""Write persistent provenance and print a guarded pilot command; never launch it."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "speed-bench/ideas/artifacts"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("model", type=Path)
    parser.add_argument("--binary", type=Path, default=REPO / "speed-bench/overthink_pilot")
    parser.add_argument("--guard", required=True, type=Path)
    parser.add_argument("--cache-gb", type=float, default=64.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--think-level", type=int, default=75)
    parser.add_argument("--suite", choices=("calibration", "heldout", "hard", "counting"), default="hard")
    parser.add_argument("--candidate", type=float)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--cases", type=int)
    parser.add_argument("--case-offset", type=int)
    parser.add_argument("--diagnostic-stride", type=int)
    parser.add_argument("--markers", choices=("paper", "hesitation"))
    parser.add_argument("--after", type=int)
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.markers is None:
        args.markers = os.environ.get("DS4_OVERTHINK_MARKERS", "paper")
    if args.markers not in ("paper", "hesitation"):
        parser.error("DS4_OVERTHINK_MARKERS must be paper or hesitation")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("run name may contain only letters, digits, underscores, and hyphens")
    if not math.isfinite(args.cache_gb) or not 0 < args.cache_gb <= 1048576:
        parser.error("cache size must be a finite positive GiB size")
    if not 1 <= args.max_tokens <= 4096 or not 1 <= args.think_level <= 100:
        parser.error("token cap or reasoning effort out of range")
    if args.candidate is not None and (not math.isfinite(args.candidate) or not 0 < args.candidate <= 4):
        parser.error("candidate must be a finite penalty in (0,4]")
    if args.native_only and args.candidate is not None:
        parser.error("native-only excludes candidate lambda")
    suite_cases = dict(calibration=4, heldout=8, hard=6, counting=4)[args.suite]
    if args.case_offset is None:
        raw = os.environ.get("DS4_PILOT_CASE_OFFSET", "0")
        if not raw.isascii() or not raw.isdecimal():
            parser.error("DS4_PILOT_CASE_OFFSET must be a nonnegative integer")
        args.case_offset = int(raw)
    if not 0 <= args.case_offset < suite_cases:
        parser.error("case offset must be a zero-based index within the selected suite")
    if args.cases is None and os.environ.get("DS4_PILOT_CASE_LIMIT") is not None:
        raw = os.environ["DS4_PILOT_CASE_LIMIT"]
        if not raw.isascii() or not raw.isdecimal():
            parser.error("DS4_PILOT_CASE_LIMIT must be a positive integer")
        args.cases = int(raw)
    if args.cases is None:
        args.cases = suite_cases - args.case_offset
    if not 1 <= args.cases <= suite_cases - args.case_offset:
        parser.error("case limit plus offset must fit within the selected suite")
    if args.diagnostic_stride is None:
        raw = os.environ.get("DS4_PILOT_DIAGNOSTIC_STRIDE", "0")
        if not raw.isascii() or not raw.isdecimal():
            parser.error("DS4_PILOT_DIAGNOSTIC_STRIDE must be a nonnegative integer")
        args.diagnostic_stride = int(raw)
    if not 0 <= args.diagnostic_stride <= 4096:
        parser.error("diagnostic stride must be from zero (off) through 4096")
    if args.after is None and os.environ.get("DS4_OVERTHINK_AFTER") is not None:
        raw = os.environ["DS4_OVERTHINK_AFTER"]
        if not raw.isascii() or not raw.isdecimal():
            parser.error("DS4_OVERTHINK_AFTER must be a nonnegative integer")
        args.after = int(raw)
    if args.after is not None and not 0 <= args.after <= 2147483647:
        parser.error("late threshold must be a nonnegative int32")
    binary, guard, model = (path.resolve(strict=True) for path in (args.binary, args.guard, args.model))
    if not binary.is_file() or not os.access(binary, os.X_OK) or not guard.is_file() or not model.is_file():
        parser.error("binary, guard, and model must exist; binary must be executable")
    if args.revision is None:
        args.revision = subprocess.check_output([
            "jj", "--ignore-working-copy", "log", "-r", "@", "--no-graph", "-T", "commit_id"
        ], cwd=REPO, text=True).strip()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output, pressure_output = ARTIFACTS / args.run_name, ARTIFACTS / (args.run_name + "-pressure")
    if output.exists() or pressure_output.exists():
        parser.error("run and pressure directories must be fresh")
    pilot_command = [str(binary), str(model), args.run_name, str(args.max_tokens),
                     str(args.think_level), args.suite]
    if args.candidate is not None:
        pilot_command.append(str(args.candidate))
    if args.native_only:
        pilot_command.append("--native-only")
    pilot_command += ["--cache-gb", str(args.cache_gb)]
    pilot_command += ["--cases", str(args.cases), "--case-offset", str(args.case_offset),
                      "--diagnostic-stride", str(args.diagnostic_stride)]
    policy_env = {name: os.environ.get(name) for name in (
        "DS4_OVERTHINK_PENALTY", "DS4_OVERTHINK_SCOPE", "DS4_OVERTHINK_AFTER", "DS4_OVERTHINK_MARKERS", "DS4_PILOT_CACHE_GIB", "DS4_PILOT_CASE_LIMIT", "DS4_PILOT_CASE_OFFSET", "DS4_PILOT_DIAGNOSTIC_STRIDE")}
    policy_env["DS4_OVERTHINK_MARKERS"] = args.markers
    if args.after is not None:
        policy_env["DS4_OVERTHINK_AFTER"] = str(args.after)
    command = ["python3", str(guard), str(pressure_output), "--", *pilot_command]
    inherited_policy = [f"{name}={value}" for name, value in policy_env.items() if value is not None]
    if inherited_policy:
        command = ["env", *inherited_policy, *command]
    model_stat = model.stat()
    sources = {name: digest(REPO / name) for name in (
        "ds4.c", "ds4.h", "ds4_overthink_markers.h", "Makefile", "speed-bench/overthink_pilot.c", "speed-bench/prepare_overthink_run.py", "speed-bench/overthink_diagnostics.h", "speed-bench/overthink_gold.py", "speed-bench/overthink_counting_cases.h", "speed-bench/overthink_counting_gold.py", "speed-bench/experiments/ds41-ssd/overthink-provenance/counting-study/cases-v1.json")}
    manifest = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        revision=args.revision, cwd=str(REPO), argv=command, pilot_argv=pilot_command,
        binary=dict(path=str(binary), sha256=digest(binary), bytes=binary.stat().st_size),
        guard=dict(path=str(guard), sha256=digest(guard)), source_sha256=sources,
        model=dict(path=str(model), bytes=model_stat.st_size, mtime_ns=model_stat.st_mtime_ns,
                   sha256="not read: model payload is intentionally not hashed"),
        cache_gib_requested=args.cache_gb, max_tokens=args.max_tokens,
        think_level=args.think_level, suite=args.suite, candidate=args.candidate,
        native_only=args.native_only,
        case_limit=args.cases, case_offset=args.case_offset,
        diagnostic_stride=args.diagnostic_stride, overthink_after=args.after or 0,
        marker_profile=args.markers,
        policy_environment=policy_env,
        results_dir=str(output), pressure_dir=str(pressure_output))
    manifest_path = ARTIFACTS / (args.run_name + "-manifest.json")
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2); stream.write("\n")
    print(f"Manifest: {manifest_path}")
    print("Run from the recorded cwd only after the model slot is granted:")
    print(shlex.join(command))


if __name__ == "__main__":
    main()
