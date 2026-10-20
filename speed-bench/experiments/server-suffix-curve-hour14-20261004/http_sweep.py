#!/usr/bin/env python3
"""Four serial HTTP workflow arms; run only inside ROOT's pressure guard."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LAUNCH = ROOT / "speed-bench/experiments/server-workflow-hour12-20261004/launch.py"
ARMS = (("tail64", "reuse", 64), ("tail256", "reuse", 256),
        ("tail128", "reuse", 128), ("native", "native", 128))


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HERE / "http")
    p.add_argument("--cache-root", type=Path,
                   default=ROOT / ".workspaces/server-agent-day-20261004")
    p.add_argument("--shared", type=Path,
                   default=ROOT / ".workspaces/server-agent-day-20261004/workflow-shared")
    p.add_argument("--skip-tail128", action="store_true",
                   help="Explicitly omit this point if ROOT's one-hour deadline requires it.")
    a = p.parse_args()
    output, shared, caches = a.output.resolve(), a.shared.resolve(), a.cache_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arms = [arm for arm in ARMS if not (a.skip_tail128 and arm[0] == "tail128")]
    save(output / "sweep-execution.json", {
        "argv": sys.argv, "cwd": str(ROOT), "runtime_revision": "3fb450f7",
        "launcher": str(LAUNCH), "shared_visible_paths": str(shared),
        "order": [arm[0] for arm in arms], "omitted_tail128": a.skip_tail128,
        "interpretation": "HTTP task curve; histories may differ. Separate from core NLL curve.",
        "guard": "One external ROOT pressure guard; no nested guards."})
    rows = []
    for name, mode, fresh in arms:
        command = [sys.executable, "-B", str(LAUNCH), "--mode", mode,
                   "--fresh", str(fresh), "--output", str(output / name),
                   "--cache", str(caches / ("curve-http-cache-" + name)),
                   "--shared", str(shared)]
        save(output / (name + "-command.json"), {"argv": command, "cwd": str(ROOT),
             "mode": mode, "fresh_tail_lower_bound": fresh})
        print(json.dumps({"arm": name, "event": "start", "argv": command}), flush=True)
        started = time.monotonic()
        with (output / (name + "-stdout.log")).open("wb") as stdout, \
                (output / (name + "-stderr.log")).open("wb") as stderr:
            completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        row = {"arm": name, "mode": mode, "fresh_tail_lower_bound": fresh,
               "seconds": time.monotonic() - started, "returncode": completed.returncode,
               "output": str(output / name)}
        rows.append(row)
        save(output / "sweep-result.json", rows)
        print(json.dumps(row), flush=True)
    return 0 if all(row["returncode"] == 0 for row in rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
