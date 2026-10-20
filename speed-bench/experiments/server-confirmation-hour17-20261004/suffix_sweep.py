#!/usr/bin/env python3
"""Fresh-facts confirmation: reuse64 then native, with one external ROOT guard."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARMS = (("reuse64", "reuse", 64), ("native", "native", 64))


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HERE / "suffix-http")
    p.add_argument("--cache-root", type=Path,
                   default=ROOT / ".workspaces/server-agent-day-20261004")
    p.add_argument("--shared", type=Path,
                   default=ROOT / ".workspaces/server-agent-day-20261004/confirmation-suffix-shared")
    p.add_argument("--timeout", type=float, default=600)
    a = p.parse_args()
    output, shared, caches = a.output.resolve(), a.shared.resolve(), a.cache_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    save(output / "sweep-execution.json", {"argv": sys.argv, "cwd": str(ROOT),
         "order": [arm[0] for arm in ARMS], "shared_visible_paths": str(shared),
         "wall_cap_seconds": a.timeout, "model": "deepseek-v4.1-flash",
         "scope": "New facts in existing workflow family; no generated-history paired NLL.",
         "guard": "One external ROOT pressure guard; no nested guards."})
    rows = []
    for name, mode, fresh in ARMS:
        remaining = a.timeout - (time.monotonic()-started)
        if remaining <= 0:
            rows.append({"arm": name, "error": "campaign wall cap before arm", "returncode": None})
            break
        command = [sys.executable, "-B", str(HERE / "suffix_launch.py"), "--mode", mode,
                   "--fresh", str(fresh), "--output", str(output / name),
                   "--cache", str(caches / ("confirmation-suffix-cache-" + name)),
                   "--store", str(caches / ("confirmation-suffix-store-" + name)),
                   "--shared", str(shared), "--timeout", str(max(.1, remaining-5))]
        save(output / (name + "-command.json"), {"argv": command, "cwd": str(ROOT),
             "mode": mode, "fresh_tail_lower_bound": fresh})
        print(json.dumps({"arm": name, "event": "start", "argv": command}), flush=True)
        arm_start, timed_out = time.monotonic(), False
        with (output / (name + "-stdout.log")).open("wb") as stdout, \
                (output / (name + "-stderr.log")).open("wb") as stderr:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                rc = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Still owned and unreaped: terminate all launcher/server/client descendants.
                import os
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(.2)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                rc = proc.wait(timeout=2)
        row = {"arm": name, "mode": mode, "fresh_tail_lower_bound": fresh,
               "seconds": time.monotonic()-arm_start, "returncode": rc,
               "timed_out": timed_out, "output": str(output / name)}
        rows.append(row)
        save(output / "sweep-result.json", rows)
        print(json.dumps(row), flush=True)
        if timed_out:
            break
    save(output / "sweep-result.json", rows)
    return 0 if len(rows) == 2 and all(row.get("returncode") == 0 for row in rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
