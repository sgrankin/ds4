#!/usr/bin/env python3
"""Existing/canonical suffix materialization on the unchanged item19 workflow."""
import argparse
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DRIVER = ROOT / 'speed-bench/experiments/concurrent-sessions-hour19-20261004/run.py'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("existing-before", "canonical", "existing-after"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--shared", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=600)
    a = p.parse_args()
    a.mode, a.fresh = "reuse", 64
    a.canonical = 1 if a.arm == "canonical" else 0
    a.output.mkdir(parents=True, exist_ok=False)
    a.cache.mkdir(parents=True, exist_ok=False)
    a.store.mkdir(parents=True, exist_ok=False)
    a.shared.mkdir(parents=True, exist_ok=True)
    arm_started = time.monotonic()
    deadline = arm_started + a.timeout
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DS4", "PT_"))}
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                              (ROOT / "ds4_metal.m").read_text()))
    env.update({k: str(ROOT / v) for k, v in shaders.items()})
    env.update(DS4_STATE_CACHE_DIR=str(a.cache.resolve()), DS4_STATE_CACHE_HOT_MB="0",
               DS4_STATE_CACHE_SSD_MB="4096", DS4_SERVER_RESPONSE_STORE_DIR=str(a.store.resolve()),
               DS4_METAL_V41_RESIDENT_ROUTE_GAP="0", DS4_METAL_V41_SELECTED_TILE="64",
               DS4_METAL_V41_SELECTED_LIMIT="1024",
               DS4_SERVER_CONTEXT_SUFFIX_REUSE="1" if a.mode == "reuse" else "0",
               DS4_SERVER_CONTEXT_SUFFIX_FRESH_TAIL=str(a.fresh),
               DS4_SERVER_CONTEXT_SUFFIX_CANONICAL=str(a.canonical))
    outcomes = []
    for phase in ("before", "after"):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        command = [str(ROOT / "ds4-server"), "--model", str(ROOT / "gguf/DeepSeek-V4.1-Flash-Q2.gguf"),
                   "--host", "127.0.0.1", "--port", str(port), "--ssd-streaming",
                   "--ssd-streaming-cache-experts", "64GB", "--ctx", "8192",
                   "--prefill-chunk", "512", "--batched-session", "2", "--tokens", "128"]
        save(a.output / f"{phase}-execution.json", {"command": command, "cwd": str(ROOT),
             "environment": {k: v for k, v in env.items() if k.startswith("DS4")},
             "arm": a.arm, "canonical_suffix_requested": a.canonical,
             "hot_mb": 0, "fixed_fresh_tail": 64, "driver": str(DRIVER)})
        started, rc, error = time.monotonic(), None, None
        startup_seconds = client_seconds = shutdown_seconds = None
        client_started = None
        client_command = [sys.executable, "-B", str(DRIVER),
            "--base-url", f"http://127.0.0.1:{port}", "--phase", phase, "--mode", a.mode,
            "--output", str(a.output / "http"), "--shared", str(a.shared),
            "--server-log", str(a.output / f"{phase}-server.stderr.log")]
        save(a.output / f"{phase}-client-command.json", {"argv": client_command, "cwd": str(ROOT)})
        with (a.output / f"{phase}-server.stdout.log").open("wb") as stdout, \
                (a.output / f"{phase}-server.stderr.log").open("wb") as stderr:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            try:
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"server exited during startup: {proc.returncode}")
                    if time.monotonic() >= min(started + 120, deadline):
                        raise TimeoutError("server startup/campaign timeout")
                    health = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                    try:
                        health.request("GET", "/v1/models")
                        response = health.getresponse()
                        response.read()
                        if response.status == 200:
                            break
                    except OSError:
                        pass
                    finally:
                        health.close()
                    time.sleep(.2)
                startup_seconds = time.monotonic() - started
                save(a.output / f"{phase}-startup.json", {
                     "seconds": startup_seconds, "pid": proc.pid, "port": port})
                client_started = time.monotonic()
                rc = subprocess.run(client_command, cwd=ROOT,
                    timeout=max(.1, deadline-time.monotonic())).returncode
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if client_started is not None:
                    client_seconds = time.monotonic() - client_started
                shutdown_started = time.monotonic()
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
                shutdown_seconds = time.monotonic() - shutdown_started
        outcomes.append({"phase": phase, "returncode": rc,
                         "seconds": time.monotonic()-started, "startup_seconds": startup_seconds,
                         "client_seconds": client_seconds, "shutdown_seconds": shutdown_seconds,
                         "server_returncode": proc.returncode, "error": error})
        print(json.dumps(outcomes[-1]), flush=True)
        save(a.output / "result.json", outcomes)
        if error or time.monotonic() >= deadline:
            break
    save(a.output / "launcher-summary.json", {"mode": a.mode, "fresh_tail": a.fresh, "hot_mb": 0, "arm": a.arm, "canonical_suffix_requested": a.canonical, "driver": str(DRIVER),
         "seconds": time.monotonic()-arm_started, "phases": outcomes,
         "shared": str(a.shared.resolve()), "cache": str(a.cache.resolve()),
         "store": str(a.store.resolve()), "timeout_seconds": a.timeout})
    return 0 if len(outcomes) == 2 and all(x["returncode"] == 0 for x in outcomes) else 3


if __name__ == "__main__":
    raise SystemExit(main())
