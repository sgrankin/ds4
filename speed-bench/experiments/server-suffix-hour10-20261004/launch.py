#!/usr/bin/env python3
"""Own one server and run the fixed HTTP screen; the existing guard wraps us."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("native", "reuse"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--fresh", type=int, default=128)
    p.add_argument("--delete-a", type=int, default=48)
    p.add_argument("--delete-b", type=int, default=32)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    a.cache.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DS4", "PT_"))}
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^"]+)"\]',
                              (ROOT / "ds4_metal.m").read_text()))
    env.update({k: str(ROOT / v) for k, v in shaders.items()})
    env.update(DS4_STATE_CACHE_DIR=str(a.cache.resolve()), DS4_STATE_CACHE_HOT_MB="0",
               DS4_STATE_CACHE_SSD_MB="1024", DS4_METAL_V41_RESIDENT_ROUTE_GAP="0",
               DS4_METAL_V41_SELECTED_TILE="64", DS4_METAL_V41_SELECTED_LIMIT="1024",
               DS4_SERVER_CONTEXT_SUFFIX_REUSE="1" if a.mode == "reuse" else "0",
               DS4_SERVER_CONTEXT_SUFFIX_FRESH_TAIL=str(a.fresh))
    command = [str(ROOT / "ds4-server"), "--model", str(ROOT / "gguf/DeepSeek-V4.1-Flash-Q2.gguf"),
               "--host", "127.0.0.1", "--port", str(port), "--ssd-streaming",
               "--ssd-streaming-cache-experts", "64GB", "--ctx", "8192",
               "--prefill-chunk", "512", "--batched-session", "2", "--tokens", "32"]
    (a.output / "server-execution.json").write_text(json.dumps(
        {"command": command, "cwd": str(ROOT), "environment": {k: v for k, v in env.items()
          if k.startswith("DS4")}}, indent=2) + "\n")
    started = time.monotonic()
    with (a.output / "server.stdout.log").open("wb") as stdout, \
            (a.output / "server.stderr.log").open("wb") as stderr:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        try:
            deadline = started + 120
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited during startup: {proc.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("server startup timeout")
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
            (a.output / "startup.json").write_text(json.dumps(
                {"seconds": time.monotonic() - started, "pid": proc.pid, "port": port}) + "\n")
            completed = subprocess.run([sys.executable, "-B", str(HERE / "run.py"),
                "--base-url", f"http://127.0.0.1:{port}", "--mode", a.mode,
                "--output", str(a.output / "http"), "--server-log", str(a.output / "server.stderr.log"),
                "--delete-a", str(a.delete_a), "--delete-b", str(a.delete_b)], cwd=ROOT, timeout=900)
            return completed.returncode
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
