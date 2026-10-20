#!/usr/bin/env python3
"""One ROOT-guarded model server for the two fresh policy retrieval checks."""
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


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HERE / "http")
    p.add_argument("--final-cap", type=int, default=32)
    args = p.parse_args()
    parent = subprocess.check_output(["ps", "-p", str(os.getppid()), "-o", "command="], text=True)
    if "speed-bench/ideas/pressure_run.py" not in parent:
        p.error("launch as the direct child of MAIN speed-bench/ideas/pressure_run.py")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {k:v for k,v in os.environ.items() if not k.startswith(("DS4", "PT_"))}
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                             (ROOT / "ds4_metal.m").read_text()))
    env.update({k:str(ROOT/v) for k,v in shaders.items()})
    env.update(DS4_STATE_CACHE_DIR=str(out / "cache"), DS4_STATE_CACHE_HOT_MB="0",
               DS4_STATE_CACHE_SSD_MB="1024", DS4_SERVER_RESPONSE_STORE_DIR=str(out / "store"),
               DS4_METAL_V41_RESIDENT_ROUTE_GAP="0", DS4_METAL_V41_SELECTED_TILE="64",
               DS4_METAL_V41_SELECTED_LIMIT="1024", DS4_SERVER_CONTEXT_SUFFIX_REUSE="0")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [str(ROOT / "ds4-server"), "--model", str(ROOT / "gguf/DeepSeek-V4.1-Flash-Q2.gguf"),
               "--host", "127.0.0.1", "--port", str(port), "--ssd-streaming",
               "--ssd-streaming-cache-experts", "64GB", "--ctx", "8192",
               "--prefill-chunk", "512", "--batched-session", "2", "--tokens", "128"]
    save(out / "execution.json", {"command":command, "cwd":str(ROOT),
        "environment":{k:v for k,v in env.items() if k.startswith("DS4")},
        "source_warmup":False, "server_count":1, "retrieval_cases":2,
        "final_cap":args.final_cap,
        "scope":"Fresh file-retrieval controls; no suffix reuse or core latency comparison."})
    started = time.monotonic()
    with (out / "server.stdout.log").open("wb") as stdout, (out / "server.stderr.log").open("wb") as stderr:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        try:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited during startup: {proc.returncode}")
                if time.monotonic()-started >= 120:
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
            save(out / "startup.json", {"seconds":time.monotonic()-started, "pid":proc.pid, "port":port})
            runner = [sys.executable, "-B", str(HERE / "http_run.py"), "--base-url",
                      f"http://127.0.0.1:{port}", "--output", str(out / "checks"),
                      "--final-cap", str(args.final_cap)]
            completed = subprocess.run(runner, cwd=ROOT, timeout=600)
            save(out / "result.json", {"argv":runner, "returncode":completed.returncode,
                                      "seconds":time.monotonic()-started})
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
