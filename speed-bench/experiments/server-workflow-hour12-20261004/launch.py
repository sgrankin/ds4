#!/usr/bin/env python3
"""One guarded model server at a time, with restart between edit families."""
import argparse
import hashlib
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


def native_prefix_probe(output, port, server_log):
    """Replay one stored native input; one generated token is enough to load it."""
    payload = json.loads((output / "http/before/A-source-0-request.json").read_text())
    payload["max_output_tokens"] = 1
    (output / "native-prefix-probe-request.json").write_text(json.dumps(payload, indent=2) + "\n")
    offset = server_log.stat().st_size
    started = time.monotonic()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=240)
    try:
        connection.request("POST", "/v1/responses", body=json.dumps(payload).encode(),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        status, raw = response.status, response.read()
    finally:
        connection.close()
    (output / "native-prefix-probe-response.json").write_bytes(raw)
    with server_log.open("rb") as stream:
        stream.seek(offset)
        log = stream.read().decode(errors="replace")
    (output / "native-prefix-probe-server.log").write_text(log)
    hits = [line for line in log.splitlines() if "state cache exact hit" in line]
    result = {"http_status": status, "seconds": time.monotonic() - started,
              "exact_hit_lines": hits, "pass": status == 200 and bool(hits),
              "scope": "Unchanged native-prefix restore only; one-token output is not a task grade."}
    (output / "native-prefix-probe-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("native", "reuse"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--shared", type=Path, required=True)
    p.add_argument("--fresh", type=int, default=128)
    p.add_argument("--model", type=Path, default=ROOT / "gguf/DeepSeek-V4.1-Flash-Q2.gguf")
    p.add_argument("--compressed-kv-f16", action="store_true")
    p.add_argument("--native-prefix-probe", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    a.cache.parent.mkdir(parents=True, exist_ok=True)
    a.shared.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DS4", "PT_"))}
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                              (ROOT / "ds4_metal.m").read_text()))
    env.update({k: str(ROOT / v) for k, v in shaders.items()})
    env.update(DS4_STATE_CACHE_DIR=str(a.cache.resolve()), DS4_STATE_CACHE_HOT_MB="0",
               DS4_STATE_CACHE_SSD_MB="1024", DS4_METAL_V41_RESIDENT_ROUTE_GAP="0",
               DS4_METAL_V41_SELECTED_TILE="64", DS4_METAL_V41_SELECTED_LIMIT="1024",
               DS4_SERVER_CONTEXT_SUFFIX_REUSE="1" if a.mode == "reuse" else "0",
               DS4_SERVER_CONTEXT_SUFFIX_FRESH_TAIL=str(a.fresh))
    if a.compressed_kv_f16:
        env.update(DS4_V41_COMPRESSED_KV_F16="1",
                   DS4_SERVER_CONTEXT_SUFFIX_CANONICAL="1", DS4_SERVER_CONTEXT_SUFFIX_PAD="1")
    server_sha256 = hashlib.sha256((ROOT / "ds4-server").read_bytes()).hexdigest()
    outcomes = []
    for phase in ("before", "after"):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        command = [str(ROOT / "ds4-server"), "--model", str(a.model.resolve()),
                   "--host", "127.0.0.1", "--port", str(port), "--ssd-streaming",
                   "--ssd-streaming-cache-experts", "64GB", "--ctx", "8192",
                   "--prefill-chunk", "512", "--batched-session", "2", "--tokens", "128"]
        (a.output / f"{phase}-execution.json").write_text(json.dumps(
            {"command": command, "cwd": str(ROOT), "environment": {k: v for k, v in env.items()
             if k.startswith("DS4")}, "server_sha256": server_sha256,
             "native_prefix_probe": a.native_prefix_probe}, indent=2) + "\n")
        started = time.monotonic()
        with (a.output / f"{phase}-server.stdout.log").open("wb") as stdout, \
                (a.output / f"{phase}-server.stderr.log").open("wb") as stderr:
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
                (a.output / f"{phase}-startup.json").write_text(json.dumps(
                    {"seconds": time.monotonic() - started, "pid": proc.pid, "port": port}) + "\n")
                probe = None
                if phase == "after" and a.native_prefix_probe:
                    probe = native_prefix_probe(a.output, port, a.output / f"{phase}-server.stderr.log")
                completed = subprocess.run([sys.executable, "-B", str(HERE / "run.py"),
                    "--base-url", f"http://127.0.0.1:{port}", "--phase", phase, "--mode", a.mode,
                    "--output", str(a.output / "http"), "--shared", str(a.shared),
                    "--server-log", str(a.output / f"{phase}-server.stderr.log")], cwd=ROOT, timeout=1200)
                outcomes.append({"phase": phase, "returncode": completed.returncode,
                                 "native_prefix_probe": probe})
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
    (a.output / "result.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    return int(any(x["returncode"] or
                   (x["native_prefix_probe"] is not None and not x["native_prefix_probe"]["pass"])
                   for x in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
