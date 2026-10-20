#!/usr/bin/env python3
"""Rightward context expansion with frozen left-only controls; ROOT launches."""
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
DRIVER = HERE / 'run.py'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("native-before", "context-right", "native-after"), required=True)
    p.add_argument("--server", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--visible", type=Path, default=ROOT / ".workspaces/recipe-20261008/right-visible")
    p.add_argument("--global-cap", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--fresh-tail", type=int, choices=(64, 128, 256), default=64)
    a = p.parse_args()
    a.mode, a.fresh = ("reuse" if a.arm == "context-right" else "native"), a.fresh_tail
    a.right = int(a.arm == 'context-right')
    if a.server is None:
        a.server = ROOT / ('ds4-server' if a.right else '.workspaces/recipe-20261008/right-native-server')
    substitutions = 0
    route_settings = {"enabled": bool(substitutions), "gap": .90 if substitutions else 0,
        "max_mass": .25, "max_rank": 64, "max_substitutions": substitutions,
        "max_total_mass": .50, "policy": "replace", "native_weights": bool(substitutions)}
    a.canonical, a.pad = 1, 1
    a.output.mkdir(parents=True, exist_ok=False)
    a.cache.mkdir(parents=True, exist_ok=False)
    a.store.mkdir(parents=True, exist_ok=False)
    a.visible.mkdir(parents=True, exist_ok=True)
    arm_started = time.monotonic()
    deadline = arm_started + a.timeout
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DS4", "PT_"))}
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                              (ROOT / "ds4_metal.m").read_text()))
    env.update({k: str(ROOT / v) for k, v in shaders.items()})
    env.update(DS4_STATE_CACHE_DIR=str(a.cache.resolve()), DS4_STATE_CACHE_HOT_MB="0",
               DS4_STATE_CACHE_SSD_MB="4096", DS4_SERVER_RESPONSE_STORE_DIR=str(a.store.resolve()),
               DS4_METAL_V41_RESIDENT_ROUTE_GAP="0", DS4_METAL_V41_RESIDENT_PREFILL_GAP="0",
               DS4_METAL_V41_SELECTED_TILE="64",
               DS4_METAL_V41_SELECTED_LIMIT="1024",
               DS4_SERVER_CONTEXT_SUFFIX_REUSE="1",
               DS4_SERVER_CONTEXT_SUFFIX_RIGHT=str(a.right),
               DS4_SERVER_CONTEXT_SUFFIX_FRESH_TAIL=str(a.fresh),
               DS4_SERVER_CONTEXT_SUFFIX_CANONICAL=str(a.canonical),
               DS4_SERVER_CONTEXT_SUFFIX_PAD=str(a.pad))
    env["DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS"] = "0"
    outcomes = []
    for phase in ("dedupe", "bucket"):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        command = [str(a.server.resolve()), "--model", str(ROOT / "gguf/DeepSeek-V4.1-Flash-Q2.gguf"),
                   "--host", "127.0.0.1", "--port", str(port), "--ssd-streaming",
                   "--ssd-streaming-cache-experts", "64GB", "--ctx", "8192",
                   "--prefill-chunk", "512", "--batched-session", "2", "--tokens", str(a.global_cap)]
        save(a.output / f"{phase}-execution.json", {"command": command, "cwd": str(ROOT),
             "environment": {k: v for k, v in env.items() if k.startswith("DS4")},
             "mode": a.mode, "arm": a.arm, "route_settings": route_settings, "server": str(a.server.resolve()), "canonical_suffix_requested": a.canonical, "padding_suffix_requested": a.pad,
             "hot_mb": 0, "fixed_fresh_tail": a.fresh, "right_shift_enabled": bool(a.right), "driver": str(DRIVER)})
        started, rc, error = time.monotonic(), None, None
        startup_seconds = client_seconds = shutdown_seconds = None
        client_started = None
        client_command = [sys.executable, "-B", str(DRIVER),
            "--base-url", f"http://127.0.0.1:{port}", "--arm", a.arm, "--case", phase,
            "--output", str(a.output / "http" / phase), "--visible", str(a.visible),
            "--global-cap", str(a.global_cap)]
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
    save(a.output / "launcher-summary.json", {"mode": a.mode, "arm": a.arm, "route_settings": route_settings, "fresh_tail": a.fresh, "hot_mb": 0, "server": str(a.server.resolve()), "canonical_suffix_requested": a.canonical, "padding_suffix_requested": a.pad, "driver": str(DRIVER),
         "right_shift_enabled": bool(a.right), "seconds": time.monotonic()-arm_started, "phases": outcomes,
         "visible": str(a.visible.resolve()), "cache": str(a.cache.resolve()),
         "store": str(a.store.resolve()), "timeout_seconds": a.timeout})
    return 0 if len(outcomes) == 2 and all(x["returncode"] == 0 for x in outcomes) else 3


if __name__ == "__main__":
    raise SystemExit(main())
