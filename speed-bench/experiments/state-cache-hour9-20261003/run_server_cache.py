#!/usr/bin/env python3
"""Two simultaneous clients, then disk-cache replay after an owned restart.

The existing pressure guard wraps this driver. GET /v1/models is the server's
startup health endpoint. No tokenizer or model metadata is opened by the driver.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
HIT = r"state cache exact hit slot=(\d+) tokens=(\d+)"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--server", default=str(ROOT / "ds4-server"))
    p.add_argument("--output", required=True)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--prefix-lines", type=int, default=160)
    a = p.parse_args()
    if not 0 < a.timeout <= 600 or a.prefix_lines < 1:
        p.error("timeout must be in (0,600]; prefix-lines must be positive")
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cache = out / "state-cache"
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DS4", "PT_"))}
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                              (ROOT / "ds4_metal.m").read_text()))
    env.update({k: str(ROOT / v) for k, v in shaders.items()})
    env.update(DS4_STATE_CACHE_DIR=str(cache), DS4_STATE_CACHE_HOT_MB="0", DS4_STATE_CACHE_SSD_MB="512",
               DS4_METAL_V41_RESIDENT_ROUTE_GAP="0", DS4_METAL_V41_SELECTED_TILE="64",
               DS4_METAL_V41_SELECTED_LIMIT="1024")
    command = [str(Path(a.server).resolve()), "--model", a.model, "--host", "127.0.0.1", "--port", str(port),
               "--ssd-streaming", "--ssd-streaming-cache-experts", "64GB", "--ctx", "8192",
               "--prefill-chunk", "512", "--batched-session", "2", "--tokens", "16"]
    common = ("All following lines are irrelevant reference material. For every request reply exactly READY "
              "and nothing else.\n" +
              "Unchanging reference note: the archive contains ordinary historical records.\n" * a.prefix_lines +
              "Ignore the reference material for your answer. Follow the user's instruction.")
    requests = [{"model": "deepseek-chat", "messages": [
                    {"role": "system", "content": common},
                    {"role": "user", "content": f"This is request {name}. Reply exactly READY."}],
                 "temperature": 0, "max_tokens": 16, "seed": 20261003, "stream": False,
                 "chat_template_kwargs": {"enable_thinking": False}}
                for name in ("A", "B")]
    dump(out / "requests.json", requests)
    dump(out / "execution.json", {"server_command_each_start": command, "cwd": str(ROOT), "port": port,
                                 "env": {k: v for k, v in env.items() if k.startswith("DS4")},
                                 "health": "GET /v1/models", "common_prefix_token_count": "approximately 2K; usage is actual",
                                 "timeout_seconds": a.timeout, "owned_restarts": 1})
    begin = time.monotonic()
    deadline = begin + a.timeout
    phases, error, proc = [], None, None

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("600-second driver deadline reached")
        return value

    def stop_owned():
        nonlocal proc
        if proc is None:
            return None
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        status = proc.returncode
        proc = None
        return status

    def interrupted(signum, _frame):
        raise InterruptedError(f"driver interrupted by signal {signum}")

    previous_signals = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for phase in ("first", "restart"):
            started = time.monotonic()
            with (out / f"{phase}-server.stdout.log").open("wb") as stdout, \
                    (out / f"{phase}-server.stderr.log").open("wb") as stderr:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
                pid = proc.pid
                while True:
                    remaining()
                    if proc.poll() is not None:
                        raise RuntimeError(f"{phase} server exited during startup: {proc.returncode}")
                    health = http.client.HTTPConnection("127.0.0.1", port, timeout=min(1, remaining()))
                    try:
                        health.request("GET", "/v1/models", headers={"Connection": "close"})
                        response = health.getresponse()
                        payload = response.read()
                        if response.status == 200 and isinstance(json.loads(payload).get("data"), list):
                            break
                    except (OSError, http.client.HTTPException, ValueError):
                        pass
                    finally:
                        health.close()
                    time.sleep(min(0.2, remaining()))
                startup_seconds = time.monotonic() - started
                barrier = threading.Barrier(2)

                def complete(index):
                    barrier.wait()
                    t = time.monotonic()
                    row = {"request": index, "status": None, "error": None}
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=min(240, remaining()))
                    try:
                        connection.request("POST", "/v1/chat/completions",
                                           body=json.dumps(requests[index]).encode(),
                                           headers={"Content-Type": "application/json", "Connection": "close"})
                        response = connection.getresponse()
                        row["status"] = response.status
                        row["body"] = response.read().decode("utf-8", errors="replace")
                        parsed = json.loads(row["body"])
                        choice = parsed.get("choices", [{}])[0]
                        row.update(content=choice.get("message", {}).get("content"),
                                   finish_reason=choice.get("finish_reason"), usage=parsed.get("usage"))
                        row["ready"] = isinstance(row["content"], str) and row["content"].strip() == "READY"
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        row["ready"] = False
                    finally:
                        connection.close()
                        row["seconds"] = time.monotonic() - t
                    return row

                pair_start = time.monotonic()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(complete, i) for i in (0, 1)]
                    rows = [future.result(timeout=remaining()) for future in futures]
                pair_seconds = time.monotonic() - pair_start
                returncode = stop_owned()
            log = (out / f"{phase}-server.stderr.log").read_text(errors="replace")
            hits = [{"slot": int(slot), "tokens": int(tokens)} for slot, tokens in re.findall(HIT, log)]
            record = {"phase": phase, "pid": pid, "startup_seconds": startup_seconds,
                      "concurrent_pair_seconds": pair_seconds, "server_returncode": returncode,
                      "requests": rows, "state_cache_exact_hits": hits,
                      "state_cache_enabled": "state cache enabled" in log}
            phases.append(record)
            dump(out / f"{phase}-results.json", record)
            if not all(r["status"] == 200 and not r["error"] for r in rows):
                raise RuntimeError(f"{phase} completion HTTP failure; dependent restart stopped")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        final_returncode = stop_owned()
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
    logs = []
    for phase in ("first", "restart"):
        path = out / f"{phase}-server.stderr.log"
        if path.exists():
            logs.append(f"===== {phase} =====\n" + path.read_text(errors="replace"))
    (out / "server.log").write_text("\n".join(logs))
    requests_ok = len(phases) == 2 and all(r["status"] == 200 and r["ready"] and not r["error"]
                                         and r["finish_reason"] != "length" for phase in phases for r in phase["requests"])
    cache_hit = len(phases) == 2 and bool(phases[1]["state_cache_exact_hits"])
    result = {"seconds": time.monotonic() - begin, "error": error, "phases": phases,
              "expected_answer": "READY", "requests_correct": requests_ok,
              "disk_restart_exact_hit": cache_hit, "hit_regex": HIT,
              "cleanup_returncode": final_returncode,
              "pass": bool(not error and requests_ok and cache_hit
                           and all(p["state_cache_enabled"] and p["server_returncode"] == 0 for p in phases))}
    dump(out / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
