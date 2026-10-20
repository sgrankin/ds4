#!/usr/bin/env python3
"""Four scheduler arms, to be wrapped ONCE by the existing pressure_run.py.

No tokenizer, warmup request, model retry, or application SSD KV cache. Each
arm owns one fresh server; the outer guard owns the full driver/process tree.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
QUANTA = (128, 512, 2048, 128)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("arm deadline reached")
    return seconds


def stop_owned(proc):
    if proc is None:
        return None
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
    return proc.returncode


def streamed(label, payload, base, out, origin, deadline, first_content=None):
    row = {"label": label, "started_s": time.monotonic() - origin,
           "content_events": [], "usage": None, "finish_reason": None,
           "done": False, "error": None}
    pieces = []
    request = urllib.request.Request(base + "/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Connection": "close"})
    try:
        with (out / f"{label}.sse").open("wb") as raw, \
                (out / f"{label}.events.jsonl").open("w") as events, \
                urllib.request.urlopen(request, timeout=remaining(deadline)) as response:
            row["http_status"] = response.status
            for line in response:
                at = time.monotonic() - origin
                raw.write(line)
                raw.flush()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                record = {"received_s": at}
                if data == b"[DONE]":
                    record["data"] = "[DONE]"
                    row["done"] = True
                else:
                    obj = json.loads(data)
                    record["data"] = obj
                    if obj.get("error"):
                        raise RuntimeError(str(obj["error"]))
                    if obj.get("usage") is not None:
                        row["usage"] = obj["usage"]
                    for choice in obj.get("choices", []):
                        if choice.get("finish_reason") is not None:
                            row["finish_reason"] = choice["finish_reason"]
                        content = choice.get("delta", {}).get("content")
                        if content:
                            pieces.append(content)
                            row["content_events"].append({"received_s": at, "text": content})
                            if first_content is not None:
                                first_content.set()
                events.write(json.dumps(record, ensure_ascii=False) + "\n")
                events.flush()
                remaining(deadline)
                if row["done"]:
                    break
    except urllib.error.HTTPError as exc:
        row["http_status"] = exc.code
        row["error"] = f"HTTPError {exc.code}: {exc.read().decode(errors='replace')}"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["finished_s"] = time.monotonic() - origin
        row["seconds"] = row["finished_s"] - row["started_s"]
        row["content"] = "".join(pieces)
        row["ttft_seconds"] = (row["content_events"][0]["received_s"] - row["started_s"]
                                if row["content_events"] else None)
        (out / f"{label}.content.txt").write_text(row["content"])
        dump(out / f"{label}.result.json", row)
    return row


def run_arm(args, index, quantum, long_prompt, short_prompt):
    out = args.output / f"{index:02d}-q{quantum}"
    out.mkdir()
    for name in ("state", "cache", "responses"):
        (out / name).mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DS4", "PT_"))}
    env.update(DS4_METAL_V41_RESIDENT_ROUTE_GAP="0",
               DS4_METAL_V41_RESIDENT_PREFILL_GAP="0",
               DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS="0",
               DS4_SERVER_CONTEXT_SUFFIX_REUSE="0",
               DS4_SERVER_RESPONSE_STORE_DIR=str(out / "responses"))
    # State/cache paths are reserved artifacts only. With DS4_STATE_CACHE_DIR
    # absent and no --kv-disk-dir, both application SSD KV cache paths stay off.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [str(args.server), "--model", str(args.model), "--host", "127.0.0.1",
               "--port", str(port), "--ssd-streaming", "--ssd-streaming-cache-experts", "64GB",
               "--batched-session", "2", "--ctx", "16384", "--prefill-chunk", "2048",
               "--mixed-prefill-quantum", str(quantum), "--mtp-draft", "1", "--tokens", "512"]
    dump(out / "execution.json", {"command": command, "cwd": str(ROOT),
        "env": {k: v for k, v in env.items() if k.startswith("DS4")},
        "application_ssd_kv_cache": "disabled: no DS4_STATE_CACHE_DIR or --kv-disk-dir",
        "health": "GET /v1/models", "warmup_requests": 0,
        "timeout_seconds": args.timeout, "reserved_state_dir": str(out / "state"),
        "reserved_cache_dir": str(out / "cache")})
    origin = time.monotonic()
    deadline = origin + args.timeout
    proc = None
    pool = None
    rows = {}
    result = {"arm": index, "mixed_prefill_quantum": quantum, "error": None}
    try:
        with (out / "server.stdout.log").open("wb") as stdout, \
                (out / "server.stderr.log").open("wb") as stderr:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            result["server_pid"] = proc.pid
            base = f"http://127.0.0.1:{port}"
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"owned server exited during startup: {proc.returncode}")
                try:
                    with urllib.request.urlopen(base + "/v1/models", timeout=min(1, remaining(deadline))) as response:
                        models = json.load(response)["data"]
                    if models:
                        break
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(min(.2, remaining(deadline)))
            result["startup_seconds"] = time.monotonic() - origin
            requests = {}
            for label, prompt, limit in (("B", short_prompt, 512), ("A", long_prompt, 32)):
                requests[label] = {"model": models[0]["id"], "stream": True,
                    "stream_options": {"include_usage": True}, "temperature": 0,
                    "max_tokens": limit, "seed": 20261020, "reasoning_effort": "none",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": prompt}]}
            dump(out / "requests.json", requests)
            encoded = {label: json.dumps(value).encode() for label, value in requests.items()}
            first_b = threading.Event()
            pool = ThreadPoolExecutor(max_workers=2)
            b = pool.submit(streamed, "B", encoded["B"], base, out, origin, deadline, first_b)
            while not first_b.wait(timeout=min(.02, remaining(deadline))):
                if b.done():
                    rows["B"] = b.result()
                    raise RuntimeError("B ended before any content; see B.result.json")
            a = pool.submit(streamed, "A", encoded["A"], base, out, origin, deadline)
            rows["B"] = b.result(timeout=remaining(deadline))
            rows["A"] = a.result(timeout=remaining(deadline))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["server_returncode"] = stop_owned(proc)
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        result["arm_seconds"] = time.monotonic() - origin
    # Preserve worker results even if an outer timeout interrupted collection.
    for label in ("B", "A"):
        path = out / f"{label}.result.json"
        if path.exists():
            rows[label] = json.loads(path.read_text())
    result["requests"] = rows
    if "A" in rows and "B" in rows:
        a_start = rows["A"]["started_s"]
        times = [event["received_s"] for event in rows["B"]["content_events"]]
        gaps = [right - left for left, right in zip(times, times[1:]) if right >= a_start]
        result["B_max_content_gap_after_A_start_seconds"] = max(gaps) if gaps else None
        result["B_content_events_after_A_start"] = sum(t >= a_start for t in times)
        result["overlap_observed"] = bool(times and times[-1] >= a_start)
        try:
            parsed_b = json.loads(rows["B"]["content"].strip())
            result["B_correct"] = (parsed_b == list(range(1, 81)) and
                                   all(type(n) is int for n in parsed_b))
        except ValueError:
            result["B_correct"] = False
        result["A_correct"] = rows["A"]["content"].strip() == "CORAL-8431"
        result["A_prompt_tokens"] = (rows["A"].get("usage") or {}).get("prompt_tokens")
        n = result["A_prompt_tokens"]
        result["A_prompt_in_8k_10k_target"] = n is not None and 8000 <= n <= 10000
    result["pass"] = bool(not result["error"] and result.get("A_correct") and
        result.get("B_correct") and result.get("overlap_observed") and
        all(row.get("done") and not row.get("error") and row.get("finish_reason") != "length"
            for row in rows.values()) and len(rows) == 2)
    dump(out / "result.json", result)
    print(json.dumps({"arm": index, "quantum": quantum, "pass": result["pass"],
                      "error": result["error"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--server", type=Path, default=ROOT / "ds4-server")
    parser.add_argument("--output", type=Path, default=ROOT / ".workspaces/upgrade-20261020/quantum")
    parser.add_argument("--timeout", type=float, default=900, help="seconds per arm, including startup")
    parser.add_argument("--log-lines", type=int, default=300, help="fixed log size; actual token count comes from usage")
    args = parser.parse_args()
    if args.timeout <= 0 or args.log_lines < 1:
        parser.error("timeout and log-lines must be positive")
    args.model = args.model.resolve()
    args.server = args.server.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    modules = ("router", "cache", "session", "decode", "prefill", "index", "archive", "transport")
    logs = "".join(f"checkpoint {i:04d}: module={modules[i % len(modules)]} batch={(i * 37) % 997:03d} "
        f"retained={(i * 7919) % 1000000:06d} status=reviewed immutable=true note=historical log only.\n"
        for i in range(args.log_lines))
    long_prompt = ("The following synthetic historical log is context only. Do not execute anything. "
        "Ignore its details when answering the instruction after the log.\n<log>\n" + logs +
        "</log>\nReply only CORAL-8431. No other text, explanation, tools, or Markdown.")
    short_prompt = ("Reply with exactly one JSON array containing the integers 1 through 80 in ascending order. "
                    "Include every integer exactly once. No prose, explanation, tools, or Markdown.")
    dump(args.output / "fixture.json", {"quanta": list(QUANTA), "log_lines": args.log_lines,
        "long_prompt": long_prompt, "short_prompt": short_prompt,
        "long_prompt_sha256": hashlib.sha256(long_prompt.encode()).hexdigest(),
        "expected_A": "CORAL-8431", "expected_B": list(range(1, 81)),
        "timing": "client monotonic timestamps at receipt of SSE data lines; content events can contain multiple characters/tokens",
        "gap_definition": "largest adjacent B content-event gap whose right endpoint is at/after A request start, including the straddling gap",
        "limitations": "Fresh server/expert cache each arm; OS/SSD page cache and machine state carry across arms. No warmup. B TTFT precedes A and is a cold/order diagnostic, not a scheduler benefit. Ordered one-fixture scheduler screen, not general throughput or quality evidence."})
    def interrupted(signum, _frame):
        raise InterruptedError(f"driver interrupted by signal {signum}")
    saved = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    arms = []
    error = None
    try:
        for index, quantum in enumerate(QUANTA, 1):
            row = run_arm(args, index, quantum, long_prompt, short_prompt)
            arms.append(row)
            if row["error"]:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    metrics = {"B_ttft_seconds": lambda r: r["requests"]["B"]["ttft_seconds"],
        "B_total_seconds": lambda r: r["requests"]["B"]["seconds"],
        "B_max_gap_seconds": lambda r: r["B_max_content_gap_after_A_start_seconds"],
        "A_ttft_seconds": lambda r: r["requests"]["A"]["ttft_seconds"],
        "A_total_seconds": lambda r: r["requests"]["A"]["seconds"]}
    comparisons = {}
    if len(arms) == 4 and all(r["error"] is None for r in arms):
        for name, get in metrics.items():
            values = [get(r) for r in arms]
            if any(v is None for v in values):
                continue
            baseline = (values[0] + values[3]) / 2
            comparisons[name] = {"opening_128": values[0], "closing_128": values[3],
                "bracket_mean_128": baseline, "closing_minus_opening_pct":
                (100 * (values[3] / values[0] - 1) if values[0] else None),
                "candidate_512": values[1], "candidate_2048": values[2],
                "saving_vs_bracket_mean_pct": {
                    str(q): (100 * (1 - value / baseline) if baseline else None)
                    for q, value in ((512, values[1]), (2048, values[2]))}}
    result = {"error": error, "arms": arms, "comparisons": comparisons,
              "pass": len(arms) == 4 and not error and all(r["pass"] for r in arms)}
    dump(args.output / "result.json", result)
    print(json.dumps({"output": str(args.output), "pass": result["pass"], "error": error}), flush=True)
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
