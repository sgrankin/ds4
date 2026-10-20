#!/usr/bin/env python3
"""Full-input Responses replay against an already ROOT-owned, guarded server."""
import argparse
import csv
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def output_text(body):
    return "".join(part.get("text", "") for item in body.get("output", [])
                   if item.get("type") == "message"
                   for part in item.get("content", []) if part.get("type") == "output_text")


def grade(text, gold):
    if gold == "READY":
        return text.strip() == gold
    try:
        answer, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
        return all(answer.get(k) == v for k, v in gold.items())
    except (ValueError, AttributeError):
        return False


def log_tail(path, offset):
    if path is None or not path.exists():
        return "", offset
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read()
        return data.decode("utf-8", errors="replace"), stream.tell()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--mode", choices=("native", "reuse"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--server-log", type=Path)
    p.add_argument("--session-id", default="ds4-hour10-ordinary-context")
    p.add_argument("--delete-a", type=int, default=48)
    p.add_argument("--delete-b", type=int, default=32)
    p.add_argument("--timeout", type=float, default=240)
    a = p.parse_args()
    if not 0 <= a.delete_a <= 48 or not 0 <= a.delete_b <= 32:
        p.error("delete-a must be 0..48 and delete-b 0..32")
    a.output.mkdir(parents=True, exist_ok=False)
    first = (HERE / "fixtures/disposable-a.txt").read_text().splitlines(keepends=True)
    second = (HERE / "fixtures/disposable-b.txt").read_text().splitlines(keepends=True)
    active = (HERE / "fixtures/active.txt").read_text()
    stable = (HERE / "fixtures/unchanged-suffix.txt").read_text()
    intro = "REFERENCE DOCUMENT. Obsolete notes may be deleted; active facts remain authoritative.\n"
    contexts = [intro + "".join(first) + "".join(second) + active + stable,
                intro + "".join(first[a.delete_a:]) + "".join(second) + active + stable,
                intro + "".join(first[a.delete_a:]) + "".join(second[a.delete_b:]) + active + stable]
    questions = ["Read the reference document. Reply exactly READY.",
                 "Using the active facts, calculate packages * units_per_package - damaged_units. "
                 "Reply only a JSON object with the integer key total.",
                 "Retrieve the archive_code and reserve_count from the retained active facts. "
                 "Reply only JSON with keys code (string) and reserve (integer)."]
    golds = ["READY", {"total": 416}, {"code": "MAPLE-42", "reserve": 73}]
    instructions = "Follow the newest user question. Use the active reference facts; ignore obsolete notes."
    replies, rows = [], []
    offset = a.server_log.stat().st_size if a.server_log and a.server_log.exists() else 0
    dump(a.output / "execution.json", {"mode": a.mode, "base_url": a.base_url,
         "session_id": a.session_id, "delete_a_lines": a.delete_a,
         "delete_b_lines": a.delete_b, "source_and_edited_contexts": "context-*.txt",
         "server_started_by_driver": False, "reuse_server_env": {
             "DS4_SERVER_CONTEXT_SUFFIX_REUSE": "1", "fresh_tail_tokens": 128},
         "source_warmup_in_total": True, "max_output_tokens": 32})
    started = time.monotonic()
    for index, (context, question, gold) in enumerate(zip(contexts, questions, golds)):
        name = ("source", "edited-1", "edited-2")[index]
        (a.output / f"context-{index}.txt").write_text(context)
        inputs = [{"role": "user", "content": contexts[index] + "\n" + questions[0]}]
        for previous in range(index):
            inputs.append({"role": "assistant", "content": replies[previous]})
            if previous + 1 <= index:
                inputs.append({"role": "user", "content": questions[previous + 1]})
        payload = {"model": "deepseek-chat", "instructions": instructions, "input": inputs,
                   "prompt_cache_key": a.session_id, "store": False, "stream": False,
                   "temperature": 0, "seed": 20261004, "max_output_tokens": 32,
                   "reasoning": {"effort": "none"}}
        dump(a.output / f"{name}-request.json", payload)
        request = urllib.request.Request(a.base_url.rstrip("/") + "/v1/responses",
                  data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        t = time.monotonic()
        body, error, status = {}, None, None
        try:
            with urllib.request.urlopen(request, timeout=a.timeout) as response:
                status = response.status
                raw = response.read()
            (a.output / f"{name}-response.json").write_bytes(raw)
            body = json.loads(raw)
        except urllib.error.HTTPError as exc:
            status = exc.code
            (a.output / f"{name}-response.json").write_bytes(exc.read())
            error = str(exc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.monotonic() - t
        text = output_text(body)
        replies.append(text)
        new_log, offset = log_tail(a.server_log, offset)
        (a.output / f"{name}-server.log").write_text(new_log)
        suffix_lines = [line for line in new_log.splitlines() if "suffix" in line.lower()]
        usage = body.get("usage") or {}
        complete = body.get("status") == "completed"
        row = {"mode": a.mode, "phase": name, "seconds": seconds, "http_status": status,
               "response_status": body.get("status"), "text": text, "gold": gold,
               "correct": grade(text, gold), "complete": complete, "error": error,
               "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
               "usage": usage, "incomplete_details": body.get("incomplete_details"),
               "suffix_log_lines": suffix_lines}
        row["pass"] = status == 200 and not error and row["correct"] and complete
        rows.append(row)
        dump(a.output / f"{name}-result.json", row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if status != 200 or error:
            break
    result = {"mode": a.mode, "requests": rows, "seconds": time.monotonic() - started,
              "source_warmup_seconds": rows[0]["seconds"],
              "edited_seconds": sum(row["seconds"] for row in rows[1:]),
              "pass": len(rows) == 3 and all(row["pass"] for row in rows)}
    dump(a.output / "result.json", result)
    fields = ["mode", "phase", "seconds", "http_status", "response_status", "input_tokens",
              "output_tokens", "correct", "complete", "pass", "error", "text"]
    with (a.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
