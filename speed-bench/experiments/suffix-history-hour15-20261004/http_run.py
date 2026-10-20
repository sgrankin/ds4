#!/usr/bin/env python3
"""Two fresh retrieval checks against ROOT's already guarded HTTP server."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
TOOLS = [{"type":"function", "name":"read", "description":"Read the advertised JSON policy file in full.",
          "parameters":{"type":"object", "properties":{
              "path":{"type":"string"}, "whole":{"type":"boolean"}, "raw":{"type":"boolean"}},
              "required":["path", "whole", "raw"], "additionalProperties":False}}]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def final_text(body):
    return "".join(part.get("text", "") for item in body.get("output", [])
                   if item.get("type") == "message"
                   for part in item.get("content", []) if part.get("type") == "output_text")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--final-cap", type=int, default=32)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, gold in (("helpful-history", 456), ("obsolete-history", 450)):
        out = args.output / name
        out.mkdir()
        policy = (HERE / "fixtures" / name / "policy.json").resolve()
        history = [{"role":"user", "content":
            f"Read the current shipping policy from {policy} using read with whole=true and raw=true. "
            "The new shipment has 18 batches, 26 units_per_batch, and 9 reserved_units. "
            "Compute net units, apply the current file's rounding policy, and reply only with the final integer."}]
        calls, reads, replies, errors = [], [], [], []
        text, complete = "", False
        started = time.monotonic()
        for index in range(3):
            cap = 128 if not reads else args.final_cap
            payload = {"model":"deepseek-v4.1-flash", "input":history,
                       "instructions":"Read the advertised file before calculating. Follow its current rule; never guess from a previous answer.",
                       "tools":TOOLS, "tool_choice":"auto", "prompt_cache_key":"ds4-hour15-retrieval-" + name,
                       "store":False, "stream":False, "temperature":0, "seed":20261004,
                       "max_output_tokens":cap, "reasoning":{"effort":"none"}}
            save(out / f"request-{index}.json", payload)
            request = urllib.request.Request(args.base_url.rstrip("/") + "/v1/responses",
                data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
            body, status = {}, None
            t = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=240) as response:
                    status, raw = response.status, response.read()
                (out / f"response-{index}.json").write_bytes(raw)
                body = json.loads(raw)
            except urllib.error.HTTPError as exc:
                status = exc.code
                (out / f"response-{index}.json").write_bytes(exc.read())
                errors.append(str(exc))
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
            replies.append({"seconds":time.monotonic()-t, "http_status":status,
                            "max_output_tokens":cap, "response_status":body.get("status"),
                            "usage":body.get("usage"), "incomplete_details":body.get("incomplete_details"),
                            "stop_token_observed":None})
            output = body.get("output") or []
            history.extend(output)
            text = final_text(body)
            complete = status == 200 and body.get("status") == "completed"
            pending = [item for item in output if item.get("type") == "function_call"]
            if not complete or not pending:
                break
            for call in pending:
                record = {"name":call.get("name"), "call_id":call.get("call_id"),
                          "arguments":call.get("arguments"), "read":False, "error":None}
                result = "ERROR: only the advertised policy read is permitted."
                try:
                    supplied = json.loads(call.get("arguments", "{}"))
                    path = Path(supplied["path"]).resolve()
                    if (call.get("name") != "read" or path != policy or
                            supplied.get("whole") is not True or supplied.get("raw") is not True):
                        raise ValueError("read must target the advertised policy with whole=true/raw=true")
                    read_started = time.monotonic()
                    result = path.read_text()
                    record.update(read=True, path=str(path), text=result,
                                  read_ms=(time.monotonic()-read_started)*1000)
                    reads.append(str(path))
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    errors.append(record["error"])
                calls.append(record)
                history.append({"type":"function_call_output", "call_id":call.get("call_id"), "output":result})
            save(out / "history.json", history)
        else:
            errors.append("three HTTP responses exhausted without a final answer")
            complete = False
        answer = int(text.strip()) if text.strip().isascii() and text.strip().isdigit() else None
        row = {"case":name, "gold":gold, "answer":answer, "text":text,
               "correct":answer == gold, "complete":complete, "required_file":str(policy),
               "actual_read_count":len(reads), "calls":calls, "replies":replies,
               "seconds":time.monotonic()-started, "errors":errors,
               "source_warmup":False, "context":"fresh question plus actual policy tool result",
               "scope":"Retrieval correctness; independent of core fresh/reuse contexts and timings."}
        row["pass"] = bool(row["correct"] and complete and len(reads) == 1 and not errors)
        save(out / "history.json", history)
        save(out / "result.json", row)
        rows.append(row)
        save(args.output / "results.json", rows)
        print(json.dumps({k:row[k] for k in ("case", "seconds", "answer", "gold", "complete", "actual_read_count", "pass")}), flush=True)
    return 0 if all(row["pass"] for row in rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
