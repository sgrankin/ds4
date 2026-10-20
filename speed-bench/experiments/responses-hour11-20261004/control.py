#!/usr/bin/env python3
"""Four greedy diagnostic requests: full public replay versus stored continuation.

Uses the saved child because the qualification deliberately deleted its parent.
No server lifecycle, fixture edits, retries, or generated-output repairs.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

from run import TOOL, call_item, final_gold, instruction, output_text, response_id


def user(text):
    return {"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": text}]}


def pairs(manifest):
    f = manifest["fixture"]
    parent, child, pending = manifest["parent"], manifest["child"], manifest["pending_tool"]
    if not response_id(child) or not response_id(pending) or call_item(pending) is None:
        raise ValueError("saved child and pending tool must have valid response/call ids")
    original = "Memory literal is " + f["literal"] + ". Current count is 37. Report these facts."
    recall = "Report the remembered literal and current count."
    question = "The current count is now 82. Report memory and current count."
    history = [user(original), *parent["output"], user(recall), *child["output"]]
    count_common = {"instructions": instruction(f["fork_marker"])}
    count = {"name": "changed-instructions-count", "anchor": "saved-child",
             "gold": {"memory": f["literal"], "count": 82, "marker": f["fork_marker"]},
             "full": count_common | {"input": [*history, user(question)]},
             "previous": count_common | {"previous_response_id": child["id"], "input": question}}
    call = call_item(pending)
    tool_output = {"type": "function_call_output", "call_id": call["call_id"], "output": '{"count":61}'}
    tool_question = 'Call lookup_count with key "subject" now. Wait for its result.'
    tool_common = {"instructions": instruction("TOOL_FINAL_" + f["nonce"]),
                   "tools": [TOOL], "tool_choice": "auto"}
    tool = {"name": "changed-instructions-tool-output", "anchor": "pending-tool",
            "gold": {"memory": f["literal"], "count": 61, "marker": "TOOL_FINAL_" + f["nonce"]},
            "full": tool_common | {"input": [user(original), *parent["output"], user(tool_question),
                                                *pending["output"], tool_output]},
            "previous": tool_common | {"previous_response_id": pending["id"], "input": [tool_output]}}
    return [count, tool]


def post(base, payload):
    start = time.monotonic()
    request = urllib.request.Request(base + "/responses", data=json.dumps(payload).encode(), method="POST",
                                    headers={"Content-Type": "application/json", "Connection": "close"})
    try:
        with urllib.request.urlopen(request, timeout=120) as reply:
            status, wire = reply.status, reply.read().decode("utf-8")
        error = None
    except urllib.error.HTTPError as exc:
        status, wire, error = exc.code, exc.read().decode("utf-8", errors="replace"), None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        status, wire, error = None, "", type(exc).__name__ + ": " + str(exc)
    try: body = json.loads(wire)
    except ValueError: body = None
    usage = body.get("usage") or {} if isinstance(body, dict) else {}
    return {"request": payload, "http_status": status, "seconds": time.monotonic()-start,
            "error": error, "wire": wire, "body": body, "actual_text": output_text(body),
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "response_status": body.get("status") if isinstance(body, dict) else None,
            "incomplete_details": body.get("incomplete_details") if isinstance(body, dict) else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    planned = pairs(manifest)  # Assemble the independent golds before the first request.
    base = args.base_url.rstrip("/"); base = base if base.endswith("/v1") else base + "/v1"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        results = []
        for pair in planned:
            arms = {}
            for arm in ("full", "previous"):
                payload = {"model": manifest["model"], "store": False, "temperature": 0,
                           "reasoning": {"effort": "none"}, "max_output_tokens": 96} | pair[arm]
                arms[arm] = post(base, payload)
                arms[arm]["gold_pass"] = arms[arm]["http_status"] == 200 and final_gold(arms[arm]["body"], pair["gold"])
                print(json.dumps({"pair": pair["name"], "arm": arm, "http_status": arms[arm]["http_status"],
                                  "input_tokens": arms[arm]["input_tokens"], "gold_pass": arms[arm]["gold_pass"],
                                  "actual_text": arms[arm]["actual_text"]}), flush=True)
            a, b = arms["full"], arms["previous"]
            result = {"name": pair["name"], "anchor": pair["anchor"], "independent_gold": pair["gold"], "arms": arms,
                      "pair_agreement": a["response_status"] == b["response_status"] == "completed" and a["actual_text"] == b["actual_text"],
                      "prompt_input_tokens_match": a["input_tokens"] is not None and a["input_tokens"] == b["input_tokens"],
                      "input_token_delta_previous_minus_full": b["input_tokens"] - a["input_tokens"] if a["input_tokens"] is not None and b["input_tokens"] is not None else None,
                      "both_gold": a["gold_pass"] and b["gold_pass"]}
            results.append(result)
            print(json.dumps({key: result[key] for key in ("name", "pair_agreement", "prompt_input_tokens_match", "input_token_delta_previous_minus_full", "both_gold")}), flush=True)
        result = {"utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
                  "manifest": str(args.manifest.resolve()), "base_url": base, "pairs": results,
                  "interpretation": "Greedy first attempts; same current instructions and facts, full public input versus stored replay. Original sampled failures retained separately. No retries or repairs; timings diagnostic."}
        json.dump(result, stream, indent=2); stream.write("\n")
    return 0 if all(r["both_gold"] and r["pair_agreement"] and r["prompt_input_tokens_match"] for r in results) else 3


if __name__ == "__main__": sys.exit(main())
