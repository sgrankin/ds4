#!/usr/bin/env python3
"""Small real-HTTP durable Responses qualification; server/restart owned by ROOT.

Run --phase before, restart the owned server, then --phase after with the same
output directory. This driver starts no server/model, uses no SDK, and records
first attempts including wrong text, incomplete responses and HTTP failures.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import sys
import time
import urllib.error
import urllib.request

TOOL = {"type": "function", "name": "lookup_count",
        "description": "Get the current count for a named record.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                       "required": ["key"], "additionalProperties": False}}


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2); stream.write("\n")


def instruction(marker):
    return ('When answering, return only JSON with keys "memory", "count", "marker". '
            'Use the known facts. The current instruction marker is "' + marker + '".')


def response_id(body):
    return body.get("id") if isinstance(body, dict) else None


def output_text(body):
    if not isinstance(body, dict): return ""
    return "".join(part.get("text", "") for item in body.get("output", [])
                   if item.get("type") == "message"
                   for part in item.get("content", []) if part.get("type") == "output_text")


def final_gold(body, gold):
    if not isinstance(body, dict) or body.get("status") != "completed": return False
    try: actual = json.loads(output_text(body).strip())
    except (ValueError, TypeError): return False
    return actual == gold


def call_item(body):
    if not isinstance(body, dict): return None
    calls = [item for item in body.get("output", []) if item.get("type") == "function_call"]
    if len(calls) != 1: return None
    call = calls[0]
    try: arguments = json.loads(call.get("arguments", ""))
    except (ValueError, TypeError): return None
    if call.get("name") != "lookup_count" or arguments != {"key": "subject"} or not call.get("call_id"):
        return None
    return call


class Campaign:
    def __init__(self, args):
        self.args = args; self.rows = []; self.start = time.monotonic()
        base = args.base_url.rstrip("/")
        self.base = base if base.endswith("/v1") else base + "/v1"
        self.output = args.output_dir.resolve(); self.output.mkdir(parents=True, exist_ok=True)

    def http(self, method, route, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.base + route, data=data, method=method,
                    headers={"Content-Type": "application/json", "Connection": "close"})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.args.timeout) as reply:
                status, wire = reply.status, reply.read().decode("utf-8")
            error = None
        except urllib.error.HTTPError as exc:
            status, wire, error = exc.code, exc.read().decode("utf-8", errors="replace"), None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            status, wire, error = None, "", type(exc).__name__ + ": " + str(exc)
        try: body = json.loads(wire)
        except ValueError: body = None
        return {"method": method, "route": route, "request": payload, "http_status": status,
                "seconds": time.monotonic() - start, "body": body, "wire": wire, "error": error}

    def record(self, name, exchange, passed, **extra):
        body = exchange.get("body")
        row = dict(name=name, result="ERROR" if exchange.get("error") else "PASS" if passed else "FAIL",
                   **exchange, **extra)
        if isinstance(body, dict):
            row.update(response_id=response_id(body), response_status=body.get("status"),
                       incomplete_details=body.get("incomplete_details"), actual_text=output_text(body),
                       usage=body.get("usage"), store=body.get("store"),
                       previous_response_id=body.get("previous_response_id"))
            row["cap_hit"] = (body.get("status") == "incomplete" and
                              (body.get("incomplete_details") or {}).get("reason") in ("max_tokens", "max_output_tokens"))
        self.rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("name", "result", "http_status", "response_id",
              "response_status", "seconds", "actual_text")}), flush=True)
        return body

    def skip(self, name, reason):
        self.rows.append(dict(name=name, result="SKIP", reason=reason))
        print(json.dumps(self.rows[-1]), flush=True)

    def post(self, name, payload, gold=None, tool=False):
        payload = {"model": self.args.model, "max_output_tokens": 96,
                   "reasoning": {"effort": "none"}} | payload
        exchange = self.http("POST", "/responses", payload)
        body = exchange["body"]
        passed = exchange["http_status"] == 200 and bool(response_id(body))
        if isinstance(body, dict):
            passed &= body.get("store") is payload.get("store", True)
            passed &= body.get("previous_response_id") == payload.get("previous_response_id")
        if gold is not None: passed &= final_gold(body, gold)
        if tool: passed &= isinstance(body, dict) and body.get("status") == "completed" and call_item(body) is not None
        return self.record(name, exchange, passed, independent_gold=gold, expects_function_call=tool)

    def get_exact(self, name, original):
        rid = response_id(original)
        if not rid:
            self.skip(name, "parent POST produced no response id"); return None
        exchange = self.http("GET", "/responses/" + rid)
        body = exchange["body"]
        passed = (exchange["http_status"] == 200 and isinstance(body, dict) and body.get("id") == rid
                  and body.get("output") == original.get("output") and body.get("status") == original.get("status"))
        return self.record(name, exchange, passed, expected_response_id=rid,
                           expected_output=original.get("output"))

    def tool_parent(self, name, parent, marker):
        rid = response_id(parent)
        if not rid: self.skip(name, "missing parent response id"); return None
        return self.post(name, {"store": True, "previous_response_id": rid,
            "instructions": instruction(marker), "input": 'Call lookup_count with key "subject" now. Wait for its result.',
            "tools": [TOOL], "tool_choice": "auto"}, tool=True)

    def tool_continue(self, name, parent, literal, marker, count):
        rid, call = response_id(parent), call_item(parent)
        if not rid or call is None:
            self.skip(name, "first tool request did not produce exactly one valid lookup_count call"); return None
        return self.post(name, {"store": True, "previous_response_id": rid, "instructions": instruction(marker),
            "input": [{"type": "function_call_output", "call_id": call["call_id"],
                       "output": json.dumps({"count": count})}], "tools": [TOOL], "tool_choice": "auto"},
            gold={"memory": literal, "count": count, "marker": marker})

    def stream(self, parent, fixture):
        rid = response_id(parent)
        if not rid: self.skip("sse-terminal", "missing parent response id"); return
        marker = "STREAM_" + fixture["nonce"]
        payload = {"model": self.args.model, "store": True, "stream": True, "previous_response_id": rid,
                   "instructions": instruction(marker), "input": "Report the remembered literal and current count.",
                   "reasoning": {"effort": "none"}, "max_output_tokens": 96}
        exchange = self.http("POST", "/responses", payload)
        events = []
        for frame in exchange["wire"].replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:"))
            if not data or data == "[DONE]": continue
            try: events.append(json.loads(data))
            except ValueError: events.append({"unparsed_data": data})
        terminals = [event for event in events if event.get("type") in
                     ("response.completed", "response.incomplete", "response.failed", "response.cancelled")]
        terminal = terminals[-1].get("response") if len(terminals) == 1 else None
        terminal_id = response_id(terminal)
        ids = [response_id(event["response"]) for event in events if isinstance(event.get("response"), dict)]
        gold = {"memory": fixture["literal"], "count": 37, "marker": marker}
        passed = (exchange["http_status"] == 200 and bool(terminal_id) and len(terminals) == 1
                  and all(value == terminal_id for value in ids) and final_gold(terminal, gold))
        exchange["body"] = terminal
        self.record("sse-terminal", exchange, passed, events=events, independent_gold=gold,
                    terminal_response_id=terminal_id, response_ids=ids)
        self.get_exact("sse-get-exact", terminal)

    def inspect_store(self):
        path = self.args.store_dir
        try:
            return [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(path.iterdir()) if p.is_file()]
        except OSError as exc: return {"inspection_error": str(exc)}

    def private_reasoning(self, parent):
        rid = response_id(parent)
        if not rid: self.skip("private-reasoning-stored-hidden", "missing low-effort parent id"); return
        path = self.args.store_dir / (rid + ".json")
        error = None; record = None
        try: record = json.loads(path.read_text())
        except (OSError, ValueError) as exc: error = str(exc)
        reasoning = [item.get("reasoning_content", "") for item in (record or {}).get("replay", [])
                     if item.get("role") == "assistant" and item.get("reasoning_content")]
        stored = (record or {}).get("response", {})
        public_has_reasoning = any(item.get("type") == "reasoning" for item in parent.get("output", []))
        passed = (parent.get("status") == "completed" and bool(reasoning) and not public_has_reasoning
                  and stored.get("id") == rid and stored.get("output") == parent.get("output"))
        self.record("private-reasoning-stored-hidden", {"body": parent, "error": error}, passed,
                    store_file=str(path), private_reasoning_chars=sum(map(len, reasoning)),
                    public_has_reasoning_item=public_has_reasoning)

    def finish(self, manifest=None):
        result = dict(phase=self.args.phase, utc=datetime.now(timezone.utc).isoformat(),
            argv=sys.argv, base_url=self.base, store_dir=str(self.args.store_dir.resolve()),
            model=self.args.model, total_seconds=time.monotonic()-self.start, rows=self.rows,
            counts={kind: sum(r["result"] == kind for r in self.rows) for kind in ("PASS", "FAIL", "ERROR", "SKIP")},
            store_files=self.inspect_store(), interpretation="First actual HTTP attempts; no retries/repairs. Server startup/restart time is owned by the outer launcher and excluded here.")
        write_json(self.output / (self.args.phase + ".json"), result)
        if manifest is not None: write_json(self.output / "manifest.json", manifest)
        print(json.dumps({k: result[k] for k in ("phase", "total_seconds", "counts")}), flush=True)
        return 0 if all(r["result"] == "PASS" for r in self.rows) else 3


def before(c):
    nonce = secrets.token_hex(5)
    fixture = {"nonce": nonce, "literal": "MEM_" + secrets.token_hex(8),
               "private_literal": "PRIVATE_" + secrets.token_hex(8), "old_marker": "OLD_" + nonce,
               "new_marker": "NEW_" + nonce, "fork_marker": "FORK_" + nonce}
    write_json(c.output / "fixtures.json", fixture)  # Fixed before the first HTTP request.
    parent = c.post("parent-remember", {"store": True, "max_output_tokens": 64,
        "instructions": instruction(fixture["old_marker"]),
        "input": 'Memory literal is ' + fixture["literal"] + '. Current count is 37. Report these facts.'},
        gold={"memory": fixture["literal"], "count": 37, "marker": fixture["old_marker"]})
    c.get_exact("parent-get-exact", parent)
    child = None
    if response_id(parent):
        child = c.post("child-new-instructions-default-store", {"previous_response_id": parent["id"],
            "instructions": instruction(fixture["new_marker"]), "input": "Report the remembered literal and current count."},
            gold={"memory": fixture["literal"], "count": 37, "marker": fixture["new_marker"]})
        fork = c.post("fork-new-count", {"store": True, "previous_response_id": parent["id"],
            "instructions": instruction(fixture["fork_marker"]), "input": "The current count is now 82. Report memory and current count."},
            gold={"memory": fixture["literal"], "count": 82, "marker": fixture["fork_marker"]})
        c.get_exact("fork-get-exact", fork)
        c.get_exact("parent-unchanged-after-fork", parent)
        c.get_exact("sibling-unchanged-after-fork", child)
    else:
        c.skip("child-new-instructions-default-store", "missing parent id"); c.skip("fork-new-count", "missing parent id")
    transient = c.post("store-false", {"store": False, "instructions": 'Reply exactly "TRANSIENT".', "input": "Acknowledge.", "max_output_tokens": 64})
    if response_id(transient):
        exchange = c.http("GET", "/responses/" + transient["id"])
        c.record("store-false-get-404", exchange, exchange["http_status"] == 404)
    else: c.skip("store-false-get-404", "missing transient id")
    exchange = c.http("POST", "/responses", {"model": c.args.model, "store": True,
        "reasoning": {"effort": "none"}, "max_output_tokens": 1,
        "instructions": "Write the entire requested list without omissions.",
        "input": "Write integers 1 through 20 separated by commas."})
    capped = exchange["body"]
    c.record("cap1-incomplete", exchange, exchange["http_status"] == 200 and bool(response_id(capped))
        and capped.get("status") == "incomplete"
        and (capped.get("incomplete_details") or {}).get("reason") in ("max_tokens", "max_output_tokens")
        and capped.get("store") is True)
    c.get_exact("cap1-get-exact", capped)
    if response_id(capped):
        exchange = c.http("POST", "/responses", {"model": c.args.model, "store": True,
            "previous_response_id": capped["id"], "input": "Continue.",
            "reasoning": {"effort": "none"}, "max_output_tokens": 64})
        c.record("incomplete-parent-chain-rejected", exchange, exchange["http_status"] == 400)
    else: c.skip("incomplete-parent-chain-rejected", "missing capped response id")
    tools = c.tool_parent("tool-auto-parent", parent, "TOOL_" + nonce)
    c.get_exact("tool-parent-get-exact", tools)
    exchange = c.http("POST", "/responses", {"model": c.args.model, "store": True,
        "tools": [TOOL], "tool_choice": {"type": "function", "name": "lookup_count"},
        "input": 'Call lookup_count with key "subject".', "max_output_tokens": 64,
        "reasoning": {"effort": "none"}})
    c.record("forced-tool-choice-rejected", exchange, exchange["http_status"] == 400)
    if response_id(tools) and call_item(tools):
        exchange = c.http("POST", "/responses", {"model": c.args.model, "store": True,
            "previous_response_id": tools["id"], "reasoning": {"effort": "none"}, "max_output_tokens": 64,
            "input": [{"type": "function_call_output", "call_id": "call_missing_" + nonce, "output": '{"count":61}'}]})
        c.record("unknown-call-id-rejected", exchange, exchange["http_status"] == 400)
    else: c.skip("unknown-call-id-rejected", "missing valid tool call")
    c.tool_continue("tool-output-continuation", tools, fixture["literal"], "TOOL_FINAL_" + nonce, 61)
    pending = c.tool_parent("pending-tool-for-restart", parent, "PENDING_" + nonce)
    private_literal = fixture["private_literal"]
    private = c.post("low-effort-private-parent", {"store": True, "reasoning": {"effort": "low"},
        "max_output_tokens": 256, "instructions": instruction("PRIVATE_" + nonce),
        "input": "Memory literal is " + private_literal + ". Current count is 19. Report these exact facts."},
        gold={"memory": private_literal, "count": 19, "marker": "PRIVATE_" + nonce})
    c.private_reasoning(private)
    c.get_exact("private-parent-get-exact", private)
    c.stream(parent, fixture)
    # Delete a disposable stored child; keep the parent and pending tool alive.
    deleted = c.post("delete-child", {"store": True, "instructions": 'Reply exactly "DELETE_ME".', "input": "Acknowledge.", "max_output_tokens": 64})
    if response_id(deleted):
        rid = deleted["id"]; c.get_exact("delete-child-get-before", deleted)
        exchange = c.http("DELETE", "/responses/" + rid); body = exchange["body"]
        c.record("delete-child-delete", exchange, exchange["http_status"] == 200 and isinstance(body, dict)
                 and body.get("id") == rid and body.get("deleted") is True)
        exchange = c.http("GET", "/responses/" + rid)
        c.record("delete-child-get-404", exchange, exchange["http_status"] == 404)
    else: c.skip("delete-child-delete", "missing disposable id")
    return c.finish(dict(fixture=fixture, parent=parent, child=child, pending_tool=pending, private_parent=private,
                         model=c.args.model, store_dir=str(c.args.store_dir.resolve()),
                         before_base_url=c.base, server_restart_owned_by="ROOT"))


def after(c):
    manifest = json.loads((c.output / "manifest.json").read_text())
    if manifest["store_dir"] != str(c.args.store_dir.resolve()) or manifest["model"] != c.args.model:
        raise ValueError("after phase must use the before-phase model and durable store directory")
    fixture, parent, pending = manifest["fixture"], manifest["parent"], manifest["pending_tool"]
    c.get_exact("restart-parent-get-exact", parent)
    if response_id(parent):
        marker = "AFTER_" + fixture["nonce"]
        c.post("restart-child-memory", {"store": True, "previous_response_id": parent["id"],
            "instructions": instruction(marker), "input": "Report the remembered literal and current count."},
            gold={"memory": fixture["literal"], "count": 37, "marker": marker})
    else: c.skip("restart-child-memory", "missing before-phase parent id")
    c.get_exact("restart-pending-tool-get-exact", pending)
    c.tool_continue("restart-pending-tool-output", pending, fixture["literal"], "AFTER_TOOL_" + fixture["nonce"], 73)
    c.get_exact("restart-parent-still-unchanged", parent)
    private = manifest["private_parent"]
    c.get_exact("restart-private-parent-get-exact", private)
    c.private_reasoning(private)
    if response_id(private) and private.get("status") == "completed":
        marker = "AFTER_PRIVATE_" + fixture["nonce"]
        c.post("restart-private-parent-child", {"store": True, "previous_response_id": private["id"],
            "instructions": instruction(marker), "input": "Report the remembered literal and current count."},
            gold={"memory": fixture["private_literal"], "count": 19, "marker": marker})
    else: c.skip("restart-private-parent-child", "low-effort parent did not complete; no retry")
    child = manifest["child"]
    c.get_exact("restart-existing-child-get-exact", child)
    if response_id(parent) and response_id(child):
        rid = parent["id"]
        exchange = c.http("DELETE", "/responses/" + rid); body = exchange["body"]
        c.record("restart-parent-delete", exchange, exchange["http_status"] == 200 and isinstance(body, dict)
                 and body.get("id") == rid and body.get("deleted") is True)
        exchange = c.http("GET", "/responses/" + rid)
        c.record("restart-parent-deleted-get-404", exchange, exchange["http_status"] == 404)
        c.get_exact("child-unchanged-after-parent-delete", child)
        marker = "AFTER_DELETE_" + fixture["nonce"]
        c.post("child-continues-after-parent-delete", {"store": True, "previous_response_id": child["id"],
            "instructions": instruction(marker), "input": "Report the remembered literal and current count."},
            gold={"memory": fixture["literal"], "count": 37, "marker": marker})
    else: c.skip("child-continues-after-parent-delete", "missing before-phase parent or child id")
    return c.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--phase", choices=("before", "after"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="deepseek-v4.1-flash")
    p.add_argument("--timeout", type=float, default=180)
    args = p.parse_args()
    c = Campaign(args)
    if (c.output / (args.phase + ".json")).exists(): p.error("phase output already exists; retain first attempts")
    return before(c) if args.phase == "before" else after(c)

if __name__ == "__main__": sys.exit(main())
