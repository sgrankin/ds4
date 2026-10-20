#!/usr/bin/env python3
"""File-backed HTTP agent edits. ROOT launches/restarts the guarded server."""
import argparse
import csv
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

BEGIN, END = "[DISPOSABLE_OLD_BEGIN]", "[DISPOSABLE_OLD_END]"
FACTS = {"A": (17, 25, 9, "RIVER-73", "RIVER-74"),
         "B": (19, 11, 7, "CEDAR-91", "CEDAR-92")}
TOOLS = [{"type": "function", "name": "read", "description": "Read one advertised JSON file in full.",
          "parameters": {"type": "object", "properties": {
              "path": {"type": "string"}, "whole": {"type": "boolean"}, "raw": {"type": "boolean"}},
              "required": ["path", "whole", "raw"], "additionalProperties": False}}]
INSTRUCTIONS = ("Follow the newest request. Use the read tool for advertised files. "
                "Never guess a file's current content from an earlier answer. "
                "Active inventory tool results remain authoritative when obsolete audit prose is deleted.")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def final_text(body):
    return "".join(part.get("text", "") for item in body.get("output", [])
                   if item.get("type") == "message"
                   for part in item.get("content", []) if part.get("type") == "output_text")


def read_log(path, offset):
    if not path or not path.exists():
        return "", offset
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read()
        return data.decode("utf-8", errors="replace"), stream.tell()


def answer_grade(text, gold):
    if gold == "READY":
        return text.strip() == "READY", None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
        return (value == gold and type(value.get("active_total")) is int
                and type(value.get("archived_code")) is str), value
    except (ValueError, AttributeError):
        return False, None


def turn(a, out, name, session, history, wanted_read, gold, offset, tools=True):
    start = time.monotonic()
    calls, reads, errors, replies = [], [], [], []
    text, complete = "", False
    for index in range(3):
        # Tool request gets128 tokens; after the read, only the final JSON needs64.
        cap = 128 if tools and not reads else 64
        payload = {"model": "deepseek-v4.1-flash", "instructions": INSTRUCTIONS,
                   "input": history, "prompt_cache_key": "ds4-hour12-" + session,
                   "store": False, "stream": False, "temperature": 0,
                   "seed": 20261004, "max_output_tokens": cap, "reasoning": {"effort": "none"}}
        if tools:
            payload["tools"] = TOOLS
        dump(out / f"{name}-{index}-request.json", payload)
        request = urllib.request.Request(a.base_url.rstrip("/") + "/v1/responses",
                    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        t, body, status = time.monotonic(), {}, None
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                status, raw = response.status, response.read()
            (out / f"{name}-{index}-response.json").write_bytes(raw)
            body = json.loads(raw)
        except urllib.error.HTTPError as exc:
            status = exc.code
            (out / f"{name}-{index}-response.json").write_bytes(exc.read())
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        replies.append({"seconds": time.monotonic()-t, "http_status": status,
                        "response_status": body.get("status"), "usage": body.get("usage"),
                        "incomplete_details": body.get("incomplete_details")})
        output = body.get("output") or []
        history.extend(output)  # Preserve actual message, call, reasoning and IDs.
        text = final_text(body)
        complete = status == 200 and body.get("status") == "completed"
        new_calls = [item for item in output if item.get("type") == "function_call"]
        if not complete or not new_calls:
            break
        for call in new_calls:
            record = {"name": call.get("name"), "call_id": call.get("call_id"),
                      "arguments": call.get("arguments"), "read": False, "error": None}
            result = "ERROR: only the requested advertised read is permitted."
            try:
                args = json.loads(call.get("arguments", "{}"))
                path = Path(args["path"]).resolve()
                if (not wanted_read or call.get("name") != "read" or path != wanted_read
                        or args.get("whole") is not True or args.get("raw") is not True):
                    raise ValueError("read must target this turn's advertised file with whole=true/raw=true")
                result = path.read_text()
                record.update(read=True, path=str(path), text=result)
                reads.append(str(path))
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                errors.append(record["error"])
            calls.append(record)
            history.append({"type": "function_call_output", "call_id": call.get("call_id"), "output": result})
        dump(out / f"{name}-history.json", history)
    else:
        errors.append("three HTTP responses exhausted without a final answer")
        complete = False
    correct, answer = answer_grade(text, gold)
    fresh_read = len(reads) == 1 and (not wanted_read or reads[0] == str(wanted_read))
    log, offset = read_log(a.server_log, offset)
    (out / f"{name}-server.log").write_text(log)
    row = {"step": name, "session": session, "seconds": time.monotonic()-start,
           "text": text, "answer": answer, "gold": gold, "correct": correct,
           "fresh_read": fresh_read, "complete": complete, "calls": calls,
           "required_read": str(wanted_read) if wanted_read else None,
           "replies": replies, "errors": errors,
           "suffix_log_lines": [line for line in log.splitlines() if "suffix" in line.lower()],
           "exact_hit_log_lines": [line for line in log.splitlines() if "state cache exact hit" in line],
           "slot_eviction_log_lines": [line for line in log.splitlines() if "evicting checkpoint" in line]}
    row["pass"] = bool(correct and (fresh_read or not tools) and complete and not errors)
    dump(out / f"{name}-history.json", history)
    dump(out / f"{name}-result.json", row)
    print(json.dumps({key: row[key] for key in ("step", "session", "seconds", "text",
          "correct", "fresh_read", "complete", "pass", "suffix_log_lines")},
          ensure_ascii=False), flush=True)
    return row, offset


def remove_first_block(history, destination):
    # Only the known marked span in the original user message is edited.
    before = history[0]["content"]
    lo, hi = before.index(BEGIN), before.index(END) + len(END)
    after = before[:lo] + "[old history archived]" + before[hi:]
    destination.with_suffix(".before.txt").write_text(before)
    destination.with_suffix(".after.txt").write_text(after)
    history[0]["content"] = after
    return {"removed_bytes": len(before[lo:hi].encode()), "remaining_blocks": after.count(BEGIN)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--phase", choices=("before", "after"), required=True)
    p.add_argument("--mode", choices=("native", "reuse"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shared", type=Path, required=True)
    p.add_argument("--server-log", type=Path)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.shared = a.shared.resolve()
    state_file = a.output / "histories.json"
    if a.phase == "before" and state_file.exists():
        p.error("before phase requires a fresh output namespace")
    a.output.mkdir(parents=True, exist_ok=True)
    out = a.output / a.phase
    out.mkdir(exist_ok=False)
    offset = a.server_log.stat().st_size if a.server_log and a.server_log.exists() else 0
    rows, edits, started = [], [], time.monotonic()
    dump(out / "execution.json", {"argv_phase": a.phase, "mode": a.mode, "base_url": a.base_url,
         "shared": str(a.shared), "output": str(a.output), "tools": TOOLS,
         "server_started_by_driver": False, "source_context_tokens": "estimated about2K; usage is actual",
         "turn_order": ["A-source", "B-source", "A-edit1", "B-edit1", "C-turnover"]
                      if a.phase == "before" else ["A-edit2", "B-edit2"]})
    if a.phase == "before":
        state = {"mode": a.mode, "shared": str(a.shared), "histories": {}, "results": []}
        for session, (batches, items, reserved, old_code, new_code) in FACTS.items():
            directory = a.shared / session
            directory.mkdir(parents=True, exist_ok=True)
            active = directory / "active.json"
            archive = directory / "archive.json"
            data = {"batches": batches, "items_per_batch": items, "reserved_items": reserved,
                    "notes": [f"Active note {i:02d}: retain current inventory; older audit prose is disposable." for i in range(32)]}
            dump(active, data)
            dump(archive, {"archived_code": None})
            block = BEGIN + "\nOld audit code: " + old_code + ".\n" + "old audit log unchanged.\n" * 96 + END
            initial = ("These two marked audit spans are obsolete prose; their useful code will be archived.\n"
                       + block + "\n" + block + "\n"
                       + f"Read {active} once using read with whole=true and raw=true. "
                       "Retain the inventory numbers and complete unchanged tool result. Then reply READY only.")
            history = [{"role": "user", "content": initial}]
            state["histories"][session] = history
            row, offset = turn(a, out, session+"-source", session, history, active, "READY", offset)
            rows.append(row)
            state["results"].append(row)
            dump(state_file, state)
    else:
        state = json.loads(state_file.read_text())
        if state["mode"] != a.mode or state["shared"] != str(a.shared):
            p.error("after phase must continue the same mode/shared paths")
    for session, (batches, items, reserved, old_code, new_code) in FACTS.items():
        history = state["histories"][session]
        archive = a.shared / session / "archive.json"
        code = old_code if a.phase == "before" else new_code
        edit_start = time.monotonic()
        dump(archive, {"archived_code": code})
        name = session + ("-edit1" if a.phase == "before" else "-edit2")
        edit = remove_first_block(history, out / name)
        edit.update(session=session, seconds=time.monotonic()-edit_start)
        edits.append(edit)
        query = (f"Question {1 if a.phase == 'before' else 2}: read {archive} again now with whole=true and raw=true. "
                 "A fresh read is required even if an earlier answer contained a code. "
                 "Use the unchanged active inventory tool result to calculate batches * items_per_batch - reserved_items. "
                 "Reply only JSON with active_total (integer) and archived_code (string) from the current file.")
        history.append({"role": "user", "content": query})
        row, offset = turn(a, out, name, session, history, archive,
                           {"active_total": batches*items-reserved, "archived_code": code}, offset)
        row["seconds_including_edit"] = time.monotonic()-edit_start
        row["edit_seconds"] = edit["seconds"]
        dump(out / f"{name}-result.json", row)
        rows.append(row)
        state["results"].append(row)
        dump(state_file, state)
    if a.phase == "before":
        row, offset = turn(a, out, "C-turnover", "C", [{"role": "user", "content": "Reply READY only."}],
                           None, "READY", offset, tools=False)
        rows.append(row)
        state["results"].append(row)
        dump(state_file, state)
    result = {"phase": a.phase, "mode": a.mode, "steps": rows, "edits": edits,
              "seconds": time.monotonic()-started,
              "source_warmup_seconds": sum(r["seconds"] for r in rows if r["step"].endswith("source")),
              "edited_seconds": sum(r["seconds_including_edit"] for r in rows if "-edit" in r["step"]),
              "suffix_grafts_observed": sum("suffix APPROXIMATE" in line for r in rows for line in r["suffix_log_lines"]),
              "approximate_ssd_sources_observed": sum("suffix APPROXIMATE" in line and "source_ssd=1" in line
                  and "source_approximate=1" in line for r in rows for line in r["suffix_log_lines"]),
              "pass": all(r["pass"] for r in rows), "restart_owned_by_ROOT": True}
    dump(out / "result.json", result)
    fields = ["step", "session", "seconds", "seconds_including_edit", "correct", "fresh_read", "complete", "pass", "text"]
    with (out / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
