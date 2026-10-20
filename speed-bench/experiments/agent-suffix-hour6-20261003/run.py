#!/usr/bin/env python3
"""Four real agent turns and one idle context deletion; ROOT schedules models."""
import argparse
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BEGIN, END = "[DISPOSABLE_OLD_BEGIN]", "[DISPOSABLE_OLD_END]"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--agent", default=str(ROOT / ".workspaces/idea-agent-suffix-hour6/build/ds4-agent"))
    p.add_argument("--output", required=True)
    p.add_argument("--shared", default=str(HERE / "shared"))
    p.add_argument("--mode", choices=("native", "suffix"), default="native")
    p.add_argument("--tail", type=int, choices=(128, 256), default=128)
    p.add_argument("--notes", type=int, default=24)
    p.add_argument("--blob-units", type=int, default=128)
    p.add_argument("--breadcrumb", default="[old history archived]")
    p.add_argument("--timeout", type=float, default=420)
    a = p.parse_args()
    out, shared = Path(a.output).resolve(), Path(a.shared).resolve()
    out.mkdir(parents=True, exist_ok=False)
    work, cache = shared / "work", shared / "cache"
    work.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    context, active, archive = shared / "context.txt", work / "active.json", work / "archive.json"
    context.unlink(missing_ok=True)
    active_data = {"batches": 17, "items_per_batch": 25, "reserved_items": 9,
                   "notes": [f"Active processing note {i}: retain the current inventory rules; old audit prose is disposable."
                             for i in range(a.notes)]}
    dump(active, active_data)
    dump(archive, {"archived_code": None})
    first = ("The following marked span is disposable old history; later its useful fact will be archived. "
             + BEGIN + "\nOld audit code: RIVER-73.\n"
             + "old audit log unchanged.\n" * a.blob_units + END
             + "\nDo not use tools now. Reply READY only.")
    second = (f"Use the read tool once to read {active}, with whole=true and raw=true. "
              "This file is the self-contained active inventory specification. Retain the three numeric rules "
              "and the notes from its tool result for a later calculation. Do not read other files. Then reply READY only.")
    third = ("Reply READY only, without tools or calculations. We will next ask a short inventory question. "
             "Keep the active specification from the preceding read result. Its batches, items per batch, and reserved "
             "items define the calculation: multiply the first two quantities and subtract the reserved quantity. "
             "The older audit code will be put into permanent storage before that question. Retrieve it from storage "
             "when asked instead of trusting recollection. Preserve this entire latest request and its rules. "
             "The disposable audit log is irrelevant to the inventory. No answer is needed yet. "
             "Use only the current active inventory, keep the arithmetic in its original order, and report two fields "
             "when the final question arrives. Keep numeric values numeric and the archived code a string. "
             "Do not summarize or edit any files yourself. Wait for the final question. This is a readiness turn, "
             "so provide exactly READY and finish.")
    fourth = (f"Read {archive} using the read tool with whole=true and raw=true. "
              "Using the previously read active inventory, return only JSON with active_total "
              "(batches times items_per_batch minus reserved_items) and archived_code from this permanent file.")
    turns = [first, second, third, fourth]
    dump(out / "fixture.json", {"turns": turns, "active": active_data,
                               "gold": {"active_total": 416, "archived_code": "RIVER-73"},
                               "mode": a.mode, "tail": a.tail, "blob_units": a.blob_units,
                               "notes": a.notes, "breadcrumb": a.breadcrumb,
                               "context": str(context), "archive": str(archive)})
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                              (ROOT / "ds4_metal.m").read_text()))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DS4", "PT_"))}
    env.update({k: str(ROOT / v) for k, v in shaders.items()})
    env.update(DS4_AGENT_CACHE_DIR=str(cache), DS4_AGENT_COMPACT_PREFIX_CACHE="1",
               DS4_AGENT_CONTEXT_FRONTIERS="4", DS4_AGENT_CONTEXT_FRONTIER_MB="512",
               DS4_AGENT_CONTEXT_SUFFIX_REUSE="1" if a.mode == "suffix" else "0",
               DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL=str(a.tail), DS4_AGENT_TEST_TIME="1790954400",
               DS4_METAL_V41_RESIDENT_ROUTE_GAP="0",
               DS4_METAL_V41_SELECTED_TILE="64", DS4_METAL_V41_SELECTED_LIMIT="1024")
    command = [str(Path(a.agent).resolve()), "--model", a.model, "--ssd-streaming",
               "--ssd-streaming-cache-experts", "64GB", "--ctx", "8192", "--prefill-chunk", "512",
               "--nothink", "--temp", "0", "--tokens", "512", "--seed", "20261003",
               "--non-interactive", "--chdir", str(work), "--live-context-file", str(context),
               "--trace", str(out / "agent.trace")]
    dump(out / "execution.json", {"command": command, "cwd": str(ROOT),
                                 "env": {k: v for k, v in env.items() if k.startswith("DS4")},
                                 "cache_shared": True, "eof_saves_named_session": False})
    start, sent, seen, final_start = time.monotonic(), 0, 0, None
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    sel = selectors.DefaultSelector()
    chunks = {"stdout": bytearray(), "stderr": bytearray()}
    for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        os.set_blocking(stream.fileno(), False)
        sel.register(stream, selectors.EVENT_READ, name)
    completions, error, edit, final_end = [], None, None, None
    try:
        while sel.get_map():
            if time.monotonic() - start > a.timeout:
                raise TimeoutError("420-second arm deadline exceeded")
            for key, _ in sel.select(0.2):
                data = os.read(key.fd, 65536)
                if not data:
                    sel.unregister(key.fileobj)
                    continue
                chunks[key.data].extend(data)
                if key.data != "stderr":
                    continue
                count = chunks["stderr"].count(b"+DWARFSTAR_WAITING")
                while seen < count:
                    seen += 1
                    if sent:
                        completions.append({"turn": sent, "seconds_from_start": time.monotonic() - start})
                    if sent == 4:
                        final_end = time.monotonic()
                        proc.stdin.close()
                        continue
                    if sent == 3:
                        final_start = time.monotonic()
                        before = context.read_text()
                        if before.count(BEGIN) != 1 or before.count(END) != 1:
                            raise ValueError("expected exactly one disposable marked span")
                        lo, hi = before.index(BEGIN), before.index(END) + len(END)
                        after = before[:lo] + a.breadcrumb + before[hi:]
                        (out / "before-edit.txt").write_text(before)
                        dump(archive, {"archived_code": "RIVER-73"})
                        context.write_text(after)
                        (out / "after-edit.txt").write_text(after)
                        edit = {"removed_bytes": len(before[lo:hi].encode()),
                                "replacement": a.breadcrumb, "seconds": time.monotonic() - final_start}
                        dump(out / "edit.json", edit)
                    proc.stdin.write((turns[sent] + "\n").encode())
                    proc.stdin.flush()
                    sent += 1
        rc = proc.wait(timeout=5)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        rc = None
    finally:
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(0.2)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            rc = proc.wait(timeout=5)
        sel.close()
        for name, data in chunks.items():
            (out / (name + ".log")).write_bytes(data)
    text = context.read_text() if context.exists() else ""
    (out / "final-context.txt").write_text(text)
    final = text.rsplit("<｜Assistant｜>", 1)[-1]
    try:
        answer, _ = json.JSONDecoder().raw_decode(final[final.index("{"):])
    except (ValueError, json.JSONDecodeError):
        answer = None
    calls = re.findall(r'<｜DSML｜\s*invoke name="read">(.*?)</｜DSML｜\s*invoke>', text, re.S)
    archive_read = any(str(archive) in call for call in calls)
    active_read = any(str(active) in call for call in calls)
    correct = (isinstance(answer, dict) and type(answer.get("active_total")) is int
               and answer.get("active_total") == 416 and answer.get("archived_code") == "RIVER-73")
    trace = (out / "agent.trace").read_text() if (out / "agent.trace").exists() else ""
    result = {"returncode": rc, "error": error, "seconds": time.monotonic() - start,
              "final_turn_including_edit_seconds": final_end - final_start if final_start and final_end else None,
              "turns_sent": sent, "turn_completions": completions, "answer": answer,
              "answer_correct": correct, "incoming_question_retained": fourth in text,
              "archived_fact_removed_before_query": "RIVER-73" not in (out / "after-edit.txt").read_text() if (out / "after-edit.txt").exists() else False, "active_tool_read": active_read, "archive_tool_read": archive_read,
              "edit": edit, "mode": a.mode, "tail_lower_bound": a.tail,
              "suffix_trace": [line for line in trace.splitlines() if "suffix" in line or "frontier" in line],
              "context_edit_trace": [line for line in trace.splitlines() if "live context edit" in line],
              "final_assistant": final}
    result["task_pass"] = bool(rc == 0 and not error and len(completions) == 4 and correct
                               and active_read and archive_read and result["incoming_question_retained"])
    dump(out / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["task_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
