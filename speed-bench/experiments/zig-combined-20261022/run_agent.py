#!/usr/bin/env python3
"""Six turns, three idle deletions; every query must freshly read the archive.

Adapted from item6 run_repeat.py. The three disposable blocks precede the
unchanged inventory tool result. Token geometry comes from the actual trace;
byte lengths do not guarantee even token shifts or reuse of one frontier.
"""
import argparse
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
BEGIN, END = "[DISPOSABLE_OLD_BEGIN]", "[DISPOSABLE_OLD_END]"
GOLD = {"active_total": 416, "archived_code": "RIVER-73"}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".edit.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def read_answer(text):
    final = text.rsplit("<｜Assistant｜>", 1)[-1]
    try:
        return json.JSONDecoder().raw_decode(final[final.index("{"):])[0], final
    except (ValueError, json.JSONDecodeError):
        return None, final


def suffix_geometry(trace):
    found = []
    for line in trace.splitlines():
        if "context suffix APPROXIMATE" not in line:
            continue
        m = re.search(r"reused=(\d+) fresh=(\d+) source=(\d+):(\d+) destination=(\d+):(\d+)", line)
        if not m:
            continue
        reused, fresh, source_start, source_end, destination_start, destination_end = map(int, m.groups())
        item = dict(reused=reused, fresh=fresh, source_start=source_start,
                    source_end=source_end, destination_start=destination_start,
                    destination_end=destination_end, shift=source_start-destination_start,
                    aligned=source_start % 2 == destination_start % 2 == 0,
                    span_lengths_equal=source_end-source_start == destination_end-destination_start,
                    line=line)
        for field in ("pre_rope", "prior_approximate"):
            match = re.search(rf"\b{field}=(\d+)", line)
            item[field] = int(match[1]) if match else None
        found.append(item)
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--agent", default=str(ROOT / "ds4-agent"))
    p.add_argument("--output", required=True)
    p.add_argument("--shared", default=str(HERE / "shared"))
    p.add_argument("--mode", choices=("native", "suffix"), default="native")
    p.add_argument("--prerope", type=int, choices=(0, 1), default=1)
    p.add_argument("--tail", type=int, choices=(128, 256), default=128)
    p.add_argument("--notes", type=int, default=24)
    p.add_argument("--blob-units", type=int, default=128)
    p.add_argument("--breadcrumb", default="[old history archived]")
    p.add_argument("--timeout", type=float, default=600)
    a = p.parse_args()
    if a.notes < 0 or a.blob_units < 1 or a.timeout <= 0:
        p.error("notes must be nonnegative; blob-units and timeout must be positive")
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
    active_text = json.dumps(active_data, indent=2, ensure_ascii=False)
    dump(archive, {"archived_code": None})
    block = (BEGIN + "\nOld audit code: RIVER-73.\n"
             + "old audit log unchanged.\n" * a.blob_units + END)
    first = ("The next three disjoint marked spans are disposable old history. "
             "Their useful code will later be stored in a permanent archive.\n"
             + "\n".join([block] * 3)
             + "\nDo not use tools now. Reply READY only.")
    second = (f"Use the read tool once to read {active}, with whole=true and raw=true. "
              "This file is the self-contained active inventory specification. Retain the three numeric rules "
              "and notes from its tool result. Do not read other files. Then reply READY only.")
    third = ("Reply READY only, without tools or calculations. Keep the active specification from the preceding "
             "read result. Its batches, items per batch, and reserved items define the calculation: multiply "
             "the first two quantities and subtract the reserved quantity. The three disposable old logs "
             "will be removed one at a time between questions. Their code will be put into permanent storage. "
             "Retrieve it from storage every time you are asked, even if an earlier answer already contained it. "
             "Preserve this entire request and the unchanged active specification. Keep numeric values numeric "
             "and the archived code a string. Do not summarize or edit files yourself. Wait for each question. "
             "This is a readiness turn, so provide exactly READY and finish.")
    query = (f"Read {archive} again now using the read tool with whole=true and raw=true. "
             "A fresh read is required on this turn; do not substitute an earlier read or remembered code. "
             "Using the previously read unchanged active inventory, return only JSON with active_total "
             "(batches times items_per_batch minus reserved_items) and archived_code from this permanent file.")
    turns = [first, second, third, query, query, query]
    dump(out / "fixture.json", {"turns": turns, "disposable_blocks": 3,
                               "identical_disjoint_block_text": block,
                               "active": active_data, "active_tool_result_text": active_text,
                               "gold_each_query": GOLD, "mode": a.mode, "prerope": a.prerope,
                               "tail": a.tail, "blob_units": a.blob_units, "notes": a.notes,
                               "breadcrumb": a.breadcrumb, "context": str(context), "archive": str(archive)})
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                              (ROOT / "ds4_metal.m").read_text()))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DS4", "PT_"))}
    env.update({k: str(ROOT / v) for k, v in shaders.items()})
    env.update(DS4_AGENT_CACHE_DIR=str(cache), DS4_AGENT_COMPACT_PREFIX_CACHE="1",
               DS4_AGENT_CONTEXT_FRONTIERS="4", DS4_AGENT_CONTEXT_FRONTIER_MB="512",
               DS4_AGENT_CONTEXT_SUFFIX_REUSE="1" if a.mode == "suffix" else "0",
               DS4_AGENT_CONTEXT_SUFFIX_PREROPE=str(a.prerope) if a.mode == "suffix" else "0",
               DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL=str(a.tail), DS4_AGENT_TEST_TIME="1790954400",
               DS4_METAL_V41_RESIDENT_ROUTE_GAP="0",
               DS4_METAL_V41_SELECTED_TILE="64", DS4_METAL_V41_SELECTED_LIMIT="1024")
    env.update(DS4_STATE_CACHE_DIR=str(out.parent / "state-cache"), DS4_STATE_CACHE_HOT_MB="0", DS4_STATE_CACHE_SSD_MB="512")
    command = [str(Path(a.agent).resolve()), "--model", a.model, "--ssd-streaming",
               "--ssd-streaming-cache-experts", "32GB", "--ctx", "8192", "--prefill-chunk", "512",
               "--nothink", "--temp", "0", "--tokens", "512", "--seed", "20261003",
               "--non-interactive", "--chdir", str(work), "--live-context-file", str(context),
               "--trace", str(out / "agent.trace")]
    dump(out / "execution.json", {"command": command, "cwd": str(ROOT),
                                 "env": {k: v for k, v in env.items() if k.startswith("DS4")},
                                 "cache_shared": True, "eof_saves_named_session": False,
                                 "geometry_authority": "actual agent trace; no tokenization or parity retry"})
    start, sent, seen = time.monotonic(), 0, 0
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    sel = selectors.DefaultSelector()
    chunks = {"stdout": bytearray(), "stderr": bytearray()}
    for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        os.set_blocking(stream.fileno(), False)
        sel.register(stream, selectors.EVENT_READ, name)
    completions, edits, queries, error = [], [], [], None
    query_start, query_trace_start = None, 0
    try:
        while sel.get_map():
            if time.monotonic() - start > a.timeout:
                raise TimeoutError(f"{a.timeout:g}-second arm deadline exceeded")
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
                    if sent >= 4:
                        text = context.read_text()
                        (out / f"query-{sent-3}-context.txt").write_text(text)
                        turn_text = text[text.rfind("<｜User｜>" + query):]
                        answer, final = read_answer(turn_text)
                        calls = re.findall(r'<｜DSML｜\s*invoke name="read">(.*?)</｜DSML｜\s*invoke>', turn_text, re.S)
                        fresh_read = any(str(archive) in call for call in calls)
                        correct = (isinstance(answer, dict) and type(answer.get("active_total")) is int
                                   and answer == GOLD)
                        trace = (out / "agent.trace").read_text()
                        turn_trace = trace[query_trace_start:]
                        (out / f"query-{sent-3}.trace").write_text(turn_trace)
                        queries.append({"query": sent-3, "turn": sent, "answer": answer,
                                        "answer_correct": correct, "fresh_archive_read": fresh_read,
                                        "seconds_including_edit": time.monotonic()-query_start,
                                        "source_geometry": suffix_geometry(turn_trace),
                                        "context_edit_trace": [line for line in turn_trace.splitlines()
                                                               if "context suffix" in line or "live context edit" in line],
                                        "final_assistant": final})
                    if sent == len(turns):
                        proc.stdin.close()
                        continue
                    if sent >= 3:
                        number = sent-2
                        query_start = time.monotonic()
                        before = context.read_text()
                        trace_path = out / "agent.trace"
                        query_trace_start = len(trace_path.read_text()) if trace_path.exists() else 0
                        remaining = 4-number
                        if before.count(BEGIN) != remaining or before.count(END) != remaining:
                            raise ValueError(f"expected {remaining} remaining disjoint disposable blocks")
                        if before.count(active_text) != 1:
                            raise ValueError("expected one unchanged active inventory tool result")
                        lo, hi = before.index(BEGIN), before.index(END)+len(END)
                        if hi >= before.index(active_text):
                            raise ValueError("disposable block must precede unchanged active tool result")
                        after = before[:lo] + a.breadcrumb + before[hi:]
                        if active_text not in after:
                            raise ValueError("edit changed active inventory tool result")
                        (out / f"before-edit-{number}.txt").write_text(before)
                        (out / f"after-edit-{number}.txt").write_text(after)
                        if number == 1:
                            dump(archive, {"archived_code": "RIVER-73"})
                        atomic_text(context, after)
                        edits.append({"edit": number, "removed_bytes": len(before[lo:hi].encode()),
                                      "replacement": a.breadcrumb, "remaining_blocks": remaining-1,
                                      "active_tool_result_unchanged": True,
                                      "byte_edit_seconds": time.monotonic()-query_start})
                    proc.stdin.write((turns[sent]+"\n").encode())
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
            (out / (name+".log")).write_bytes(data)
    text = context.read_text() if context.exists() else ""
    (out / "final-context.txt").write_text(text)
    calls = re.findall(r'<｜DSML｜\s*invoke name="read">(.*?)</｜DSML｜\s*invoke>', text, re.S)
    active_read = any(str(active) in call for call in calls)
    geometries = [g for q in queries for g in q["source_geometry"]]
    result = {"returncode": rc, "error": error, "seconds": time.monotonic()-start,
              "mode": a.mode, "prerope": a.prerope if a.mode == "suffix" else 0,
              "turns_sent": sent, "turn_completions": completions, "edits": edits,
              "queries": queries, "active_tool_read": active_read,
              "successful_grafts": len(geometries),
              "grafts_from_prior_approximate": sum(g["prior_approximate"] == 1 for g in geometries),
              "pre_rope_grafts": sum(g["pre_rope"] == 1 for g in geometries),
              "source_endpoints": [g["source_end"] for g in geometries],
              "immutable_original_source_claim": False,
              "task_pass": bool(rc == 0 and not error and len(completions) == 6 and len(edits) == 3
                                and len(queries) == 3 and active_read
                                and all(q["answer_correct"] and q["fresh_archive_read"] for q in queries))}
    result["repeated_prerope_observed"] = bool(result["pre_rope_grafts"] >= 2
                                                and result["grafts_from_prior_approximate"] >= 1)
    result["experiment_pass"] = bool(result["task_pass"] and
                                      (a.mode != "suffix" or not a.prerope
                                       or result["repeated_prerope_observed"]))
    dump(out / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["experiment_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
