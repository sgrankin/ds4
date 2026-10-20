#!/usr/bin/env python3
"""Actual direct-agent left/right edits with fresh inventory and archive reads.

Adapted from Item 8 run_agent_repeat.py. The wrapper supplies the combined
preset. Actual traces decide whether either edit grafted; no parity/model retry.
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

ROOT = Path(__file__).resolve().parents[3]
BEGIN, END = "[DISPOSABLE_OLD_BEGIN]", "[DISPOSABLE_OLD_END]"
BREADCRUMB = "[old history archived]"
GOLD = [{"active_total": 416, "archived_code": "RIVER-73"},
        {"active_total": 466, "archived_code": "EMBER-29"}]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".edit.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def answer_from(text):
    final = text.rsplit("<｜Assistant｜>", 1)[-1]
    try:
        return json.JSONDecoder().raw_decode(final[final.index("{"):])[0], final
    except (ValueError, json.JSONDecodeError):
        return None, final


def geometry(trace):
    rows = []
    for line in trace.splitlines():
        if "context suffix APPROXIMATE" not in line:
            continue
        match = re.search(r"reused=(\d+) fresh=(\d+) source=(\d+):(\d+) destination=(\d+):(\d+)", line)
        if not match:
            continue
        reused, fresh, src, src_end, dst, dst_end = map(int, match.groups())
        row = {"reused": reused, "fresh": fresh, "source_start": src,
               "source_end": src_end, "destination_start": dst,
               "destination_end": dst_end, "destination_minus_source": dst-src,
               "aligned": src % 2 == dst % 2 == 0,
               "span_lengths_equal": src_end-src == dst_end-dst, "line": line}
        for field in ("pre_rope", "prior_approximate", "source_ssd", "right_shift",
                      "source_approximate", "latest_user_start", "latest_user_fresh"):
            found = re.search(rf"\b{field}=(-?\d+)", line)
            row[field] = int(found[1]) if found else None
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shared", required=True)
    parser.add_argument("--mode", choices=("native", "persistent-fast"), default="persistent-fast")
    parser.add_argument("--blob-units", type=int, default=96)
    parser.add_argument("--notes", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    out, shared = Path(args.output).resolve(), Path(args.shared).resolve()
    out.mkdir(parents=True, exist_ok=False)
    work, bootstrap, state = shared / "work", shared / "bootstrap-cache", shared / "state-cache"
    for path in (work, bootstrap, state):
        path.mkdir(parents=True, exist_ok=True)
    context, active, archive = shared / "context.txt", work / "active.json", work / "archive.json"
    context.unlink(missing_ok=True)
    active_data = {"batches": 17, "items_per_batch": 25, "reserved_items": 9,
                   "notes": [f"Active processing note {i}: current inventory is authoritative; old audit prose is disposable."
                             for i in range(args.notes)]}
    dump(active, active_data)
    active_text = json.dumps(active_data, indent=2, ensure_ascii=False)
    dump(archive, {"archived_code": None})
    block = (BEGIN + "\nOld audit code: RIVER-73.\n"
             + "old audit log unchanged.\n" * args.blob_units + END)
    # Insert inside an existing sentence: appending a word after the marker
    # also changes its newline tokenization (the first two traces retain this).
    right_replacement = block + "\n" + block.replace(
        "old audit log unchanged.", "old audit log Reply unchanged.", 1)
    first = ("The next three disjoint marked spans are disposable old history. "
             "Their useful code will later be stored in a permanent archive.\n"
             + "\n".join([block] * 3) + "\nDo not use tools now. Reply READY only.")
    second = (f"Use the read tool once to read {active}, with whole=true and raw=true. "
              "This file is the self-contained active inventory specification. Retain the numeric rules "
              "and notes from its tool result. Do not read other files. Then reply READY only.")
    third = ("Reply READY only, without tools or calculations. The marked old audit blocks are disposable. "
             "Their useful code will be stored in a permanent archive. The inventory and archive files "
             "can change between questions. On every later question, freshly read both requested files, "
             "rather than relying on remembered numeric values or code. Preserve the unchanged earlier "
             "inventory tool result and this request when history is edited. Keep numeric values numeric "
             "and archive codes strings. Do not summarize or edit files yourself. This is a readiness "
             "turn: provide exactly READY and finish.")
    query = (f"Read both {active} and {archive} again now using the read tool with whole=true and raw=true. "
             "Fresh reads of both files are required this turn; their current contents override earlier "
             "reads and remembered values. Return only JSON with active_total (current batches times "
             "items_per_batch minus reserved_items) and archived_code from the current archive file.")
    turns = [first, second, third, query, query]
    dump(out / "fixture.json", {"turns": turns, "initial_active": active_data,
                               "gold_each_query": GOLD, "block": block,
                               "second_edit_replacement": right_replacement,
                               "mode": args.mode, "blob_units": args.blob_units,
                               "notes": args.notes, "context": str(context),
                               "active": str(active), "archive": str(archive)})
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DS4", "PT_"))}
    env.update(DS4_AGENT_CACHE_DIR=str(bootstrap), DS4_STATE_CACHE_DIR=str(state),
               DS4_STATE_CACHE_HOT_MB="0", DS4_STATE_CACHE_SSD_MB="8192",
               DS4_AGENT_TEST_TIME="1791504000")
    if args.mode == "persistent-fast":
        command = [str(ROOT / "ds4-local"), "persistent-fast"]
    else:
        command = [str(ROOT / "ds4-agent"), "--ssd-streaming",
                   "--ssd-streaming-cache-experts", "64GB", "--ctx", "8192", "--prefill-chunk", "512"]
        env.update(DS4_AGENT_COMPACT_PREFIX_CACHE="1", DS4_AGENT_CONTEXT_FRONTIERS="4",
                   DS4_AGENT_CONTEXT_FRONTIER_MB="512", DS4_AGENT_CONTEXT_SUFFIX_REUSE="0",
                   DS4_AGENT_CONTEXT_SUFFIX_PREROPE="0", DS4_AGENT_CONTEXT_SUFFIX_RIGHT="0",
                   DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL="128", DS4_METAL_V41_RESIDENT_ROUTE_GAP="0",
                   DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS="0", DS4_METAL_V41_RESIDENT_PREFILL_GAP="0",
                   DS4_METAL_V41_SELECTED_TILE="64", DS4_METAL_V41_SELECTED_LIMIT="1024")
        for source in (ROOT / "metal").glob("*.metal"):
            env["DS4_METAL_" + source.stem.upper() + "_SOURCE"] = str(source)
    command += ["--model", args.model, "--nothink", "--temp", "0", "--tokens", "512",
                "--seed", "20261009", "--non-interactive", "--chdir", str(work),
                "--live-context-file", str(context), "--trace", str(out / "agent.trace")]
    dump(out / "execution.json", {"command": command, "cwd": str(ROOT),
                                 "explicit_env": {k: v for k, v in env.items() if k.startswith("DS4")},
                                 "preset_settings": "ds4-local + preset-check.json" if args.mode == "persistent-fast" else None,
                                 "eof_saves_named_session": False,
                                 "geometry_authority": "actual agent trace; no parity or model retry"})
    started, sent, seen = time.monotonic(), 0, 0
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    selector = selectors.DefaultSelector()
    chunks = {"stdout": bytearray(), "stderr": bytearray()}
    for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
    completions, edits, queries, error = [], [], [], None
    query_started, trace_start = None, 0
    try:
        while selector.get_map():
            if time.monotonic()-started > args.timeout:
                raise TimeoutError(f"{args.timeout:g}-second deadline exceeded")
            for key, _ in selector.select(0.2):
                data = os.read(key.fd, 65536)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                chunks[key.data].extend(data)
                if key.data != "stderr":
                    continue
                count = chunks["stderr"].count(b"+DWARFSTAR_WAITING")
                while seen < count:
                    seen += 1
                    if sent:
                        completions.append({"turn": sent, "seconds_from_start": time.monotonic()-started})
                    if sent >= 4:
                        number = sent-3
                        text = context.read_text()
                        (out / f"query-{number}-context.txt").write_text(text)
                        turn_text = text[text.rfind("<｜User｜>" + query):]
                        answer, final = answer_from(turn_text)
                        calls = re.findall(r'<｜DSML｜\s*invoke name="read">(.*?)</｜DSML｜\s*invoke>', turn_text, re.S)
                        fresh_active = any(str(active) in call for call in calls)
                        fresh_archive = any(str(archive) in call for call in calls)
                        turn_trace = (out / "agent.trace").read_text()[trace_start:]
                        (out / f"query-{number}.trace").write_text(turn_trace)
                        queries.append({"query": number, "answer": answer, "gold": GOLD[number-1],
                                        "answer_correct": answer == GOLD[number-1] and type(answer.get("active_total")) is int,
                                        "fresh_active_read": fresh_active, "fresh_archive_read": fresh_archive,
                                        "seconds_including_edit": time.monotonic()-query_started,
                                        "source_geometry": geometry(turn_trace), "final_assistant": final})
                    if sent == len(turns):
                        proc.stdin.close()
                        continue
                    if sent >= 3:
                        number = sent-2
                        query_started = time.monotonic()
                        before = context.read_text()
                        trace_path = out / "agent.trace"
                        trace_start = len(trace_path.read_text()) if trace_path.exists() else 0
                        if active_text not in before:
                            raise ValueError("initial inventory tool result missing before edit")
                        if number == 1:
                            lo, hi = before.index(BEGIN), before.index(END)+len(END)
                            after = before[:lo] + BREADCRUMB + before[hi:]
                            kind = "left deletion"
                        else:
                            if before.count(BREADCRUMB) != 1:
                                raise ValueError("expected exactly one breadcrumb for growing replacement")
                            lo = before.index(BREADCRUMB)
                            hi = lo + len(BREADCRUMB)
                            after = before[:lo] + right_replacement + before[hi:]
                            kind = "right growing replacement"
                        if hi >= before.index(active_text) or before.count(active_text) != after.count(active_text):
                            raise ValueError("edit must precede and preserve initial inventory result")
                        (out / f"before-edit-{number}.txt").write_text(before)
                        (out / f"after-edit-{number}.txt").write_text(after)
                        active_data["batches"] = 17 if number == 1 else 19
                        dump(active, active_data)
                        dump(archive, {"archived_code": GOLD[number-1]["archived_code"]})
                        atomic_text(context, after)
                        edits.append({"edit": number, "kind": kind,
                                      "byte_delta": len(after.encode())-len(before.encode()),
                                      "initial_active_tool_result_unchanged": True,
                                      "current_inventory_batches": active_data["batches"],
                                      "current_archive_code": GOLD[number-1]["archived_code"]})
                    proc.stdin.write((turns[sent]+"\n").encode())
                    proc.stdin.flush()
                    sent += 1
        rc = proc.wait(timeout=5)
    except Exception as exc:
        error, rc = f"{type(exc).__name__}: {exc}", None
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
        selector.close()
        for name, data in chunks.items():
            (out / (name+".log")).write_bytes(data)
    (out / "final-context.txt").write_text(context.read_text() if context.exists() else "")
    grafts = [g for q in queries for g in q["source_geometry"]]
    task_pass = (rc == 0 and error is None and len(completions) == 5 and len(edits) == 2
                 and len(queries) == 2 and all(q["answer_correct"] and q["fresh_active_read"]
                                             and q["fresh_archive_read"] for q in queries))
    left, right = ([g for g in grafts if g["destination_minus_source"] < 0],
                   [g for g in grafts if g["destination_minus_source"] > 0])
    qualified = bool(left and right and all(g["aligned"] and g["span_lengths_equal"]
                                           and g["pre_rope"] == 1 and g["source_ssd"] == 1
                                           and g["latest_user_fresh"] == 1 for g in grafts))
    stderr = chunks["stderr"].decode(errors="replace")
    routing = {}
    for kind in ("resident-route", "resident-route-native-weights", "resident-prefill"):
        lines = [line for line in stderr.splitlines() if line.startswith("ds4: " + kind + " ")]
        if lines:
            routing[kind] = {"line": lines[-1],
                             "fields": dict(re.findall(r"\b(\w+)=([0-9]+(?:\.[0-9]+)?)", lines[-1]))}
    routing_observed = all(int(routing.get(kind, {}).get("fields", {}).get("substitutions", "0")) > 0
                           for kind in ("resident-route-native-weights", "resident-prefill"))
    result = {"mode": args.mode, "returncode": rc, "error": error,
              "seconds": time.monotonic()-started, "turns_sent": sent,
              "turn_completions": completions, "edits": edits, "queries": queries,
              "successful_grafts": len(grafts), "left_grafts": len(left), "right_grafts": len(right),
              "routing_summary": routing, "decode_and_prefill_substitutions_observed": routing_observed,
              "ssd_pre_rope_both_directions_observed": qualified, "task_pass": bool(task_pass),
              "experiment_pass": bool(task_pass and (args.mode == "native" or (qualified and routing_observed)))}
    dump(out / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["experiment_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
