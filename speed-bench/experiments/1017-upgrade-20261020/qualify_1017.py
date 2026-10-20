#!/usr/bin/env python3
"""First-attempt 1017 smoke. Wrap this driver ONCE in pressure_run.py.

One native server (text, Responses, Conversations, optional image), then one
persistent-fast headless agent (opaque-ID CLM/archive/retrieval and tiny code).
Adapted from Item 42 run_edit.py and Item 11/17 HTTP smoke; no model retries.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from serve_quantum import ROOT, dump, stop_owned

HEADERS = re.compile(r"(?m)^\[\[DS4_TURN ([0-9a-f]{16}) role=(\w+)\]\]\n")
OLD_CODE, CURRENT_KEY = "EMBER-7319", "COVE-3917"


def clean_env():
    return {k: v for k, v in os.environ.items() if not k.startswith(("DS4", "PT_"))}


def response_text(body):
    return "".join(p.get("text", "") for item in body.get("output", [])
        if item.get("type") == "message" for p in item.get("content", [])
        if p.get("type") == "output_text").strip()


def server_smoke(args):
    out = args.output / "server"
    out.mkdir()
    env = clean_env()
    env.update(DS4_METAL_V41_RESIDENT_ROUTE_GAP="0", DS4_METAL_V41_RESIDENT_PREFILL_GAP="0",
               DS4_SERVER_CONTEXT_SUFFIX_REUSE="0", DS4_SERVER_RESPONSE_STORE_DIR=str(out / "responses"))
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0)); port = reservation.getsockname()[1]
    command = [str(ROOT / "ds4-server"), "--model", str(args.model), "--host", "127.0.0.1",
        "--port", str(port), "--ssd-streaming", "--ssd-streaming-cache-experts", "64GB",
        "--ctx", "8192", "--prefill-chunk", "2048", "--batched-session", "1", "--mtp-draft", "1"]
    if args.vision:
        command += ["--vision", str(args.vision)]
    dump(out / "execution.json", {"command": command, "cwd": str(ROOT),
        "env": {k: v for k, v in env.items() if k.startswith("DS4")},
        "application_ssd_kv_cache": "disabled", "warmup_requests": 0})
    proc = None
    rows, checks = [], {}
    started = time.monotonic()
    deadline = started + args.timeout
    error = None
    base = f"http://127.0.0.1:{port}/v1"

    def http(name, method, route, payload=None):
        request = urllib.request.Request(base + route, method=method,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Connection": "close"})
        row = {"name": name, "method": method, "route": route, "request": payload}
        t = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=max(.1, deadline-time.monotonic())) as reply:
                row.update(status=reply.status, wire=reply.read().decode())
        except urllib.error.HTTPError as exc:
            row.update(status=exc.code, wire=exc.read().decode(errors="replace"))
        row["seconds"] = time.monotonic() - t
        row["body"] = json.loads(row["wire"])
        rows.append(row); dump(out / (name + ".json"), row)
        if row["status"] != 200:
            raise RuntimeError(f"{name}: HTTP {row['status']}; first wire preserved")
        return row["body"]

    try:
        with (out / "stdout.log").open("wb") as stdout, (out / "stderr.log").open("wb") as stderr:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"owned server startup exit {proc.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("server smoke deadline")
                try:
                    with urllib.request.urlopen(base + "/models", timeout=1) as reply:
                        models = json.load(reply)["data"]
                    if models:
                        break
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(.2)
            common = {"model": models[0]["id"], "temperature": 0, "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 32}
            normal = http("normal-generation", "POST", "/chat/completions", common | {
                "messages": [{"role": "user", "content": "Reply only LANGUAGE_OK."}]})
            checks["normal_generation"] = normal["choices"][0]["message"]["content"].strip() == "LANGUAGE_OK"
            rc = {"model": models[0]["id"], "store": True, "temperature": 0,
                  "max_output_tokens": 48, "reasoning": {"effort": "none"}}
            parent = http("response-parent", "POST", "/responses", rc | {
                "input": "Remember the literal LAGOON-2156. Reply only READY."})
            checks["response_parent"] = parent.get("status") == "completed" and response_text(parent) == "READY"
            fetched = http("response-get", "GET", "/responses/" + parent["id"])
            checks["response_get"] = fetched.get("id") == parent["id"] and fetched.get("output") == parent.get("output")
            child = http("response-child", "POST", "/responses", rc | {
                "previous_response_id": parent["id"], "input": "Reply only with the remembered literal."})
            checks["previous_response_id"] = (child.get("status") == "completed" and
                child.get("previous_response_id") == parent["id"] and response_text(child) == "LAGOON-2156")
            conv = http("conversation-create", "POST", "/conversations", {"metadata": {"smoke": "1017"}})
            cid = conv["id"]
            first = http("conversation-first", "POST", "/responses", rc | {
                "conversation": cid, "input": "The launch key is HARBOR-9264. Reply only READY."})
            final = http("conversation-recall", "POST", "/responses", rc | {
                "conversation": cid, "input": "Reply only with the launch key."})
            listed = http("conversation-items", "GET", "/conversations/" + cid + "/items?order=asc&limit=20")
            checks["conversation_recall"] = (first.get("status") == final.get("status") == "completed" and
                response_text(first) == "READY" and response_text(final) == "HARBOR-9264" and
                len(listed.get("data", [])) >= 4)
            if args.vision:
                uri = "data:image/png;base64," + base64.b64encode(args.image.read_bytes()).decode()
                vision = http("vision-ocr", "POST", "/chat/completions", common | {
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "What is the word and number printed at the top of this image? Reply only that exact text."},
                        {"type": "image_url", "image_url": {"url": uri}}]}]})
                checks["vision_ocr"] = vision["choices"][0]["message"]["content"].strip() == "ORBIT 4729"
    except InterruptedError:
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        returncode = stop_owned(proc)
    result = {"checks": checks, "error": error, "server_returncode": returncode,
        "seconds": time.monotonic()-started, "vision_requested": bool(args.vision),
        "vision_fixture": str(args.image), "first_attempts": True,
        "pass": not error and bool(checks) and all(checks.values())}
    dump(out / "result.json", result)
    return result


GRADER = r'''
import ast,copy,pathlib,sys
p=pathlib.Path(sys.argv[1])
scope={}
exec(compile(ast.parse(p.read_text()),str(p),"exec"),scope)
f=scope["merge_intervals"]
cases=[([],[]),([[1,3]],[[1,3]]),([[1,3],[2,6],[8,10],[9,11]],[[1,6],[8,11]]),
 ([[5,7],[1,2],[2,5]],[[1,7]]),([[-4,-1],[-2,0],[1,1]],[[-4,0],[1,1]]),
 ([[1,10],[2,3],[1,10]],[[1,10]]),([[4,5],[1,2]],[[1,2],[4,5]]),
 ([[0,0],[0,0]],[[0,0]])]
for data,gold in cases:
 before=copy.deepcopy(data)
 assert f(data)==gold,(data,gold)
 assert data==before,"input mutated"
print("PASS 8 merge-interval cases and input preservation")
'''


def agent_smoke(args):
    out = args.output / "agent"
    out.mkdir(); work = out / "work"; work.mkdir()
    context, archive = work / "context.txt", work / "archived-turn.txt"
    coding_source, first_code = work / "merge_intervals.py", out / "first-merge_intervals.py"
    env = clean_env()
    env.update(DS4_AGENT_CACHE_DIR=str(out / "bootstrap"), DS4_STATE_CACHE_DIR=str(out / "state-cache"),
               DS4_STATE_CACHE_HOT_MB="0", DS4_STATE_CACHE_SSD_MB="4096", DS4_AGENT_TEST_TIME="1792490400")
    command = [str(ROOT / "ds4-local"), "persistent-fast", "--model", str(args.model),
        "--ctx", "16384", "--prefill-chunk", "2048", "--nothink", "--temp", "0",
        "--tokens", "1024", "--seed", "20261020", "--non-interactive", "--chdir", str(work),
        "--live-context-file", str(context), "--trace", str(out / "agent.trace")]
    dump(out / "execution.json", {"command": command, "cwd": str(ROOT),
        "env": {k: v for k, v in env.items() if k.startswith("DS4")},
        "profile": "persistent-fast", "first_attempts": True})
    turns = ["[OBSOLETE_CASE_A] Archived code: " + OLD_CODE + ". This entire user turn is old audit history, "
        "which may later be archived and deleted.\n" + "Old audit log: no current decisions.\n" * 48 +
        "Do not use tools now. Reply READY only.",
        "The current key is " + CURRENT_KEY + ". Preserve this current fact and this turn. Do not use tools now. Reply READY only."]
    started = time.monotonic()
    proc = None; sent = seen = 0; target = None; old_block = None; error = None
    retrieval_token = "CHECK-" + secrets.token_hex(6)
    stdout_offset = 0
    try:
        with (out / "stdout.log").open("wb") as stdout, (out / "stderr.log").open("wb") as stderr:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
            while proc.poll() is None:
                if sent == 5 and coding_source.exists() and not first_code.exists():
                    first_code.write_bytes(coding_source.read_bytes())
                if time.monotonic()-started >= args.timeout:
                    raise TimeoutError("agent qualification deadline")
                waiting = (out / "stderr.log").read_bytes().count(b"+DWARFSTAR_WAITING")
                if waiting <= seen:
                    time.sleep(.1); continue
                seen = waiting
                if sent:
                    text = context.read_text()
                    (out / f"context-{sent}.txt").write_text(text)
                if sent == 2:
                    blocks = list(HEADERS.finditer(text))
                    for i, match in enumerate(blocks):
                        end = blocks[i+1].start() if i+1 < len(blocks) else len(text)
                        if "[OBSOLETE_CASE_A]" in text[match.end():end]:
                            target = match[1]; old_block = text[match.start():end]; break
                    if target is None:
                        raise ValueError("no opaque-labelled obsolete user turn")
                    turns.append(f"Archive the complete original older audit user turn with opaque ID {target} into {archive}, "
                        "then delete only that turn from the live context using the context editing recipe. "
                        "The ID is a string, not a position. Do not read/cat the complete context or paste its old text into arguments. "
                        "Preserve the current key and latest request. Make one cleanup, then reply CLEANED only.")
                if sent == 3:
                    original_archive = archive.read_text()
                    (out / "archive-original.txt").write_text(original_archive)
                    # Original archive remains intact; this newly appended receipt
                    # exists only in storage and forces a real subsequent retrieval.
                    archive.write_text(original_archive + "\nRETRIEVAL_TOKEN=" + retrieval_token + "\n")
                    stdout_offset = (out / "stdout.log").stat().st_size
                    turns.append(f"Freshly read {archive} using read with whole=true and raw=true, or a narrow file search. "
                        "Return only JSON with archived_code, current_key, and retrieval_token (the new RETRIEVAL_TOKEN in that file). "
                        "Do not read or edit the live context file. Do not guess from memory.")
                if sent == 4:
                    (out / "retrieval.stdout.log").write_bytes((out / "stdout.log").read_bytes()[stdout_offset:])
                    turns.append(f"Write {work / 'merge_intervals.py'} containing merge_intervals(intervals). "
                        "Input is a list of integer [start,end] pairs with start<=end. Return a new sorted list merging overlapping "
                        "or touching closed intervals (next start<=current end). Do not mutate input. Use no external dependencies. "
                        "Write the file once, do not run tests or revise it; reply WROTE only. Do not edit the live context.")
                if sent == len(turns):
                    proc.stdin.close(); break
                proc.stdin.write((turns[sent] + "\n").encode()); proc.stdin.flush(); sent += 1
            proc.wait(timeout=15)
    except InterruptedError:
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        returncode = stop_owned(proc)
    dump(out / "fixture.json", {"turns": turns, "target_id": target, "retrieval_token": retrieval_token})
    def read(name):
        p = out / name
        return p.read_text() if p.exists() else ""
    before, edited = read("context-2.txt"), read("context-3.txt")
    before_ids = {m[1] for m in HEADERS.finditer(before)}
    after_ids = {m[1] for m in HEADERS.finditer(edited)}
    answer_text = read("context-4.txt").rsplit("<｜Assistant｜>", 1)[-1]
    try:
        answer = json.JSONDecoder().raw_decode(answer_text[answer_text.index("{"):])[0]
    except ValueError:
        answer = None
    archived = read("archive-original.txt")
    retrieval_stdout = read("retrieval.stdout.log")
    trace = read("agent.trace")
    if coding_source.exists() and not first_code.exists():
        first_code.write_bytes(coding_source.read_bytes())
    checks = {"five_turns_complete": sent == 5 and seen == 6 and returncode == 0 and not error,
        "opaque_target_removed": target is not None and target not in after_ids,
        "retained_ids_stable": bool(before_ids) and before_ids-{target} <= after_ids,
        "verbatim_archive": old_block is not None and archived.strip("\n") == old_block.strip("\n"),
        "fresh_storage_answer": answer == {"archived_code": OLD_CODE, "current_key": CURRENT_KEY, "retrieval_token": retrieval_token},
        "retrieval_tool_visible": "🛠" in retrieval_stdout and archive.name in retrieval_stdout,
        "one_context_edit": trace.count("live context edit committed") == 1,
        "no_automatic_compaction": "COMPACTING" not in read("stdout.log"),
        "no_edit_rejection": "live context edit rejected" not in
            (trace + read("stdout.log") + read("stderr.log")).lower()}
    grader = {"command": [sys.executable, "-c", GRADER, str(first_code)], "cwd": str(work),
        "source": str(first_code), "observation": "first file observed by existing 100ms protocol loop",
        "first_observed_matches_final": first_code.exists() and coding_source.exists()
            and first_code.read_bytes() == coding_source.read_bytes()}
    try:
        completed = subprocess.run(grader["command"], cwd=work, text=True, capture_output=True, timeout=5)
        grader.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        checks["fresh_coding_8_cases"] = completed.returncode == 0
    except Exception as exc:
        grader["error"] = f"{type(exc).__name__}: {exc}"
        checks["fresh_coding_8_cases"] = False
    dump(out / "coding-grade.json", grader)
    result = {"returncode": returncode, "error": error, "seconds": time.monotonic()-started,
        "checks": checks, "answer": answer, "suffix_activation_lines": [line for line in trace.splitlines()
        if "context suffix APPROXIMATE" in line], "first_attempts": True, "pass": all(checks.values())}
    dump(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vision", type=Path, help="optional matched 1017 vision GGUF; includes actual image OCR")
    parser.add_argument("--image", type=Path, default=ROOT / "tests/vision-fixtures/qwen38/orbit.png")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900, help="per server/agent phase deadline")
    args = parser.parse_args()
    args.model = args.model.resolve(); args.image = args.image.resolve(); args.output = args.output.resolve()
    if args.vision:
        args.vision = args.vision.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    def interrupted(signum, _frame):
        raise InterruptedError(f"driver interrupted by signal {signum}")
    saved = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    results = {}
    try:
        results["native_server"] = server_smoke(args)
        # The owned server is fully stopped before loading the agent model.
        results["persistent_agent"] = agent_smoke(args)
    except Exception as exc:
        results["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    results["pass"] = ("error" not in results and all(results.get(name, {}).get("pass")
        for name in ("native_server", "persistent_agent")))
    results["scope"] = "Tiny first-attempt compatibility/quality screen on new weights, not a baseline or speed claim. Vision is qualified only if explicitly requested and its check passes."
    dump(args.output / "result.json", results)
    print(json.dumps(results, ensure_ascii=False), flush=True)
    return 0 if results["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
