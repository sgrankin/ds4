#!/usr/bin/env python3
"""Two-turn native agent smoke; run only in a scheduled model/GPU slot."""
import argparse
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--agent", default="./ds4-agent")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--prefix-cache", choices=("on", "off"), default="on")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = output / "context.txt"
    trace = output / "agent.trace"
    if context.exists() or trace.exists():
        raise SystemExit("use a fresh output directory")
    payload = "irrelevant filler " * 500
    first = ("Secret code: 2718. Remember it for the next turn. "
             "[HIST_BLOB_7Q_BEGIN]" + payload + "[HIST_BLOB_7Q_END] "
             "The marked blob is irrelevant old history. Reply READY only.\n")
    second = (
        f"Test your live-context feature. Use one bash call with Python to read {context}, "
        "find the FIRST literal [HIST_BLOB_7Q_BEGIN] and the FIRST following literal "
        "[HIST_BLOB_7Q_END], and replace that inclusive span with [old blob removed]. "
        "Preserve every byte outside that old span, especially the initial system/tool/prefix "
        "section, the secret code outside the blob, and this latest request. Print the removed "
        "byte count. After the tool succeeds, return the remembered code as CODE=<number>. "
        "Do not repeat the old blob or print the whole context.\n")
    (output / "turns.json").write_text(json.dumps([first, second], indent=2))
    command = [args.agent, "--model", args.model, "--ssd-streaming",
               "--ctx", "8192", "--prefill-chunk", "2048", "--nothink",
               "--temp", "0", "--tokens", "512", "--seed", "41",
               "--non-interactive", "--live-context-file", str(context),
               "--trace", str(trace)]
    env = os.environ.copy()
    env.update(DS4_AGENT_CACHE_DIR=str(output / "cache"),
               DS4_AGENT_COMPACT_PREFIX_CACHE="1" if args.prefix_cache == "on" else "0",
               DS4_AGENT_TEST_TIME="1790954400")
    start = time.monotonic()
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)
    selector = selectors.DefaultSelector()
    for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
    chunks = {"stdout": bytearray(), "stderr": bytearray()}
    waiting_seen = 0
    sent = 0
    try:
        while selector.get_map():
            if time.monotonic() - start > args.timeout:
                raise TimeoutError("native CLM smoke exceeded timeout")
            for key, _ in selector.select(timeout=0.25):
                data = os.read(key.fd, 65536)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                chunks[key.data].extend(data)
                if key.data == "stderr":
                    count = chunks["stderr"].count(b"+DWARFSTAR_WAITING")
                    if count > waiting_seen:
                        waiting_seen = count
                        if sent == 0:
                            proc.stdin.write(first.encode())
                            proc.stdin.flush()
                            sent = 1
                        elif sent == 1:
                            (output / "before-edit.txt").write_bytes(context.read_bytes())
                            proc.stdin.write(second.encode())
                            proc.stdin.close()
                            sent = 2
        rc = proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for name, data in chunks.items():
            (output / (name + ".log")).write_bytes(data)
    text = context.read_text()
    trace_text = trace.read_text()
    edits = [(int(a), int(b)) for a, b in re.findall(
        r"live context edit committed tokens=(\d+)->(\d+)", trace_text)]
    assistant_parts = text.rsplit("<｜Assistant｜>", 1)
    final_assistant = assistant_parts[-1] if len(assistant_parts) == 2 else ""
    result = {
        "command": command, "returncode": rc, "seconds": time.monotonic() - start,
        "prefix_cache": args.prefix_cache,
        "turns_sent": sent, "accepted_edits": edits,
        "history_shrank": any(a - b >= 400 for a, b in edits),
        "fact_retained": "Secret code: 2718" in text,
        "blob_removed": payload not in text,
        "final_answer_correct": bool(re.search(r"CODE=2718", final_assistant)),
        "final_assistant_text": final_assistant[:512],
        "prefix_cache_hits": trace_text.count("compaction prefix cache hit"),
        "rejected_edits": chunks["stdout"].count(b"Live context edit rejected"),
    }
    result["pass"] = (rc == 0 and sent == 2 and result["history_shrank"] and
                      result["fact_retained"] and result["blob_removed"] and
                      result["final_answer_correct"])
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
