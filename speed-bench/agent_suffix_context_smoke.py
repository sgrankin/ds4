#!/usr/bin/env python3
"""Three-turn native agent suffix smoke; model execution needs a root GPU slot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import struct
import subprocess
import time


def prepare(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = output / "context.txt"
    trace = output / "agent.trace"
    if context.exists() or trace.exists() or (output / "result.json").exists():
        raise RuntimeError("use a fresh native output directory")
    updater = output / "update_context.py"
    updater.write_text(
        "from pathlib import Path\n"
        "import os\n"
        f"p=Path({str(context)!r})\n"
        "before=p.read_text()\n"
        "old='Record Otter current access code is 2718.'\n"
        "new='Record Otter current access code is 3141.'\n"
        "assert before.count(old)==1\n"
        "after=before.replace(old,new,1)\n"
        "(p.parent/'before-tool-edit.txt').write_text(before)\n"
        "(p.parent/'after-tool-edit.txt').write_text(after)\n"
        "tmp=p.with_suffix('.edited')\n"
        "tmp.write_text(after)\n"
        "os.replace(tmp,p)\n"
        "print('Updated exactly one earlier record; all other context bytes preserved.')\n"
    )
    first = (
        "Record Otter current access code is 2718. This is ordinary test data. "
        "Neutral notes never specify or change an access code. "
        "What is the current access code? Reply only CODE=<number>. Do not use tools.\n"
    )
    second = (
        "Retain these irrelevant transport notes for the next turn. They do not change any record.\n"
        + "".join(f"Neutral note {i}: pencils are on the shelf.\n" for i in range(40))
        + "Reply only READY and stop. Do not use tools.\n"
    )
    third = (
        f"Use one bash call to run python3 {updater}. Do not read or print the script. "
        "It edits one earlier record line in your live-context file and preserves the "
        "system/tool prefix and all recent turns. When it finishes, use the updated "
        "conversation to report Record Otter's current access code. Reply only CODE=<number>.\n"
        + "The following delivery notes remain irrelevant to the access code:\n"
        + "".join(f"Delivery note {i}: the blue box stays on the shelf.\n" for i in range(22))
        + "Run the bash command above now, then reply only CODE=<number> from the updated conversation.\n"
    )
    turns = [first, second, third]
    (output / "turns.json").write_text(json.dumps(turns, indent=2) + "\n")
    command = [str(Path(args.agent).resolve()), "--model", str(Path(args.model).resolve()),
               "--ssd-streaming", "--ssd-streaming-cache-experts", "64GB",
               "--ctx", "8192", "--prefill-chunk", "256", "--nothink", "--temp", "0",
               "--tokens", "512", "--seed", "41", "--non-interactive",
               "--chdir", str(output), "--live-context-file", str(context), "--trace", str(trace)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("DS4")}
    env.update(DS4_AGENT_CACHE_DIR=str(output / "cache"),
               DS4_AGENT_CONTEXT_SUFFIX_REUSE="1" if args.suffix_mode == "on" else "0",
               DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL="128",
               DS4_AGENT_CONTEXT_FRONTIER_MB="128",
               DS4_AGENT_COMPACT_PREFIX_CACHE="1", DS4_AGENT_TEST_TIME="1790974400")
    provenance = {"binary_sha256": hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
                  "updater_sha256": hashlib.sha256(updater.read_bytes()).hexdigest(),
                  "turns_sha256": hashlib.sha256((output / "turns.json").read_bytes()).hexdigest(),
                  "model_stat": {"bytes": Path(args.model).stat().st_size,
                                 "mtime_ns": Path(args.model).stat().st_mtime_ns}}
    if args.expected_agent_sha256 and provenance["binary_sha256"] != args.expected_agent_sha256:
        raise RuntimeError("frozen agent binary hash changed")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "command.json").write_text(json.dumps({"argv": command,
        "env": {k: v for k, v in env.items() if k.startswith("DS4")}}, indent=2) + "\n")
    return output, context, trace, turns, command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suffix-mode", choices=("on", "off"), default="on")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--expected-agent-sha256")
    parser.add_argument("--timeout", type=float, default=420)
    args = parser.parse_args()
    output, context, trace, turns, command, env = prepare(args)
    if args.prepare_only:
        print(json.dumps({"status": "prepared; no model opened", "output": str(output), "argv": command}))
        return 0
    start = time.monotonic()
    proc = None
    chunks = {"stdout": bytearray(), "stderr": bytearray()}
    result = {"pass": False, "suffix_mode": args.suffix_mode, "turns_sent": 0}
    old_control = ""
    try:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env, start_new_session=True)
        selector = selectors.DefaultSelector()
        for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        waiting_seen = 0
        sent = 0
        while selector.get_map():
            if time.monotonic() - start > args.timeout:
                raise TimeoutError("native suffix-agent deadline")
            for key, _ in selector.select(0.25):
                data = os.read(key.fd, 65536)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                chunks[key.data].extend(data)
                (output / (key.data + ".log")).write_bytes(chunks[key.data])
                if key.data != "stderr":
                    continue
                count = chunks["stderr"].count(b"+DWARFSTAR_WAITING")
                if count <= waiting_seen:
                    continue
                waiting_seen = count
                if sent == 1:
                    old_control = context.read_text().rsplit("<｜Assistant｜>", 1)[-1]
                    (output / "turn1-context.txt").write_bytes(context.read_bytes())
                if sent == 2:
                    (output / "before-turn3.txt").write_bytes(context.read_bytes())
                if sent < len(turns):
                    proc.stdin.write(turns[sent].encode())
                    proc.stdin.flush()
                    sent += 1
                    result["turns_sent"] = sent
                    print(f"SUBMITTED native suffix-agent turn {sent}", flush=True)
                    if sent == len(turns):
                        proc.stdin.close()
        result["agent_returncode"] = proc.wait(timeout=5)
        text = context.read_text()
        trace_text = trace.read_text()
        final = text.rsplit("<｜Assistant｜>", 1)[-1]
        reused = [{"reused": int(a), "fresh": int(b)} for a, b in re.findall(
            r"context suffix APPROXIMATE reused=(\d+) fresh=(\d+)", trace_text)]
        latest_query_checks = re.findall(r"latest_user_start=(\d+) latest_user_fresh=(\d)", trace_text)
        named = []
        for path in (output / "cache").glob("*.kv"):
            h = path.read_bytes()[:52]
            if len(h) == 52 and h[5] == 6:
                named.append({"path": str(path), "tokens": struct.unpack_from("<I", h, 8)[0],
                              "payload_bytes": struct.unpack_from("<Q", h, 40)[0]})
        before = (output / "before-tool-edit.txt").read_text()
        after = (output / "after-tool-edit.txt").read_text()
        expected_edit = before.replace("Record Otter current access code is 2718.",
                                       "Record Otter current access code is 3141.", 1)
        result.update(seconds=time.monotonic() - start,
            old_control_correct=bool(re.search(r"CODE\s*=\s*2718", old_control)),
            final_answer_correct=bool(re.search(r"CODE\s*=\s*3141", final)),
            final_assistant_text=final[:512], approximate_reuses=reused,
            fully_fresh_latest_query=bool(latest_query_checks) and all(fresh == "1" for _, fresh in latest_query_checks),
            protected_and_other_bytes_preserved=after == expected_edit,
            updated_fact_retained="Record Otter current access code is 3141." in text,
            rejected_edits=chunks["stdout"].count(b"Live context edit rejected"),
            named_session_files=named, text_only_named_save=bool(named) and all(x["payload_bytes"] == 0 for x in named),
            approximate_save_notice=b"Saved approximate-context transcript without KV" in chunks["stdout"])
        reuse_ok = bool(reused) and all(x["fresh"] >= 128 and x["reused"] >= 64 for x in reused)
        persistence_ok = result["text_only_named_save"] if args.suffix_mode == "on" else True
        result["pass"] = (result["agent_returncode"] == 0 and sent == 3 and
            result["old_control_correct"] and result["final_answer_correct"] and
            result["protected_and_other_bytes_preserved"] and result["updated_fact_retained"] and
            result["rejected_edits"] == 0 and persistence_ok and
            ((reuse_ok and result["fully_fresh_latest_query"]) if args.suffix_mode == "on" else not reused))
        return 0 if result["pass"] else 3
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - start)
        return 2
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for name, data in chunks.items():
            (output / (name + ".log")).write_bytes(data)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
