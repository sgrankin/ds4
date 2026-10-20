#!/usr/bin/env python3
"""One first-attempt CLM edit using the existing headless agent protocol."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
case = sys.argv[1] if len(sys.argv) > 1 else "live"
OUT = Path(__file__).resolve().parent / case
WORK = ROOT / ".workspaces/agent-context-structure-20261009" / case
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
context = WORK / "context.txt"
archive = WORK / "archived-turn.txt"
env = {k: v for k, v in os.environ.items() if not k.startswith(("DS4", "PT_"))}
env.update(DS4_AGENT_CACHE_DIR=str(WORK / "bootstrap"),
           DS4_STATE_CACHE_DIR=str(WORK / "state-cache"),
           DS4_AGENT_TEST_TIME="1791558000")
command = [str(ROOT / "ds4-local"), "persistent-fast", "--model",
           "gguf/DeepSeek-V4.1-Flash-Q2.gguf", "--prefill-chunk", "2048",
           "--nothink", "--temp", "0", "--tokens", "1536", "--seed", "20261009",
           "--non-interactive", "--live-context-file", str(context),
           "--trace", str(OUT / "agent.trace")]
(OUT / "execution.json").write_text(json.dumps({"command": command, "cwd": str(ROOT),
    "env": {k: v for k, v in env.items() if k.startswith("DS4")}}, indent=2) + "\n")
turns = [
    "[OBSOLETE_CASE_A] Archived code: EMBER-7319. This entire user turn is old audit history, "
    "which may later be archived and deleted.\n" + "Old audit log: no current decisions.\n" * 48
    + "Do not use tools now. Reply READY only.",
    "The current key is COVE-3917. Preserve this current task fact and this turn. "
    "Do not use tools now. Reply READY only.",
]
started = time.monotonic()
sent = seen = 0
error = None
target = None
with (OUT / "stdout.log").open("wb") as stdout, (OUT / "stderr.log").open("wb") as stderr:
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                            stdout=stdout, stderr=stderr)
    try:
        while proc.poll() is None:
            if time.monotonic() - started > 900:
                raise TimeoutError("900-second screen budget")
            waiting = (OUT / "stderr.log").read_bytes().count(b"+DWARFSTAR_WAITING")
            if waiting <= seen:
                time.sleep(.1)
                continue
            seen = waiting
            if sent:
                text = context.read_text()
                (OUT / f"context-{sent}.txt").write_text(text)
            if sent == 2:
                blocks = list(re.finditer(r"(?m)^\[\[DS4_TURN (\d+) role=(\w+)\]\]\n", text))
                for i, m in enumerate(blocks):
                    end = blocks[i+1].start() if i+1 < len(blocks) else len(text)
                    if "[OBSOLETE_CASE_A]" in text[m.end():end]:
                        target = int(m[1])
                        break
                if target is None:
                    raise ValueError("no labelled obsolete turn")
                turns += [
                    f"Archive the complete original older audit user turn into {archive}, then delete "
                    "that one turn from your live context. Identify its stable ID using headers, "
                    "a small search, or code, and use the context editing recipe. "
                    "Do not read the complete context into tool output, or paste its old contents "
                    "into tool arguments. Preserve the current key and latest request. "
                    "Make one cleanup, then reply CLEANED only.",
                    f"Recover the archived code from {archive} using a small file read or search. "
                    "Also recall the current key. Reply only a JSON object with archived_code and current_key. "
                    "Do not read or edit the live context file.",
                ]
            if sent == len(turns):
                proc.stdin.close()
                break
            proc.stdin.write((turns[sent] + "\n").encode())
            proc.stdin.flush()
            sent += 1
        rc = proc.wait(timeout=15)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        proc.terminate()
        rc = proc.wait(timeout=15)

text = context.read_text() if context.exists() else ""
trace = (OUT / "agent.trace").read_text() if (OUT / "agent.trace").exists() else ""
first = (OUT / "context-2.txt").read_text() if (OUT / "context-2.txt").exists() else ""
edited = (OUT / "context-3.txt").read_text() if (OUT / "context-3.txt").exists() else ""
before = {int(m[1]) for m in re.finditer(r"(?m)^\[\[DS4_TURN (\d+) role=\w+\]\]$", first)}
after = {int(m[1]) for m in re.finditer(r"(?m)^\[\[DS4_TURN (\d+) role=\w+\]\]$", edited)}
final = text.rsplit("<｜Assistant｜>", 1)[-1]
try:
    answer = json.JSONDecoder().raw_decode(final[final.index("{"):])[0]
except (ValueError, json.JSONDecodeError):
    answer = None
checks = {
    "four_turns_complete": sent == 4 and seen == 5 and rc == 0 and error is None,
    "original_obsolete_turn_removed": target is not None and target not in after,
    "audit_log_not_copied_back": edited.count("Old audit log: no current decisions.") < 3,
    "retained_ids_stable": bool(before) and before - {target} <= after,
    "verbatim_archive_present": archive.exists() and "EMBER-7319" in archive.read_text(),
    "fresh_archive_answer": answer == {"archived_code": "EMBER-7319", "current_key": "COVE-3917"},
    "one_context_edit": trace.count("live context edit committed") == 1,
    "no_automatic_compaction": "COMPACTING" not in (OUT / "stdout.log").read_text(),
    "no_edit_rejection": "live context edit rejected" not in
        (trace + (OUT / "stdout.log").read_text() + (OUT / "stderr.log").read_text()).lower(),
}
(OUT / "fixture.json").write_text(json.dumps({"turns": turns, "target_id": target}, indent=2) + "\n")
result = {"returncode": rc, "error": error, "seconds": time.monotonic()-started,
          "checks": checks, "answer": answer, "first_attempt_model_outputs": True,
          "case": case, "pass": all(checks.values())}
(OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
raise SystemExit(0 if result["pass"] else 3)
