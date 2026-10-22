#!/usr/bin/env python3
"""Two ordinary turns around one real interactive /compact; run under ROOT's guard.

The pipe/noninteractive driver submits slash text as a user turn. A stdlib PTY
is required here to reach the existing UI command, with no production changes.
Reuse run_agent.py's JSON/answer helpers and settings; do not count this as its
six-turn archive/edit/suffix workflow.
"""
import argparse
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pty
import re
import selectors
import signal
import struct
import termios
import time

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / 'speed-bench/experiments/zig-combined-20261022/run_agent.py'
spec = importlib.util.spec_from_file_location('combined_agent', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
GOLD = {'code': 'SABLE-7319', 'count': 37}
SUMMARY = '[ds4-agent compacted earlier conversation. Durable task-state summary follows.]'
SUMMARY_END = '[End compacted summary. Recent conversation continues verbatim below.]'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--agent', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shared')
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shared = Path(args.shared).resolve() if args.shared else out / 'shared'
    work = shared / 'work'
    work.mkdir(parents=True, exist_ok=True)
    context, trace_path = shared / 'context.txt', out / 'agent.trace'
    context.unlink(missing_ok=True)
    first = ('Remember these durable facts for our next question: code SABLE-7319; count 37. '
             'All following audit prose is disposable. Do not use tools. '
             + ' '.join(f'Old audit line {i}: routine check complete and no new task facts.' for i in range(200))
             + ' Keep the durable code and count; reply READY only.')
    query = 'Without tools, return only JSON with the remembered code and numeric count, using keys code and count.'
    helper.dump(out / 'fixture.json', {'first': first, 'slash_command': '/compact', 'query': query,
                                      'gold': GOLD, 'ordinary_user_turns': 2,
                                      'external_live_context_edits': 0})
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
    shaders = re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                         (ROOT / 'ds4_metal.m').read_text())
    env.update({k: str(ROOT / v) for k, v in shaders})
    env.update(TERM='xterm-256color', DS4_AGENT_CACHE_DIR=str(out / 'cache'),
               DS4_AGENT_COMPACT_PREFIX_CACHE='1', DS4_AGENT_CONTEXT_SUFFIX_REUSE='0',
               DS4_AGENT_CONTEXT_SUFFIX_PREROPE='0', DS4_AGENT_TEST_TIME='1790954400',
               DS4_V41_DECODER_REPLAY_ROWS='0', DS4_METAL_V41_RESIDENT_ROUTE_GAP='0',
               DS4_STATE_CACHE_DIR=str(out / 'state-cache'), DS4_STATE_CACHE_HOT_MB='0',
               DS4_STATE_CACHE_SSD_MB='512')
    agent = Path(args.agent).resolve()
    agent_command = [str(agent), '--model', str(Path(args.model).resolve()),
               '--ssd-streaming', '--ssd-streaming-cache-experts', '32GB', '--ctx', '8192',
               '--prefill-chunk', '512', '--nothink', '--temp', '0', '--tokens', '256',
               '--seed', '20261003', '--chdir', str(work), '--live-context-file', str(context),
               '--trace', str(trace_path)]
    # Interactive linenoise has no history-path override. Keep the real HOME
    # and deny only this global history write, as in the CLI qualification.
    history = Path.home() / '.ds4_agent_history'
    profile = f'(version 1) (allow default) (deny file-write* (literal {json.dumps(str(history))}))'
    command = ['/usr/bin/sandbox-exec', '-p', profile, *agent_command]
    with agent.open('rb') as binary:
        binary_sha256 = hashlib.file_digest(binary, 'sha256').hexdigest()
    helper.dump(out / 'execution.json', {'command': command, 'cwd': str(ROOT),
                                       'agent_binary': str(agent), 'binary_sha256': binary_sha256,
                                       'history_write_denied': str(history),
                                       'env': {k: v for k, v in env.items() if k.startswith('DS4')},
                                       'interactive_pty': True, 'eof_saves_named_session': False})
    started = time.monotonic()
    pid, master = pty.fork()
    if pid == 0:
        os.chdir(ROOT)
        os.execve(command[0], command, env)
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack('HHHH', 40, 120, 0, 0))
    os.set_blocking(master, False)
    selector = selectors.DefaultSelector()
    selector.register(master, selectors.EVENT_READ)
    wire, pending = bytearray(), bytearray()
    phase, rc, error, answer = 0, None, None, None
    observed, quit_at, save_prompt_start = {}, None, 0
    try:
        while rc is None:
            if time.monotonic() - started > args.timeout:
                raise TimeoutError(f'{args.timeout:g}-second arm deadline exceeded')
            if pending:
                try:
                    written = os.write(master, pending)
                    del pending[:written]
                except BlockingIOError:
                    pass
            for _, _ in selector.select(0.2):
                try:
                    data = os.read(master, 65536)
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
                    data = b''
                if not data:
                    selector.unregister(master)
                    break
                wire.extend(data)
                # Answer linenoise's optional cursor-position query as a terminal.
                if b'\x1b[6n' in data:
                    pending.extend(b'\x1b[40;1R')
            trace = trace_path.read_text() if trace_path.exists() else ''
            mirror = context.read_text() if context.exists() else ''
            generated = trace.count('generation finished tool_round=')
            if phase == 0 and 'initial_system_prompt' in trace and b'ds4-agent> ' in wire:
                pending.extend(b'\x1b[200~' + first.encode() + b'\x1b[201~\r')
                phase = 1
            elif (phase == 1 and generated >= 1 and first in mirror
                  and mirror.rsplit('<｜Assistant｜>', 1)[-1].endswith('<｜end▁of▁sentence｜>')):
                tail = mirror.rsplit('<｜Assistant｜>', 1)[-1]
                if not re.search(r'\bREADY\b', tail):
                    raise ValueError('first completed response did not contain READY')
                (out / 'before-compact.txt').write_text(mirror)
                pending.extend(b'/compact\r')
                observed['compact_requested_seconds'] = time.monotonic() - started
                phase = 2
            elif phase == 2 and 'compacted reason="user requested compaction"' in trace and SUMMARY in mirror:
                (out / 'after-compact.txt').write_text(mirror)
                summary, retained = mirror.split(SUMMARY, 1)[1].split(SUMMARY_END, 1)
                observed.update(summary_contains_code=GOLD['code'] in summary,
                                retained_tail_contains_code=GOLD['code'] in retained,
                                private_request_absent='Internal ds4-agent context compaction request.' not in mirror,
                                compact_completed_seconds=time.monotonic() - started)
                pending.extend(query.encode() + b'\r')
                phase = 3
            elif (phase == 3 and generated >= 2 and query in mirror
                  and mirror.rsplit('<｜Assistant｜>', 1)[-1].endswith('<｜end▁of▁sentence｜>')):
                answer, final = helper.read_answer(mirror[mirror.rfind(query):])
                (out / 'final-context.txt').write_text(mirror)
                observed['final_assistant'] = final
                quit_at = time.monotonic() + 0.3  # allow UI's final idle handoff
                phase = 4
            elif phase == 4 and time.monotonic() >= quit_at:
                save_prompt_start = len(wire)
                pending.extend(b'/quit\r')
                phase = 5
            elif phase == 5 and b'Save current session? (y/n)' in wire[save_prompt_start:]:
                pending.extend(b'n\n')
                phase = 6
            waited, status = os.waitpid(pid, os.WNOHANG)
            if waited:
                rc = os.waitstatus_to_exitcode(status)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if rc is None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(pid, sig)
                except ProcessLookupError:
                    break
                time.sleep(0.2)
            _, status = os.waitpid(pid, 0)
            rc = os.waitstatus_to_exitcode(status)
        selector.close()
        os.close(master)
        (out / 'pty.log').write_bytes(wire)
    trace = trace_path.read_text() if trace_path.exists() else ''
    actual = [line for line in trace.splitlines() if 'compacted reason=' in line]
    summary_lines = [line for line in trace.splitlines() if 'compaction-summary=' in line]
    proof = (len(actual) == 1 and 'reason="user requested compaction"' in actual[0]
             and len(summary_lines) == 1 and 'tokens label=compacted_transcript ' in trace
             and observed.get('summary_contains_code') and not observed.get('retained_tail_contains_code')
             and observed.get('private_request_absent'))
    result = {'returncode': rc, 'error': error, 'seconds': time.monotonic() - started,
              'ordinary_user_turns': sum(' user="' in line for line in trace.splitlines()),
              'slash_compact_requests': int(phase >= 2),
              'external_live_context_edits': 0, 'answer': answer, 'answer_correct': answer == GOLD,
              'actual_compactions': actual, 'private_summary_trace': summary_lines,
              'compaction_proved': bool(proof), **observed}
    result['pass'] = bool(rc == 0 and not error and phase >= 5 and answer == GOLD and proof)
    helper.dump(out / 'result.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['pass'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
