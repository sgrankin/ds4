#!/usr/bin/env python3
"""Serialize model jobs and stop our process tree on sustained host pressure."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE/'artifacts'


def sample():
    vm = subprocess.check_output(['vm_stat'], text=True)
    page = int(re.search(r'page size of (\d+) bytes', vm).group(1))
    counts = {key: int(value) for key, value in
              re.findall(r'^([^:\n]+):\s+(\d+)\.', vm, re.M)}
    swap = subprocess.check_output(['sysctl', '-n', 'vm.swapusage'], text=True)
    used = re.search(r'used = ([0-9.]+)([MG])', swap)
    if not used:
        raise RuntimeError('cannot parse swap usage')
    swap_gb = float(used[1]) / (1024 if used[2] == 'M' else 1)
    level = int(subprocess.check_output(
        ['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True))
    return dict(time=time.time(), level=level, swap_gb=swap_gb,
        free_gb=counts['Pages free']*page/2**30,
        wired_gb=counts['Pages wired down']*page/2**30,
        compressor_gb=counts['Pages occupied by compressor']*page/2**30)


def pressure_reason(row, initial_swap):
    if row['level'] != 1:
        return 'kernel reports elevated memory pressure'
    if row['free_gb'] < 4 and (row['compressor_gb'] > 8 or
                             row['swap_gb']-initial_swap > .5):
        return 'low free RAM with compression or growing swap'
    if row['free_gb'] < 8 and row['compressor_gb'] > 24:
        return 'compressed RAM approaching panic snapshot range'
    return None


def descendants(pid):
    rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid='], text=True)
    children = {}
    for line in rows.splitlines():
        child, parent = map(int, line.split())
        children.setdefault(parent, []).append(child)
    found, pending = [], [pid]
    while pending:
        parent = pending.pop()
        found.append(parent)
        pending.extend(children.get(parent, []))
    return found


def terminate_tree(proc):
    # Include children that create their own sessions (the live-agent runner).
    pids = descendants(proc.pid)
    for pid in reversed(pids):
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
    try: proc.wait(timeout=3)
    except subprocess.TimeoutExpired: pass
    for pid in reversed(pids):
        try: os.kill(pid, signal.SIGKILL)
        except ProcessLookupError: pass
    proc.wait()


def run(output, command, interval=1, samples=None):
    output.mkdir(parents=True, exist_ok=False)
    read = samples or sample
    initial = read()
    reason = pressure_reason(initial, initial['swap_gb'])
    if reason:
        raise RuntimeError('preflight refused: '+reason)
    (output/'command.json').write_text(json.dumps(dict(argv=command,
        initial=initial, polling_seconds=interval,
        stop_rule='two consecutive adverse samples; large RAM use alone is allowed'),
        indent=2)+'\n')
    proc = subprocess.Popen(command, start_new_session=True)
    bad = 0
    try:
        with (output/'pressure.jsonl').open('w') as log:
            while proc.poll() is None:
                row = read()
                reason = pressure_reason(row, initial['swap_gb'])
                bad = bad+1 if reason else 0
                row.update(reason=reason, consecutive_bad=bad)
                log.write(json.dumps(row)+'\n'); log.flush()
                if bad >= 2:
                    terminate_tree(proc)
                    result = dict(status='pressure-stop', reason=reason,
                                  child_returncode=proc.returncode)
                    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
                    print(json.dumps(result), flush=True)
                    return 75
                time.sleep(interval)
    except BaseException:
        terminate_tree(proc)
        raise
    (output/'result.json').write_text(json.dumps(
        dict(status='complete', child_returncode=proc.returncode), indent=2)+'\n')
    return proc.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('a command is required after --')
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS/'model-job.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('another guarded model job is already running')
        raise SystemExit(run(a.output.resolve(), command))


if __name__ == '__main__':
    main()
