#!/usr/bin/env python3
"""Serialize model jobs and stop our owned process groups on host pressure."""
import argparse
import fcntl
import functools
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / 'artifacts'
PROBE_TIMEOUT = 2


def probe(argv):
    return subprocess.check_output(argv, text=True, timeout=PROBE_TIMEOUT)


@functools.lru_cache(maxsize=1)
def physical_gb():
    return int(probe(['sysctl', '-n', 'hw.memsize'])) / 2**30


def sample():
    vm = probe(['vm_stat'])
    page = int(re.search(r'page size of (\d+) bytes', vm).group(1))
    counts = {key: int(value) for key, value in
              re.findall(r'^([^:\n]+):\s+(\d+)\.', vm, re.M)}
    swap = probe(['sysctl', '-n', 'vm.swapusage'])
    used = re.search(r'used = ([0-9.]+)([MG])', swap)
    if not used:
        raise RuntimeError('cannot parse swap usage')
    swap_gb = float(used[1]) / (1024 if used[2] == 'M' else 1)
    level = int(probe(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level']))
    return dict(time=time.time(), level=level, swap_gb=swap_gb,
        physical_gb=physical_gb(),
        free_gb=counts['Pages free'] * page / 2**30,
        file_backed_gb=counts.get('File-backed pages', 0) * page / 2**30,
        wired_gb=counts['Pages wired down'] * page / 2**30,
        compressor_gb=counts['Pages occupied by compressor'] * page / 2**30)


def pressure_reason(row, initial_swap):
    if row['level'] != 1:
        return 'kernel reports elevated memory pressure'
    nonreclaimable = row['wired_gb'] + row['compressor_gb']
    headroom = row['physical_gb'] - nonreclaimable
    reserve = max(8, .125 * row['physical_gb'])
    if row['free_gb'] < 8 and nonreclaimable > .875 * row['physical_gb']:
        return 'wired plus compressed RAM leaves little reclaimable headroom'
    if row['swap_gb'] - initial_swap > .5 and headroom < reserve:
        return 'growing swap with little nonreclaimable-memory headroom'
    return None


def process_rows():
    rows = {}
    for line in probe(['ps', '-axo', 'pid=,ppid=,pgid=,stat=,lstart=']).splitlines():
        parts = line.split(None, 4)
        if len(parts) != 5 or parts[3].startswith('Z'):
            continue
        pid, parent, group = map(int, parts[:3])
        if group > 0:
            rows[pid] = (parent, group, parts[4])
    return rows


class OwnedTree:
    """Remember birth identities and groups, including nested setsid children."""
    def __init__(self, proc):
        self.proc = proc
        self.known = {}
        self.groups = {proc.pid}

    def refresh(self):
        rows = process_rows()
        root = rows.get(self.proc.pid)
        if root and self.proc.returncode is None:
            # An unreaped Popen child cannot have its PID reused.
            self.known.setdefault(self.proc.pid, (root[1], root[2]))
        alive = {pid for pid, identity in self.known.items()
                 if pid in rows and (rows[pid][1], rows[pid][2]) == identity}
        pending = list(alive)
        children = {}
        for pid, (parent, _, _) in rows.items():
            children.setdefault(parent, []).append(pid)
        while pending:
            for pid in children.get(pending.pop(), []):
                if pid not in alive:
                    row = rows[pid]
                    self.known[pid] = (row[1], row[2])
                    alive.add(pid)
                    pending.append(pid)
        self.groups = {rows[pid][1] for pid in alive}
        if self.proc.returncode is None:
            self.groups.add(self.proc.pid)
        return alive

    def send_groups(self, sig):
        for group in sorted(self.groups):
            try:
                os.killpg(group, sig)
            except (ProcessLookupError, PermissionError):
                # Darwin can report EPERM when a signalled group has already
                # exited. The final identity check below catches live survivors.
                pass

    def safe_refresh(self):
        try:
            return self.refresh()
        except (OSError, subprocess.SubprocessError):
            # Last confirmed groups remain usable if ps itself is unavailable.
            return None

    def terminate(self, grace=2):
        # Freeze parents before enumeration so they cannot fork missed children.
        self.send_groups(signal.SIGSTOP)
        for _ in range(3):
            old = set(self.groups)
            self.safe_refresh()
            self.send_groups(signal.SIGSTOP)
            if self.groups == old:
                break
        self.send_groups(signal.SIGTERM)
        self.send_groups(signal.SIGCONT)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            alive = self.safe_refresh()
            if alive == set():
                break
            time.sleep(.05)
        # Refresh birth identities before killing; stale/reused PIDs are excluded.
        self.safe_refresh()
        self.send_groups(signal.SIGKILL)
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            raise RuntimeError('owned job did not exit after SIGKILL')
        remaining = self.safe_refresh()
        if remaining:
            raise RuntimeError('owned processes survived termination: ' + str(sorted(remaining)))


def terminate_tree(proc):
    tree = OwnedTree(proc)
    tree.safe_refresh()
    tree.terminate()


class GuardSignal(BaseException):
    def __init__(self, signum):
        self.signum = signum


def interrupted(signum, _frame):
    raise GuardSignal(signum)


def run(output, command, interval=1, samples=None):
    output.mkdir(parents=True, exist_ok=False)
    read = samples or sample
    initial = read()
    reason = pressure_reason(initial, initial['swap_gb'])
    if reason:
        raise RuntimeError('preflight refused: ' + reason)
    (output / 'command.json').write_text(json.dumps(dict(argv=command,
        initial=initial, polling_seconds=interval,
        stop_rule='two consecutive adverse samples; large RAM use alone is allowed'),
        indent=2) + '\n')
    proc = None
    tree = None
    saved = {sig: signal.getsignal(sig) for sig in
             (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    for sig in saved:
        signal.signal(sig, interrupted)
    result = None
    try:
        proc = subprocess.Popen(command, start_new_session=True)
        tree = OwnedTree(proc)
        tree.refresh()
        bad = 0
        with (output / 'pressure.jsonl').open('w') as log:
            while True:
                tree.refresh()
                if proc.poll() is not None:
                    break
                row = read()
                reason = pressure_reason(row, initial['swap_gb'])
                bad = bad + 1 if reason else 0
                row.update(reason=reason, consecutive_bad=bad)
                log.write(json.dumps(row) + '\n')
                log.flush()
                if bad >= 2:
                    tree.terminate()
                    result = dict(status='pressure-stop', reason=reason,
                                  child_returncode=proc.returncode)
                    return 75
                time.sleep(interval)
        # A failed wrapper must not leave a previously discovered agent behind.
        tree.terminate(grace=.2)
        result = dict(status='complete', child_returncode=proc.returncode)
        return proc.returncode
    except GuardSignal as stop:
        for sig in saved:
            signal.signal(sig, signal.SIG_IGN)
        if tree:
            tree.terminate()
        result = dict(status='interrupted', signal=stop.signum,
                      child_returncode=proc.returncode if proc else None)
        return 128 + stop.signum
    except BaseException as error:
        for sig in saved:
            signal.signal(sig, signal.SIG_IGN)
        if tree:
            tree.terminate()
        result = dict(status='monitor-error', error=repr(error),
                      child_returncode=proc.returncode if proc else None)
        raise
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
        if result is not None:
            (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('a command is required after --')
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS / 'model-job.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('another guarded model job is already running')
        raise SystemExit(run(a.output.resolve(), command))


if __name__ == '__main__':
    main()
