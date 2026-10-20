#!/usr/bin/env python3
"""CPU-only process cleanup checks; these never open a model."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import pressure_run as guard

ARTIFACTS = Path(__file__).resolve().parent / 'artifacts'
HEALTHY = dict(level=1, swap_gb=0, free_gb=16, compressor_gb=0,
               wired_gb=20, physical_gb=128, file_backed_gb=90)


def wrapper(path):
    return ('import subprocess,sys,time,json; from pathlib import Path; '
            'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],start_new_session=True); '
            'Path(' + repr(str(path)) + ').write_text(json.dumps({"child":p.pid})); time.sleep(30)')


def wait_pid(path):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())['child']
        time.sleep(.025)
    raise TimeoutError('mock child did not start')


def running(pid):
    state = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='],
                           capture_output=True, text=True, timeout=2).stdout.strip()
    return bool(state) and not state.startswith('Z')


def cleanup(pid):
    if running(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class GuardTests(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='pressure-cpu-', dir=ARTIFACTS)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_large_reclaimable_file_cache_is_allowed(self):
        row = dict(HEALTHY, free_gb=.098, wired_gb=71.64,
                   compressor_gb=6.02, file_backed_gb=49)
        self.assertIsNone(guard.pressure_reason(row, 0))
        self.assertIsNone(guard.pressure_reason(dict(row, compressor_gb=10), 0))
        panic = dict(row, wired_gb=93.4, compressor_gb=33.3)
        self.assertIsNotNone(guard.pressure_reason(panic, 0))
        self.assertIsNotNone(guard.pressure_reason(dict(row, level=2), 0))
        self.assertIsNotNone(guard.pressure_reason(
            dict(row, free_gb=9, wired_gb=105, compressor_gb=10, swap_gb=1), 0))

    def test_nested_session_termination(self):
        path = self.root / 'pid.json'
        proc = subprocess.Popen([sys.executable, '-c', wrapper(path)], start_new_session=True)
        child = None
        try:
            child = wait_pid(path)
            guard.terminate_tree(proc)
            self.assertFalse(running(child))
        finally:
            cleanup(proc.pid)
            if child:
                cleanup(child)
            proc.wait(timeout=3)

    def test_signals_clean_nested_agent(self):
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            with self.subTest(signal=sig):
                path = self.root / ('pid-' + str(sig) + '.json')
                out = self.root / ('run-' + str(sig))
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                         '--worker', str(out), str(path)], start_new_session=True,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                child = None
                try:
                    child = wait_pid(path)
                    os.kill(proc.pid, sig)
                    self.assertEqual(proc.wait(timeout=6), 128 + sig)
                    self.assertFalse(running(child))
                    result = json.loads((out / 'result.json').read_text())
                    self.assertEqual(result['status'], 'interrupted')
                finally:
                    cleanup(proc.pid)
                    if child:
                        cleanup(child)
                    proc.wait(timeout=3)

    def test_probe_timeout_is_bounded(self):
        start = time.monotonic()
        with patch.object(guard, 'PROBE_TIMEOUT', .1):
            with self.assertRaises(subprocess.TimeoutExpired):
                guard.probe([sys.executable, '-c', 'import time; time.sleep(10)'])
        self.assertLess(time.monotonic() - start, 1)

    def test_monitor_probe_hang_cleans_nested_child(self):
        path = self.root / 'pid.json'
        calls = 0
        def read():
            nonlocal calls
            calls += 1
            if calls >= 8:
                guard.probe([sys.executable, '-c', 'import time; time.sleep(10)'])
            return dict(HEALTHY)
        child = None
        try:
            with patch.object(guard, 'PROBE_TIMEOUT', .1):
                with self.assertRaises(subprocess.TimeoutExpired):
                    guard.run(self.root / 'run', [sys.executable, '-c', wrapper(path)],
                              interval=.025, samples=read)
            child = wait_pid(path)
            self.assertFalse(running(child))
        finally:
            if path.exists():
                cleanup(json.loads(path.read_text())['child'])

    def test_two_adverse_samples_stop_job(self):
        calls = 0
        path = self.root / 'pid.json'
        def read():
            nonlocal calls
            calls += 1
            return dict(HEALTHY, level=2 if calls >= 7 else 1)
        try:
            rc = guard.run(self.root / 'run', [sys.executable, '-c', wrapper(path)],
                           interval=.025, samples=read)
            self.assertEqual(rc, 75)
            self.assertFalse(running(wait_pid(path)))
        finally:
            if path.exists():
                cleanup(json.loads(path.read_text())['child'])

    def test_ps_failure_uses_known_groups(self):
        path = self.root / 'pid.json'
        proc = subprocess.Popen([sys.executable, '-c', wrapper(path)], start_new_session=True)
        child = None
        try:
            child = wait_pid(path)
            tree = guard.OwnedTree(proc)
            tree.refresh()
            with patch.object(guard, 'process_rows', side_effect=subprocess.TimeoutExpired('ps', 2)):
                tree.terminate(grace=.1)
            self.assertFalse(running(child))
        finally:
            cleanup(proc.pid)
            if child:
                cleanup(child)
            proc.wait(timeout=3)

    def test_reused_pid_identity_is_excluded(self):
        class Completed:
            pid = 100
            returncode = 0
        tree = guard.OwnedTree(Completed())
        tree.known = {200: (200, 'old birth')}
        with patch.object(guard, 'process_rows', return_value={200: (1, 200, 'new birth')}):
            self.assertEqual(tree.refresh(), set())
            self.assertEqual(tree.groups, set())


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        raise SystemExit(guard.run(Path(sys.argv[2]),
            [sys.executable, '-c', wrapper(Path(sys.argv[3]))],
            interval=.025, samples=lambda: dict(HEALTHY)))
    unittest.main()
