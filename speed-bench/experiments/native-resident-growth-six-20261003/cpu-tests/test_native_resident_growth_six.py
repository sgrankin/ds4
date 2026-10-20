"""Growth model-free guard/ABBA tests: toy bytes and mocked helper processes."""
import contextlib
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'speed-bench/ideas/prepare_native_resident_growth_six.py'
spec = importlib.util.spec_from_file_location('growth_six', SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class TimingTests(unittest.TestCase):
    def test_default_native_unsets_selected_overrides_and_three_policies(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ,
                {'DS4_METAL_V41_SELECTED_TILE':'128', 'DS4_METAL_V41_SELECTED_LIMIT':'768',
                 'DS4_METAL_DISABLE_V41_WIDER_SELECTED_TILES':'1', 'DS4UNKNOWN':'remove'}):
            output = Path(tmp)
            a = driver.run_environment(output, 'A')
            self.assertNotIn('DS4_METAL_V41_SELECTED_TILE', a)
            self.assertNotIn('DS4_METAL_V41_SELECTED_LIMIT', a)
            self.assertNotIn('DS4_METAL_DISABLE_V41_WIDER_SELECTED_TILES', a)
            self.assertNotIn('DS4UNKNOWN', a)
            self.assertEqual(a[driver.ROUTE+'GAP'], '0')
            b = driver.run_environment(output, 'B')
            c = driver.run_environment(output, 'C')
            for env in (b, c):
                for key,value in driver.FAST_ENV.items(): self.assertEqual(env[key],value)
            self.assertEqual(b[driver.ROUTE+'GAP'], '0')
            for key,value in driver.CANDIDATE.items(): self.assertEqual(c[key],value)
            with self.assertRaisesRegex(ValueError, 'variant'):driver.run_environment(output, 'D')

    def test_three_comparisons_keep_primary_and_secondary_distinct(self):
        metrics=('startup_prefill_ms','append_prefill_ms','decode_ms','model_ms','turn_model_ms')
        rows=[dict(variant=variant,**dict.fromkeys(metrics,value)) for variant,value in
            [('A',100),('B',80),('C',70),('C',70),('B',80),('A',100)]]
        result=driver.comparison(rows)
        self.assertEqual(result['primary'],'native_fast_vs_default')
        self.assertEqual(result['secondary'],'resident_three_vs_native_fast')
        for name,want in [('native_fast_vs_default',-20),('resident_three_vs_native_fast',-12.5),
                ('resident_three_vs_native_default',-30)]:
            for got in result['pairs'][name]['change_percent'].values():self.assertAlmostEqual(got,want)
        self.assertTrue(all(r['runs']==2 for r in result['arms'].values()))

    def test_original_nine_phase_input_fits_without_tokenizing(self):
        raw = driver.SESSION.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), driver.SESSION_SHA)
        self.assertEqual(driver.replay_contract(raw), driver.EXPECTED_PHASES)
        self.assertEqual(driver.EXPECTED_PHASES[-1][2], 26659)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            driver.replay_contract(raw, capacity=16384)

    def test_replay_history_and_token_failures(self):
        for raw, message in [(b'P 0 2 1\n', 'count/range'),
                (b'P 0 1 1\nP 2 1 2\n', 'cached prefix'),
                (b'P 0 1 1\nD 1 1 2\n', 'cached zero'),
                (b'P 0 1 129280\nD 0 1 2\n', 'count/range'),
                (b'P 0 1 -1\nD 0 1 2\n', 'integer'),
                (b'P 0 1 1\nD 0 0\n', 'count/range')]:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, message):
                driver.replay_contract(raw)

    def fixture(self, directory):
        output = directory / 'prepared'
        output.mkdir()
        (output / 'metal').mkdir()
        (output / 'metal/fake.metal').write_text('// no GPU compiler invoked')
        (output / 'replay-three-plain').write_text('never executed')
        (output / 'session.txt').write_text('P 0 2 0 1\nD 0 1 2\n')
        shutil.copy2(driver.COMMON, output / 'abba_helpers.py')
        guard = directory / 'guard.py'
        guard.write_text('never executed')
        interpreter = directory / 'frozen-python'
        interpreter.write_text('toy interpreter identity; never executed')
        identity = dict(path=str(directory / 'missing-model'), bytes=100, mtime_ns=200)
        logits, state = b'native-vectors', b'native-state'
        manifest = dict(guard_sha256=driver.digest(guard), guard_command=[str(interpreter), str(guard), str(output / 'pressure')],
            guard_interpreter_identity=dict(path=str(interpreter), sha256=driver.digest(interpreter)),
            model_from_prior=identity, order='ABCCBA', expected_logits_bytes=len(logits),
            phase_identity=[list(row) for row in driver.EXPECTED_PHASES],
            files={str(p.relative_to(output)): driver.digest(p) for p in output.rglob('*') if p.is_file()})
        driver.write(output / 'manifest.json', manifest)
        driver.write(output / 'prepared-command.json', dict(manifest_sha256=driver.digest(output / 'manifest.json')))
        return output, guard, identity, logits, state, manifest

    def trial(self, fault=None):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            output, guard, identity, native, state, manifest = self.fixture(directory)
            actual_common = driver.load_common
            calls = []
            assets = []
            model = mock.Mock(return_value=identity)
            def load(path):
                common = actual_common(path)
                common.model_identity = model
                return common
            def execute(command, cwd, env, stdout, stderr, check):
                index = len(calls)
                variant = 'ABCCBA'[index]
                ds4 = {k: v for k, v in env.items() if k.startswith('DS4') and not k.endswith('_SOURCE')}
                expected = driver.COMMON_ENV.copy()
                expected[driver.ROUTE + 'GAP'] = '0'
                if variant in 'BC': expected.update(driver.FAST_ENV)
                if variant == 'C': expected.update(driver.CANDIDATE)
                self.assertEqual(ds4, expected)
                self.assertFalse(any('PROFILE' in k or 'METRICS' in k for k in ds4))
                calls.append(variant)
                logits = native if variant == 'A' else b'fast!!-vectors' if variant == 'B' else b'edited-vectors'
                snapshot = state if variant == 'A' else b'fast!!-state' if variant == 'B' else b'edited-state'
                if fault == 'native-drift' and index == 5: logits = b'drift!-vectors'
                if fault == 'candidate-repeat' and index == 3: logits = b'drift!-vectors'
                if fault == 'native-fast-repeat' and index == 4: logits = b'drift!-vectors'
                Path(command[3]).write_bytes(logits)
                Path(command[4]).write_bytes(snapshot)
                stdout.write('phase,kind,context,tokens,ms,first_decode_ms\n')
                for phase,kind,context,tokens in driver.EXPECTED_PHASES:
                    stdout.write(f'{phase},{kind},{context},{tokens},{10 if kind=="P" else 2},{0 if kind=="P" else 2}\n')
                slots = 7000 if fault == 'cache-slots' else 6136
                for phase,kind,context,tokens in driver.EXPECTED_PHASES:
                    stderr.write(f'REPLAY_CACHE phase={phase} configured_slots={slots} resident_entries=6\n')
                stderr.write(f'ds4: streaming expert cache budget={slots} entries=6\n')
                if variant == 'C' and fault != 'route-missing':
                    stderr.write('ds4: resident-route layers=40 substitutions=3 triple_layers=1\n')
                if fault == 'profile-output': stderr.write('REPLAY_IO phase=1 pread_calls=2\n')
                if fault == 'binary-tamper': (output / 'replay-three-plain').write_text('tampered')
                if fault == 'shader-tamper': (output / 'metal/fake.metal').write_text('tampered')
                if fault == 'session-tamper': (output / 'session.txt').write_text('tampered')
                if fault == 'plan-tamper': (output / 'manifest.json').write_text('{}')
                if fault == 'model-drift': model.return_value = dict(identity, bytes=101)
                return subprocess.CompletedProcess(command, 17 if fault == 'helper-fail' else 0)
            expected = {'native-drift': 'same-arm repeated', 'candidate-repeat': 'same-arm repeated', 'native-fast-repeat': 'same-arm repeated',
                'cache-slots': 'configured cache slots', 'profile-output': 'profiles emitted',
                'binary-tamper': 'frozen asset changed', 'shader-tamper': 'frozen asset changed',
                'session-tamper': 'frozen asset changed', 'model-drift': 'metadata changed during arm',
                'plan-tamper': 'frozen plan changed', 'route-missing': 'routing was not observed',
                'helper-fail': 'failed rc=17'}
            with mock.patch.object(driver, 'GUARD', guard), mock.patch.object(driver, 'require_main_guard') as guard_check, \
                    mock.patch.object(driver, 'load_common', side_effect=load), \
                    mock.patch.object(driver.subprocess, 'run', side_effect=execute), \
                    mock.patch.dict(os.environ, {'DS4_METAL_V41_RESIDENT_BACKFILL': '1',
                        'DS4_METAL_V41_RESIDENT_ROUTE_PROFILE': '1', 'DS4_METAL_STREAMING_EXPERT_IO_METRICS': '1',
                        'DS4_METAL_V41_SELECTED_TILE': '128', 'DS4_OVERTHINK_PENALTY': '9',
                        'DS4UNLISTED': 'must be removed', 'DS4': 'must be removed'}):
                with contextlib.redirect_stdout(io.StringIO()):
                    if fault:
                        with self.assertRaisesRegex((ValueError, RuntimeError), expected[fault]):
                            driver.run(output, True, True)
                    else:
                        results = driver.run(output, True, True)
                        self.assertEqual([r['variant'] for r in results], list('ABCCBA'))
                        self.assertTrue(json.loads((output / 'run/checks.json').read_text())['passed'])
                self.assertGreaterEqual(guard_check.call_count, 2)
            if fault in ('native-drift', 'candidate-repeat', 'native-fast-repeat', 'cache-slots', 'profile-output'):
                last = output / 'run' / f'{len(calls)-1}-{calls[-1]}' / 'result.json'
                self.assertTrue(last.is_file(), 'failed audit evidence must remain')
            return len(calls)

    def test_abba_and_inherited_flags(self):
        self.assertEqual(self.trial(), 6)

    def test_failures(self):
        for fault in ['native-drift', 'candidate-repeat', 'native-fast-repeat', 'cache-slots', 'profile-output',
                      'binary-tamper', 'shader-tamper', 'session-tamper', 'model-drift', 'helper-fail']:
            with self.subTest(fault=fault): self.trial(fault)

    def test_plan_tamper_and_missing_candidate_footer(self):
        for fault in ['plan-tamper', 'route-missing']:
            with self.subTest(fault=fault): self.trial(fault)

    def test_missing_assertions_precede_any_metadata_or_subprocess(self):
        with mock.patch.object(driver, 'require_main_guard') as guard, mock.patch.object(driver, 'load_common') as common:
            for root_grant, freeze in [(False, False), (True, False), (False, True)]:
                with self.assertRaisesRegex(ValueError, 'root grant'):
                    driver.run(Path('/nonexistent'), root_grant, freeze)
            guard.assert_not_called(); common.assert_not_called()

    def test_parent_and_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, guard, identity, native, state, manifest = self.fixture(root)
            (root / 'speed-bench/ideas/artifacts').mkdir(parents=True)
            (root / 'speed-bench/ideas/artifacts/model-job.lock').touch()
            parent = ' '.join(manifest['guard_command'])
            busy = OSError(errno.EWOULDBLOCK, 'held by guard')
            with mock.patch.object(driver, 'ROOT', root), mock.patch.object(driver, 'GUARD', guard), \
                    mock.patch.object(driver.subprocess, 'check_output', return_value=parent), \
                    mock.patch.object(driver.fcntl, 'flock', side_effect=busy):
                driver.require_main_guard(output, manifest)
            with mock.patch.object(driver, 'ROOT', root), mock.patch.object(driver, 'GUARD', guard), \
                    mock.patch.object(driver.subprocess, 'check_output', return_value=parent), \
                    mock.patch.object(driver.fcntl, 'flock'):
                with self.assertRaisesRegex(ValueError, 'does not hold'):
                    driver.require_main_guard(output, manifest)
            with mock.patch.object(driver, 'GUARD', guard), \
                    mock.patch.object(driver.subprocess, 'check_output', return_value='python3 wrong-guard.py'), \
                    mock.patch.object(driver.fcntl, 'flock') as lock:
                with self.assertRaisesRegex(ValueError, 'direct parent'):
                    driver.require_main_guard(output, manifest)
                lock.assert_not_called()
            guard.write_text('changed')
            with mock.patch.object(driver, 'GUARD', guard), mock.patch.object(driver.subprocess, 'check_output') as parent_probe:
                with self.assertRaisesRegex(ValueError, 'guard changed'):
                    driver.require_main_guard(output, manifest)
                parent_probe.assert_not_called()

    def test_guard_refusal_precedes_model_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, guard, identity, native, state, manifest = self.fixture(Path(tmp))
            with mock.patch.object(driver, 'require_main_guard', side_effect=ValueError('not guarded')), \
                    mock.patch.object(driver, 'load_common') as common:
                with self.assertRaisesRegex(ValueError, 'not guarded'):
                    driver.run(output, True, True)
                common.assert_not_called()

    def test_pinned_interpreter_alias_and_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, guard, identity, native, state, manifest = self.fixture(root)
            (root / 'speed-bench/ideas/artifacts').mkdir(parents=True)
            (root / 'speed-bench/ideas/artifacts/model-job.lock').touch()
            pinned = Path(manifest['guard_interpreter_identity']['path'])
            alias = root / 'python-alias'
            alias.symlink_to(pinned)
            other = root / 'different-python'
            other.write_bytes(pinned.read_bytes())  # Equal bytes are not equal file identity.
            busy = OSError(errno.EWOULDBLOCK, 'held by guard')
            actual = [str(alias), *manifest['guard_command'][1:]]
            with mock.patch.object(driver, 'ROOT', root), mock.patch.object(driver, 'GUARD', guard), \
                    mock.patch.object(driver.subprocess, 'check_output', return_value=' '.join(actual)), \
                    mock.patch.object(driver.fcntl, 'flock', side_effect=busy):
                proof = driver.require_main_guard(output, manifest)
                self.assertTrue(proof['interpreter_samefile'])
                self.assertEqual(proof['parent_argv'], actual)
            for actual, message in [
                    ([str(other), *manifest['guard_command'][1:]], 'pinned framework'),
                    ([str(root / 'missing-python'), *manifest['guard_command'][1:]], 'pinned framework'),
                    ([str(alias), *manifest['guard_command'][1:], '--extra'], 'direct parent'),
                    ([str(alias), str(guard)+'-changed', *manifest['guard_command'][2:]], 'direct parent')]:
                with self.subTest(parent=actual), mock.patch.object(driver, 'ROOT', root), \
                        mock.patch.object(driver, 'GUARD', guard), \
                        mock.patch.object(driver.subprocess, 'check_output', return_value=' '.join(actual)), \
                        mock.patch.object(driver.fcntl, 'flock') as lock:
                    with self.assertRaisesRegex(ValueError, message):
                        driver.require_main_guard(output, manifest)
                    lock.assert_not_called()
            pinned.write_text('changed interpreter')
            with mock.patch.object(driver, 'GUARD', guard), \
                    mock.patch.object(driver.subprocess, 'check_output', return_value=' '.join(manifest['guard_command'])), \
                    mock.patch.object(driver.fcntl, 'flock') as lock:
                with self.assertRaisesRegex(ValueError, 'pinned framework'):
                    driver.require_main_guard(output, manifest)
                lock.assert_not_called()



if __name__ == '__main__':
    unittest.main()
