"""Synthetic released producer fixtures; no actual math16 outcomes or model access."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('math16_archive', Path(__file__).with_name('archive.py'))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
BASE = Path(__file__).parents[1] / 'ideas' / 'artifacts' / 'math16-archive-synthetic-cpu-v1'


def write(path, value, raw=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if raw else (json.dumps(value, allow_nan=False) + '\n').encode())


def sha(path):
    return A.file_info(path)['sha256']


class ArchiveTests(unittest.TestCase):
    counter = 0

    def setUp(self):
        type(self).counter += 1
        self.root = BASE / ('trial-%03d' % self.counter)
        self.root.mkdir(parents=True, exist_ok=False)
        self.prepared = self.root / 'prepared'
        self.pressure = self.prepared / 'pressure'
        self.output = self.prepared / 'run'
        self.fixture = self.prepared / 'fixture.json'
        self.plan_path = self.prepared / 'plan.json'
        cases = [dict(case='case-%02d' % i, prompt='independent question %d' % i, answer=str(i)) for i in range(16)]
        write(self.fixture, dict(cases=cases))
        self.plan = dict(prepared_only=True, model_grant=False, all_first_attempts=48,
            pressure=str(self.pressure), output=str(self.output), fixture=str(self.fixture),
            guard_command_template=['shim', 'guard', str(self.pressure), '--', 'shim', 'frozen-driver', '<PLAN_SHA256>'],
            assets={str(self.fixture): A.file_info(self.fixture)}, arms=[])
        reports = []
        for name, budget in zip(A.NAMES, (0, 1024, 0)):
            directory = self.root / name
            self.plan['arms'].append(dict(name=name, budget=budget, results_dir=str(directory)))
            rows, per_case = [], []
            for i, case in enumerate(cases):
                invalid = i == 2
                row = dict(**case, expected=case['answer'], status='INVALID' if invalid else 'CORRECT',
                    stopped=not invalid, cap_hit=invalid, generated_tokens=2048 if invalid else 2,
                    reasoning_tokens=2048 if invalid else 1, parsed_answer='' if invalid else case['answer'],
                    token_trace='case-%02d.tokens.jsonl' % i, prompt_trace='case-%02d.prompt.json' % i,
                    raw_trace='case-%02d.raw' % i)
                row.pop('answer')
                write(directory / row['token_trace'], b'{"synthetic_retention_only":true}\n', raw=True)
                write(directory / row['prompt_trace'], {'token_ids': [10, i]})
                write(directory / row['raw_trace'], b'original full first attempt text', raw=True)
                rows.append(row)
                saved = {k: row[k] for k in ('case', 'status', 'stopped', 'cap_hit', 'generated_tokens', 'reasoning_tokens', 'parsed_answer')}
                saved.update(force_step=None, raw_sha256=sha(directory / row['raw_trace']),
                    token_sha256=sha(directory / row['token_trace']), prompt_sha256=sha(directory / row['prompt_trace']))
                per_case.append(saved)
            write(directory / 'results.jsonl', b''.join((json.dumps(r) + '\n').encode() for r in rows), raw=True)
            write(self.output / (name + '-process.json'), dict(returncode=3))
            reports.append(dict(budget=budget, producer_returncode=3, cases=16, completed_correct=15,
                incorrect=0, invalid=1, capped_correct=0, cap_hits=1, forced_cases=0,
                generated_tokens=2078, reasoning_tokens=2063, per_case=per_case))
        write(self.plan_path, self.plan)
        self.plan_sha = sha(self.plan_path)
        self.guard = dict(status='complete', child_returncode=3)
        write(self.pressure / 'result.json', self.guard)
        write(self.pressure / 'command.json', dict(argv=['shim', 'frozen-driver', self.plan_sha],
            initial=dict(time=1.0, level=1, swap_gb=0)))
        write(self.pressure / 'pressure.jsonl', b'{"time":2.0,"level":1,"swap_gb":0,"reason":null,"consecutive_bad":0}\n', raw=True)
        write(self.output / 'complete.json', dict(status='COMPLETE', arms=A.NAMES, returncode=3))
        self.release_path = self.root / 'release.json'
        self.release = dict(guard_result=self.guard, lock_reacquired=True, live_model_processes=[],
            time=3.0, normal_pressure_samples=1, bad_pressure_samples=0, swap_growth_gb=0)
        write(self.release_path, self.release)
        self.summary_path = self.root / 'summary.json'
        self.summary = dict(status='COMPLETE_FIRST_ATTEMPTS', plan_sha256=self.plan_sha,
            release_sha256=sha(self.release_path), guard_child_returncode=3, arms=reports,
            native_before_and_after_exact_full_typed_fields=True, fixed_unused16_cases=True,
            all_first_attempts=48, policy_chosen_before_budget_outcomes=True,
            paired=dict(native_correct_budget_failed=0, native_failed_budget_correct=0))
        write(self.summary_path, self.summary)
        self.audit_source = self.root / 'audit-source.py'
        write(self.audit_source, b'# independent frozen synthetic audit source\n', raw=True)
        self.audit_path = self.root / 'audit.json'
        self.audit = dict(status=A.AUDIT_STATUS, plan_sha256=self.plan_sha, release_sha256=sha(self.release_path),
            producer_summary_sha256=sha(self.summary_path), source_sha256=sha(self.audit_source), all48_attempts=True,
            counts=[{k: r[k] for k in A.COUNTS} for r in reports], native_correct_budget_failed=0,
            native_failed_budget_correct=0, model_or_helper_or_guard_executed=False)
        write(self.audit_path, self.audit)
        runtime = self.root / 'main'
        values = {}
        for i in range(100):
            path = runtime / ('runtime-%03d' % i)
            write(path, ('unchanged-%d' % i).encode(), raw=True)
            values[path.name] = sha(path)
        authority = self.root / 'runtime100.json'
        write(authority, dict(runtime_and_shaders_unchanged=values))
        self.registry = dict(plan=str(self.plan_path), plan_sha256=self.plan_sha, asset_count=1,
            static_files={str(self.fixture): A.file_info(self.fixture)}, audit_source=str(self.audit_source),
            audit_source_sha256=sha(self.audit_source), root=str(runtime), runtime_authority=str(authority),
            runtime_authority_sha256=sha(authority))
        self.destination = self.root / 'archive'

    def call(self, grant=True):
        return A.archive(self.registry, self.destination, self.release_path, sha(self.release_path),
                         self.summary_path, sha(self.summary_path), self.audit_path, sha(self.audit_path), grant)

    def update_chain(self):
        self.summary['release_sha256'] = sha(self.release_path)
        write(self.summary_path, self.summary)
        self.audit.update(release_sha256=sha(self.release_path), producer_summary_sha256=sha(self.summary_path))
        write(self.audit_path, self.audit)

    def rows(self, arm=0):
        path = Path(self.plan['arms'][arm]['results_dir']) / 'results.jsonl'
        return path, A.jsonl(path)

    def rewrite_rows(self, path, rows):
        write(path, b''.join((json.dumps(r) + '\n').encode() for r in rows), raw=True)

    def test_complete_preserves_rc3_caps_and_all48(self):
        result = self.call()
        obs = A.load(self.destination / 'archive-observations.json')
        self.assertEqual(obs['original_guard_child_returncode'], 3)
        self.assertEqual(obs['first_attempts'], 48)
        self.assertFalse(obs['rescored'])
        for name in A.NAMES:
            self.assertEqual(len(A.jsonl(self.destination / 'original' / name / 'results.jsonl')), 16)
        self.assertEqual(result['files'], A.physical_index(self.destination)['files'])

    def test_no_grant_or_bad_release_never_opens_attempt_inventory(self):
        with mock.patch.object(A, 'validate_attempt_inventory', side_effect=AssertionError('outcome access')):
            with self.assertRaisesRegex(ValueError, 'explicit ROOT'):
                self.call(False)
            self.release['lock_reacquired'] = False
            write(self.release_path, self.release)
            with self.assertRaisesRegex(ValueError, 'released ROOT'):
                self.call()
        self.assertFalse(self.destination.exists())

    def test_release_and_pressure_mutations(self):
        for key, value in [('normal_pressure_samples', True), ('time', False), ('bad_pressure_samples', 1),
                           ('swap_growth_gb', 1), ('live_model_processes', ['live'])]:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.release); changed[key] = value
                write(self.release_path, changed)
                with self.assertRaises(ValueError): self.call()
        write(self.release_path, self.release)
        write(self.pressure / 'pressure.jsonl', b'{"time":4,"level":1,"swap_gb":0,"reason":null,"consecutive_bad":0}\n', raw=True)
        with self.assertRaisesRegex(ValueError, 'ordered'): self.call()

    def test_exact_child_command(self):
        command = A.load(self.pressure / 'command.json'); command['argv'].append('--changed')
        write(self.pressure / 'command.json', command)
        with self.assertRaisesRegex(ValueError, 'exact frozen'): self.call()

    def test_audit_summary_mutations_block_before_attempts(self):
        with mock.patch.object(A, 'validate_attempt_inventory', side_effect=AssertionError('outcome access')):
            for key, value in [('all48_attempts', 1), ('model_or_helper_or_guard_executed', True),
                               ('status', 'PASS'), ('source_sha256', '0' * 64), ('producer_summary_sha256', '0' * 64)]:
                changed = copy.deepcopy(self.audit); changed[key] = value; write(self.audit_path, changed)
                with self.assertRaises(ValueError): self.call()
            write(self.audit_path, self.audit)
            self.summary['arms'][1]['invalid'] = True; write(self.summary_path, self.summary)
            self.audit['producer_summary_sha256'] = sha(self.summary_path); write(self.audit_path, self.audit)
            with self.assertRaisesRegex(ValueError, 'literal nonnegative'): self.call()

    def test_both_discordances_and_rc3_cannot_be_rewritten(self):
        self.audit['native_correct_budget_failed'] = 1; write(self.audit_path, self.audit)
        with self.assertRaisesRegex(ValueError, 'discordances'): self.call()
        self.audit['native_correct_budget_failed'] = 0
        self.summary['guard_child_returncode'] = 0; self.update_chain()
        with self.assertRaisesRegex(ValueError, 'producer summary'): self.call()

    def test_duplicate_missing_or_reordered_case(self):
        path, original = self.rows()
        for rows in (original[:-1], original[::-1], original[:15] + [original[0]]):
            self.rewrite_rows(path, rows)
            with self.assertRaisesRegex(ValueError, 'ordered first'): self.call()

    def test_trace_drift_missing_escape_or_role_reuse(self):
        path, original = self.rows()
        for name in ('../outside', original[1]['token_trace'], original[0]['prompt_trace'], 'missing'):
            rows = copy.deepcopy(original); rows[0]['token_trace'] = name; self.rewrite_rows(path, rows)
            with self.assertRaises((ValueError, FileNotFoundError)): self.call()
        self.rewrite_rows(path, original)
        write(Path(self.plan['arms'][0]['results_dir']) / original[0]['raw_trace'], b'changed', raw=True)
        with self.assertRaisesRegex(ValueError, 'trace identity'): self.call()

    def test_duplicate_json_and_truncated_result(self):
        path, rows = self.rows()
        write(path, b'{"case":"a","case":"a"}\n', raw=True)
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.call()
        write(path, json.dumps(rows[0]).encode(), raw=True)
        with self.assertRaisesRegex(ValueError, 'newline'): self.call()

    def test_all_files_nested_manifests_modes_and_large_boundary(self):
        directory = Path(self.plan['arms'][0]['results_dir'])
        write(directory / 'nested' / 'manifest.json', dict(first_refusal='retained'))
        write(directory / 'executable', b'#!/bin/sh\nexit 3\n', raw=True); os.chmod(directory / 'executable', 0o755)
        write(directory / 'at-limit.bin', b'x' * A.LIMIT, raw=True)
        write(directory / 'over-limit.bin', b'y' * (A.LIMIT + 1), raw=True)
        result = self.call()
        self.assertIn('original/native-before/nested/manifest.json', result['files'])
        self.assertEqual(result['files']['original/native-before/executable']['mode'], 0o755)
        self.assertIn('original/native-before/at-limit.bin', result['files'])
        self.assertNotIn('original/native-before/over-limit.bin', result['files'])
        self.assertEqual([x['bytes'] for x in result['external_references']], [A.LIMIT + 1])

    def test_alias_model_special_file_and_destination_refusal(self):
        directory = Path(self.plan['arms'][0]['results_dir'])
        (directory / 'alias').symlink_to('results.jsonl')
        with self.assertRaisesRegex(ValueError, 'alias'): self.call()
        (directory / 'alias').unlink()
        write(directory / 'forbidden.gguf', b'not a model, must still refuse', raw=True)
        with self.assertRaisesRegex(ValueError, 'model access'): self.call()
        self.assertTrue((self.destination / 'ARCHIVE-REFUSAL.json').exists())
        with self.assertRaisesRegex(ValueError, 'exclusive fresh'): self.call()

    def test_main100_drift_before_and_after_copy(self):
        runtime = Path(self.registry['root']) / 'runtime-000'
        write(runtime, b'drift', raw=True)
        with self.assertRaisesRegex(ValueError, 'MAIN runtime drift'): self.call()
        self.assertFalse(self.destination.exists())
        write(runtime, b'unchanged-0', raw=True)
        original = A.check_runtime
        calls = []
        def late_drift(registry):
            calls.append(1)
            if len(calls) == 2: write(runtime, b'late drift', raw=True)
            return original(registry)
        with mock.patch.object(A, 'check_runtime', side_effect=late_drift):
            with self.assertRaisesRegex(ValueError, 'MAIN runtime drift'): self.call()
        self.assertTrue((self.destination / 'ARCHIVE-REFUSAL.json').exists())

    def test_copy_failure_retained_no_overwrite(self):
        with mock.patch.object(A.shutil, 'copy2', side_effect=OSError('injected copy failure')):
            with self.assertRaisesRegex(OSError, 'injected'): self.call()
        self.assertEqual(A.load(self.destination / 'ARCHIVE-REFUSAL.json')['status'], 'REFUSED')
        with self.assertRaisesRegex(ValueError, 'exclusive fresh'): self.call()

    def test_static_mode_pin_and_source_tree_overlap(self):
        os.chmod(self.fixture, 0o755)
        with self.assertRaisesRegex(ValueError, 'static pins and modes'): self.call()
        os.chmod(self.fixture, 0o644)
        self.destination = self.prepared / 'archive'
        with self.assertRaisesRegex(ValueError, 'tree overlap'): self.call()


if __name__ == '__main__':
    unittest.main(verbosity=2)
