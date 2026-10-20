"""Production runner contracts; small fake assets, no model or subprocess job."""
import ast
import copy
import csv
import errno
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT/'speed-bench/ideas/artifacts/suffix-pair-tape-pilot-cpu-v1'
SPEC = importlib.util.spec_from_file_location('suffix_pilot', ROOT/'speed-bench/ideas/run_suffix_pair_tape_pilot.py')
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)
TRIALS = 0
REQUIRED = ['baseline-probe', 'new-probe', 'build-manifest.json', 'auxiliary-object-proof.json',
            'baseline-core-proof.json', 'final-link-cpu-proof.json', 'run_suffix_pair_tape_pilot.py', 'corpus.txt',
            'source/ds4.c', 'source/ds4.h', 'source/ds4_suffix_pair_tape.h',
            'source/speed-bench/suffix_pair_tape_bench.c']
FIELDS = ['step', 'target', 'delta_nll', 'kl', 'tv', 'argmax_flip', 'bit_equal']
TARGETS = tuple(range(100, 116))
TOKEN_HEX = struct.pack('<16I', *TARGETS).hex()
REAL_SHA = d.sha


def summary(mode='reuse'):
    reused = mode == 'reuse'
    return dict(mode=mode, prefix=1409, shift=65, span=512, tail=128, decode=16,
                raw_reference_rows=18, raw_candidate_rows=17 if reused else 0,
                tape_buffer_and_metadata_bytes=12582992 if mode != 'native' else 0,
                immutable_tape_bytes=12582912 if mode != 'native' else 0,
                peak_staging_bytes_including_reserve=55000000, staging_budget_bytes=128<<20,
                coverage_rejections_preserve_state=reused, precancel_preserves_state=reused,
                partial_cancel_invalidates=reused, approximate_export_blocked=reused,
                pair_inputs_reused=1536 if reused else 0, pair_rows_rebuilt=768 if reused else 0,
                pair_boundary_rows=3 if reused else 0, reused_tokens=512 if reused else 0,
                fresh_tail_tokens=128 if reused else 0, fresh_ms_diagnostic=1.5,
                reuse_ms_diagnostic=1.0 if reused else 0, mean_kl=0.0, mean_tv=0.0,
                sum_delta_nll=0.0, argmax_flips=0, bit_equal_rows=0)


class CPU(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='case-', dir=ARTIFACTS)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        # Any accidental call into the model identity function is a test error.
        self.model = patch.object(d, 'model_identity', side_effect=AssertionError('Model access forbidden')).start()
        self.addCleanup(patch.stopall)

    def reject(self, function, *args, **kwargs):
        global TRIALS
        TRIALS += 1
        with self.assertRaises((ValueError, RuntimeError, KeyError, struct.error)):
            function(*args, **kwargs)

    def guarded_sha(self, path):
        path = Path(path)
        if path == d.GUARD: return d.GUARD_SHA
        if path == d.PYTHON: return d.PYTHON_SHA
        self.assertNotEqual(path.suffix.lower(), '.gguf')
        return REAL_SHA(path)

    def fixture(self):
        out = self.base/'prepared'; out.mkdir()
        names = REQUIRED + [f'metal/shader{i}.metal' for i in range(26)]
        for name in names:
            path = out/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('fake immutable '+name).encode())
            if name in ('baseline-probe', 'new-probe'):
                path.chmod(0o755)
        plan = dict(geometry=d.GEOMETRY, arms=[list(x) for x in d.ARMS], policy=d.POLICY,
                    model_from_prior=d.MODEL, source_revision='cpu-fixture-only',
                    files={name:dict(bytes=(out/name).stat().st_size, sha256=REAL_SHA(out/name)) for name in names})
        digest = self.write_plan(out, plan)
        return out, plan, digest

    def write_plan(self, out, plan):
        (out/'manifest.json').write_text(json.dumps(plan))
        digest = REAL_SHA(out/'manifest.json')
        (out/'prepared-command.json').write_text(json.dumps(dict(manifest_sha256=digest, command=d.command(out, digest), cwd=str(d.ROOT))))
        return digest

    def test_valid_inventory_and_command_authority(self):
        out, plan, digest = self.fixture()
        with patch.object(d, 'sha', self.guarded_sha):
            self.assertEqual(d.verify(out, digest), plan)
        self.model.assert_not_called()

    def test_explicit_manifest_wrong_before_asset_read(self):
        out, plan, digest = self.fixture()
        with patch.object(d, 'sha', self.guarded_sha), patch.object(Path, 'stat', side_effect=AssertionError('No asset/model stat before manifest authority')):
            self.reject(d.verify, out, 'wrong-explicit-digest')

    def test_each_probe_must_be_executable(self):
        out, plan, digest = self.fixture()
        with patch.object(d, 'sha', self.guarded_sha):
            for name in ('baseline-probe', 'new-probe'):
                (out/name).chmod(0o644)
                self.reject(d.verify, out, digest)
                (out/name).chmod(0o755)
            self.assertEqual(d.verify(out, digest), plan)

    def test_each_mandatory_asset_omission(self):
        out, plan, digest = self.fixture()
        with patch.object(d, 'sha', self.guarded_sha):
            for name in REQUIRED + ['metal/shader0.metal']:
                changed = copy.deepcopy(plan); del changed['files'][name]
                self.reject(d.verify, out, self.write_plan(out, changed))

    def test_absolute_and_parent_path_escape(self):
        out, plan, digest = self.fixture()
        for name in ['/outside-secret', '../outside-secret', 'source/../outside-secret']:
            changed = copy.deepcopy(plan); changed['files'][name] = dict(bytes=0, sha256='bad')
            with patch.object(d, 'sha', self.guarded_sha):
                self.reject(d.verify, out, self.write_plan(out, changed))

    def test_same_length_pin_drift(self):
        out, plan, digest = self.fixture()
        path = out/'new-probe'; before = path.read_bytes()
        path.write_bytes(b'x' * len(before))
        with patch.object(d, 'sha', self.guarded_sha): self.reject(d.verify, out, digest)

    def test_size_pin_drift(self):
        out, plan, digest = self.fixture(); (out/'corpus.txt').write_bytes(b'changed')
        with patch.object(d, 'sha', self.guarded_sha): self.reject(d.verify, out, digest)

    def test_fixed_controls_cannot_change(self):
        out, plan, digest = self.fixture()
        for key, changed_value in [('geometry', dict(d.GEOMETRY, shift=64)), ('arms', list(reversed(plan['arms']))),
                                   ('policy', dict(d.POLICY, DS4_METAL_V41_RESIDENT_ROUTE_GAP='.30')),
                                   ('model_from_prior', dict(d.MODEL, bytes=1))]:
            changed = copy.deepcopy(plan); changed[key] = changed_value
            with patch.object(d, 'sha', self.guarded_sha): self.reject(d.verify, out, self.write_plan(out, changed))

    def test_prepared_command_digest_suffix_and_cwd(self):
        out, plan, digest = self.fixture()
        original = json.loads((out/'prepared-command.json').read_text())
        variants = []
        value = copy.deepcopy(original); value['manifest_sha256'] = 'wrong'; variants.append(value)
        value = copy.deepcopy(original); value['command'][-1] = 'wrong'; variants.append(value)
        value = copy.deepcopy(original); value['command'].append('--extra'); variants.append(value)
        value = copy.deepcopy(original); value['cwd'] = str(out); variants.append(value)
        for value in variants:
            (out/'prepared-command.json').write_text(json.dumps(value))
            with patch.object(d, 'sha', self.guarded_sha): self.reject(d.verify, out, digest)

    def test_guard_interpreter_hash_pins(self):
        out, plan, digest = self.fixture()
        for target in [d.GUARD, d.PYTHON]:
            with patch.object(d, 'sha', lambda p: 'wrong' if Path(p) == target else self.guarded_sha(p)):
                self.reject(d.verify, out, digest)

    def test_run_manifest_failure_never_calls_guard_or_model(self):
        with patch.object(d, 'verify', side_effect=ValueError('manifest')), patch.object(d, 'require_guard') as guard:
            self.reject(d.run, self.base/'uncreated', 'wrong')
            guard.assert_not_called(); self.model.assert_not_called()
        self.assertFalse((self.base/'uncreated').exists())

    def test_run_guard_failure_never_calls_model(self):
        with patch.object(d, 'verify', return_value={}), patch.object(d, 'require_guard', side_effect=RuntimeError('no ownership')):
            self.reject(d.run, self.base/'uncreated', 'digest')
            self.model.assert_not_called()
        self.assertFalse((self.base/'uncreated').exists())

    def test_literal_ds4_prefix_clear_uses_production_expression(self):
        tree = ast.parse((ROOT/'speed-bench/ideas/run_suffix_pair_tape_pilot.py').read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        expression = next(n.value for n in run.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'environment' for t in n.targets))
        env = {'DS4': 'bad', 'DS4_POLICY': 'bad', 'DS4OTHER': 'bad', 'PATH': '/cpu-only'}
        cleared = eval(compile(ast.Expression(expression), '<production-environment-expression>', 'eval'), {'os': SimpleNamespace(environ=env)})
        self.assertEqual(cleared, {'PATH': '/cpu-only'})

    def raw_file(self, approximate=False):
        path = self.base/'fake.raw.bin'
        header = (0x53505431, 129280, 16, 1409, 65, 512, 128, int(approximate))
        with path.open('wb') as stream:
            stream.write(struct.pack('<8I', *header)); stream.write(bytes.fromhex(TOKEN_HEX))
            stream.truncate(32 + 16*4 + (18 + (17 if approximate else 0))*129280*4)
        return path

    def test_actual_raw_geometry_both_extents(self):
        for approximate in [False, True]:
            path = self.raw_file(approximate)
            result = d.raw_reference(path, approximate)
            self.assertEqual(result['reference_rows'], 18)
            self.assertEqual(result['teacher_token_hex'], TOKEN_HEX)
            self.assertEqual(result['reference_sha256'], hashlib.sha256(b'\0'*(18*129280*4)).hexdigest())
            self.assertEqual(result['full_sha256'], REAL_SHA(path))

    def test_raw_header_and_target_contract(self):
        for index in range(8):
            path = self.raw_file(True)
            with path.open('r+b') as stream:
                stream.seek(index*4); value = 0 if index else 1; stream.write(struct.pack('<I', value))
            self.reject(d.raw_reference, path, True)
        path = self.raw_file()
        with path.open('r+b') as stream: stream.seek(32); stream.write(struct.pack('<I', 129280))
        self.reject(d.raw_reference, path, False)

    def test_raw_native_and_candidate_truncation_and_extra(self):
        for approximate, extent in [(False, 31), (False, 33), (False, 32+16*4+18*129280*4-1),
                                    (True, 32+16*4+18*129280*4), (True, 32+16*4+35*129280*4-1),
                                    (True, 32+16*4+35*129280*4+1)]:
            path = self.raw_file(approximate)
            with path.open('r+b') as stream: stream.truncate(extent)
            self.reject(d.raw_reference, path, approximate)

    def csv_fixture(self):
        rows = [dict(step=j, target=TARGETS[j] if j < 16 else -1,
                     delta_nll=(j+1)*.001 if j < 16 else 0, kl=(j+1)*.002,
                     tv=(j+1)*.001, argmax_flip=j % 2, bit_equal=int(j % 3 == 0)) for j in range(17)]
        s = summary()
        s.update(mean_kl=sum(r['kl'] for r in rows)/17, mean_tv=sum(r['tv'] for r in rows)/17,
                 sum_delta_nll=sum(r['delta_nll'] for r in rows),
                 argmax_flips=sum(r['argmax_flip'] for r in rows), bit_equal_rows=sum(r['bit_equal'] for r in rows))
        return rows, s

    def write_csv(self, rows):
        path = self.base/'quality.csv'
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
        return path

    def test_csv_ordered_teacher_rows_and_terminal(self):
        rows, s = self.csv_fixture()
        result = d.validate_quality_csv(self.write_csv(rows), s, TOKEN_HEX)
        self.assertEqual((result['rows'], result['teacher_rows']), (17, 16))

    def test_csv_duplicate_missing_reordered_rows_targets(self):
        rows, s = self.csv_fixture()
        variants = [rows[:-1], rows + rows[-1:], [rows[0]] + rows[:-1], list(reversed(rows))]
        for field, value in [('step', 2), ('target', 99)]:
            bad = copy.deepcopy(rows); bad[1][field] = value; variants.append(bad)
        for bad in variants: self.reject(d.validate_quality_csv, self.write_csv(bad), s, TOKEN_HEX)

    def test_csv_nonfinite_ranges_flags_terminal_nll(self):
        rows, s = self.csv_fixture()
        for field, value in [('delta_nll', float('nan')), ('kl', float('inf')), ('tv', -1), ('tv', 1.01),
                             ('kl', -.001), ('argmax_flip', 2), ('bit_equal', -1)]:
            bad = copy.deepcopy(rows); bad[0][field] = value
            self.reject(d.validate_quality_csv, self.write_csv(bad), s, TOKEN_HEX)
        bad = copy.deepcopy(rows); bad[-1]['delta_nll'] = .001
        self.reject(d.validate_quality_csv, self.write_csv(bad), s, TOKEN_HEX)

    def test_csv_summary_arithmetic(self):
        rows, s = self.csv_fixture(); path = self.write_csv(rows)
        for field in ['mean_kl', 'mean_tv', 'sum_delta_nll', 'argmax_flips', 'bit_equal_rows']:
            bad = dict(s); bad[field] += 1
            self.reject(d.validate_quality_csv, path, bad, TOKEN_HEX)

    def test_summary_three_valid_modes(self):
        for mode in ['native', 'tape-native', 'reuse']: d.validate_summary(summary(mode), mode)

    def test_summary_counts_are_literal_nonnegative_ints(self):
        s = summary()
        for field, original in s.items():
            if type(original) is int:
                for value in [False, -1, '0', float(original)]:
                    bad = dict(s); bad[field] = value
                    self.reject(d.validate_summary, bad, 'reuse')

    def test_summary_literal_control_booleans(self):
        s = summary()
        for field, original in s.items():
            if type(original) is bool:
                for value in [False, 'false', 1, None]:
                    bad = dict(s); bad[field] = value
                    self.reject(d.validate_summary, bad, 'reuse')

    def test_summary_finite_metrics_exclude_bool(self):
        s = summary()
        for field in ['fresh_ms_diagnostic', 'reuse_ms_diagnostic', 'mean_kl', 'mean_tv', 'sum_delta_nll']:
            for value in [False, '0', float('nan'), float('inf'), -float('inf')]:
                bad = dict(s); bad[field] = value
                self.reject(d.validate_summary, bad, 'reuse')

    def test_summary_geometry_rows_budget_controls(self):
        s = summary()
        for field in ['prefix', 'shift', 'span', 'tail', 'decode', 'raw_reference_rows', 'raw_candidate_rows',
                      'staging_budget_bytes', 'pair_inputs_reused', 'pair_rows_rebuilt', 'pair_boundary_rows',
                      'reused_tokens', 'fresh_tail_tokens']:
            bad = dict(s); bad[field] += 1
            self.reject(d.validate_summary, bad, 'reuse')
        bad = dict(s); bad['mode'] = 'native'; self.reject(d.validate_summary, bad, 'reuse')
        for value in [128<<20, (128<<20)+1]:
            bad = dict(s); bad['peak_staging_bytes_including_reserve'] = value
            self.reject(d.validate_summary, bad, 'reuse')
        valid = dict(s); valid['peak_staging_bytes_including_reserve'] = (128<<20)-1
        d.validate_summary(valid, 'reuse')

    def states(self):
        canonical, private = self.base/'native.bin', self.base/'diagnostic.bin'
        h = struct.pack('<13I', 0x34565344, 2, 4096, 1, 128, 128, 4097, 2049, 40, 512, 128, 129280, 0x413431)
        canonical.write_bytes(h+b'fake-native-body'); private.write_bytes(b'DSSD'+h[4:]+b'fake-diagnostic!')
        self.assertEqual(canonical.stat().st_size, private.stat().st_size)
        return canonical, private

    def test_diagnostic_private_tag_header_size_and_hash(self):
        canonical, private = self.states()
        result = d.diagnostic_state(private, canonical)
        self.assertEqual(result['sha256'], REAL_SHA(private)); self.assertFalse(result['canonical_export'])
        self.assertEqual(result['diagnostic_magic'], 'DSSD')

    def test_diagnostic_rejects_canonical_or_private_wrong_magic(self):
        for which, magic in [('canonical', b'DSSD'), ('canonical', b'BAD!'), ('private', b'DSV4'), ('private', b'BAD!')]:
            canonical, private = self.states(); path = canonical if which == 'canonical' else private
            path.write_bytes(magic + path.read_bytes()[4:]); self.reject(d.diagnostic_state, private, canonical)

    def test_diagnostic_context_layout_and_extent(self):
        for offset in range(4, 52, 4):
            canonical, private = self.states(); data = bytearray(private.read_bytes()); data[offset] ^= 1
            private.write_bytes(data); self.reject(d.diagnostic_state, private, canonical)
        for length in [0, 51, 52]:
            canonical, private = self.states(); private.write_bytes(private.read_bytes()[:length])
            self.reject(d.diagnostic_state, private, canonical)
        canonical, private = self.states(); private.write_bytes(private.read_bytes()+b'extra')
        self.reject(d.diagnostic_state, private, canonical)

    def guard_mocks(self, args=None, samefile=True, lock_errno=errno.EWOULDBLOCK, hashes=None):
        output, digest = self.base/'uncreated', 'cpu-explicit-manifest'
        expected = d.command(output, digest)
        observed = expected if args is None else args(expected)
        ps = patch.object(d.subprocess, 'check_output', return_value=shlex.join(observed)).start()
        patch.object(d.os.path, 'samefile', return_value=samefile).start()
        patch.object(d, 'sha', hashes or self.guarded_sha).start()
        handle = MagicMock(); handle.__enter__.return_value.fileno.return_value = 73
        opened = patch.object(Path, 'open', return_value=handle).start()
        failure = OSError(lock_errno, 'CPU mock lock') if lock_errno is not None else None
        lock = patch.object(d.fcntl, 'flock', side_effect=failure).start()
        return output, digest, ps, opened, lock

    def test_guard_exact_parent_framework_held_lock(self):
        output, digest, ps, opened, lock = self.guard_mocks()
        d.require_guard(output, digest)
        ps.assert_called_once_with(['/bin/ps', '-p', str(d.os.getppid()), '-o', 'command='], text=True, timeout=2)
        opened.assert_called_once_with('r+')
        lock.assert_called_once_with(opened.return_value.__enter__.return_value, d.fcntl.LOCK_EX|d.fcntl.LOCK_NB)
        self.model.assert_not_called()

    def test_guard_exact_suffix_cannot_change(self):
        variants = [lambda a: a[:-1]+['wrong-manifest'], lambda a: a+['--extra'],
                    lambda a: a[:1]+['/wrong/pressure_run.py']+a[2:], lambda a: []]
        for variant in variants:
            output, digest, ps, opened, lock = self.guard_mocks(args=variant)
            self.reject(d.require_guard, output, digest); lock.assert_not_called()
            patch.stopall()

    def test_guard_foreign_interpreter_alias(self):
        output, digest, ps, opened, lock = self.guard_mocks(samefile=False)
        self.reject(d.require_guard, output, digest); lock.assert_not_called()

    def test_guard_framework_samefile_alias_allowed(self):
        output, digest, ps, opened, lock = self.guard_mocks(args=lambda a: ['/verified-samefile/framework-alias']+a[1:])
        d.require_guard(output, digest)

    def test_guard_unheld_lock_and_unexpected_error(self):
        output, digest, ps, opened, lock = self.guard_mocks(lock_errno=None)
        self.reject(d.require_guard, output, digest)
        self.assertEqual(lock.call_count, 2)
        patch.stopall()
        output, digest, ps, opened, lock = self.guard_mocks(lock_errno=errno.EPERM)
        with self.assertRaises(OSError): d.require_guard(output, digest)

    def test_guard_pinned_framework_and_guard_hash(self):
        for target in [d.GUARD, d.PYTHON]:
            output, digest, ps, opened, lock = self.guard_mocks(hashes=lambda p: 'wrong' if Path(p)==target else self.guarded_sha(p))
            self.reject(d.require_guard, output, digest); lock.assert_not_called()
            patch.stopall()

    def test_model_hash_refusal_before_open(self):
        with patch.object(Path, 'open', side_effect=AssertionError('Model open forbidden')):
            self.reject(REAL_SHA, Path(d.MODEL['path']))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CPU)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(pass_=result.wasSuccessful(), tests=result.testsRun, rejected_cases=TRIALS,
                         models=0, model_metadata_reads=0, gpu_devices=0, artifact_root=str(ARTIFACTS))))
    raise SystemExit(0 if result.wasSuccessful() else 1)
