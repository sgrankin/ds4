#!/usr/bin/env python3
"""Independent CPU fixtures: synthetic bytes/metadata and tiny Python producers."""
import contextlib
import copy
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'speed-bench/ideas'))
import run_ragged_native_tail_qualification as d
import analyze_ragged_native_tail_qualification as a

ARTIFACTS = WORK / 'speed-bench/ideas/artifacts/ragged-qualifier-cpu-fixtures'
ARTIFACTS.mkdir(exist_ok=True, parents=True)


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False) + '\n')


def packed(values):
    return struct.pack('<' + str(len(values)) + 'f', *values)


class Fixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='case-', dir=ARTIFACTS)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def refuse(self, fn, *args):
        with self.assertRaises((ValueError, KeyError, TypeError, OSError)):
            fn(*args)

    def csv(self, name, header, rows):
        path = self.root / name
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_strict_json_and_number_types(self):
        for data in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":'):
            self.refuse(a.contract.load_json, data)
        for value in (True, -1, 1.0, '1', 1 << 64):
            self.refuse(a.integer, value, 'value')
        for value in (True, '1', math.nan, math.inf, -math.inf):
            self.refuse(a.number, value, 'value')
        for value in ('-1', '+1', '01', '1.0', '1 ', 'true', str(1 << 64)):
            self.refuse(a.decimal, value)
        self.assertEqual(a.decimal('0'), 0)

    def test_main_csv_geometry_duplicates_truncation_nonfinite(self):
        header = ['phase', 'kind', 'context', 'tokens', 'ms', 'first_decode_ms']
        rows = [[p, k, pos, n, 10, 1 if k == 'D' else 0] for p, (k, pos, n) in enumerate(zip(a.contract.KINDS, a.contract.POSITIONS, a.contract.COUNTS))]
        path = self.csv('main.csv', header, rows)
        self.assertEqual(len(a.main_rows(path)), 9)
        mutations = []
        for index, column, value in [(3, 2, 10213), (5, 3, 16384), (0, 0, 1), (0, 4, 'NaN'), (0, 5, 1), (2, 5, 11), (2, 4, -1)]:
            changed = copy.deepcopy(rows); changed[index][column] = value; mutations.append(changed)
        mutations += [rows[:-1], rows + [rows[-1]], rows[:3] + [rows[2]] + rows[4:]]
        for changed in mutations:
            self.refuse(a.main_rows, self.csv('main.csv', header, changed))
        self.csv('main.csv', header, rows)
        path.write_bytes(path.read_bytes().rstrip(b'\r\n'))
        self.refuse(a.main_rows, path)
        self.refuse(a.csv_rows, self.csv('extra.csv', header + ['extra'], [rows[0] + ['x']]), header)

    def cache_log(self):
        return '\n'.join([f'REPLAY_CACHE phase={p} configured_slots=6136 resident_entries={p}' for p in range(9)] +
            ['ragged main final, branches excluded',
             'ds4:   streaming expert cache budget=6136 experts entries=9 expert=1.25 MiB target=64.00 GiB live=0.10 GiB, hits=3 misses=9 hit_rate=0.250 wraps=1 evictions=2 buffer_allocs=9 buffer_reuses=0 evict_dontneed=0.01 GiB miss_willneed=0.02 GiB miss_pread=0.02 GiB pread_ms=0.300',
             'RAGGED_BRANCH_CACHE phase=3 configured_slots=6136 resident_entries=3',
             'RAGGED_BRANCH_CACHE phase=5 configured_slots=6136 resident_entries=5']) + '\n'

    def test_cache_all_phases_and_footer_scope(self):
        log = self.cache_log()
        parsed = a.cache_records(log)
        self.assertEqual(parsed['main_footer']['misses'], 9)
        self.assertFalse(parsed['exact_main_IO_available'])
        for bad in (log.replace('phase=8 ', 'phase=7 '), log.replace('slots=6136', 'slots=7680', 1),
                    log.replace('resident_entries=3', 'resident_entries=6137'), log + 'REPLAY_CACHE bad\n',
                    log.replace('hits=3', 'hits=3 hits=4'), log.replace('pread_ms=0.300', 'pread_ms=nan'),
                    log.replace('hits=3', 'hits=-3'), log.replace('ragged main final, branches excluded', ''),
                    log + 'ragged main final, branches excluded\n'):
            self.refuse(a.cache_records, bad)
        # A later branch memory footer must not replace the main-only one.
        self.assertEqual(a.cache_records(log + 'ds4:   streaming expert cache budget=2\n')['main_footer']['budget'], 6136)

    def test_branch_io_typed_complete_order(self):
        head = ['phase', 'pread_calls', 'requested_bytes', 'returned_bytes', 'hits', 'misses', 'evictions']
        rows = [[3, 1, 4096, 4096, 2, 1, 0], [5, 2, 8192, 8192, 3, 2, 1]]
        self.assertEqual(a.branch_io(self.csv('io.csv', head, rows))[1]['returned_bytes'], 8192)
        for changed in (rows[::-1], rows[:1], rows + [rows[0]], [[3, 1, 4096, 4095, 2, 1, 0], rows[1]],
                        [[3, True, 4096, 4096, 2, 1, 0], rows[1]]):
            self.refuse(a.branch_io, self.csv('io.csv', head, changed))

    def snapshot_bytes(self, history, logits):
        # Independent concrete serialization formula: three compressed pair
        # owners, one per-token owner,40 sliding owners and three pending pairs.
        pos = len(history); vocab = len(logits)
        n = 52 + pos * 4 + vocab * 4 + 40 * min(pos, 128) * 512 * 4
        n += 3 * (pos // 2) * 640 * 4 + pos * 640 * 4
        if pos % 2:
            n += 3 * 2 * 512 * 4
        head = struct.pack('<13I', 0x34565344, 2, 32768, 1, 128, 128, 32769, pos, 40, 512, 128, vocab, 0x413431)
        prefix = head + struct.pack('<' + str(pos) + 'i', *history) + packed(logits)
        # Opaque hidden regions may legally contain nonfinite sentinel bits.
        return prefix + b'\xff' * (n - len(prefix))

    def test_canonical_complete_bytes_header_history_and_opaque_sentinels(self):
        for history in ([], [11], [11, 12], list(range(17))):
            path = self.root / 'state.bin'; data = self.snapshot_bytes(history, [0, 1, 2, 3]); path.write_bytes(data)
            pin, logits = a.snapshot(path, history, packed([0, 1, 2, 3]), 4)
            self.assertEqual(pin['bytes'], len(data)); self.assertEqual(logits, packed([0, 1, 2, 3]))
            for changed in (data[:-1], data + b'x', b'DSSD' + data[4:], data[:28] + struct.pack('<I', len(history) + 1) + data[32:]):
                path.write_bytes(changed); self.refuse(a.snapshot, path, history, None, 4)
            path.write_bytes(data)
            self.refuse(a.snapshot, path, history, packed([1, 1, 2, 3]), 4)
            if history:
                self.refuse(a.snapshot, path, [history[0] + 1] + history[1:], None, 4)
            bad = bytearray(data); offset = 52 + len(history) * 4
            bad[offset:offset+4] = struct.pack('<I', 0x7fc00000); path.write_bytes(bad)
            self.refuse(a.snapshot, path, history, None, 4)

    def test_raw_extents_nonfinite_and_independent_metrics(self):
        path = self.root / 'raw.f32'; path.write_bytes(packed([0, 1, 2, 3]) * 2)
        self.assertEqual(a.finite_vectors(path, 2, 4)['bytes'], 32)
        self.assertEqual(a.vector(path, 1, 4), packed([0, 1, 2, 3]))
        for data in (path.read_bytes()[:-1], path.read_bytes() + b'x', packed([0, math.nan, 2, 3]) * 2, packed([0, math.inf, 2, 3]) * 2):
            path.write_bytes(data); self.refuse(a.finite_vectors, path, 2, 4)
        metric = a.contract.logit_metrics(packed([0, 0]), packed([math.log(3), 0]), 1, 2)
        self.assertAlmostEqual(metric['delta_nll'], math.log(2), places=6)
        self.assertAlmostEqual(metric['kl'], .5 * math.log(4 / 3), places=6)
        self.assertAlmostEqual(metric['tv'], .25, places=6)
        self.assertEqual(a.contract.logit_metrics(packed([2, 1]), packed([2, 1]), 0, 2)['delta_nll'], 0)

    def dispatch_rows(self, tails):
        values = []
        for p in (0, 1, 3, 5, 7):
            seq = tails.get(p, [a.contract.COUNTS[p]])
            cursor = a.contract.PREFIXES[p]
            for n in seq:
                values.append(dict(phase=p, start=cursor, end=cursor+n, total=a.contract.POSITIONS[p], count=n)); cursor += n
        return values

    def test_actual_dispatch_completion_and_difference(self):
        native = self.dispatch_rows({3: [8192, 64, 64], 5: [16384] + [64]*7 + [63]})
        adaptive = self.dispatch_rows({3: [8192] + [8]*16, 5: [16384, 511]})
        data = lambda rows: ''.join(json.dumps(r) + '\n' for r in rows).encode()
        p = a.contract.dispatch(data(native)); q = a.contract.dispatch(data(adaptive))
        self.assertTrue(a.contract.dispatch_gate(p, q))
        self.refuse(a.contract.dispatch_gate, p, p)
        partial = copy.deepcopy(q); partial[5] = p[5]; self.refuse(a.contract.dispatch_gate, p, partial)
        for rows in (native[:-1], native + [native[-1]], native[::-1]):
            self.refuse(a.contract.dispatch, data(rows))
        for field, value in [('count', True), ('start', 1), ('total', 0), ('count', 0)]:
            rows = copy.deepcopy(native); rows[0][field] = value; self.refuse(a.contract.dispatch, data(rows))
        self.refuse(a.contract.dispatch, data(native)[:-1])
        crossing = copy.deepcopy(p); crossing[3] = [8256, 64]
        self.refuse(a.contract.dispatch_gate, crossing, q)

    def test_teacher_rows_score_before_target_no_duplicates(self):
        targets = {3: list(range(32)), 5: list(range(32, 64))}
        rows = [[p, j, a.contract.POSITIONS[p]+j, t] for p in (3,5) for j,t in enumerate(targets[p])]
        head = ['phase', 'step', 'pos', 'target']
        path = self.csv('target.csv', head, rows)
        self.assertEqual(len(a.contract.branch_rows(path, targets)), 64)
        for changed in (rows[:-1], rows + [rows[0]], rows[:1] + [rows[0]] + rows[2:]):
            self.refuse(a.contract.branch_rows, self.csv('target.csv', head, changed), targets)
        for column in (0, 1, 2, 3):
            changed = copy.deepcopy(rows); changed[0][column] += 1
            self.refuse(a.contract.branch_rows, self.csv('target.csv', head, changed), targets)

    def report(self, drift=False):
        return dict(artifacts={'raw': dict(sha256='different' if drift else 'same', bytes=16)},
                    cache=dict(main=[[0,6136,0]], branches=[[3,6136,0]], main_footer={'hits':1}),
                    branch_IO=[{'returned_bytes':0}], dispatch={3:[8192,64,64],5:[16384]+[64]*7+[63]})

    def test_native_controls_require_raw_state_cache_io_dispatch(self):
        old = self.report()
        self.assertTrue(a.exact_control(old, copy.deepcopy(old)))
        for field in ('artifacts', 'branch_IO', 'dispatch'):
            changed = copy.deepcopy(old); changed[field] = {'different':1}
            self.refuse(a.exact_control, old, changed)
        for field in ('main', 'branches', 'main_footer'):
            changed = copy.deepcopy(old); changed['cache'][field] = {'different':1}
            self.refuse(a.exact_control, old, changed)

    def fake_plan(self, rc_name=None):
        # The actual driver launches only this tiny Python producer in these
        # tests. A nonexistent model string is never resolved or inspected.
        producer = self.root / 'producer.py'
        producer.write_text('import json,os,sys\nfrom pathlib import Path\np=Path(sys.argv[1]);p.mkdir()\n(p/"cpu.json").write_text(json.dumps({"name":p.name,"env":{k:v for k,v in os.environ.items() if k.startswith("DS4")}}))\nraise SystemExit(int(sys.argv[2]))\n')
        replay = self.root / 'replay.txt'; replay.write_text('CPU ONLY\n')
        plan = dict(output=str(self.root / 'run'), pressure=str(self.root / 'pressure'), run_cwd=str(self.root),
                    replay=str(replay), assets={}, arms=[])
        for name in a.NAMES:
            plan['arms'].append(dict(name=name, output=str(self.root / 'run' / name),
                argv=[sys.executable, str(producer), str(self.root / 'run' / name), str(7 if name==rc_name else 0)],
                environment={'DS4_CPU_FAKE':'only'}, exact_reference= 'old0-default' if name in ('new0-default','new1-default') else ('old0-fast' if name=='new0-fast' else None)))
        path = self.root / 'plan.json'; write(path,plan)
        return path,plan

    def simulated(self, rc_name=None, drift_name=None, asset_failure_call=None):
        path,plan = self.fake_plan(rc_name)
        events=[]; asset_calls=0
        def assets(_):
            nonlocal asset_calls
            asset_calls+=1; events.append('assets')
            if asset_failure_call == asset_calls: raise ValueError('static tamper')
        def model(_):
            self.assertIn(events[-3:], (['guard','grant','assets'], ['assets','guard','grant'])); events.append('FAKE_METADATA')
            return {'CPU_ONLY':True}
        def validate(output, stderr, replay, targets, rc):
            a.require(type(rc) is int and rc==0,'simulated producer refusal')
            return self.report(Path(output).name == drift_name)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(d,'validate_plan'))
            stack.enter_context(patch.object(d,'require_guard',side_effect=lambda *_: events.append('guard') or {'CPU':True}))
            stack.enter_context(patch.object(d,'require_grant',side_effect=lambda *_: events.append('grant')))
            stack.enter_context(patch.object(d,'check_assets',side_effect=assets))
            stack.enter_context(patch.object(d,'model_identity',side_effect=model))
            stack.enter_context(patch.object(d,'fixture_check',return_value={}))
            stack.enter_context(patch.object(a,'validate_arm',side_effect=validate))
            stack.enter_context(patch.object(a,'candidate_gate',return_value={'CPU':True}))
            stack.enter_context(patch.dict(os.environ, {'DS4_X_INHERITED':'bad','DS4WRONG':'bad'}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            if rc_name or drift_name or asset_failure_call:
                self.refuse(d.run,path,a.sha(path),True)
            else:
                self.assertEqual(d.run(path,a.sha(path),True),0)
                self.refuse(d.run,path,a.sha(path),True) # preserved first run, never overwrite/retry
        return path,plan,events

    def test_simulated_six_producer_order_controls_env_no_retry(self):
        _,plan,events=self.simulated()
        complete=a.load(Path(plan['output'])/'complete.json')
        self.assertEqual(complete['arms'],a.NAMES)
        self.assertEqual(events.count('FAKE_METADATA'),8)
        for name in a.NAMES:
            self.assertEqual(a.load(Path(plan['output'])/name/'cpu.json'),{'name':name,'env':{'DS4_CPU_FAKE':'only'}})
        self.assertFalse((Path(plan['output'])/'first-failure.json').exists())

    def test_simulated_first_failure_stops_no_candidate_or_retry(self):
        _,plan,_=self.simulated(rc_name='new0-default')
        root=Path(plan['output']); fail=a.load(root/'first-failure.json')
        self.assertEqual(fail['completed_arms'],['old0-default'])
        self.assertEqual(a.load(root/'new0-default-process.json')['returncode'],7)
        self.assertFalse((root/'new1-default-started.json').exists()); self.assertFalse((root/'complete.json').exists())

    def test_simulated_native_drift_stops_before_candidate(self):
        _,plan,_=self.simulated(drift_name='new0-default')
        root=Path(plan['output'])
        self.assertEqual(a.load(root/'first-failure.json')['completed_arms'],['old0-default'])
        self.assertFalse((root/'new1-default-started.json').exists())

    def test_simulated_asset_drift_blocks_next_launch_before_metadata(self):
        _,plan,events=self.simulated(asset_failure_call=3)
        self.assertEqual(events.count('FAKE_METADATA'),2)
        self.assertEqual(a.load(Path(plan['output'])/'first-failure.json')['completed_arms'],['old0-default'])
        self.assertFalse((Path(plan['output'])/'new0-default-started.json').exists())

    def test_no_explicit_grant_has_no_child_or_model(self):
        path,plan=self.fake_plan()
        with patch.object(d,'model_identity',side_effect=AssertionError('must not inspect model')), patch.object(d.subprocess,'Popen',side_effect=AssertionError('must not launch')):
            self.refuse(d.run,path,a.sha(path),False)
        self.assertFalse(Path(plan['output']).exists())

    def guard_fixture(self):
        root=self.root/'MAIN'; (root/'speed-bench/ideas/artifacts').mkdir(parents=True)
        lock=root/'speed-bench/ideas/artifacts/model-job.lock'; lock.touch()
        framework=self.root/'framework'; framework.write_text('CPU interpreter identity')
        guard=self.root/'pressure.py'; guard.write_text('CPU guard identity')
        plan={'guard_command_template':['shim',str(guard),'pressure','--','shim','driver','--plan-sha256',d.PLACEHOLDER]}
        return root,lock,framework,guard,plan

    def test_direct_parent_framework_alias_exact_suffix_and_held_lock(self):
        root,lock,framework,guard,plan=self.guard_fixture()
        actual=[str(framework)]+d.command(plan,'pin')[1:]
        with lock.open('r+') as owned, contextlib.ExitStack() as stack:
            fcntl.flock(owned,fcntl.LOCK_EX|fcntl.LOCK_NB)
            stack.enter_context(patch.object(d,'ROOT',root)); stack.enter_context(patch.object(d,'FRAMEWORK',framework))
            stack.enter_context(patch.object(d,'GUARD',guard)); stack.enter_context(patch.object(d,'FRAMEWORK_SHA',a.sha(framework)))
            stack.enter_context(patch.object(d,'GUARD_SHA',a.sha(guard))); stack.enter_context(patch.object(d.Path,'cwd',return_value=root))
            with patch.object(d.subprocess,'check_output',return_value=shlex.join(actual)) as ps:
                self.assertTrue(d.require_guard(plan,'pin')['lock_held'])
                self.assertEqual(ps.call_args.kwargs['timeout'],2)
            for changed in (actual[:-1]+['altered'], ['missing-framework']+actual[1:], actual+["extra"]):
                with patch.object(d.subprocess,'check_output',return_value=shlex.join(changed)):
                    self.refuse(d.require_guard,plan,'pin')
            with patch.object(d.subprocess,'check_output',return_value=shlex.join(actual)), patch.object(d.Path,'cwd',return_value=self.root):
                self.refuse(d.require_guard,plan,'pin')
            fcntl.flock(owned,fcntl.LOCK_UN)
            with patch.object(d.subprocess,'check_output',return_value=shlex.join(actual)):
                self.refuse(d.require_guard,plan,'pin')

    def test_explicit_root_proof_exact_command_and_typed_grant(self):
        plan={'guard_command_template':['python','guard',d.PLACEHOLDER]}; path=self.root/'plan.json'
        proof={'status':'GRANTED','sole_model_grant':True,'manifest_sha256':'pin','command':d.command(plan,'pin'),'cwd':str(d.ROOT),'time':1.0}
        write(self.root/'root-launch-proof.json',proof); d.require_grant(plan,path,'pin')
        for field,value in [('status','queued'),('sole_model_grant',1),('manifest_sha256','other'),('command',['changed']),('cwd','elsewhere'),('time',True),('time',0)]:
            bad=copy.deepcopy(proof);bad[field]=value;write(self.root/'root-launch-proof.json',bad)
            self.refuse(d.require_grant,plan,path,'pin')

    def test_static_asset_alias_mode_and_model_hash_refusal(self):
        path=self.root/'asset';path.write_bytes(b'static');path.chmod(0o640)
        self.assertEqual(d.asset(path)['mode'],0o640)
        alias=self.root/'alias';alias.symlink_to(path);self.refuse(d.asset,alias)
        missing=self.root/'never-open.gguf'
        self.refuse(d.asset,missing);self.refuse(a.sha,missing)
        first=d.asset(path);path.write_bytes(b'drift')
        with patch.object(d,'build_check'),patch.object(d,'fixture_check'):
            self.refuse(d.check_assets,{'assets':{str(path):first}})

    def test_fixed_plan_inventory_authority_and_controls(self):
        # The validator reads only the original frozen small teacher/spec
        # files. Assets/build execution are replaced by a declared inventory;
        # actual preparation subsequently verifies every real pin separately.
        with tempfile.TemporaryDirectory(prefix='cpu-plan-',dir=ARTIFACTS.parent) as directory:
            parent=Path(directory);path=parent/'plan.json'
            for name in ('qualification-spec.json','ragged-growth.txt','branch-targets.txt','original-growth.txt'):
                shutil.copy2(d.SPEC/name,parent/name)
            spec=a.load(parent/'qualification-spec.json');arms=[]
            for frozen in spec['arms']:
                env={'DS4_METAL_STREAMING_EXPERT_IO_METRICS':'1','DS4_METAL_V41_RESIDENT_ROUTE_GAP':'0',
                     'DS4_METAL_V41_RESIDENT_ROUTE_MODE':'native',d.FLAG:str(frozen['flag'])}
                env.update({'DS4_METAL_'+n.upper()+'_SOURCE':str(parent/'metal'/(n+'.metal')) for n in d.SHADERS})
                if frozen['policy']=='fast':env.update(DS4_METAL_V41_SELECTED_TILE='64',DS4_METAL_V41_SELECTED_LIMIT='1024')
                arms.append(dict(name=frozen['name'],binary_variant=frozen['binary'],flag=frozen['flag'],policy=frozen['policy'],
                    exact_reference=frozen['exact_reference'],output=str(parent/'run'/frozen['name']),
                    argv=[d.BINARIES[frozen['binary']],d.MODEL['path'],str(parent/'ragged-growth.txt'),str(parent/'branch-targets.txt'),str(parent/'run'/frozen['name'])],environment=env))
            authorities={d.BUILD/'build-manifest.json',*[Path(p) for p in d.BINARIES.values()]}
            mandatory=authorities|{d.GUARD,d.PYTHON,d.PYTHON_TARGET,d.FRAMEWORK}
            mandatory|={parent/n for n in d.SCRIPTS+['runner-tests.py','cpu-proof.json','cpu-tests.log','qualification-spec.json','ragged-growth.txt','branch-targets.txt','original-growth.txt']}
            mandatory|={parent/'metal'/(n+'.metal') for n in d.SHADERS}
            assets={str(p):dict(sha256='0'*64,bytes=0,mode=0o644) for p in mandatory}
            pins={str(d.GUARD):d.GUARD_SHA,str(d.PYTHON):d.PYTHON_SHA,str(d.PYTHON_TARGET):d.PYTHON_SHA,str(d.FRAMEWORK):d.FRAMEWORK_SHA,
                  str(d.BUILD/'build-manifest.json'):d.BUILD_SHA,str(parent/'qualification-spec.json'):d.SPEC_SHA,
                  str(parent/'ragged-growth.txt'):d.REPLAY_SHA,str(parent/'branch-targets.txt'):d.TARGET_SHA,
                  str(parent/'original-growth.txt'):d.ORIGINAL_SHA,str(parent/d.SCRIPTS[2]):d.VALIDATOR_SHA,
                  **{d.BINARIES[k]:v for k,v in d.BINARY_SHAS.items()}}
            for p,digest in pins.items():assets[p]['sha256']=digest
            plan=dict(schema='ragged-native-tail-six-qualifier-v1',prepared_only=True,model_grant=False,quiet_grant=False,no_automatic_retry=True,
                root_cwd=str(d.ROOT),run_cwd=str(d.WORK),output=str(parent/'run'),pressure=str(parent/'pressure'),model=d.MODEL,
                binaries=d.BINARIES,binary_shas=d.BINARY_SHAS,build_manifest_sha256=d.BUILD_SHA,spec=str(parent/'qualification-spec.json'),
                replay=str(parent/'ragged-growth.txt'),targets=str(parent/'branch-targets.txt'),original=str(parent/'original-growth.txt'),
                guard=str(d.GUARD),guard_sha256=d.GUARD_SHA,framework_identity=dict(path=str(d.FRAMEWORK),sha256=d.FRAMEWORK_SHA),
                guard_command_template=[str(d.PYTHON),str(d.GUARD),str(parent/'pressure'),'--',str(d.PYTHON),str(parent/d.SCRIPTS[0]),'--run',str(path),'--plan-sha256',d.PLACEHOLDER,'--root-granted'],arms=arms,assets=assets)
            with patch.object(d,'authority_paths',return_value=authorities),patch.object(d,'shader_pins',return_value={n+'.metal':'0'*64 for n in d.SHADERS}):
                d.validate_plan(plan,path)
                changes=[]
                for field,value in [('model_grant',0),('prepared_only',1),('quiet_grant',True),('output',str(self.root/'escape')),('binaries',{'old':'other'}),('build_manifest_sha256','0'*64)]:
                    bad=copy.deepcopy(plan);bad[field]=value;changes.append(bad)
                for field,value in [('flag',True),('argv',['altered']),('exact_reference',None),('policy','fast')]:
                    bad=copy.deepcopy(plan);bad['arms'][1][field]=value;changes.append(bad)
                bad=copy.deepcopy(plan);bad['guard_command_template'][-2]='not-the-plan-hash';changes.append(bad)
                bad=copy.deepcopy(plan);bad['arms'][0]['environment']['DS4_METAL_V41_SELECTED_TILE']='64';changes.append(bad)
                bad=copy.deepcopy(plan);bad['arms'][4]['environment']['DS4_METAL_V41_SELECTED_LIMIT']='256';changes.append(bad)
                bad=copy.deepcopy(plan);del bad['assets'][str(parent/'runner-tests.py')];changes.append(bad)
                bad=copy.deepcopy(plan);bad['assets'][str(self.root/'escape')]=dict(sha256='0'*64,bytes=0,mode=0);changes.append(bad)
                bad=copy.deepcopy(plan);bad['assets'][str(d.GUARD)]['sha256']='0'*64;changes.append(bad)
                for field,value in [('bytes',True),('mode',True),('mode',0o1000),('sha256','untyped')]:
                    bad=copy.deepcopy(plan);bad['assets'][str(parent/'runner-tests.py')][field]=value;changes.append(bad)
                bad=copy.deepcopy(plan);bad['arms']=bad['arms'][::-1];changes.append(bad)
                for bad in changes:self.refuse(d.validate_plan,bad,path)

    def test_release_gate_precedes_every_outcome_read(self):
        path=self.root/'plan.json';write(path,{'output':str(self.root/'NONEXISTENT_OUTCOMES')})
        with patch.object(d,'validate_plan'),patch.object(a,'validate_release',side_effect=ValueError('unreleased')),patch.object(a,'validate_arm',side_effect=AssertionError('outcome read forbidden')):
            self.refuse(a.analyze,path,a.sha(path),'no-release','pin',self.root/'result.json')
        self.assertFalse((self.root/'result.json').exists())

    def release_fixture(self):
        pressure=self.root/'pressure';pressure.mkdir()
        plan={'pressure':str(pressure),'guard_command_template':['shim','guard',str(pressure),'--','shim','driver',d.PLACEHOLDER]}
        result={'status':'complete','child_returncode':0};write(pressure/'result.json',result)
        write(pressure/'command.json',{'argv':d.command(plan,'pin')[4:],'initial':{'level':1,'swap_gb':0,'time':1}})
        (pressure/'pressure.jsonl').write_text(''.join(json.dumps({'time':t,'level':1,'swap_gb':0,'reason':None,'consecutive_bad':0})+'\n' for t in (2,3)))
        release={'guard_result':result,'lock_reacquired':True,'live_model_processes':[],'time':4,'normal_pressure_samples':2,'bad_pressure_samples':0,'swap_growth_gb':0}
        path=self.root/'release.json';write(path,release)
        return plan,path,release

    def test_complete_release_pressure_child_and_type_balance(self):
        plan,path,release=self.release_fixture()
        a.validate_release(plan,'pin',path,a.sha(path))
        for key,value in [('lock_reacquired',1),('live_model_processes',['live']),('time',True),('normal_pressure_samples',True),('bad_pressure_samples',1),('swap_growth_gb',1)]:
            bad=copy.deepcopy(release);bad[key]=value;write(path,bad);self.refuse(a.validate_release,plan,'pin',path,a.sha(path))
        write(path,release)
        pressure=Path(plan['pressure'])
        for field,value in [('level',2),('swap_gb',1),('reason','bad'),('consecutive_bad',True),('time',5)]:
            rows=[{'time':2,'level':1,'swap_gb':0,'reason':None,'consecutive_bad':0},{'time':3,'level':1,'swap_gb':0,'reason':None,'consecutive_bad':0}]
            rows[1][field]=value;(pressure/'pressure.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.refuse(a.validate_release,plan,'pin',path,a.sha(path))
        self.refuse(a.validate_release,plan,'pin',path,'0'*64)

    def test_artifact_inventory_refusal_before_raw_or_snapshot_reads(self):
        out=self.root/'arm';out.mkdir()
        self.refuse(a.validate_arm,out,self.root/'stderr','',{},0)
        for name in a.FILES:(out/name).touch()
        (out/'extra').touch();self.refuse(a.validate_arm,out,self.root/'stderr','',{},0)
        (out/'extra').unlink();(out/'main.csv').unlink();(out/'main.csv').symlink_to(self.root/'missing')
        self.refuse(a.validate_arm,out,self.root/'stderr','',{},0)
        self.refuse(a.validate_arm,out,self.root/'stderr','',{},True)

    def test_final_phase_has_no_invented_target_nll(self):
        targets={3:list(range(32)),5:list(range(32))}
        def vec(path,index):return packed([0,0])
        original=a.contract.logit_metrics
        with patch.object(a,'vector',side_effect=vec),patch.object(a.contract,'ragged',return_value=[{'ids':[1]} for _ in range(9)]),patch.object(a.contract,'logit_metrics',side_effect=lambda x,y,t:original(x,y,t%2,2)):
            value=a.diagnostics('native','candidate',targets,'CPU')
        self.assertEqual(value['predictions'],64)
        self.assertEqual(len(value['phase_end_rows']),9)
        self.assertEqual(value['phase_end_rows'][7]['target'],1)
        self.assertIsNone(value['phase_end_rows'][8]['target'])
        self.assertNotIn('delta_nll',value['phase_end_rows'][8])


if __name__ == '__main__':
    unittest.main(verbosity=2)
