import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
a=load('archive',ROOT/'speed-bench/ideas/analyze_closing_first_attempt.py')
fixture_source=load('fixture_source',ROOT/'tests/test_summarize_closing.py')
c=fixture_source.c

def put(path,value):path.write_text(json.dumps(value)+'\n')

class FirstAttempt(unittest.TestCase):
    def setUp(self):
        base=ROOT/'speed-bench/ideas/artifacts/closing-analysis-cpu-fixtures';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.base=Path(self.temp.name)
        self.maker=fixture_source.CheckedClosing();self.maker.setUp()
        self.reference=self.base/'reference';self.reference.mkdir();self.maker.out=self.reference
        self.rows=self.maker.make(0);self.fixture=self.maker.fixture
        self.fixturepath=self.base/'fixture.json';put(self.fixturepath,self.fixture)
        self.out=self.base/'output';self.out.mkdir();self.pressure=self.base/'pressure';self.pressure.mkdir()
        self.arms=[]
        for name,boost in a.ARMS:
            target=self.base/name;shutil.copytree(self.reference,target)
            if boost:
                self.maker.out=target;rows=self.maker.make(2,cap=True)
                # Circular is naturally below gate and must preserve original.
                circle=copy.deepcopy(self.rows[0]);circle['closing_boost']=2
                for n in [circle['raw_trace'],circle['diagnostics']['token_trace']]:shutil.copy2(self.reference/n,target/n)
                rows[0]=circle;fixture_source.lines(target/'results.jsonl',rows)
            self.arms.append(dict(name=name,boost=boost,results_dir=str(target),argv=['opaque-cpu-fixture',name]))
        self.plan=dict(output=str(self.out),pressure=str(self.pressure),fixture=str(self.fixturepath),reference=str(self.reference),
                       arms=self.arms,run_cwd=str(self.base),environment={'DS4_OVERTHINK_PENALTY':'0'})
        self.release=json.loads((ROOT/'tests/fixtures/closing-root-release-proof-20261003.json').read_text())
        put(self.out/'root-release-proof.json',self.release);put(self.pressure/'result.json',dict(status='complete',child_returncode=3))
        reports=[]
        for arm in self.arms:
            rc=3 if arm['boost'] else 0
            report,_,_=a.inspect_arm(c,arm,self.fixture,self.reference,rc);report['arm']=arm['name'];reports.append(report)
            put(self.out/(arm['name']+'-process.json'),dict(returncode=rc))
            put(self.out/(arm['name']+'-command.json'),dict(argv=arm['argv'],cwd=self.plan['run_cwd'],ds4_environment=self.plan['environment']))
        put(self.out/'summary.json',dict(status='COMPLETE',all_first_attempts_retained=True,arms=reports))
    def tearDown(self):self.maker.tearDown();self.temp.cleanup()
    def check(self):return a.analyze_completed(self.plan,c)
    def test_complete_caps_invalids_and_raw_metrics_preserved(self):
        r=self.check();self.assertEqual(r['original_guard_returncode'],3)
        candidate=r['arms'][1];self.assertEqual((candidate['invalid'],candidate['cap_hits'],candidate['changed_cases']),(3,3,3))
        self.assertEqual(candidate['completed_correct'],1);self.assertEqual(candidate['active_steps'],3072)
        self.assertEqual(candidate['per_case'][0]['active_steps'],0)
        self.assertTrue(candidate['per_case'][0]['token_rng_piece_status_parity_vs_native25'])
        self.assertEqual(candidate['per_case'][1]['active_prefilter_samples'],32)
        self.assertGreater(candidate['per_case'][1]['mean_active_kl_raw_to_policy_prefilter'],0)
        self.assertIn('Not post-top-p',r['distribution']);self.assertFalse(r['model_opened'])
    def test_release_fail_closed(self):
        for key,value in [('guard_result',None),('guard_result',dict(status='complete',child_returncode=0)),
                          ('guard_result',dict(status='complete',child_returncode=True)),
                          ('guard_result',dict(status='running',child_returncode=3)),
                          ('lock_reacquired',False),('live_model_processes',[123])]:
            bad=dict(self.release,**{key:value})
            with self.subTest(key=key),self.assertRaises(ValueError):a.validate_release(bad,dict(status='complete',child_returncode=3))
        for rc in [1,2,True,None]:
            with self.subTest(rc=rc),self.assertRaises(ValueError):a.validate_release(self.release,dict(status='complete',child_returncode=rc))
    def test_no_release_no_live_judgment(self):
        (self.out/'root-release-proof.json').unlink()
        with self.assertRaises(FileNotFoundError):self.check()
    def test_old_mock_schema_rejected(self):
        old=dict(guard_returncode=3,root_lock_reacquired=True,matching_live_model_processes=[])
        with self.assertRaises(ValueError):a.validate_release(old,dict(status='complete',child_returncode=3))
    def test_actual_metadata_fixture_and_typed_pressure_time_gate(self):
        a.validate_release(self.release,dict(status='complete',child_returncode=3))
        for key,value in [('time',True),('time',float('nan')),('normal_pressure_samples',False),
                          ('normal_pressure_samples',0),('bad_pressure_samples',False),('bad_pressure_samples',1),
                          ('swap_growth_gb',True),('swap_growth_gb',float('inf')),('swap_growth_gb',.1),
                          ('peak_wired_compressor_gib',float('nan'))]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                a.validate_release(dict(self.release,**{key:value}),dict(status='complete',child_returncode=3))
    def test_release_gate_precedes_missing_or_unfinished_live_outcome(self):
        put(self.out/'root-release-proof.json',dict(self.release,guard_result=dict(status='running',child_returncode=3)))
        (self.out/'summary.json').unlink()
        with self.assertRaisesRegex(ValueError,'Original guard returncode differs'):self.check()
    def test_unfinished_summary_not_judged(self):
        put(self.out/'summary.json',dict(status='RUNNING',all_first_attempts_retained=True,arms=[]))
        with self.assertRaises(ValueError):self.check()
    def test_missing_duplicate_reordered_aggregate(self):
        original=json.loads((self.out/'summary.json').read_text())
        for arms in [original['arms'][:-1],original['arms'][:2]+[original['arms'][0]],list(reversed(original['arms']))]:
            put(self.out/'summary.json',dict(original,arms=arms))
            with self.assertRaises(ValueError):self.check()
    def test_wrong_first_attempt_command(self):
        p=self.out/'closing-add2-command.json';v=json.loads(p.read_text());v['argv'].append('retry');put(p,v)
        with self.assertRaises(ValueError):self.check()
    def test_cap_rc_not_repaired(self):
        put(self.out/'closing-add2-process.json',dict(returncode=0))
        with self.assertRaises(ValueError):self.check()
    def test_typed_original_counts_required(self):
        p=self.out/'summary.json';v=json.loads(p.read_text());v['arms'][1]['completed_correct']=True;put(p,v)
        with self.assertRaises(ValueError):self.check()
    def test_original_native_rng_drift_fails(self):
        arm=self.arms[2];row=c.jsonl(Path(arm['results_dir'])/'results.jsonl')[0]
        p=Path(arm['results_dir'])/row['diagnostics']['token_trace'];trace=c.jsonl(p);trace[0]['rng_after']='0000000000000000';fixture_source.lines(p,trace)
        with self.assertRaises(ValueError):self.check()
    def test_circular_changed_or_active_fails(self):
        target=Path(self.arms[1]['results_dir']);self.maker.out=target;self.maker.make(2,cap=True)
        with self.assertRaises(ValueError):self.check()
    def test_active_actual_delta_temperature_diagnostics_fail_tamper(self):
        target=Path(self.arms[1]['results_dir']);row=c.jsonl(target/'results.jsonl')[1]
        p=target/row['diagnostics']['token_trace'];trace=c.jsonl(p);trace[1024]['kl_raw_to_policy_pre_filter']=1.;fixture_source.lines(p,trace)
        with self.assertRaises(ValueError):self.check()
    def test_forbidden_model_hash(self):
        with self.assertRaises(ValueError):a.sha('/unused/model.gguf')
    def test_saved_cpu_fixture(self):
        # Retain one synthetic, model-free inactive-native + active-cap set.
        saved=ROOT/'speed-bench/ideas/artifacts/closing-analysis-saved-cpu-fixture-v2'
        if saved.exists():return
        saved.mkdir();put(saved/'fixture.json',self.fixture);put(saved/'expected.json',self.check())
        shutil.copytree(self.reference,saved/'native-reference')
        for arm in self.arms:shutil.copytree(arm['results_dir'],saved/arm['name'])

if __name__=='__main__':unittest.main()
