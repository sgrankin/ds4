import copy
import errno
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'speed-bench'))
import run_native_effort_holdout as r
spec=importlib.util.spec_from_file_location('synthetic',ROOT/'tests/test_analyze_native_effort_holdout.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
PREP=ROOT/'speed-bench/ideas/artifacts/native-effort-holdout-prepared-20261003-v1'

class Driver(unittest.TestCase):
    def setUp(self):
        base=ROOT/'speed-bench/ideas/artifacts/holdout-driver-cpu';base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.base=Path(self.tmp.name)
        self.plan=json.loads((PREP/'plan.json').read_text());self.plan['workspace']=str(self.base);self.plan['run_cwd']=str(self.base)
        self.plan['output']=str(self.base/'output');self.plan['pressure']=str(self.base/'pressure')
        for arm in self.plan['arms']:arm['results_dir']=str(self.base/'speed-bench/ideas/artifacts'/arm['run_name'])
        self.plan_path=self.base/'plan.json';self.plan_path.write_text(json.dumps(self.plan)+'\n')
        self.expected=hashlib.sha256(self.plan_path.read_bytes()).hexdigest()
        self.patches=[mock.patch.object(r,'__file__',str(PREP/'run_native_effort_holdout.py')),
                      mock.patch.object(r.analysis,'__file__',str(PREP/'analyze_native_effort_holdout.py'))]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()
    def test_complete_36_attempts_all_first_caps_wrong_and_repeats(self):
        validated={}
        for level in [0,25,75]:
            p=self.base/str(level);p.mkdir();s.synthetic(p,level);validated[level]=r.analysis.validate_arm(p,level,s.fixture,3)
        calls=[]
        def guard(*args):calls.append('guard');return {'mock_guard':True}
        def assets(*args):calls.append('assets')
        def identity(*args):self.assertEqual(calls[-2:],['guard','assets']);calls.append('modelstat');return self.plan['model']
        process=mock.Mock(pid=123);process.wait.return_value=3
        with mock.patch.object(r,'require_guard',side_effect=guard),mock.patch.object(r,'check_assets',side_effect=assets),\
             mock.patch.object(r,'model_identity',side_effect=identity),mock.patch.object(r.subprocess,'Popen',return_value=process) as popen,\
             mock.patch.object(r.analysis,'validate_arm',side_effect=lambda path,level,fixture,rc:copy.deepcopy(validated[level])),\
             mock.patch.dict(r.os.environ,{'DS4':'poison','DS4_UNEXPECTED':'poison'},clear=False):
            self.assertEqual(r.run(self.plan_path,self.expected),3)
        self.assertEqual(popen.call_count,6)
        for call,arm in zip(popen.call_args_list,self.plan['arms']):
            self.assertEqual(call.args[0],arm['argv']);env=call.kwargs['env']
            self.assertNotIn('DS4',env);self.assertNotIn('DS4_UNEXPECTED',env)
            self.assertEqual({k:v for k,v in env.items() if k.startswith('DS4')},self.plan['environment'])
        result=json.loads((self.base/'output/summary.json').read_text())
        self.assertEqual(result['exact_repeated_native_levels'],[75,25,0]);self.assertEqual(result['first_attempts'],36)
        self.assertTrue(result['all_first_attempts_retained'])
    def test_unpinned_manifest_rejected_before_guard_or_modelstat(self):
        self.plan_path.write_text('{}\n')
        with mock.patch.object(r,'require_guard') as guard,mock.patch.object(r,'model_identity') as stat:
            with self.assertRaisesRegex(ValueError,'authoritative manifest'):r.run(self.plan_path,self.expected)
            guard.assert_not_called();stat.assert_not_called()
    def test_missing_mandatory_or_changed_settings_rejected(self):
        changes=[('binary',None),('fixture',None),('guard',None),('build_manifest',None)]
        for key,_ in changes:
            bad=copy.deepcopy(self.plan);bad['asset_sha256'].pop(bad[key])
            with self.subTest(key=key),self.assertRaises(ValueError):r.validate_plan(bad)
        bad=copy.deepcopy(self.plan);bad['settings']['native_penalty']=False
        with self.assertRaises(ValueError):r.validate_plan(bad)
        bad=copy.deepcopy(self.plan);bad['arms'][0]['level']=False
        with self.assertRaises(ValueError):r.validate_plan(bad)
    def test_exact_guard_suffix_framework_lock_and_absolute_ps_timeout(self):
        frame=self.plan['guard_interpreter_identity']['path'];tail=[self.plan['guard'],self.plan['pressure'],'--',self.plan['launch_python'],
            str(PREP/'run_native_effort_holdout.py'),str(self.plan_path.resolve()),self.expected]
        good=[frame]+tail
        locked=OSError(errno.EAGAIN,'mock held MAIN lock')
        with mock.patch.object(r,'sha',side_effect=lambda p:self.plan['guard_sha256'] if p==self.plan['guard'] else self.plan['guard_interpreter_identity']['sha256']),\
             mock.patch.object(r.os.path,'samefile',return_value=True),mock.patch.object(Path,'open',return_value=mock.MagicMock()),\
             mock.patch.object(r.fcntl,'flock',side_effect=locked),mock.patch.object(r.subprocess,'check_output',return_value=shlex.join(good)) as ps:
            self.assertTrue(r.require_guard(self.plan,self.plan_path,self.expected)['guard_lock_held'])
            self.assertEqual(ps.call_args.args[0][0],'/bin/ps');self.assertEqual(ps.call_args.kwargs['timeout'],2)
            for wrong in [[frame,'unrelated',*tail],[frame,*tail,'extra'],['/foreign-python',*tail]]:
                ps.return_value=shlex.join(wrong)
                # Foreign interpreter is independently rejected even with exact tail.
                with mock.patch.object(r.os.path,'samefile',return_value=wrong[0]==frame):
                    with self.assertRaises(ValueError):r.require_guard(self.plan,self.plan_path,self.expected)
    def test_guard_rejection_precedes_any_model_access(self):
        with mock.patch.object(r,'require_guard',side_effect=ValueError('no guard')),mock.patch.object(r,'check_assets') as assets,\
             mock.patch.object(r,'model_identity') as stat,mock.patch.object(r.subprocess,'Popen') as process:
            with self.assertRaisesRegex(ValueError,'no guard'):r.run(self.plan_path,self.expected)
            assets.assert_not_called();stat.assert_not_called();process.assert_not_called()
    def test_control_failure_stops_dependent_arms_retains_first_attempt(self):
        p=self.base/'synthetic';p.mkdir();s.synthetic(p,75);valid=r.analysis.validate_arm(p,75,s.fixture,3)
        process=mock.Mock(pid=123);process.wait.return_value=3
        with mock.patch.object(r,'require_guard',return_value={}),mock.patch.object(r,'check_assets'),\
             mock.patch.object(r,'model_identity',return_value=self.plan['model']),mock.patch.object(r.subprocess,'Popen',return_value=process) as popen,\
             mock.patch.object(r.analysis,'validate_arm',side_effect=lambda path,level,fixture,rc:copy.deepcopy(valid)),\
             mock.patch.object(r.analysis,'require_repeat',side_effect=ValueError('repeat drift')):
            with self.assertRaisesRegex(ValueError,'repeat drift'):r.run(self.plan_path,self.expected)
        self.assertEqual(popen.call_count,4);self.assertTrue((self.base/'output/effort75-after-process.json').is_file())
        self.assertTrue((self.base/'output/first-failure.json').is_file());self.assertFalse((self.base/'output/summary.json').exists())

if __name__=='__main__':unittest.main()
