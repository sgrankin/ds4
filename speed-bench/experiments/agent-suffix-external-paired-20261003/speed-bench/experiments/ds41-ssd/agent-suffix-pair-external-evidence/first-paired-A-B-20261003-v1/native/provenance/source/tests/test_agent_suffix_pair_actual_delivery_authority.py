"""Corrective delivery authority fixtures; original seven are retained unchanged."""
import copy,importlib.util,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
WORK=Path(__file__).resolve().parents[1];HERE=WORK/'speed-bench/ideas'
spec=importlib.util.spec_from_file_location('actual_authority',HERE/'run_agent_suffix_pair_actual.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
class Authority(unittest.TestCase):
    def setUp(self):self.p=Path(tempfile.mkdtemp(prefix='actual-authority-',dir=HERE/'artifacts'))
    def grant(self):return dict(status='GRANTED',sole_model_grant=True,manifest_sha256='e'*64,command=c.command(self.p,'e'*64),time=1790980000.5)
    def save_grant(self,v):c.put(self.p/'root-launch-proof.json',v)
    def test_missing_or_forged_grant_before_first_model_identity(self):
        with patch.object(c,'verify',return_value={}),patch.object(c,'require_guard',return_value=None),patch.object(c,'model_identity',side_effect=AssertionError('model path touched')):
            with self.assertRaises(FileNotFoundError):c.run(self.p,'e'*64)
            v=self.grant();v['command']+=['forged'];self.save_grant(v)
            with self.assertRaises(c.ControlFailure):c.run(self.p,'e'*64)
    def test_exact_typed_finite_grant(self):
        good=self.grant();self.save_grant(good);self.assertEqual(c.require_root_grant(self.p,'e'*64),good)
        for key,value in [('status','PREPARED'),('sole_model_grant',1),('manifest_sha256','f'*64),('time',True),('time',0),('time',float('inf'))]:
            v=copy.deepcopy(good);v[key]=value;(self.p/'root-launch-proof.json').write_text(json.dumps(v))
            with self.assertRaises(c.ControlFailure):c.require_root_grant(self.p,'e'*64)
    def fixture(self,missing=None):
        names={'agent','task.json','grader.py','update_context.py','run_grader.py','run_agent_suffix_pair_actual.py','first.txt','second.txt','third.txt','caller-build-manifest.json'}|{'metal/'+n for n in c.SHADERS.values()}
        for name in names:
            f=self.p/name;f.parent.mkdir(parents=True,exist_ok=True)
            f.write_bytes((HERE/'agent_suffix_pair_actual_v1/task.json').read_bytes() if name=='task.json' else b'CPU fixture only; never executed\n')
        (self.p/'agent').chmod(0o755)
        files={name:dict(bytes=(self.p/name).stat().st_size,sha256=c.sha(self.p/name)) for name in names if name!=missing}
        plan=dict(corrective_infrastructure_attempt=c.CORRECTION,task_sha256=c.TASK_SHA,model=c.MODEL,policy=c.POLICY,turns=3,timeout_seconds=900,files=files,shader_env={k:str(self.p/'metal'/n) for k,n in c.SHADERS.items()})
        (self.p/'manifest.json').write_text(json.dumps(plan));digest=c.sha(self.p/'manifest.json');(self.p/'prepared-command.json').write_text(json.dumps(dict(manifest_sha256=digest,command=c.command(self.p,digest),cwd=str(c.ROOT))));return digest
    def test_all26_shader_payloads_mandatory_before_model_identity(self):
        digest=self.fixture('metal/dense.metal')
        with patch.object(c,'model_identity',side_effect=AssertionError('model touched')):
            with self.assertRaisesRegex(ValueError,'Required assets missing'):c.run(self.p,digest)
    def test_complete_nonmodel_manifest_accepted(self):
        digest=self.fixture();self.assertEqual(len(c.verify(self.p,digest)['files']),36)
    def test_manifest_duplicate_and_nonfinite_fields_rejected(self):
        for text in ['{"files":{},"files":{}}','{"files":NaN}']:
            (self.p/'manifest.json').write_text(text);digest=c.sha(self.p/'manifest.json')
            with self.assertRaises(c.ControlFailure):c.verify(self.p,digest)
    def test_prepared_command_duplicate_field_rejected(self):
        digest=self.fixture();(self.p/'prepared-command.json').write_text('{"command":[],"command":[]}')
        with self.assertRaises(c.ControlFailure):c.verify(self.p,digest)
    def test_single_child_prelaunch_guard_and_grant_recheck(self):
        body=(HERE/'run_agent_suffix_pair_actual.py').read_text().split('def run(p,digest):',1)[1].split("if __name__",1)[0]
        before=body.split('identity=model_identity()',1)[0];self.assertIn('require_guard(p,digest);require_root_grant(p,digest)',before)
        before_drive=body.split('rc=drive(',1)[0];self.assertIn('require_guard(p,digest);require_root_grant(p,digest) # Recheck the one child',before_drive)
    def test_missing_or_forged_corrective_attempt_tag_refused(self):
        digest=self.fixture();plan=c.load_json(self.p/'manifest.json')
        for kind in ('missing','original','different'):
            changed=copy.deepcopy(plan)
            if kind=='missing':changed.pop('corrective_infrastructure_attempt')
            elif kind=='original':changed['corrective_infrastructure_attempt']['attempt']=0
            else:changed['corrective_infrastructure_attempt']['original_manifest_sha256']='f'*64
            (self.p/'manifest.json').write_text(json.dumps(changed));digest=c.sha(self.p/'manifest.json')
            with self.assertRaisesRegex(ValueError,'provenance'):c.verify(self.p,digest)
if __name__=='__main__':unittest.main(verbosity=2)
