"""Fixed160MiB admission experiment authority and budget CPU controls."""
import copy,importlib.util,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
WORK=Path(__file__).resolve().parents[1];HERE=WORK/'speed-bench/ideas'
spec=importlib.util.spec_from_file_location('actual_authority',HERE/'run_agent_suffix_pair_actual_memory160.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
class Authority(unittest.TestCase):
    def setUp(self):self.p=Path(tempfile.mkdtemp(prefix='actual-authority-',dir=Path(os.environ['PAIR_MEMORY160_CPU_CASES'])))
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
        names={'agent','task.json','grader.py','update_context.py','run_grader.py','run_agent_suffix_pair_actual_memory160.py','first.txt','second.txt','third.txt','caller-build-manifest.json'}|{'metal/'+n for n in c.SHADERS.values()}
        for name in names:
            f=self.p/name;f.parent.mkdir(parents=True,exist_ok=True)
            f.write_bytes((HERE/'agent_suffix_pair_memory160_v1/task.json').read_bytes() if name=='task.json' else b'CPU fixture only; never executed\n')
        (self.p/'agent').chmod(0o755)
        files={name:dict(bytes=(self.p/name).stat().st_size,sha256=c.sha(self.p/name)) for name in names if name!=missing}
        plan=dict(memory_admission_experiment=c.EXPERIMENT,task_sha256=c.TASK_SHA,model=c.MODEL,policy=c.POLICY,turns=3,timeout_seconds=900,files=files,shader_env={k:str(self.p/'metal'/n) for k,n in c.SHADERS.items()})
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
        body=(HERE/'run_agent_suffix_pair_actual_memory160.py').read_text().split('def run(p,digest):',1)[1].split("if __name__",1)[0]
        before=body.split('identity=model_identity()',1)[0];self.assertIn('require_guard(p,digest);require_root_grant(p,digest)',before)
        before_drive=body.split('rc=drive(',1)[0];self.assertIn('require_guard(p,digest);require_root_grant(p,digest) # Recheck the one child',before_drive)
    def test_missing_or_forged_corrective_attempt_tag_refused(self):
        digest=self.fixture();plan=c.load_json(self.p/'manifest.json')
        for kind in ('missing','original','different'):
            changed=copy.deepcopy(plan)
            if kind=='missing':changed.pop('memory_admission_experiment')
            elif kind=='original':changed['memory_admission_experiment']['status']='CORRECTIVE_INFRASTRUCTURE_ATTEMPT'
            else:changed['memory_admission_experiment']['prior_fixed128_prepared_manifest_sha256']='f'*64
            (self.p/'manifest.json').write_text(json.dumps(changed));digest=c.sha(self.p/'manifest.json')
            with self.assertRaisesRegex(ValueError,'provenance'):c.verify(self.p,digest)

    def test_only_task_budget_and_provenance_change_all_literals_golds_identical(self):
        original=c.load_json(HERE/'agent_suffix_pair_actual_v1/task.json');new=c.load_json(HERE/'agent_suffix_pair_memory160_v1/task.json')
        reconstructed=copy.deepcopy(new);reconstructed.pop('memory_admission_experiment');reconstructed['policy']['DS4_AGENT_CONTEXT_FRONTIER_MB']='128'
        for key in [('edit','geometry_selection_stage'),('success_components','structural')]:
            reconstructed[key[0]][key[1]]=reconstructed[key[0]][key[1]].replace('<=160MiB','<=128MiB')
        self.assertEqual(original,reconstructed)
        for name in ['first.txt','second.txt','third.txt','grader.py','update_context.py','run_grader.py']:
            self.assertEqual((HERE/'agent_suffix_pair_actual_v1'/name).read_bytes(),(HERE/'agent_suffix_pair_memory160_v1'/name).read_bytes())
        self.assertEqual(c.sha(HERE/'agent_suffix_pair_memory160_v1/task.json'),c.TASK_SHA)
        self.assertEqual(c.POLICY['DS4_AGENT_CONTEXT_FRONTIER_MB'],'160');self.assertEqual(c.POLICY['DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL'],'128')

    def test_private_caller_delta_is_only_include_and_budget_macro(self):
        old=(WORK/'speed-bench/agent_suffix_pair_actual.c').read_text();new=(WORK/'speed-bench/agent_suffix_pair_actual_memory160.c').read_text()
        self.assertEqual(new,old.replace('agent_suffix_pair_actual_hook.inc','agent_suffix_pair_actual_memory160_hook.inc'))
        old=(WORK/'speed-bench/agent_suffix_pair_actual_hook.inc').read_text();new=(WORK/'speed-bench/agent_suffix_pair_actual_memory160_hook.inc').read_text()
        self.assertEqual(new,old.replace('(UINT64_C(128)<<20)','(UINT64_C(160)<<20)'))
        for name in ['ds4_agent.c','ds4.c','ds4_metal.m','ds4.h','speed-bench/agent_suffix_pair_actual_literals.h','speed-bench/agent_suffix_pair_actual_ownership.h','speed-bench/agent_suffix_pair_actual_admission.h']:
            historical=Path('/Volumes/Models/ds4/.workspaces/idea-agent-suffix-pair-delivery')/name
            self.assertEqual((WORK/name).read_bytes(),historical.read_bytes(),name)

    def test_shared_admission_writer160_boundaries_and_actual128_failure(self):
        import subprocess,sys
        root=self.p/'memory';root.mkdir()
        probe=Path(os.environ['PAIR_MEMORY160_CPU_PROBE']);r=subprocess.run([str(probe),'--memory-selftest',str(root)],capture_output=True,text=True,timeout=5)
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn('old128 rejection: PASS',r.stdout)
        rows=[c.decode_json(line) for line in (root/'admission-ledger.jsonl').read_bytes().splitlines()];self.assertEqual(len(rows),3)
        self.assertEqual([r['admitted'] for r in rows],[True,True,False])
        for row in rows:
            terms=[row[k] for k in ('worker_owned_bytes','text_requested_bytes','splice_forecast_bytes','canonical_undo_forecast_bytes','undo_terminator_forecast_bytes','checkpoint_token_forecast_bytes','scratch_reserved_bytes')]
            self.assertTrue(all(type(x) is int and x>=0 for x in terms));self.assertEqual(row['budget_bytes'],160<<20)
            self.assertEqual(row['admitted'],sum(terms)<=160<<20);self.assertIs(row['canonical_undo_allocated_by_selector'],False);self.assertIs(row['observed_global_peak'],False)
        self.assertEqual(sum(rows[0][k] for k in ('worker_owned_bytes','temporary_forecast_bytes','scratch_reserved_bytes')),137150794)
        self.assertGreater(137150794,128<<20);self.assertLess(137150794,160<<20)
        proof=Path('/Volumes/Models/ds4/.workspaces/idea-agent-suffix-pair-delivery/speed-bench/ideas/artifacts/agent-suffix-pair-actual-delivery-prepared-20261003-v2/run/private/admission-ledger.jsonl')
        self.assertEqual(c.sha(proof),c.EXPERIMENT['prior_fixed128_admission_ledger_sha256'])
        old=[c.decode_json(x) for x in proof.read_bytes().splitlines()];self.assertIs(old[1]['admitted'],False);self.assertEqual(old[1]['budget_bytes'],128<<20)

    def test_production_protocol_gate_and_delivered_bytes_stay_exact(self):
        import subprocess,re
        probe=Path(os.environ['PAIR_MEMORY160_CPU_PROBE']);r=subprocess.run([str(probe),'--gate-selftest'],capture_output=True,text=True,timeout=5)
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn('PASS10',r.stdout)
        body=lambda s:re.search(r'^def delivery_payload[(]prompt[)]:.*?(?=^def )',s,re.M|re.S).group(0)
        self.assertEqual(body((HERE/'run_agent_suffix_pair_actual.py').read_text()),body((HERE/'run_agent_suffix_pair_actual_memory160.py').read_text()))
        task=(HERE/'agent_suffix_pair_memory160_v1/third.txt').read_bytes();self.assertEqual(len(task),1798)
        literal=c.decode_json(re.search(r'pair_actual_third_user = (".*");',(WORK/'speed-bench/agent_suffix_pair_actual_literals.h').read_text()).group(1)).encode()
        self.assertEqual(c.delivery_payload(task.decode()),literal);self.assertEqual(literal,task+b'\n')

    def test_budget160_parser_rejects_old128_and_accepts_new_bound(self):
        import struct
        p=self.p/'selection';p.mkdir();user=128000;old=list(range(100))+[12,13,14]+list(range(200,264))+[user]+list(range(400,527));edited=list(range(100))+list(range(200,264))+[user]+list(range(400,527))
        for name,tokens in [('selected-old.ids',old),('selected-edited.ids',edited)]:
            (p/name).write_bytes(struct.pack('<4I',0x41535031,1,len(tokens),user)+struct.pack('<'+'i'*len(tokens),*tokens))
        task=c.load_json(HERE/'agent_suffix_pair_memory160_v1/task.json');before=task['first_user_text']+'\nRECENT\n'+task['third_user_text'];after=c.apply_frozen_text(before,task,0)
        (p/'selected-before.txt').write_text(before);(p/'selected-after.txt').write_text(after)
        selection=dict(status='SELECTED_BEFORE_THIRD_EVAL',candidate=0,displacement=3,source_start=103,source_end=167,destination_start=100,destination_end=164,reused=64,fresh=128,latest_user_start=164,user_id=user,old_tokens=len(old),edited_tokens=len(edited),budget=160<<20,preallocation_peak_with_reserves=137150794,current_payload_forecast=30140128,actual_commit_rechecks_admission=True,old_text_bytes=len(before.encode()),edited_text_bytes=len(after.encode()))
        c.put(p/'selection.json',selection);c.put(p/'candidate-00.json',dict(candidate=0,replaced=True,spliced=True,displacement=3,eligible=True));self.assertEqual(c.validate_selection(p,task)['selection']['budget'],160<<20)
        for key,value in [('budget',128<<20),('preallocation_peak_with_reserves',(160<<20)+1),('budget',True)]:
            bad=copy.deepcopy(selection);bad[key]=value;(p/'selection.json').write_text(json.dumps(bad))
            with self.assertRaises(c.ControlFailure):c.validate_selection(p,task)

if __name__=='__main__':unittest.main(verbosity=2)
