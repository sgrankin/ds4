#!/usr/bin/env python3
"""Model-free actual-task protocol, frozen-edit and geometry fixtures.

Tiny fake children and disposable project files only. No model path metadata,
engine, compiler, native caller or sampler is touched.
"""
import copy,hashlib,importlib.util,json,os,struct,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
WORK=HERE.parent.parent
DRAFT=HERE/'agent_suffix_pair_external_v1'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ctl=module('actual_controller',HERE/'run_agent_suffix_pair_external.py')
grader=module('actual_grader',DRAFT/'grader.py')
TASK=json.loads((DRAFT/'task.json').read_text())
ROOT_CASES=Path(os.environ.get('PAIR_ACTUAL_CPU_CASES',str(HERE/'artifacts/agent-suffix-pair-external-cpu-v1/cases')))
ROOT_CASES.mkdir(parents=True,exist_ok=True)

def row(turn=1,round=0,tool=False):
    return dict(turn=turn,tool_round=round,generated=1,carried=0,limit=512,configured_limit=512,
      context_limited=False,model_stopped=not tool,got_tool=tool,malformed_tool=False,
      early_tool_error=False,interrupted=False,at_limit=False,
      calls=[dict(name='bash',args=[dict(name='command',value='python3 update_context.py',is_string=True)])] if tool else [])

def write_ids(path,ids,user=128000):
    path.write_bytes(struct.pack('<4I',0x41535031,1,len(ids),user)+struct.pack('<'+'i'*len(ids),*ids))

def selection_fixture(p,candidate=2):
    # Distinct small IDs plus entire latest User region fresh; deletion is odd.
    user=128000;prefix=list(range(100));tail=list(range(200,264))+[user]+list(range(400,527))
    old=prefix+[12,13,14]+tail;edited=prefix+tail
    write_ids(p/'selected-old.ids',old,user);write_ids(p/'selected-edited.ids',edited,user)
    before=TASK['first_user_text']+'\nRECENT TOOL AND USER HISTORY\n'+TASK['third_user_text']
    after=ctl.apply_frozen_text(before,TASK,candidate)
    (p/'selected-before.txt').write_text(before);(p/'selected-after.txt').write_text(after)
    s=dict(status='SELECTED_BEFORE_THIRD_EVAL',candidate=candidate,displacement=3,source_start=103,source_end=167,
      destination_start=100,destination_end=164,reused=64,fresh=128,latest_user_start=164,user_id=user,
      old_tokens=len(old),edited_tokens=len(edited),budget=160<<20,preallocation_peak_with_reserves=110000000,
      current_payload_forecast=40000000,actual_commit_rechecks_admission=True,old_text_bytes=len(before.encode()),edited_text_bytes=len(after.encode()))
    (p/'selection.json').write_text(json.dumps(s))
    for i in range(candidate+1):
        (p/f'candidate-{i:02d}.json').write_text(json.dumps(dict(candidate=i,replaced=True,spliced=True,displacement=3,eligible=i==candidate)))
    return s

FAKE=r'''
import hashlib,json,os,sys,time
from pathlib import Path
p=Path(os.environ['FAKE_PRIVATE']);trace=Path(os.environ['FAKE_TRACE']);project=Path(os.environ['FAKE_PROJECT']);mode=os.environ['FAKE_MODE']
def marker():
 os.write(1,b'+DWARFSTAR_');time.sleep(.003);os.write(1,b'WAITING\n')
def record(turn,answer):
 r=dict(turn=turn,tool_round=0,generated=1,carried=0,limit=512,configured_limit=512,context_limited=False,model_stopped=True,got_tool=False,malformed_tool=False,early_tool_error=False,interrupted=False,at_limit=False,calls=[])
 if mode=='cap' and turn==2:r.update(generated=512,at_limit=True)
 if not(mode=='missinground' and turn==2):
  with (p/'rounds.jsonl').open('a') as f:json.dump(r,f);f.write('\n')
 with trace.open('a') as f:
  f.write('trace user="literal"\n');f.write('prefill sync done tool_round=0\n')
  raw=answer.encode();f.write(' token index=1 id=3 bytes='+str(len(raw))+' text="literal" hex='+raw.hex()+'\n')
  if not(mode=='truncatedtrace' and turn==1):f.write('generation finished tool_round=0 generated=1\n')
 os.write(1,b'PAIR_ACTUAL_ROUND native fake protocol fixture\n')
marker()
for turn in range(1,4):
 payload=os.read(0,4096)
 if not payload:sys.exit(11)
 (p/('prompt-'+str(turn)+'.bin')).write_bytes(payload)
 answer='CODE=9999' if mode=='wrong' and turn==1 else 'CODE=2718' if turn==1 else 'READY' if turn==2 else 'PASS'
 if mode=='mutation' and turn==2:(project/'run_grader.py').write_text('modified')
 if mode=='eligiblefail' and turn==3:
  (p/'selection-failure.txt').write_text('No frozen candidate eligible')
  sys.exit(1)
 if mode=='infraexit' and turn==2:sys.exit(9)
 record(turn,answer);marker()
 if mode=='duplicate' and turn==3:marker()
if mode=='hangeof':time.sleep(30)
else:
 while os.read(0,4096):pass
'''

class Disposable(unittest.TestCase):
    def setUp(self):self.p=Path(tempfile.mkdtemp(prefix=self._testMethodName+'-',dir=ROOT_CASES))
    def bad(self,fn,*args):
        with self.assertRaises((ctl.ControlFailure,ValueError,KeyError,TypeError)):fn(*args)

class TypedGeometry(Disposable):
    def test_valid_odd_full_user_and_first_eligible(self):
        s=selection_fixture(self.p);self.assertEqual(ctl.validate_selection(self.p,TASK)['selection'],s)
    def test_malformed_geometry_and_budget(self):
        baseline=selection_fixture(self.p)
        for key,value in [('displacement',2),('candidate',True),('fresh',127),('latest_user_start',165),('destination_end',165),('budget',129<<20),('preallocation_peak_with_reserves',(160<<20)+1),('current_payload_forecast',0),('actual_commit_rechecks_admission',1)]:
            s=copy.deepcopy(baseline);s[key]=value;(self.p/'selection.json').write_text(json.dumps(s));self.bad(ctl.validate_selection,self.p,TASK)
        (self.p/'selection.json').write_text(json.dumps(baseline))
    def test_truncated_ids_and_literal_corruption(self):
        selection_fixture(self.p);data=(self.p/'selected-edited.ids').read_bytes();(self.p/'selected-edited.ids').write_bytes(data[:-1]);self.bad(ctl.validate_selection,self.p,TASK)
        (self.p/'selected-edited.ids').write_bytes(data);(self.p/'selected-after.txt').write_text('different');self.bad(ctl.validate_selection,self.p,TASK)
    def test_candidate_order_cannot_skip_or_fake_splice(self):
        selection_fixture(self.p)
        p=self.p/'candidate-00.json';c=json.loads(p.read_text());c['eligible']=True;p.write_text(json.dumps(c));self.bad(ctl.validate_selection,self.p,TASK)
        c['eligible']=False;p.write_text(json.dumps(c));p=self.p/'candidate-02.json';c=json.loads(p.read_text());c['spliced']=False;p.write_text(json.dumps(c));self.bad(ctl.validate_selection,self.p,TASK)
    def test_opaque_id_bounds_and_duplicate_json(self):
        selection_fixture(self.p);data=bytearray((self.p/'selected-old.ids').read_bytes());struct.pack_into('<i',data,16,-1);(self.p/'selected-old.ids').write_bytes(data);self.bad(ctl.validate_selection,self.p,TASK)
        self.bad(ctl.decode_json,'{"x":1,"x":2}');self.bad(ctl.decode_json,'{"x":NaN}')
    def test_actual_commit_geometry_bound(self):
        s=selection_fixture(self.p);trace=self.p/'trace';line='context pair suffix APPROXIMATE reused=64 fresh=128 source=103:167 destination=100:164 rebuilt_pairs=96 boundary_pairs=3 copied_bytes=12 unique_peak=115000000 budget=167772160 latest_user_start=164 latest_user_fresh=1\n';trace.write_text(line)
        self.assertEqual(ctl.structural_trace(trace,s)['fresh'],128)
        for changed in [line.replace('destination=100','destination=102'),line+line,line.replace('latest_user_fresh=1','latest_user_fresh=0'),'safe exact fallback']:
            trace.write_text(changed);self.bad(ctl.structural_trace,trace,s)

class NativeObservations(Disposable):
    def test_round_order_and_stop(self):
        records=[row(1),row(2),row(3,0,True),row(3,1)];self.assertEqual(ctl.validate_rounds(records,3),records)
        for bad in [records[:1]+records[:1],records[1:],records+[row(3,1)],records[:-1]]:self.bad(ctl.validate_rounds,bad,3)
    def test_typed_limit_and_tool_flags(self):
        baseline=row()
        for key,value in [('generated',True),('limit','512'),('model_stopped',1),('context_limited',True),('interrupted',True),('malformed_tool',True),('early_tool_error',True)]:
            r=copy.deepcopy(baseline);r[key]=value;self.bad(ctl.validate_rounds,[r])
        r=row();r.update(generated=512,at_limit=True);self.bad(ctl.validate_rounds,[r])
        r=row();r['got_tool']=True;self.bad(ctl.validate_rounds,[r])
    def test_jsonl_truncation_duplicate_fields(self):
        path=self.p/'rounds';path.write_text(json.dumps(row()));self.bad(ctl.rounds,path)
        path.write_text('{"turn":1,"turn":2}\n');self.bad(ctl.rounds,path)
    def test_token_trace_lossless_and_truncated(self):
        path=self.p/'trace';path.write_text('trace user="x"\nprefill sync done tool_round=0\n token index=1 id=3 bytes=9 text="x" hex=434f44453d32373138\ngeneration finished tool_round=0 generated=1\n')
        self.assertEqual(ctl.generated_text(path)[0]['text'],'CODE=2718')
        for text in [path.read_text().replace('index=1','index=2'),path.read_text().replace('id=3','id=-1'),path.read_text().replace('bytes=9','bytes=8'),path.read_text().split('generation finished')[0],path.read_text().replace('434f44453d32373138','ff') .replace('bytes=9','bytes=1')]:
            path.write_text(text);self.bad(ctl.generated_text,path)

class TinyProtocol(Disposable):
    def invoke(self,mode='good',checker=None,timeout=3,cleanup_reserve=1.2):
        output=self.p/'output';output.mkdir();private=self.p/'private';private.mkdir();project=self.p/'project';project.mkdir();trace=self.p/'trace';trace.write_text('');(project/'run_grader.py').write_text('immutable')
        fake=self.p/'fake.py';fake.write_text(FAKE)
        env=dict(os.environ,FAKE_PRIVATE=str(private),FAKE_TRACE=str(trace),FAKE_PROJECT=str(project),FAKE_MODE=mode,PYTHONDONTWRITEBYTECODE='1')
        prompts=['first\nmultiline\n','second\nmultiline\n','third\nmultiline\n']
        inputs={'run_grader.py':ctl.sha(project/'run_grader.py')}
        check=checker or (lambda records,deadline:dict(turn=3,passed=True))
        rc=ctl.drive([sys.executable,str(fake)],prompts,output,project,private,trace,env,inputs,check,timeout,cleanup_reserve)
        result=json.loads((output/'controller-outcome.json').read_text())
        return rc,result,private,output
    def test_three_ordered_completions_single_multiline_write(self):
        rc,r,p,_=self.invoke();self.assertEqual(rc,0);self.assertEqual(r['completed'],[1,2,3]);self.assertTrue(r['overall_success']);self.assertTrue(r['cleanup']['leader_reaped'])
        for i,name in enumerate(['first','second','third'],1):self.assertEqual((p/f'prompt-{i}.bin').read_bytes(),(name+'\nmultiline\n\n').encode())
        self.assertEqual(r['persistence'],'NOT_EXERCISED')
    def test_wrong_control_first_failure_not_repaired(self):
        rc,r,p,o=self.invoke('wrong');self.assertEqual(rc,3);self.assertEqual(r['submitted'],1);self.assertFalse(r['overall_success']);self.assertFalse((p/'prompt-2.bin').exists());self.assertTrue((o/'traceback.txt').exists())
    def test_fixture_mutation_is_separate_overall_failure(self):
        rc,r,_,_=self.invoke('mutation');self.assertEqual(rc,3);self.assertEqual(r['completed'],[1]);self.assertIn('Fixture mutation',r['error'])
    def test_observed_round_limit(self):
        rc,r,_,_=self.invoke('cap');self.assertEqual(rc,3);self.assertEqual(r['completed'],[1]);self.assertIn('generation-limit',r['error'])
    def test_missing_round_and_selection_failure(self):
        rc,r,_,_=self.invoke('missinground');self.assertEqual(rc,3);self.assertEqual(r['completed'],[1])
    def test_eligibility_failure_is_retained(self):
        rc,r,p,_=self.invoke('eligiblefail');self.assertEqual(rc,3);self.assertTrue((p/'selection-failure.txt').exists());self.assertEqual(r['completed'],[1,2])
    def test_duplicate_completed_marker(self):
        rc,r,_,_=self.invoke('duplicate');self.assertEqual(rc,3);self.assertEqual(r['completed'],[1,2,3]);self.assertFalse(r['overall_success']);self.assertIn('Duplicate WAITING',r['error'])
    def test_infrastructure_exit_not_semantic_failure(self):
        rc,r,_,_=self.invoke('infraexit');self.assertEqual(rc,1);self.assertEqual(r['agent_returncode'],9)
    def test_truncated_trace(self):
        rc,r,_,_=self.invoke('truncatedtrace');self.assertEqual(rc,3);self.assertEqual(r['completed'],[])
    def test_whole_deadline_includes_eof_cleanup(self):
        begin=time.monotonic();rc,r,_,_=self.invoke('hangeof',timeout=1.8,cleanup_reserve=1.2);self.assertEqual(rc,124);self.assertEqual(r['completed'],[1,2,3]);self.assertTrue(r['cleanup']['leader_reaped']);self.assertLess(time.monotonic()-begin,1.8)
    def test_whole_deadline_includes_external_checker(self):
        def check(records,deadline):ctl.checked_command([sys.executable,'-c','import time;time.sleep(30)'],deadline)
        begin=time.monotonic();rc,r,_,_=self.invoke(checker=check,timeout=2.1,cleanup_reserve=1.2);self.assertEqual(rc,124);self.assertEqual(r['completed'],[1,2]);self.assertTrue(r['cleanup']['leader_reaped']);self.assertLess(time.monotonic()-begin,2.1)

class OwnershipAndAuthority(Disposable):
    def test_timeout_pipe_holder_keeps_birth_anchor_and_closes_handles(self):
        actual_popen=ctl.subprocess.Popen;owned=[]
        def capture(*args,**kwargs):
            p=actual_popen(*args,**kwargs);owned.append(p);return p
        # Leader exits without being reaped while its sleeping child holds the
        # pipes. The owned group is signaled before wait; no old reaped PGID is
        # reused, and both checker pipes close on the timeout path.
        script='import subprocess,sys;subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);sys.exit(0)'
        events=[];actual_kill=ctl.os.killpg
        def kill(pid,sig):
            self.assertIsNone(owned[0].returncode);events.append((pid,sig));return actual_kill(pid,sig)
        with patch.object(ctl.subprocess,'Popen',capture),patch.object(ctl.os,'killpg',kill):
            with self.assertRaises(subprocess.TimeoutExpired):ctl.checked_command([sys.executable,'-c',script],time.monotonic()+.15)
        self.assertTrue(events);self.assertIsNotNone(owned[0].returncode);self.assertTrue(owned[0].stdout.closed);self.assertTrue(owned[0].stderr.closed)
    def test_signals_all_before_first_reap(self):
        events=[]
        class Proc:
            pid=123;returncode=None
            def wait(self,timeout):events.append(('wait',timeout));self.returncode=-9
        with patch.object(ctl.os,'killpg',lambda pid,sig:events.append(('signal',pid,sig))),patch.object(ctl.time,'sleep',lambda seconds:events.append(('grace',seconds))):ctl.stop_process(Proc())
        self.assertEqual([e[0] for e in events],['signal','grace','signal','wait']);self.assertEqual(events[-1],('wait',5))
    def test_already_reaped_leader_never_signaled(self):
        class Proc:pid=123;returncode=0
        with patch.object(ctl.os,'killpg',side_effect=AssertionError('unowned old PGID')),patch.object(ctl.time,'sleep',side_effect=AssertionError('unneeded sleep')):ctl.stop_process(Proc())
    def test_guard_rejection_precedes_model_access(self):
        with patch.object(ctl,'verify',return_value={}),patch.object(ctl,'require_guard',side_effect=RuntimeError('unguarded')),patch.object(ctl,'model_identity',side_effect=AssertionError('model metadata touched')):
            with self.assertRaisesRegex(RuntimeError,'unguarded'):ctl.run(self.p,'pin')
    def test_parent_inspection_absolute_bounded(self):
        def ps(argv,**kwargs):
            self.assertEqual(argv[0],'/bin/ps');self.assertEqual(kwargs['timeout'],2);return 'unowned parent'
        with patch.object(ctl.subprocess,'check_output',ps):
            with self.assertRaises(RuntimeError):ctl.require_guard(self.p,'pin')



class ExternalSpecific(Disposable):
    def test_frozen_new_delivery_exact_literal_and_one_protocol_LF(self):
        import ast,re
        prompt=(DRAFT/'third.txt').read_bytes()
        self.assertEqual(len(prompt),1906);self.assertEqual(hashlib.sha256(prompt).hexdigest(),'acb5e79e7d5991099be0788b252b09d0ee6c147e02266f69e1ec5848486f149f')
        text=(WORK/'speed-bench/agent_suffix_pair_external_literals.h').read_text()
        match=re.search(r'static const char \*pair_actual_third_user =\s*("(?:[^"\\]|\\.)*");',text,re.S)
        actual=''.join(ast.literal_eval(x) for x in re.findall(r'"(?:[^"\\]|\\.)*"',match[1])).encode()
        self.assertEqual(actual,ctl.delivery_payload(prompt.decode()));self.assertEqual(actual,prompt+b'\n')
        for extra in [b'',b'\n\n',b'\0',b'\r\n']:self.assertNotEqual(actual,prompt+extra)
        self.bad(ctl.delivery_payload,'text\0tail');self.assertEqual(ctl.delivery_payload('a\nb\n'),b'a\nb\n\n')
    def test_private_caller_restore_three_seams_to_immutable_production(self):
        old=(WORK/'ds4_agent.c').read_text();new=(WORK/'speed-bench/agent_suffix_pair_external.c').read_text()
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),'3c0dc3d9141f33cea40878c3724f9893d23c229b492434ea9455bdcac7464806')
        raw=new;new=new.replace('#include "agent_suffix_pair_external_hook.inc"\n\n','',1)
        a=new.index('    if (!pair_actual_select_after_user(');b=new.index('\n',new.index('    }',a))+1;new=new[:a]+new[b:]
        a=new.index('        pair_actual_round_observation(');b=new.index('\n',new.index(';',a))+1;new=new[:a]+new[b:]
        self.assertEqual(new,old)
        self.assertLess(raw.index('if (!pair_actual_select_after_user('),raw.index('uint64_t rng =',raw.index('if (!pair_actual_select_after_user(')))
    def test_control_serial_free_and_commit_before_any_sampling(self):
        hook=(WORK/'speed-bench/agent_suffix_pair_external_hook.inc').read_text();control=(WORK/'speed-bench/agent_suffix_pair_external_controls.inc').read_text()
        for forbidden in ['ds4_session_sync','ds4_session_generate','ds4_session_decode','ds4_session_sample','ds4_sample']:self.assertNotIn(forbidden,hook+control)
        pre=control.split('static bool pair_external_preedit_controls')[1].split('static bool pair_external_commit')[0]
        self.assertLess(pre.index('ds4_session_snapshot_free(&state);'),pre.index('logits=malloc'))
        commit=control.split('static bool pair_external_commit')[1].split('static bool pair_external_native_edit')[0]
        self.assertLess(commit.index('free(before);before=NULL;free(after);after=NULL;'),commit.index('agent_worker_import_live_context'))
        self.assertLess(commit.index('agent_worker_export_live_context'),commit.index('agent_replace_file'));self.assertIn('rc!=(pair_external_reuse?2:1)',commit)
        self.assertIn('return ok&&pair_external_commit',hook)
    def test_raw_control_strict_full_finite_extent(self):
        f=self.p/'raw';valid=struct.pack('<4I',0x45585231,1,129280,2582)+struct.pack('<f',0.25)*129280
        f.write_bytes(valid);self.assertEqual(ctl.validate_raw_control(f)['pos'],2582)
        for bad in [b'',valid[:15],valid[:-4],valid+b'\0\0\0\0',struct.pack('<4I',0x45585231,1,129280,0)+valid[16:],valid[:-4]+struct.pack('<f',float('nan')),valid[:-4]+struct.pack('<f',float('inf'))]:
            f.write_bytes(bad);self.bad(ctl.validate_raw_control,f)
    def test_nonutf8_ordinary_piece_trace_is_lossless(self):
        f=self.p/'trace';f.write_bytes(b'trace user="x"\nprefill sync done tool_round=0\n token index=1 id=3 bytes=1 text="\xc3" hex=c3\n token index=2 id=4 bytes=1 text="\xa9" hex=a9\ngeneration finished tool_round=0 generated=2\n')
        self.assertEqual(ctl.generated_text(f)[0]['text'],'é')
    def test_actual_fresh_count_must_equal_selection(self):
        s=selection_fixture(self.p);f=self.p/'trace';f.write_text('context pair suffix APPROXIMATE reused=64 fresh=129 source=103:167 destination=100:164 rebuilt_pairs=96 boundary_pairs=3 copied_bytes=12 unique_peak=115000000 budget=167772160 latest_user_start=164 latest_user_fresh=1\n');self.bad(ctl.structural_trace,f,s)
    def test_commit_strict_mode_no_exact_fallback_success(self):
        value=dict(status='EXTERNAL_EDIT_COMMITTED_BEFORE_THIRD_FORWARD',arm='reuse',import_returncode=2,approximate=True,selector_scratch_freed_before_import=True,before_third_RNG=True,before_third_sampling=True,postimport_ids_exact=True)
        ctl.require_commit(value,'reuse')
        for key,bad in [('import_returncode',1),('import_returncode',True),('approximate',False),('before_third_RNG',1),('before_third_sampling',False)]:
            changed=value.copy();changed[key]=bad;self.bad(ctl.require_commit,changed,'reuse')
    def test_task_old_facts_candidates_and_first_two_unchanged(self):
        old=json.loads((HERE/'agent_suffix_pair_memory160_v1/task.json').read_bytes())
        for key in ['first_user_text','second_user','edit']:self.assertEqual(TASK[key],old[key])
        self.assertEqual(TASK['external_experiment'],ctl.EXPERIMENT);self.assertEqual(ctl.sha(DRAFT/'task.json'),ctl.TASK_SHA)
        self.assertNotIn('3141',TASK['third_user_text']);self.assertNotIn('2718',(DRAFT/'run_grader.py').read_text());self.assertNotIn('3141',(DRAFT/'run_grader.py').read_text())
    def test_tool_audit_single_generic_execution_only(self):
        records=[row(1),row(2),row(3,0,True),row(3,1)]
        records[2]['calls'][0]['args'][0]['value']='python3 run_grader.py'
        self.assertEqual(ctl.audit_calls(records,self.p)['grader_wrapper_invocations'],1)
        for command in ['cat run_grader.py','cat live-context.txt','echo x > live-context.txt','python3 -c \'open("grader.py").read()\'','env','printenv','python3 run_grader.py; python3 run_grader.py','python3 run_grader.py&&python3 run_grader.py']:
            rr=copy.deepcopy(records);rr[2]['calls'][0]['args'][0]['value']=command;self.bad(ctl.audit_calls,rr,self.p)
    def test_grant_missing_forged_or_bool_before_model_identity(self):
        digest='d'*64;value=dict(status='GRANTED',sole_model_grant=True,manifest_sha256=digest,command=ctl.command(self.p,digest),time=1)
        for key,bad in [('status','APPROVED'),('sole_model_grant',1),('manifest_sha256','e'*64),('command',[]),('time',True),('time',0),('time',float('inf'))]:
            proof=value.copy();proof[key]=bad;(self.p/'root-launch-proof.json').write_text(json.dumps(proof))
            with patch.object(ctl,'verify',return_value={}),patch.object(ctl,'require_guard',return_value=None),patch.object(ctl,'model_identity',side_effect=AssertionError('model metadata touched')):
                self.bad(ctl.run,self.p,digest)
        (self.p/'root-launch-proof.json').unlink()
        with patch.object(ctl,'verify',return_value={}),patch.object(ctl,'require_guard',return_value=None),patch.object(ctl,'model_identity',side_effect=AssertionError('model metadata touched')):
            with self.assertRaises(FileNotFoundError):ctl.run(self.p,digest)
    def test_arm_policy_changes_only_two_reuse_flags(self):
        a,b=ctl.policy('reuse'),ctl.policy('native');self.assertEqual({k for k in a if a[k]!=b[k]},{'DS4_AGENT_CONTEXT_SUFFIX_REUSE','DS4_AGENT_CONTEXT_SUFFIX_PAIR_TAPE'})
        self.bad(ctl.policy,'fallback')
    def test_first_wrong_grading_code_never_repaired(self):
        private=self.p/'private';private.mkdir();project=self.p/'project';project.mkdir();(project/'access.py').write_text('def current_access_line(): return "LEGACY:2718"\n')
        env=dict(os.environ,DS4_AGENT_PAIR_ACTUAL_OUTPUT=str(private),DS4_AGENT_PAIR_ACTUAL_PYTHON=sys.executable,DS4_AGENT_PAIR_ACTUAL_GRADER=str(DRAFT/'grader.py'))
        first=subprocess.run([sys.executable,str(DRAFT/'run_grader.py')],cwd=project,env=env,capture_output=True,text=True,timeout=5)
        self.assertEqual((first.returncode,first.stdout,first.stderr),(3,'FAIL\n',''))
        (project/'access.py').write_text('def current_access_line(): return "CODE=3141"\n')
        again=subprocess.run([sys.executable,str(DRAFT/'run_grader.py')],cwd=project,env=env,capture_output=True,text=True,timeout=5)
        self.assertEqual((again.returncode,again.stdout,again.stderr),(3,'FAIL\n',''));self.assertEqual(len((private/'grader-attempts.jsonl').read_bytes().splitlines()),1)

class SharedOwnership(Disposable):
    def setUp(self):
        super().setUp();self.shared=self.p/'owned/shared';self.patch=patch.object(ctl,'SHARED',self.shared);self.patch.start();self.addCleanup(self.patch.stop)
        self.a=self.p/'A';self.a.mkdir();(self.a/'run_grader.py').write_bytes((DRAFT/'run_grader.py').read_bytes());self.digest='a'*64
    def setup_A_snapshot(self):
        project,live=ctl.initialize_shared(self.a,dict(arm='reuse'),self.digest);(project/'access.py').write_text('OLD A SOURCE');live.write_text('OLD A LIVE');out=self.p/'A-output';out.mkdir();ctl.snapshot_shared(out,self.digest)
        b=self.p/'B';b.mkdir();(b/'run_grader.py').write_bytes((DRAFT/'run_grader.py').read_bytes());ref=b/'reference';ref.mkdir();snapshot=(out/'shared-snapshot.json').read_bytes();(ref/'shared-snapshot.json').write_bytes(snapshot)
        release=dict(time=1,guard_result=dict(status='complete',child_returncode=3),lock_reacquired=True,live_model_processes=[],bad_pressure_samples=0)
        (ref/'root-release-proof.json').write_text(json.dumps(release));proof=dict(status='ROOT_SHARED_SNAPSHOT_ARCHIVED',archived_files_equal=True,A_manifest_sha256=self.digest,snapshot_sha256=ctl.sha(ref/'shared-snapshot.json'),release_sha256=ctl.sha(ref/'root-release-proof.json'))
        (ref/'root-archive-proof.json').write_text(json.dumps(proof));return b,dict(arm='native',native_source_manifest_sha256=self.digest),ref,out
    def test_new_A_exclusive_and_shared_paths_constant(self):
        project,live=ctl.initialize_shared(self.a,dict(arm='reuse'),self.digest);self.assertEqual(project,self.shared/'project');self.assertEqual(live,self.shared/'live-context.txt')
        with self.assertRaises(FileExistsError):ctl.initialize_shared(self.a,dict(arm='reuse'),self.digest)
    def test_archived_released_owned_namespace_resets_only_records(self):
        b,plan,ref,out=self.setup_A_snapshot();project,live=ctl.initialize_shared(b,plan,'b'*64)
        self.assertFalse((project/'access.py').exists());self.assertFalse(live.exists());self.assertEqual((project/'run_grader.py').read_bytes(),(DRAFT/'run_grader.py').read_bytes());self.assertTrue((out/'shared-final/project/access.py').exists())
        self.assertEqual(ctl.load_json(self.shared/'.owner.json')['arm'],'native')
    def test_unrecorded_file_refuses_before_any_unlink(self):
        b,plan,ref,_=self.setup_A_snapshot();extra=self.shared/'foreign';extra.write_text('untouched');before=ctl.shared_inventory()
        self.bad(ctl.initialize_shared,b,plan,'b'*64);self.assertEqual(ctl.shared_inventory(),before)
    def test_changed_record_refuses_before_any_unlink(self):
        b,plan,ref,_=self.setup_A_snapshot();(self.shared/'project/access.py').write_text('CHANGED');before=ctl.shared_inventory()
        self.bad(ctl.initialize_shared,b,plan,'b'*64);self.assertEqual(ctl.shared_inventory(),before)
    def test_release_archive_birth_and_symlink_refusal(self):
        b,plan,ref,_=self.setup_A_snapshot();path=ref/'root-release-proof.json';value=ctl.load_json(path);value['lock_reacquired']=False;path.write_text(json.dumps(value));self.bad(ctl.initialize_shared,b,plan,'b'*64);self.assertTrue((self.shared/'project/access.py').exists())
        (self.shared/'alias').symlink_to('project/access.py');self.bad(ctl.shared_inventory)
    def test_unknown_arm_no_shared_creation(self):
        self.bad(ctl.initialize_shared,self.a,dict(arm='invalid'),self.digest);self.assertFalse(self.shared.exists())
    def test_snapshot_deadline_and_fixed_size_bound(self):
        ctl.initialize_shared(self.a,dict(arm='reuse'),self.digest);self.bad_deadline=False
        with self.assertRaises(TimeoutError):ctl.shared_inventory(time.monotonic()-1)
        large=self.shared/'oversize';with_open=large.open('wb');with_open.truncate((16<<20)+1);with_open.close();self.bad(ctl.shared_inventory)

class ExplicitAuthority(Disposable):
    def test_correct_new_reference_requires_every_candidate_and_copied_A(self):
        ref=self.p/'reference';ref.mkdir();selected=selection_fixture(ref)
        commit=dict(status='EXTERNAL_EDIT_COMMITTED_BEFORE_THIRD_FORWARD',arm='reuse',import_returncode=2,approximate=True,selector_scratch_freed_before_import=True,before_third_RNG=True,before_third_sampling=True,postimport_ids_exact=True)
        (ref/'commit.json').write_text(json.dumps(commit));(ref/'preedit-state.bin').write_bytes(b'CPU canonical fixture')
        (ref/'preedit-logits.bin').write_bytes(struct.pack('<4I',0x45585231,1,129280,167)+bytes(4*129280));(ref/'choice.bin').write_bytes(struct.pack('<4I',0x45584331,1,selected['candidate'],selected['old_tokens']))
        (self.p/'task.json').write_text(json.dumps(TASK))
        authority=dict(arm='reuse',task_sha256=ctl.TASK_SHA,shared_paths=ctl.SHARED_PATHS,policy=ctl.policy('reuse'),external_experiment=ctl.EXPERIMENT)
        (ref/'A-manifest.json').write_text(json.dumps(authority));digest=ctl.sha(ref/'A-manifest.json');a_path=self.p/'new-A'
        (ref/'A-prepared-command.json').write_text(json.dumps(dict(manifest_sha256=digest,command=ctl.command(a_path,digest),cwd=str(ctl.ROOT))))
        names=['selection.json','selected-before.txt','selected-after.txt','selected-old.ids','selected-edited.ids','preedit-state.bin','preedit-logits.bin','commit.json','choice.bin']+[f'candidate-{i:02d}.json' for i in range(selected['candidate']+1)]
        pins={n:dict(bytes=(ref/n).stat().st_size,sha256=ctl.sha(ref/n)) for n in names}
        binding=dict(status='NEW_A_PRETHIRD_STRUCTURAL_REFERENCE_ONLY',task_sha256=ctl.TASK_SHA,shared_paths=ctl.SHARED_PATHS,generated_or_grader_outcomes_used_for_choice=False,A_manifest_sha256=digest,A_prepared_path=str(a_path),files=pins)
        (ref/'binding.json').write_text(json.dumps(binding));plan=dict(native_source_manifest_sha256=digest,files={'reference/'+n:{} for n in names})
        self.assertEqual(ctl.validate_reference(self.p,plan)['selection'],selected)
        for key,value in [('generated_or_grader_outcomes_used_for_choice',True),('A_manifest_sha256','4'*64),('files',{n:p for n,p in pins.items() if n!='candidate-00.json'})]:
            altered=binding.copy();altered[key]=value;(ref/'binding.json').write_text(json.dumps(altered));self.bad(ctl.validate_reference,self.p,plan)
        (ref/'binding.json').write_text(json.dumps(binding));changed=plan.copy();changed['files']={n:p for n,p in plan['files'].items() if n!='reference/candidate-00.json'};self.bad(ctl.validate_reference,self.p,changed)
    def test_native_build_plan_cwd_source_headers_and_object_scope(self):
        plan=json.loads((DRAFT/'native-build-plan.json').read_bytes())
        self.assertEqual(len(plan['commands']),2);self.assertEqual(len(plan['objects']),14);self.assertEqual(len(plan['shaders']),26)
        for command in plan['commands']:self.assertEqual(command['cwd'],str(WORK));self.assertIn('-I.',command['argv']);self.assertIn('-isysroot',command['argv'])
        argv=plan['commands'][0]['argv'];source=argv[argv.index('-c')+1];self.assertEqual(source,'speed-bench/agent_suffix_pair_external.c');self.assertTrue((Path(plan['commands'][0]['cwd'])/source).is_file())
        self.assertIn('speed-bench/agent_suffix_pair_external_controls.inc',plan['sources']);self.assertIn('speed-bench/agent_suffix_pair_external_literals.h',plan['sources']);self.assertIn('ds4_ssd.h',plan['sources'])
        self.assertEqual(plan['commands'][1]['argv'].count(plan['commands'][0]['argv'][-1]),1)
    def test_missing_or_malformed_build_refused(self):
        plan=dict(caller_source_freeze='1'*40,cpu_evidence_freeze='2'*40,files={'agent':dict(bytes=3,sha256='3'*64,mode=493)})
        self.bad(ctl.validate_build_authority,self.p,plan) if (self.p/'caller-build-manifest.json').exists() else None
        with self.assertRaises(FileNotFoundError):ctl.validate_build_authority(self.p,plan)
        for name in ['caller-build-manifest.json','native-build-plan.json','cpu-proof.json','source-provenance.json']:(self.p/name).write_text('{}')
        self.bad(ctl.validate_build_authority,self.p,plan)

if __name__=='__main__':unittest.main(verbosity=2)
