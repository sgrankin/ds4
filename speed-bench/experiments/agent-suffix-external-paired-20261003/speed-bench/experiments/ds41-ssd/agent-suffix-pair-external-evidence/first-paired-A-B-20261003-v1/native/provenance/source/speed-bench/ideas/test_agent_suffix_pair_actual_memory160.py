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
DRAFT=HERE/'agent_suffix_pair_memory160_v1'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ctl=module('actual_controller',HERE/'run_agent_suffix_pair_actual_memory160.py')
update=module('actual_update',DRAFT/'update_context.py')
grader=module('actual_grader',DRAFT/'grader.py')
TASK=json.loads((DRAFT/'task.json').read_text())
ROOT_CASES=Path(os.environ.get('PAIR_ACTUAL_CPU_CASES',str(HERE/'artifacts/agent-suffix-pair-actual-cpu-draft-v1/cases')))
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

class FrozenEdit(Disposable):
    def test_exact_all_candidates_preserve_appended_history(self):
        before=TASK['first_user_text']+'\nNEW APPENDED TOOL HISTORY <literal>\n'
        for i in range(16):
            self.assertEqual(update.apply(before,TASK,i),ctl.apply_frozen_text(before,TASK,i))
            self.assertTrue(update.apply(before,TASK,i).endswith('NEW APPENDED TOOL HISTORY <literal>\n'))
    def test_updater_atomic_current_not_old_selection_and_no_echo(self):
        (self.p/'selection.json').write_text(json.dumps(dict(status='SELECTED_BEFORE_THIRD_EVAL',candidate=0)))
        live=self.p/'live';before=TASK['first_user_text']+'\nAPPENDED AFTER SELECTION\n';live.write_text(before)
        result=update.execute(self.p,live,DRAFT/'task.json')
        self.assertEqual(live.read_text(),update.apply(before,TASK,0));self.assertEqual(result['before_sha256'],hashlib.sha256(before.encode()).hexdigest())
        self.assertEqual(list(self.p.glob('live.edit-*')),[])
        with self.assertRaises(FileExistsError):update.execute(self.p,live,DRAFT/'task.json')
    def test_updater_failure_is_retained_and_no_repair(self):
        (self.p/'selection.json').write_text(json.dumps(dict(status='SELECTED_BEFORE_THIRD_EVAL',candidate=0)))
        live=self.p/'live';live.write_text('missing fixed text')
        with self.assertRaises(ValueError):update.execute(self.p,live,DRAFT/'task.json')
        self.assertEqual(live.read_text(),'missing fixed text');self.assertEqual(json.loads((self.p/'update-result.json').read_text())['status'],'FIRST_UPDATE_FAILURE_RETAINED')
        with self.assertRaises(FileExistsError):update.execute(self.p,live,DRAFT/'task.json')
    def test_duplicate_or_wrong_type_operation_rejected(self):
        self.bad(update.apply,TASK['first_user_text']+TASK['edit']['replace_exact_unique_fact'][0],TASK,0)
        self.bad(update.apply,TASK['first_user_text'],TASK,True)
    def test_gold_current_and_old_and_unsafe(self):
        for text,expected in [('def current_access_line():\n    return "CODE=3141"\n',True),('def current_access_line():\n    return "LEGACY:2718"\n',False),('import os\ndef current_access_line():\n    return "CODE=3141"\n',False),('def current_access_line():\n    return open("gold").read()\n',False)]:
            (self.p/'access.py').write_text(text);self.assertEqual(grader.evaluate(self.p)['passed'],expected)
    def test_project_wrapper_contains_no_gold_and_prompts_one_write(self):
        text=(DRAFT/'run_grader.py').read_text();self.assertNotIn('3141',text);self.assertNotIn('2718',text)
        for n in ['first.txt','second.txt','third.txt']:self.assertLessEqual(len(((DRAFT/n).read_text()+'\n').encode()),4096)
        self.assertNotIn('3141',TASK['third_user_text'])
    def test_actual_wrapper_first_wrong_file_cannot_be_repaired(self):
        private=self.p/'private';private.mkdir();project=self.p/'project';project.mkdir()
        source='def current_access_line():\n    return "LEGACY:2718"\n';(project/'access.py').write_text(source)
        env=dict(os.environ,DS4_AGENT_PAIR_ACTUAL_OUTPUT=str(private),DS4_AGENT_PAIR_ACTUAL_PYTHON=sys.executable,DS4_AGENT_PAIR_ACTUAL_GRADER=str(DRAFT/'grader.py'))
        first=subprocess.run([sys.executable,str(DRAFT/'run_grader.py')],cwd=project,env=env,capture_output=True,text=True,timeout=5)
        self.assertEqual((first.returncode,first.stdout,first.stderr),(3,'FAIL\n',''))
        self.assertEqual((private/'first-graded-access.py').read_text(),source)
        (project/'access.py').write_text('def current_access_line():\n    return "CODE=3141"\n')
        second=subprocess.run([sys.executable,str(DRAFT/'run_grader.py')],cwd=project,env=env,capture_output=True,text=True,timeout=5)
        self.assertEqual((second.returncode,second.stdout,second.stderr),(3,'FAIL\n',''))
        self.assertEqual(len((private/'grader-attempts.jsonl').read_text().splitlines()),1);self.assertTrue((private/'grader-command-failure.json').exists())
    def test_actual_wrapper_timeout_preserves_claim_before_any_later_pass(self):
        private=self.p/'private';private.mkdir();project=self.p/'project';project.mkdir();(project/'access.py').write_text('def current_access_line(): return "CODE=3141"\n')
        fake=self.p/'hanging-grader.py';fake.write_text('import time;time.sleep(30)\n')
        env=dict(os.environ,DS4_AGENT_PAIR_ACTUAL_OUTPUT=str(private),DS4_AGENT_PAIR_ACTUAL_PYTHON=sys.executable,DS4_AGENT_PAIR_ACTUAL_GRADER=str(fake))
        first=subprocess.run([sys.executable,str(DRAFT/'run_grader.py')],cwd=project,env=env,capture_output=True,text=True,timeout=5)
        self.assertEqual((first.returncode,first.stdout,first.stderr),(3,'FAIL\n',''));self.assertTrue((private/'grader-started.json').exists());self.assertTrue((private/'grader-command-failure.json').exists())
        self.assertFalse((private/'grader-attempts.jsonl').exists())

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
    def test_read_contamination_and_single_updater(self):
        r=[row(1),row(2),row(3,0,True),row(3,1)]
        r[-1]['calls']=[dict(name='bash',args=[dict(name='command',value='python3 run_grader.py',is_string=True)])]
        self.assertTrue(ctl.audit_calls(r,self.p)['explicit_read_audit_passed'])
        for cmd in ['cat update_context.py','python3 -c \'open("grader.py").read()\'','rg 3141 task.json']:
            rr=copy.deepcopy(r);rr[-1]['calls'].append(dict(name='bash',args=[dict(name='command',value=cmd,is_string=True)]));self.bad(ctl.audit_calls,rr,self.p)
        rr=copy.deepcopy(r);rr[2]['calls'][0]['args'][0]['value']='cat update_context.py';self.bad(ctl.audit_calls,rr,self.p)

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
    def test_private_caller_three_observational_seams_original_unchanged(self):
        old=(WORK/'ds4_agent.c').read_text();new=(WORK/'speed-bench/agent_suffix_pair_actual.c').read_text()
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),'3c0dc3d9141f33cea40878c3724f9893d23c229b492434ea9455bdcac7464806')
        include='#include "agent_suffix_pair_actual_hook.inc"\n\n'
        new=new.replace(include,'',1)
        start=new.index('    if (!pair_actual_select_after_user(');end=new.index('\n',new.index('    }',start))+1;new=new[:start]+new[end:]
        start=new.index('        pair_actual_round_observation(');end=new.index('\n',new.index(';',start))+1;new=new[:start]+new[end:]
        # Removing the exact inserted blocks must restore production bytes.
        self.assertEqual(new,old)
        raw=(WORK/'speed-bench/agent_suffix_pair_actual.c').read_text();a=raw.index('if (!pair_actual_select_after_user(');b=raw.index('if (!w->session_title)',a);self.assertLess(a,b)
        hook=(WORK/'speed-bench/agent_suffix_pair_actual_hook.inc').read_text()
        for forbidden in ['ds4_session_sync','ds4_session_generate','ds4_session_decode','ds4_session_sample','ds4_sample']:self.assertNotIn(forbidden,hook)

if __name__=='__main__':unittest.main(verbosity=2)
