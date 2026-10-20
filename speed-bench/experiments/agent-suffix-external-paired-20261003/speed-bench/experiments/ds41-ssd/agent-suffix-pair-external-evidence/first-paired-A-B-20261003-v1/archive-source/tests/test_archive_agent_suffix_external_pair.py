"""Disposable model-free archive/analysis controls; no completed outcomes read."""
import copy,hashlib,importlib.util,json,math,os,stat,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).absolute().parents[1]/'speed-bench/ideas/archive_agent_suffix_external_pair.py'
S=importlib.util.spec_from_file_location('paired_archive',P);a=importlib.util.module_from_spec(S);S.loader.exec_module(a)
CASES=Path(os.environ['PAIR_ARCHIVE_CPU_CASES']);CASES.mkdir(parents=True,exist_ok=True)

def row(t=1,r=0,n=1,tool=False,cap=False):
    return dict(turn=t,tool_round=r,generated=512 if cap else n,carried=0,limit=512,configured_limit=512,
      context_limited=False,model_stopped=not tool and not cap,got_tool=tool,malformed_tool=False,
      early_tool_error=False,interrupted=False,at_limit=cap,
      calls=[dict(name='bash',args=[dict(name='command',value='python3 run_grader.py',is_string=True)])] if tool else [])

def trace(rows,raw=b'x'):
    out=[];turn=0
    for r in rows:
        if r['turn']!=turn:out.append(b'trace user="literal"');turn=r['turn']
        out.append(b'prefill sync done tool_round='+str(r['tool_round']).encode())
        for i in range(r['generated']):out.append(b' token index='+str(i+1).encode()+b' id=3 bytes='+str(len(raw)).encode()+b' text="raw" hex='+raw.hex().encode())
        out.append(b'generation finished tool_round='+str(r['tool_round']).encode()+b' generated='+str(r['generated']).encode()+b' carried=0 context_limited=0')
    return b'\n'.join(out)+b'\n'

def outcome(completed=(1,2),status='FIRST_FAILURE_RETAINED'):
    return dict(first_attempt=True,no_retries=True,persistence='NOT_EXERCISED',overall_success=status=='SUCCESS',status=status,
      submitted=3,completed=list(completed),partial_gold_checks=[dict(turn=t,passed=True) for t in completed],final_returncode=0 if status=='SUCCESS' else 3,error='first failure')

def pressure():
    prepared=dict(manifest_sha256='a'*64,command=['framework','guard','pressure','--','framework','driver'])
    grant=dict(status='GRANTED',sole_model_grant=True,manifest_sha256='a'*64,command=prepared['command'],time=1)
    rows=[dict(time=2,level=1,swap_gb=0,physical_gb=128,free_gb=.1,file_backed_gb=50,wired_gb=72,compressor_gb=6,reason=None,consecutive_bad=0)]
    command=dict(argv=['framework','driver'],initial=dict(swap_gb=0))
    result=dict(status='complete',child_returncode=3)
    release=dict(time=3,guard_result=result,lock_reacquired=True,live_model_processes=[],normal_pressure_samples=1,bad_pressure_samples=0,
      swap_growth_gb=0,peak_wired_compressor_gib=78,fresh_ps_args_and_comm=True,all_owned_ancestors_excluded=True)
    return release,result,grant,prepared,command,rows

class Case(unittest.TestCase):
    def setUp(self):self.p=Path(tempfile.mkdtemp(prefix=self._testMethodName+'-',dir=CASES)).absolute()
    def bad(self,fn,*args):
        with self.assertRaises((a.Refusal,ValueError,KeyError,TypeError)):fn(*args)
    def put(self,n,v):p=self.p/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return p
    def test_json_duplicate_nonfinite_and_truncation(self):
        self.bad(a.decode,'{"x":1,"x":2}');self.bad(a.decode,'{"x":NaN}')
        p=self.p/'r';p.write_bytes(b'{"x":1}');self.bad(a.jsonl,p);p.write_bytes(b'[]\n');self.bad(a.jsonl,p)
    def test_cap_preserved_not_success_or_stop(self):
        r=[row(1),row(2),row(3,0,tool=True),row(3,1,cap=True)];v=a.rounds_control(r)
        self.assertEqual(v['generated_tokens'],515);self.assertEqual(v['cap_rounds'][0]['model_stopped'],False);self.assertFalse(v['stop_or_cap_cause_inferred'])
        o=outcome();c=a.controller_control(o,o,3,r);self.assertFalse(c['overall_success']);self.assertEqual(c['final_returncode'],3)
    def test_typed_round_order_bounds_and_calls(self):
        for k,v in [('generated',True),('configured_limit',511),('at_limit',1),('got_tool',True),('turn',0),('tool_round',-1),('carried',513)]:
            r=row();r[k]=v;self.bad(a.rounds_control,[r])
        self.bad(a.rounds_control,[row(1),row(1)]);self.bad(a.rounds_control,[row(2)]);self.bad(a.rounds_control,[row(1),row(3)])
        r=row(tool=True);r['calls'][0]['args']*=2;self.bad(a.rounds_control,[r])
    def test_lossless_nonUTF8_trace_and_actual_count(self):
        rows=[row(1),row(2),row(3,0,cap=True)];v=a.token_trace(trace(rows,b'\xff'),rows)
        self.assertFalse(v['rounds'][0]['valid_utf8']);self.assertFalse(v['full_generated_logits_available']);self.assertEqual(v['rounds'][-1]['generated'],512)
        self.bad(a.token_trace,trace(rows).replace(b'index=1 ',b'index=2 ',1),rows)
        self.bad(a.token_trace,trace(rows).replace(b'id=3 ',b'id=-1 ',1),rows)
        self.bad(a.token_trace,trace(rows).replace(b'bytes=1 ',b'bytes=2 ',1),rows)
        self.bad(a.token_trace,trace(rows).rsplit(b'generation finished',1)[0],rows)
    def test_round_and_trace_order_bound(self):
        self.bad(a.token_trace,trace([row(1),row(2)]),[row(1)])
        self.bad(a.token_trace,trace([row(1)]).replace(b'tool_round=0',b'tool_round=1'),[row(1)])
    def test_parsed_call_audit_keeps_forbidden_and_wrapper(self):
        r=row(3,tool=True);r['calls'][0]['args'][0]['value']='cat live-context.txt; python3 run_grader.py; env'
        v=a.calls_audit(a.rounds_control([row(1),row(2),r])['parsed_calls'],'/pinned/Python')
        self.assertEqual(v['grader_wrapper_parsed_invocations'],1);self.assertEqual(len(v['findings']),1);self.assertEqual(len(v['findings'][0]['findings']),2);self.assertFalse(v['hard_OS_sandbox_claimed'])
    def test_wrapper_path_not_gold_read_false_positive(self):
        r=[row(1),row(2),row(3,tool=True)];v=a.calls_audit(a.rounds_control(r)['parsed_calls'],'/pinned/Python')
        self.assertEqual(v['findings'],[]);self.assertEqual(v['grader_wrapper_parsed_invocations'],1)
    def test_controller_partial_success_and_strict_three(self):
        r=[row(1),row(2),row(3)];o=outcome(status='SUCCESS',completed=(1,2,3));self.assertTrue(a.controller_control(o,o,0,r)['overall_success'])
        for k,v in [('completed',[2,1]),('overall_success',1),('first_attempt',False),('no_retries',False),('submitted',2),('final_returncode',True)]:
            x=copy.deepcopy(o);x[k]=v;self.bad(a.controller_control,x,o,0,r)
        x=outcome(status='SUCCESS');self.bad(a.controller_control,x,x,0,r)
    def test_failed_first_grader_preserved_without_rescore(self):
        p=self.p/'first-graded-access.py';p.write_bytes(b'def current_access_line():return "CODE=2718"\n');h=a.sha(p)
        self.put('grader-started.json',dict(status='FIRST_GRADING_STARTED',code_sha256=h))
        self.put('grader-command-result.json',dict(returncode=3,passed=False,stdout='FAIL\n',stderr=''))
        checks=dict(source_exists=True,pure_module=True,callable_exists=True,current_value=False,obsolete_absent=False)
        (self.p/'grader-attempts.jsonl').write_text(json.dumps(dict(passed=False,checks=checks,source_sha256=h))+'\n')
        v=a.grader_control(self.p,dict(grader_wrapper_parsed_invocations=1));self.assertFalse(v['attempts'][0]['passed']);self.assertFalse(v['rescore_or_repair_performed'])
        p.write_bytes(b'mutated');self.bad(a.grader_control,self.p,dict(grader_wrapper_parsed_invocations=1))
    def test_missing_grader_stays_unavailable_and_no_grade(self):
        v=a.grader_control(self.p,dict(grader_wrapper_parsed_invocations=0));self.assertEqual(v['status'],'NOT_AVAILABLE');self.assertTrue(v['no_grade_inferred'])
        self.put('grader-command-result.json',dict(passed=True));self.bad(a.grader_control,self.p,dict(grader_wrapper_parsed_invocations=1))
    def test_duplicate_first_grader_rejected(self):
        p=self.p/'first-graded-access.py';p.write_bytes(b'x');h=a.sha(p);self.put('grader-started.json',dict(status='FIRST_GRADING_STARTED',code_sha256=h))
        (self.p/'grader-attempts.jsonl').write_text((json.dumps(dict(passed=False,checks={},source_sha256=h))+'\n')*2)
        self.bad(a.grader_control,self.p,dict(grader_wrapper_parsed_invocations=2))
    def test_release_exact_and_healthy_reclaimable_memory(self):
        v=a.release_control(*pressure());self.assertEqual(v['child_returncode'],3);self.assertEqual(v['normal'],1)
        for k,v in [('lock_reacquired',1),('time',True),('time',float('inf')),('bad_pressure_samples',1),('live_model_processes',[123]),('normal_pressure_samples',2),('swap_growth_gb',.1),('peak_wired_compressor_gib',79)]:
            args=list(pressure());args[0][k]=v;self.bad(a.release_control,*args)
    def test_forged_grant_or_guard_command_rejected(self):
        for k,v in [('sole_model_grant',1),('status','DONE'),('manifest_sha256','b'*64),('time',True),('time',4),('command',['other'])]:
            args=list(pressure());args[2][k]=v;self.bad(a.release_control,*args)
        args=list(pressure());args[4]['argv']=['other'];self.bad(a.release_control,*args)
    def test_pressure_failure_nonfinite_or_counts_not_hidden(self):
        for k,v in [('level',2),('level',True),('wired_gb',float('nan')),('time',-1),('reason','adverse'),('consecutive_bad',True),('consecutive_bad',1)]:
            args=list(pressure());args[-1][0][k]=v;self.bad(a.release_control,*args)
    def test_archive_nested_manifest_complete_and_large_reference(self):
        source=self.p/'source';source.mkdir();(source/'manifest.json').write_bytes(b'nested');(source/'tiny').write_bytes(b'x');(source/'big').write_bytes(b'abcdefgh')
        dest=self.p/'archive';files,refs=a.archive_roots({'source':source},dest,6)
        self.assertEqual(set(files),{'source/manifest.json','source/tiny'});self.assertEqual(set(refs),{'source/big'});self.assertEqual(a.pin(dest/'source/manifest.json'),a.pin(source/'manifest.json'))
        self.bad(a.archive_roots,{'source':source},dest,6)
    def test_symlink_file_ancestor_and_model_forbidden(self):
        p=self.p/'real';p.write_bytes(b'x');alias=self.p/'alias';alias.symlink_to(p);self.bad(a.pin,alias)
        model=self.p/'fake.gguf';model.write_bytes(b'notmodel');self.bad(a.pin,model)
        root=self.p/'root';root.mkdir();(root/'bad').symlink_to(p);self.bad(a.inventory,root)
    def test_mutation_during_copy_retained_refusal(self):
        source=self.p/'source';source.mkdir();p=source/'a';p.write_bytes(b'original');old=a.shutil.copy2
        def mutate(src,dst):v=old(src,dst);p.write_bytes(b'changed');return v
        with patch.object(a.shutil,'copy2',mutate):self.bad(a.archive_roots,{'source':source},self.p/'archive',1024)
        self.assertTrue((self.p/'archive/source/a').exists())
    def test_snapshot_exact_bytes_modes_and_inventory(self):
        p=self.p/'run/shared-final/a';p.parent.mkdir(parents=True);p.write_bytes(b'owned');p.chmod(0o600)
        self.put('run/shared-snapshot.json',dict(status='OWNED_SHARED_SNAPSHOT_RETAINED',manifest_sha256='a'*64,leader_cleanup_completed=True,global_model_release_claimed=False,files={'a':a.pin(p)},root='/literal',root_identity=dict(device=1,inode=2)))
        self.assertFalse(a.snapshot_control(self.p,'a'*64)['current_mutable_shared_root_read_or_reset']);p.chmod(0o644);self.bad(a.snapshot_control,self.p,'a'*64)
    def test_explicit_old_archive_authority_before_outcomes(self):
        p=self.p/'index.json';p.write_bytes(b'archived before new native arm');j=dict(required_input_files={str(p):a.sha(p)})
        a.required_inputs(j);j['required_input_files'][str(p)]='a'*64;self.bad(a.required_inputs,j)
        j['required_input_files'][str(p)]=True;self.bad(a.required_inputs,j)

if __name__=='__main__':unittest.main()
