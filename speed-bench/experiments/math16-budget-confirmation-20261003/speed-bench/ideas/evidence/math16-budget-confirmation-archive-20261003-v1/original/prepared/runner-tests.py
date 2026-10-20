"""Model-free independent math16 traces; no real helper/model/guard is executed."""
import contextlib,copy,errno,hashlib,io,json,os
from pathlib import Path
import re,shutil,subprocess,sys,time,unittest
from unittest.mock import patch
WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'speed-bench/ideas'))
import analyze_reasoning_budget_confirmation as a
import run_reasoning_budget_confirmation as d
ART=WORK/'speed-bench/ideas/artifacts'
CPU=ART/('math16-budget-cpu-fixtures-'+str(time.time_ns()));CPU.mkdir(parents=True)
FIXTURE=a.load(d.INPUT/'fixture.json')
assert a.sha(d.INPUT/'fixture.json')=='f0eeb1effe60fb75af9572d6e81b7a102f1eb83cfe995a2e919109e7b2ed8703'
SERIAL=0


def write(path,value):
    path=Path(path)
    if path.exists():path.unlink()  # Mutation hardlinks must not modify originals.
    path.write_text(json.dumps(value,allow_nan=False)+'\n')


def lines(path,values):
    path=Path(path)
    if path.exists():path.unlink()
    path.write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in values))


def fork(root):
    global SERIAL
    SERIAL+=1;target=CPU/('mutation-'+str(SERIAL));shutil.copytree(root,target,copy_function=os.link);return target


def state_next(word):
    word=word if word else 11400714819323198485
    word^=word//4096;word^=(word*33554432)%18446744073709551616;word^=word//134217728
    return word%18446744073709551616


def scripted(root,budget):
    root.mkdir();rows=[]
    for i,case in enumerate(FIXTURE['cases']):
        kind=i%4
        close=[2,1100,None,1500][kind] if not budget else [2,1024,1024,1024][kind]
        forced=close if budget and kind else None
        count=[4,1102,2048,2048][kind] if not budget else [4,1026,1026,2048][kind]
        final=case['answer'] if kind in (0,3) or kind==1 and not budget or i==6 and budget else '999' if kind==1 else '' if not budget else 'not-an-integer'
        stopped=kind in (0,1) or kind==2 and bool(budget)
        parse_ok=kind!=2 or i==6 and bool(budget)
        status='INVALID' if not parse_ok else 'INCORRECT' if kind==1 and budget else 'CORRECT'
        raw=bytearray();tokens=[];rng=20261002
        for step in range(count+int(stopped)):
            stop=step==count;before=rng;isforce=step==forced
            if not isforce:rng=state_next(rng)
            piece=b'<eos>' if stop else b'</think>' if step==close else final.encode() if close is not None and step==close+1 else b' ' if close is not None and step>close else b'r'
            tokens.append(dict(step=step,token_id=1 if stop else 128822 if step==close else 10 if piece==b'r' else 11,
                stop=stop,inside_thinking=close is None or step<=close,generated_thinking_tokens=step if close is None else min(step,close),
                policy_active=isforce,penalty=0,rng_before=f'{before:016x}',rng_after=f'{rng:016x}',
                intervention='forced-thinking-close' if isforce else 'native-sample',piece_hex=piece.hex()))
            if not stop:raw.extend(piece)
        if kind==3:final+=' '*(count-close-2)
        row=dict(case=case['case'],policy_type='protocol-boundary-token',reasoning_budget=budget,forced_think_close=forced is not None,
            suite='math-budget-confirmation16-v1',think_level=25,native_penalty=0,think_enabled=True,prompt_supplied_think_close=False,
            generated_think_close=close is not None,status=status,parse_ok=bool(parse_ok),parse_reason='none' if parse_ok else 'unclosed_reasoning' if close is None else 'unsupported_final_format',
            scorer='strict-final-v2',seed=20261002,temperature_float_value=0.6000000238418579,top_p_float_value=0.949999988079071,min_p=0,
            max_tokens=2048,cache_gib_requested=64,context_size=8192,prefill_chunk=512,prompt_tokens=3,generated_tokens=count,
            reasoning_tokens=count if close is None else close+1,think_closed=close is not None,stopped=bool(stopped),cap_hit=count==2048,
            correct_exact=status=='CORRECT',prefill_seconds=.1,decode_seconds=.2,parsed_answer=final.strip() if parse_ok else '',final=final,
            expected=case['answer'],prompt=case['prompt'],raw_trace=case['case']+'.txt',token_trace=case['case']+'.tokens.jsonl',prompt_trace=case['case']+'.prompt.ids')
        (root/row['raw_trace']).write_bytes(raw);lines(root/row['token_trace'],tokens);(root/row['prompt_trace']).write_text('128000\n42\n128821\n');rows.append(row)
    lines(root/'results.jsonl',rows);return root


NATIVE=scripted(CPU/'native',0);CANDIDATE=scripted(CPU/'budget1024',1024)
FAKE=CPU/'fake_cpu_producer.py'
FAKE.write_text("""import json,os,pathlib,shutil,sys
source,target,rc,audit=sys.argv[1:]
shutil.copytree(source,target)
with pathlib.Path(audit).open('a') as f:f.write(json.dumps({'DS4X':os.environ.get('DS4X'),'DS4_UNEXPECTED':os.environ.get('DS4_UNEXPECTED'),'penalty':os.environ.get('DS4_OVERTHINK_PENALTY')})+'\\n')
sys.exit(int(rc))
""")


def synthetic(name):
    p=CPU/name;p.mkdir();write(p/'fixture.json',FIXTURE)
    arms=[]
    for n,b,s in zip(a.NAMES,[0,1024,0],[NATIVE,CANDIDATE,NATIVE]):
        output=p/(n+'-result');arms.append(dict(name=n,budget=b,results_dir=str(output),argv=[sys.executable,str(FAKE),str(s),str(output),'3',str(p/'audit.jsonl')]))
    plan=dict(output=str(p/'run'),fixture=str(p/'fixture.json'),arms=arms,run_cwd=str(WORK),environment=dict(d.ENV),pressure=str(p/'pressure'),guard_command_template=['cpu','guard','pressure','--','child',d.PLACEHOLDER])
    write(p/'plan.json',plan);return p/'plan.json',plan


class Study(unittest.TestCase):
    def reject(self,call):
        with self.assertRaises((ValueError,OSError,KeyError,TypeError)):call()

    def arm(self,path,budget=1024,rc=3):return a.validate_arm(path,budget,FIXTURE,rc)

    def row(self,updates,index=1):
        p=fork(CANDIDATE);rows=a.jsonl(p/'results.jsonl');rows[index].update(updates);lines(p/'results.jsonl',rows);self.reject(lambda:self.arm(p))

    def token(self,updates,step=1024,index=1):
        p=fork(CANDIDATE);row=a.jsonl(p/'results.jsonl')[index];tokens=a.jsonl(p/row['token_trace']);tokens[step].update(updates);lines(p/row['token_trace'],tokens);self.reject(lambda:self.arm(p))

    def test_all16_correct_wrong_invalid_caps_and_both_discordances(self):
        n=self.arm(NATIVE,0);c=self.arm(CANDIDATE)
        self.assertEqual(n[0]['cases'],16);self.assertEqual(c[0]['cases'],16);self.assertTrue(a.exact_native(n,self.arm(NATIVE,0)))
        gates=a.pre_force(n,c);self.assertEqual(sum(x['force_step']==1024 for x in gates),12)
        self.assertEqual((c[0]['completed_correct'],c[0]['incorrect'],c[0]['invalid'],c[0]['capped_correct']),(5,4,3,4))
        paired=a.paired(n,c);self.assertEqual((paired['native_correct_budget_failed'],paired['native_failed_budget_correct']),(4,1))
        self.assertEqual(c[1][1]['reasoning_tokens'],1025)
        self.assertEqual(state_next(20261002),0x00026a77b178752e);self.assertEqual(a.next_rng(20261002),0x00026a77b178752e)

    def test_complete_order_extent_and_strict_scoring(self):
        for kind in ('missing','duplicate','reverse'):
            p=fork(CANDIDATE);r=a.jsonl(p/'results.jsonl')
            if kind=='missing':r.pop()
            elif kind=='duplicate':r[-1]=r[0]
            else:r.reverse()
            lines(p/'results.jsonl',r);self.reject(lambda:self.arm(p))
        for updates in [dict(expected='1'),dict(prompt='changed'),dict(status='CORRECT'),dict(parse_ok=False),dict(parsed_answer='43'),dict(suite='counting-v1'),dict(think_level=True),dict(seed=True),dict(reasoning_budget=True),dict(temperature_float_value=.6),dict(prefill_chunk=256),dict(cap_hit=1),dict(stopped=False),dict(decode_seconds=float('inf'))]:
            if updates.get('decode_seconds')==float('inf'):
                self.reject(lambda:a.decode_json('{"decode_seconds":Infinity}'))
            else:self.row(updates)
        self.reject(lambda:self.arm(CANDIDATE,rc=0));self.reject(lambda:self.arm(CANDIDATE,rc=True))
        self.assertIsNone(a.scorer.parsed_integer('9'*5000));self.assertEqual(a.scorer.parsed_integer('0'*5000+'43'),'43')

    def test_force_source_rng_phase_and_no_reopening(self):
        for updates in [dict(policy_active=False),dict(token_id=1),dict(piece_hex=b'</think>!'.hex()),dict(rng_after='0000000000000000'),dict(inside_thinking=False),dict(generated_thinking_tokens=1023),dict(stop=True),dict(intervention='native-sample'),dict(penalty=.1)]:self.token(updates)
        self.token(dict(policy_active=True,intervention='forced-thinking-close'),1023)
        self.token(dict(policy_active=True,intervention='forced-thinking-close'),1025)
        self.token(dict(inside_thinking=True),1025);self.token(dict(piece_hex=b'<think>'.hex()),1025)
        self.token(dict(generated_thinking_tokens=1025),1025)

    def test_tokens_raw_and_prompt_fail_closed(self):
        for kind in ('duplicate','missing','truncated','duplicate-key','NaN','raw','prompt','escape'):
            p=fork(CANDIDATE);row=a.jsonl(p/'results.jsonl')[1];path=p/row['token_trace'];data=path.read_bytes()
            if kind in ('duplicate','missing'):
                values=a.jsonl(path)
                if kind=='duplicate':values[10]=values[9]
                else:values.pop(10)
                lines(path,values)
            elif kind in ('truncated','duplicate-key','NaN'):
                path.unlink();path.write_bytes(data[:-1] if kind=='truncated' else data.replace(b'"step": 0',b'"step": 0, "step": 0',1) if kind=='duplicate-key' else data.replace(b'"penalty": 0',b'"penalty": NaN',1))
            elif kind in ('raw','prompt'):
                target=p/row['raw_trace' if kind=='raw' else 'prompt_trace'];target.unlink();target.write_bytes(b'wrong' if kind=='raw' else b'1\n129280\n3\n')
            else:
                rows=a.jsonl(p/'results.jsonl');rows[1]['raw_trace']='../escape';lines(p/'results.jsonl',rows)
            self.reject(lambda:self.arm(p))
        for change in [dict(step=True),dict(token_id=129280),dict(policy_active=1),dict(piece_hex='0'),dict(penalty=True)]:self.token(change)
        self.reject(lambda:a.sha(CPU/'do-not-open.gguf'))

    def test_native_and_preforce_exact_identity(self):
        n=self.arm(NATIVE,0);c=self.arm(CANDIDATE)
        changed=copy.deepcopy(n);changed[2][0]['trace'][0]['token_id']+=1;self.reject(lambda:a.exact_native(n,changed))
        changed=copy.deepcopy(c);changed[2][1]['trace'][0]['rng_after']='0'*16;self.reject(lambda:a.pre_force(n,changed))
        changed=copy.deepcopy(c);changed[2][0]['raw_sha256']='0'*64;self.reject(lambda:a.pre_force(n,changed))
        changed=copy.deepcopy(c);changed[2][0]['prompt_sha256']='0'*64;self.reject(lambda:a.pre_force(n,changed))

    def test_source_qualified_fixed_input_and_completed_build(self):
        d.input_check();d.qualify_build(a.load(d.BUILD/'build-manifest.json'),d.BUILD_MANIFEST_SHA)
        text=(d.SOURCE/'speed-bench/reasoning_budget_confirmation_probe.c').read_text()
        branch='if (argc==2 && !strcmp(argv[1],"--helper-self-test")) return self_test();'
        self.assertLess(text.index(branch),text.index('ds4_engine_open('))
        old=text.replace('reasoning_budget_confirmation_cases.h','reasoning_budget_cases.h').replace('math-budget-confirmation16-v1','counting-v1')
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),'48bf48e6ae5b16adffa53129571c9ff3279ed1e5372e3ca8d56738f0aeed519c')
        changed=a.load(d.BUILD/'build-manifest.json');changed['core_or_Metal_rebuild']=True
        self.reject(lambda:d.qualify_build(changed,d.BUILD_MANIFEST_SHA))

    def test_exact_SDK_library_alias_only(self):
        self.assertEqual(d.asset(d.SDK_LIBRARY),dict(sha256=d.SDK_LIBRARY_SHA,bytes=347931))
        self.assertEqual(d.asset(d.SDK_TARGET),dict(sha256=d.SDK_LIBRARY_SHA,bytes=347931))
        with patch.object(d.os,'readlink',return_value='foreign.tbd'):self.reject(lambda:d.asset(d.SDK_LIBRARY))
        foreign=CPU/'foreign-symlink';foreign.symlink_to(d.SDK_TARGET);self.reject(lambda:d.asset(foreign))

    def test_fixed_plan_schema_inventory_and_tamper_controls(self):
        p=ART/('math16-budget-shape-'+str(time.time_ns()));p.mkdir();write(p/'fixture.json',FIXTURE)
        bp=a.load(d.BUILD/'build-plan.json');refs={str(p/'source'/n):pin for n,pin in d.SOURCES.items()}
        objects={path:pin for path,pin in bp['borrowed_pins'].items() if path.endswith('.o')}
        env=dict(d.ENV);env.update({'DS4_METAL_'+n+'_SOURCE':str(p/'metal'/(n.lower()+'.metal')) for n in d.SHADER_NAMES})
        files={str(p/n) for n in d.SCRIPTS+['fixture.json','cpu-proof.json','cpu-tests.log','runner-tests.py']}
        files|={str(d.INPUT/'input-freeze.json')}|{str(d.INPUT/n) for n in a.load(d.INPUT/'input-freeze.json')['files']}
        files|={str(d.BUILD/n) for n in d.BUILD_FILES}|set(bp['borrowed_pins'])|set(refs)|{v for k,v in env.items() if k.endswith('_SOURCE')}
        files|={str(d.GUARD),str(d.PYTHON),str(d.PYTHON_TARGET),str(d.FRAMEWORK),str(d.SDK_TARGET)}
        assets={path:dict(sha256='0'*64,bytes=1) for path in files}
        fixed={str(d.GUARD):d.GUARD_SHA,str(d.PYTHON):d.PYTHON_SHA,str(d.PYTHON_TARGET):d.PYTHON_SHA,str(d.FRAMEWORK):d.FRAMEWORK_SHA,
            str(d.SDK_TARGET):d.SDK_LIBRARY_SHA,str(d.SDK_LIBRARY):d.SDK_LIBRARY_SHA,str(p/'reasoning_budget_signed64.py'):d.SCORER_SHA,
            str(d.INPUT/'input-freeze.json'):d.INPUT_SHA,str(d.BUILD/'build-plan.json'):d.BUILD_PLAN_SHA,str(d.BUILD/'build-manifest.json'):d.BUILD_MANIFEST_SHA,
            str(d.BUILD/'reasoning-budget-confirmation-probe'):d.BINARY_SHA,**objects,**refs}
        for path,pin in fixed.items():assets[path]['sha256']=pin
        arms=[]
        for name,budget in zip(a.NAMES,[0,1024,0]):
            rn=p.name+'-'+name;arms.append(dict(name=name,budget=budget,run_name=rn,argv=[str(d.BUILD/'reasoning-budget-confirmation-probe'),d.MODEL['path'],rn,'25',str(budget)],results_dir=str(d.WORK/'speed-bench/ideas/artifacts'/rn)))
        path=p/'plan.json'
        plan=dict(schema='math-budget-confirmation16-v1',prepared_only=True,model_grant=False,analysis_requires_root_release=True,no_automatic_retry_or_scheduler=True,
            settings=copy.deepcopy(d.SETTINGS),sdk_library_alias=copy.deepcopy(d.SDK_ALIAS),all_first_attempts=48,model=copy.deepcopy(d.MODEL),workspace=str(d.WORK),run_cwd=str(d.WORK),root_cwd=str(d.ROOT),
            binary=str(d.BUILD/'reasoning-budget-confirmation-probe'),binary_sha256=d.BINARY_SHA,build_manifest=str(d.BUILD/'build-manifest.json'),build_manifest_sha256=d.BUILD_MANIFEST_SHA,build_plan_sha256=d.BUILD_PLAN_SHA,input_freeze_sha256=d.INPUT_SHA,
            guard=str(d.GUARD),guard_sha256=d.GUARD_SHA,launch_python=str(d.PYTHON),guard_interpreter_identity=dict(path=str(d.FRAMEWORK),sha256=d.FRAMEWORK_SHA),
            guard_command_template=[str(d.PYTHON),str(d.GUARD),str(p/'pressure'),'--',str(d.PYTHON),str(p/d.SCRIPTS[0]),'--run',str(path),'--plan-sha256',d.PLACEHOLDER,'--root-granted'],
            output=str(p/'run'),pressure=str(p/'pressure'),fixture=str(p/'fixture.json'),arms=arms,environment=env,core_objects=objects,source_refs=refs,assets=assets)
        d.validate_plan(plan,path)
        for mutate in [lambda x:x['assets'].pop(x['binary']),lambda x:x['assets'].pop(str(d.SDK_TARGET)),lambda x:x['assets'].pop(next(iter(x['source_refs']))),
            lambda x:x.update(model_grant=True),lambda x:x.update(prepared_only=1),lambda x:x.update(all_first_attempts=True),lambda x:x.update(build_manifest_sha256='a'*64),
            lambda x:x['settings'].update(seed=True),lambda x:x['arms'].reverse(),lambda x:x['arms'][1].update(budget=True),lambda x:x['arms'][1]['argv'].append('--retry'),
            lambda x:x['guard_command_template'].__setitem__(-2,'a'*64),lambda x:x['environment'].update(DS4X='1'),lambda x:x['sdk_library_alias'].update(link_text='other'),
            lambda x:x['assets'][x['binary']].update(bytes=True)]:
            bad=copy.deepcopy(plan);mutate(bad);self.reject(lambda:d.validate_plan(bad,path))

    def test_exact_parent_framework_lock_and_explicit_grant(self):
        path,plan=synthetic('parent');plan['guard_interpreter_identity']=dict(path=str(d.FRAMEWORK),sha256=d.FRAMEWORK_SHA)
        root=CPU/'fake-lock-root';(root/'speed-bench/ideas/artifacts').mkdir(parents=True);(root/'speed-bench/ideas/artifacts/model-job.lock').touch()
        command=d.rendered_command(plan,'a'*64);actual=[str(d.FRAMEWORK)]+command[1:]
        def digest(path):return d.GUARD_SHA if str(path)==str(d.GUARD) else d.FRAMEWORK_SHA
        def held(*args):raise BlockingIOError(errno.EAGAIN,'CPU held')
        with patch.object(d,'ROOT',root),patch.object(a,'sha',side_effect=digest),patch.object(d.os.path,'samefile',return_value=True),patch.object(d.fcntl,'flock',side_effect=held),patch.object(d.subprocess,'check_output',return_value=' '.join(actual)) as ps:
            self.assertTrue(d.require_guard(plan,path,'a'*64)['lock_held']);self.assertEqual(ps.call_args.args[0][0],'/bin/ps');self.assertEqual(ps.call_args.kwargs['timeout'],2)
            ps.return_value=' '.join(actual+['--alias']);self.reject(lambda:d.require_guard(plan,path,'a'*64))
            ps.return_value=' '.join(actual)
            with patch.object(d.os.path,'samefile',return_value=False):self.reject(lambda:d.require_guard(plan,path,'a'*64))
            with patch.object(d.fcntl,'flock',return_value=None):self.reject(lambda:d.require_guard(plan,path,'a'*64))
        grant=dict(status='GRANTED',sole_model_grant=True,manifest_sha256='a'*64,command=command,time=1);write(path.parent/'root-launch-proof.json',grant);d.require_grant(plan,path,'a'*64)
        for k,v in [('sole_model_grant',1),('manifest_sha256','b'*64),('command',[]),('time',False)]:
            bad=dict(grant);bad[k]=v;write(path.parent/'root-launch-proof.json',bad);self.reject(lambda:d.require_grant(plan,path,'a'*64))

    def test_release_and_pressure_gate_before_outcomes(self):
        result=dict(status='complete',child_returncode=3);proof=dict(guard_result=result,lock_reacquired=True,live_model_processes=[],time=1,normal_pressure_samples=1,bad_pressure_samples=0,swap_growth_gb=0)
        a.validate_release(proof,result)
        for k,v in [('lock_reacquired',1),('live_model_processes',['helper']),('normal_pressure_samples',0),('bad_pressure_samples',1),('swap_growth_gb',.1),('time',float('nan'))]:
            bad=copy.deepcopy(proof);bad[k]=v;self.reject(lambda:a.validate_release(bad,result))
        path,plan=synthetic('pressure');pressure=Path(plan['pressure']);pressure.mkdir();pin=a.sha(path)
        write(pressure/'command.json',dict(argv=d.rendered_command(plan,pin)[4:],initial=dict(time=.1,level=1,swap_gb=0)))
        rows=[dict(time=.2,level=1,swap_gb=0,reason=None,consecutive_bad=0)];lines(pressure/'pressure.jsonl',rows);a.validate_pressure(plan,pin,proof)
        for k,v in [('level',2),('time',.01),('swap_gb',.1),('consecutive_bad',True)]:
            bad=copy.deepcopy(rows);bad[0][k]=v;lines(pressure/'pressure.jsonl',bad);self.reject(lambda:a.validate_pressure(plan,pin,proof))
        # Pure AST order gate; no live completed outcomes are opened.
        import ast
        tree=ast.parse(Path(a.__file__).read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='analyze')
        calls={}
        for node in ast.walk(function):
            if isinstance(node,ast.Call):
                name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
                calls.setdefault(name,[]).append(node.lineno)
        self.assertLess(min(calls['validate_release']),min(calls['validate_arm']))

    def invoke(self,path,plan,guard_failure=False):
        events=[]
        def guard(*args):
            events.append('guard')
            if guard_failure:raise ValueError('CPU guard refusal')
            return dict(cpu=True)
        def grant(*args):events.append('grant')
        def assets(*args):events.append('assets')
        def model(*args):
            self.assertIn('guard',events);self.assertIn('grant',events);self.assertIn('assets',events);events.append('FAKE-model');return dict(cpu=True)
        with patch.object(d,'validate_plan'),patch.object(d,'require_guard',side_effect=guard),patch.object(d,'require_grant',side_effect=grant),patch.object(d,'check_assets',side_effect=assets),patch.object(d,'model_identity',side_effect=model),patch.dict(os.environ,{'DS4X':'remove','DS4_UNEXPECTED':'remove'}),contextlib.redirect_stdout(io.StringIO()):
            rc=d.run(path,a.sha(path),root_granted=True)
        return rc,events

    def test_real_CPU_subprocesses_all48_no_auto_retry_and_environment(self):
        path,plan=synthetic('48-attempts');rc,events=self.invoke(path,plan);self.assertEqual(rc,3)
        complete=a.load(Path(plan['output'])/'complete.json');self.assertEqual(complete['arms'],a.NAMES);self.assertEqual(complete['returncode'],3)
        audit=a.jsonl(path.parent/'audit.jsonl');self.assertEqual(len(audit),3);self.assertTrue(all(x=={'DS4X':None,'DS4_UNEXPECTED':None,'penalty':'0'} for x in audit))
        self.assertLess(events.index('guard'),events.index('FAKE-model'))
        self.reject(lambda:self.invoke(path,plan));self.assertEqual(len(a.jsonl(path.parent/'audit.jsonl')),3)

    def test_fail_first_before_candidates(self):
        path,plan=synthetic('infrastructure-failure');plan['arms'][0]['argv'][-2]='2';write(path,plan)
        self.reject(lambda:self.invoke(path,plan));self.assertEqual(len(a.jsonl(path.parent/'audit.jsonl')),1)
        failure=a.load(Path(plan['output'])/'first-failure.json');self.assertEqual(failure['completed_arms'],[])
        self.assertFalse(Path(plan['arms'][1]['results_dir']).exists())
        path,plan=synthetic('guard-refusal');self.reject(lambda:self.invoke(path,plan,True));self.assertFalse(Path(plan['output']).exists());self.assertFalse((path.parent/'audit.jsonl').exists())


if __name__=='__main__':unittest.main()
