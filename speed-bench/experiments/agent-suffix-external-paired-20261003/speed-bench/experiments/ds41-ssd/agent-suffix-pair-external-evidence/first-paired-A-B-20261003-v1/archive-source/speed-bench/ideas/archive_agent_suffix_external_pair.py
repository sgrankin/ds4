#!/usr/bin/env python3
"""Model-free first-attempt archival. Never run helpers, graders or model probes.

The frozen producer status is preserved; analysis does not repair or rescore it.
Generated pieces are byte traces, not full generated-step logit distributions.
"""
import argparse,hashlib,json,math,re,shlex,shutil,stat,struct,types
from pathlib import Path

class Refusal(RuntimeError):pass

def require(value,message):
    if not value:raise Refusal(message)

def obj(pairs):
    out={}
    for k,v in pairs:
        require(k not in out,'duplicate JSON field '+k);out[k]=v
    return out

def decode(data):
    return json.loads(data,object_pairs_hook=obj,parse_constant=lambda x:(_ for _ in ()).throw(Refusal('nonfinite JSON '+x)))

def load(p):return decode(Path(p).read_bytes())

def safe_file(p):
    p=Path(p);require(p.is_absolute() and p.suffix.lower()!='.gguf','nonmodel absolute file required')
    for q in [p,*p.parents]:require(not q.is_symlink(),'file/ancestor alias '+str(q))
    require(stat.S_ISREG(p.stat().st_mode),'regular file required '+str(p));return p

def sha(p):
    p=safe_file(p)
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def pin(p):
    p=safe_file(p);s=p.stat();return dict(bytes=s.st_size,mode=stat.S_IMODE(s.st_mode),sha256=sha(p))

def same(a,b):
    a=safe_file(a);b=safe_file(b)
    if a.stat().st_size!=b.stat().st_size:return False
    with a.open('rb') as x,b.open('rb') as y:
        while True:
            u=x.read(1<<20);v=y.read(1<<20)
            if u!=v:return False
            if not u:return True

def put(p,data):
    with Path(p).open('x') as f:json.dump(data,f,indent=2,ensure_ascii=True);f.write('\n')

def finite(x):return type(x) in (int,float) and math.isfinite(x)

def jsonl(p):
    data=Path(p).read_bytes();require(data.endswith(b'\n'),'truncated JSONL '+str(p))
    rows=[decode(x) for x in data.splitlines()];require(all(type(x) is dict for x in rows),'JSONL objects required');return rows

def release_control(release,result,grant,prepared,pressure_command,rows):
    require(type(release) is dict and release.get('guard_result')==result,'literal ROOT/guard result differs')
    require(result.get('status')=='complete' and type(result.get('child_returncode')) is int,'guard did not complete')
    for k in ['lock_reacquired','fresh_ps_args_and_comm','all_owned_ancestors_excluded']:
        require(release.get(k) is True,'ROOT release proof missing '+k)
    require(release.get('live_model_processes')==[] and finite(release.get('time')) and release['time']>0,'ROOT release time/processes')
    for k in ['normal_pressure_samples','bad_pressure_samples']:
        require(type(release.get(k)) is int and release[k]>=0,'typed pressure count')
    require(grant.get('status')=='GRANTED' and grant.get('sole_model_grant') is True and grant.get('manifest_sha256')==prepared['manifest_sha256'] and grant.get('command')==prepared['command'],'ROOT literal sole grant differs')
    require(finite(grant.get('time')) and 0<grant['time']<release['time'],'ROOT grant time')
    argv=prepared['command'];require(argv.count('--')==1 and pressure_command.get('argv')==argv[argv.index('--')+1:],'guard child argv differs')
    initial=pressure_command.get('initial',{});require(finite(initial.get('swap_gb')) and initial['swap_gb']>=0,'initial swap')
    require(rows,'no pressure samples');bad=0;peak=0;swap=initial['swap_gb'];prev=0;consecutive=0
    for r in rows:
        keys=['time','swap_gb','physical_gb','free_gb','file_backed_gb','wired_gb','compressor_gb']
        require(all(finite(r.get(k)) and r[k]>=0 for k in keys) and r['physical_gb']>0 and r['time']>=prev,'typed finite ordered pressure fields');prev=r['time']
        require(type(r.get('level')) is int and type(r.get('consecutive_bad')) is int and r['consecutive_bad']>=0,'typed kernel pressure')
        used=r['wired_gb']+r['compressor_gb'];headroom=r['physical_gb']-used
        adverse=r['level']!=1 or r['free_gb']<8 and used>.875*r['physical_gb'] or r['swap_gb']-initial['swap_gb']>.5 and headroom<max(8,.125*r['physical_gb'])
        consecutive=consecutive+1 if adverse else 0
        require((r.get('reason') is not None)==adverse and (r.get('reason') is None or type(r.get('reason')) is str) and r['consecutive_bad']==consecutive,'pressure rule/reason/count differs')
        bad+=int(adverse);peak=max(peak,used);swap=max(swap,r['swap_gb'])
    require(bad==0 and release['bad_pressure_samples']==bad and release['normal_pressure_samples']==len(rows),'pressure/release counts differ')
    require(finite(release.get('swap_growth_gb')) and abs(release['swap_growth_gb']-(swap-initial['swap_gb']))<1e-9,'swap growth differs')
    require(finite(release.get('peak_wired_compressor_gib')) and abs(release['peak_wired_compressor_gib']-peak)<1e-9,'pressure peak differs')
    return dict(child_returncode=result['child_returncode'],normal=len(rows),bad=bad,swap_growth_gb=swap-initial['swap_gb'],peak_wired_compressor_gib=peak,root_release_time=release['time'])

def rounds_control(rows):
    require(rows,'no actual native rounds');previous=None;cap=[];flags=[];counts={};calls=[]
    for i,r in enumerate(rows):
        for k in ['turn','tool_round','generated','carried','limit','configured_limit']:
            require(type(r.get(k)) is int,'untyped round '+k)
        require(1<=r['turn']<=3 and r['tool_round']>=0 and r['configured_limit']==512 and 0<=r['generated']<=r['limit']<=512 and 0<=r['carried']<=512,'round bounds')
        expected=(1,0) if previous is None else (previous[0],previous[1]+1) if r['turn']==previous[0] else (previous[0]+1,0)
        require((r['turn'],r['tool_round'])==expected,'round order/duplicate');previous=expected
        for k in ['context_limited','model_stopped','got_tool','malformed_tool','early_tool_error','interrupted','at_limit']:
            require(type(r.get(k)) is bool,'untyped round flag '+k)
        require(r['at_limit']==(r['generated']==r['limit']),'cap observation inconsistent')
        if r['at_limit']:cap.append(dict(turn=r['turn'],tool_round=r['tool_round'],generated=r['generated'],limit=r['limit'],model_stopped=r['model_stopped'],context_limited=r['context_limited']))
        require(type(r.get('calls')) is list and bool(r['calls'])==r['got_tool'],'parsed-call observation differs')
        for j,c in enumerate(r['calls']):
            require(type(c) is dict and type(c.get('name')) is str and type(c.get('args')) is list,'typed parsed call')
            names=[]
            for a in c['args']:
                require(type(a) is dict and type(a.get('name')) is str and type(a.get('value')) is str and type(a.get('is_string')) is bool,'typed parsed argument');names.append(a['name'])
            require(len(set(names))==len(names),'duplicate parsed argument')
            calls.append(dict(turn=r['turn'],tool_round=r['tool_round'],call_index=j,call=c))
        counts[str(r['turn'])]=counts.get(str(r['turn']),0)+r['generated']
        flags.append({k:r[k] for k in ['turn','tool_round','context_limited','model_stopped','got_tool','malformed_tool','early_tool_error','interrupted','at_limit']})
    return dict(round_count=len(rows),generated_tokens=sum(r['generated'] for r in rows),tokens_by_turn=counts,cap_rounds=cap,round_flags=flags,parsed_calls=calls,stop_or_cap_cause_inferred=False)

def token_trace(data,rows):
    active=None;turn=0;done=[];digest=hashlib.sha256();pieces=[]
    for line in data.split(b'\n'):
        if b' user="' in line:turn+=1
        match=re.search(rb'prefill sync done tool_round=(\d+)',line)
        if match:
            require(active is None,'overlapping generation trace');active=(turn,int(match[1]));pieces=[];continue
        if active is not None and b' token index=' in line:
            m=re.search(rb' token index=(\d+) id=(-?\d+) bytes=(\d+) text=.* hex=([0-9a-f]*)$',line)
            require(m is not None,'malformed token trace')
            index,ident,length=int(m[1]),int(m[2]),int(m[3]);require(index==len(pieces)+1 and 0<=ident<129280 and len(m[4])==2*length,'token trace index/ID/extent')
            raw=bytes.fromhex(m[4].decode('ascii'));pieces.append(raw);digest.update(struct.pack('<iiI',active[0],active[1],ident));digest.update(struct.pack('<I',length));digest.update(raw)
        elif active is not None and b'generation finished tool_round=' in line:
            m=re.search(rb'generation finished tool_round=(\d+) generated=(\d+)',line)
            require(m is not None and int(m[1])==active[1] and int(m[2])==len(pieces),'generation trace count differs')
            raw=b''.join(pieces)
            try:raw.decode('utf-8');utf8=True
            except UnicodeError:utf8=False
            v=dict(turn=active[0],tool_round=active[1],generated=len(pieces),piece_bytes=len(raw),piece_sha256=hashlib.sha256(raw).hexdigest(),valid_utf8=utf8)
            if active[0] in (1,2):v['control_answer']=raw.decode('utf-8') if utf8 else None
            done.append(v);active=None
    require(active is None,'truncated generation trace')
    require([(r['turn'],r['tool_round'],r['generated']) for r in rows]==[(r['turn'],r['tool_round'],r['generated']) for r in done],'native round/token trace order differs')
    return dict(rounds=done,ordered_ID_piece_stream_sha256=digest.hexdigest(),full_generated_logits_available=False,trace_text_is_byte_pieces=True)

def calls_audit(calls,python):
    forbidden=['live-context','task.json','selection.json','grader.py','selected-','preedit-','controller','choice.bin','.owner.json','DS4_AGENT_PAIR_']
    findings=[];wrappers=0
    for row in calls:
        if row['turn']!=3:continue
        c=row['call'];args={a['name']:a['value'] for a in c['args']};value=' '.join(args.values());tokens=[]
        kinds=[]
        if any(x in value.replace('run_grader.py','') for x in forbidden):kinds.append('private/live-source-access-attempt')
        if c['name']=='bash':
            try:
                lexer=shlex.shlex(args.get('command',''),posix=True,punctuation_chars=True);lexer.whitespace_split=True;tokens=list(lexer)
            except ValueError:kinds.append('malformed-shell-audit')
            if any(t in ('env','printenv','set') or 'os.environ' in t for t in tokens):kinds.append('private-environment-access-attempt')
            interpreters=('python3',python)
            wrappers+=sum(tokens[i] in interpreters and tokens[i+1] in ('run_grader.py','./run_grader.py') for i in range(len(tokens)-1))
            wrappers+=sum(t=='./run_grader.py' and (i==0 or tokens[i-1] not in interpreters) for i,t in enumerate(tokens))
        if kinds:findings.append(dict(turn=row['turn'],tool_round=row['tool_round'],call_index=row['call_index'],findings=kinds,call=c))
    return dict(findings=findings,grader_wrapper_parsed_invocations=wrappers,hard_OS_sandbox_claimed=False,audit_scope='actual parsed call arguments; indirect/unparsed access is not proven absent')

def controller_control(outcome,model_phase,rc,rows):
    for v in [outcome,model_phase]:
        require(v.get('first_attempt') is True and v.get('no_retries') is True and v.get('persistence')=='NOT_EXERCISED','controller first-attempt scope')
        require(type(v.get('overall_success')) is bool and type(v.get('submitted')) is int and type(v.get('completed')) is list,'typed controller status')
        require(v['completed']==list(range(1,len(v['completed'])+1)) and all(type(x) is int for x in v['completed']) and 0<=len(v['completed'])<=v['submitted']<=3,'controller completion order')
        require(type(v.get('partial_gold_checks')) is list and [x.get('turn') for x in v['partial_gold_checks']]==v['completed'] and all(x.get('passed') is True for x in v['partial_gold_checks']),'partial control checks')
        require(v.get('status') in ('SUCCESS','TIMEOUT','FIRST_FAILURE_RETAINED'),'producer controller status')
        require(v['overall_success']==(v['status']=='SUCCESS'),'controller success/status differs')
        require(not v['overall_success'] or v['completed']==[1,2,3],'strict overall success requires three completions')
        for turn in v['completed']:
            selected=[r for r in rows if r['turn']==turn];require(selected and selected[-1]['model_stopped'] and not selected[-1]['got_tool'] and not any(r['at_limit'] for r in selected),'completion lacks actual observed stop')
    require(type(outcome.get('final_returncode')) is int and outcome['final_returncode']==rc,'controller/guard returncode differs')
    return dict(status=outcome['status'],model_phase_status=model_phase['status'],overall_success=outcome['overall_success'],completed=outcome['completed'],submitted=outcome['submitted'],error=outcome.get('error'),final_returncode=rc,partial_gold_checks=outcome['partial_gold_checks'],persistence='NOT_EXERCISED',producer_status_preserved=True)

def grader_control(private,audit):
    p=Path(private);names=['grader-started.json','grader-command-result.json','grader-command-failure.json','grader-attempts.jsonl','first-graded-access.py'];present={n:(p/n).exists() for n in names}
    if not any(present.values()):return dict(status='NOT_AVAILABLE',records=present,parsed_wrapper_invocations=audit['grader_wrapper_parsed_invocations'],no_grade_inferred=True)
    require(present['grader-started.json'] and present['first-graded-access.py'],'partial first grader must retain source/start')
    started=load(p/'grader-started.json');require(started.get('status')=='FIRST_GRADING_STARTED' and started.get('code_sha256')==sha(p/'first-graded-access.py'),'first graded file binding')
    attempts=jsonl(p/'grader-attempts.jsonl') if present['grader-attempts.jsonl'] else []
    require(len(attempts)<=1,'duplicate grading attempt retained refusal')
    for r in attempts:
        keys={'source_exists','pure_module','callable_exists','current_value','obsolete_absent'}
        require(type(r.get('passed')) is bool and type(r.get('checks')) is dict and set(r['checks'])==keys and all(type(x) is bool for x in r['checks'].values()) and r['passed']==all(r['checks'].values()) and r.get('source_sha256')==started['code_sha256'],'typed first grader result/source')
    result=load(p/'grader-command-result.json') if present['grader-command-result.json'] else None
    if result is not None:
        require(type(result.get('returncode')) is int and type(result.get('passed')) is bool and type(result.get('stdout')) is str and type(result.get('stderr')) is str,'typed grader command')
        require(result['passed']==(result['returncode']==0 and result['stdout'].strip()=='PASS' and not result['stderr']),'producer grader command logic differs')
        require(len(attempts)==1 and attempts[0]['passed']==result['passed'],'first grader command/attempt differs')
    return dict(status='FIRST_ATTEMPT_PRESERVED',records=present,started=started,attempts=attempts,command_result=result,command_failure=load(p/'grader-command-failure.json') if present['grader-command-failure.json'] else None,parsed_wrapper_invocations=audit['grader_wrapper_parsed_invocations'],rescore_or_repair_performed=False,post_first_check_only_is_separate_audit=True)

def snapshot_control(p,digest):
    p=Path(p);s=load(p/'run/shared-snapshot.json');require(s.get('status')=='OWNED_SHARED_SNAPSHOT_RETAINED' and s.get('manifest_sha256')==digest and s.get('leader_cleanup_completed') is True and s.get('global_model_release_claimed') is False,'shared snapshot authority')
    files=s.get('files');require(type(files) is dict,'snapshot files')
    base=p/'run/shared-final';actual={str(f.relative_to(base)) for f in base.rglob('*') if f.is_file()}
    require(actual==set(files),'shared snapshot complete physical inventory')
    for name,value in files.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe shared snapshot name')
        require(all(pin(base/name)[k]==value[k] for k in ['bytes','mode','sha256']),'shared snapshot bytes/mode differ '+name)
    return dict(snapshot_sha256=sha(p/'run/shared-snapshot.json'),root=s.get('root'),root_identity=s.get('root_identity'),files=files,current_mutable_shared_root_read_or_reset=False)

def import_controller(plan):
    p=Path(plan['controller']['path']);require(sha(p)==plan['controller']['sha256'],'frozen pure controller source differs')
    m=types.ModuleType('frozen_paired_controller');m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m

def required_inputs(plan):
    for path,digest in plan['required_input_files'].items():
        require(type(digest) is str and re.fullmatch('[0-9a-f]{64}',digest) is not None and sha(path)==digest,'literal archived/qualified input authority differs '+path)

def check_authorities(plan,ctl):
    checked={};released={}
    for arm in ['reuse','native']:
        value=plan['arms'][arm];p=Path(value['path']);require(sha(p/'manifest.json')==value['manifest_sha256'] and sha(p/'prepared-command.json')==value['command_sha256'],'literal prepared authority '+arm)
        require(sha(p/'root-release-proof.json')==value['release_sha256'],'literal release authority '+arm)
        prepared=load(p/'prepared-command.json');release=load(p/'root-release-proof.json');result=load(p/'pressure/result.json');grant=load(p/'root-launch-proof.json')
        released[arm]=release_control(release,result,grant,prepared,load(p/'pressure/command.json'),jsonl(p/'pressure/pressure.jsonl'))
        checked[arm]=ctl.verify(p,value['manifest_sha256'])
    require(released['native']['root_release_time']>released['reuse']['root_release_time'],'arm release order')
    a,b=checked['reuse'],checked['native'];require(a['arm']=='reuse' and b['arm']=='native','frozen arm authority')
    require(a['task_sha256']==b['task_sha256']==plan['task_sha256'] and a['shared_paths']==b['shared_paths'],'literal same task/shared paths')
    common=set(a['files'])&set(b['files'])
    require(common==set(a['files']),'all A assets must be common native authority')
    for name in common:require(same(Path(plan['arms']['reuse']['path'])/name,Path(plan['arms']['native']['path'])/name),'common A/B prepared bytes differ '+name)
    return checked,released

def analysis(plan,ctl,checked,released):
    arms={}
    for arm in ['reuse','native']:
        p=Path(plan['arms'][arm]['path']);private=p/'run/private';trace=p/'run/trace.log';rows=jsonl(private/'rounds.jsonl');r=rounds_control(rows);audit=calls_audit(r['parsed_calls'],str(ctl.PYTHON))
        structure=ctl.validate_external_commit(p,private,load(p/'task.json'),trace,checked[arm])
        outcome=controller_control(load(p/'run/controller-outcome.json'),load(p/'run/controller-model-phase-outcome.json'),released[arm]['child_returncode'],rows)
        controls={};pieces=token_trace(trace.read_bytes(),rows)
        for turn in [1,2]:
            v=load(p/f'run/turn-{turn}-check.json');observed=[x for x in pieces['rounds'] if x['turn']==turn]
            require(type(v.get('turn')) is int and v['turn']==turn and v.get('passed') is True and type(v.get('answer')) is str and v['answer'].strip()==['CODE=2718','READY'][turn-1] and len(observed)==1 and observed[0]['control_answer']==v['answer'],'frozen initial semantic control/actual bytes differ');controls[str(turn)]=v
        arms[arm]=dict(controller=outcome,pressure=released[arm],structure=structure,initial_controls=controls,rounds=r,token_piece_trace=pieces,parsed_call_audit=audit,first_grader=grader_control(private,audit),shared_snapshot=snapshot_control(p,plan['arms'][arm]['manifest_sha256']))
    names=['selected-old.ids','selected-edited.ids','selected-before.txt','selected-after.txt','preedit-state.bin','preedit-logits.bin']
    a=Path(plan['arms']['reuse']['path'])/'run/private';b=Path(plan['arms']['native']['path'])/'run/private'
    parity={n:dict(byte_exact=same(a/n,b/n),A=pin(a/n),B=pin(b/n)) for n in names}
    require(all(v['byte_exact'] for v in parity.values()),'actual pre-forward whole source/raw/canonical parity differs')
    invalid=not all(x['controller']['overall_success'] for x in arms.values())
    return dict(status='FIRST_ATTEMPTS_ARCHIVED_NO_RESCORING',arms=arms,preedit_control_parity=parity,task_comparison_valid=not invalid,semantic_cache_regression_or_success_claimed=False,speed_claimed=False,forgetting_claimed=False,full_generated_logit_distribution_available=False,persistence='NOT_EXERCISED',scope='one externally edited task; initial native source parity and transaction structure are distinct from failed task outcomes')

def inventory(root):
    root=Path(root);require(root.is_absolute() and root.is_dir() and not root.is_symlink(),'literal source root required')
    out={}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),'source alias '+str(p))
        if p.is_file():out[str(p.relative_to(root))]=pin(p)
        else:require(p.is_dir(),'unexpected source entry')
    return out

def archive_roots(roots,dest,threshold):
    dest=Path(dest);require(not dest.exists(),'exclusive archive destination exists');dest.mkdir(parents=True,exist_ok=False);files={};external={};total=0
    for label,root in roots.items():
        require(re.fullmatch('[A-Za-z0-9_-]+',label) is not None,'unsafe root label');root=Path(root);items=inventory(root)
        for name,value in items.items():
            key=label+'/'+name;original=root/name
            if value['bytes']>threshold:external[key]=dict(path=str(original),**value,reason='retained original exceeds fixed compact-copy threshold');continue
            total+=value['bytes'];require(total<=64<<20,'compact archive exceeds fixed64MiB storage envelope')
            target=dest/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target);require(pin(target)==value and same(original,target),'archive byte/mode copy differs');files[key]=dict(original_path=str(original),**value)
        require(inventory(root)==items,'source inventory changed during copying')
    return files,external

def execute(plan_path,digest):
    plan_path=Path(plan_path).absolute();require(sha(plan_path)==digest,'explicit plan digest required');plan=load(plan_path)
    require(plan.get('status')=='FROZEN_MODEL_FREE_PAIRED_ARCHIVE_SOURCE_V1' and plan.get('no_model_helper_guard_grader_or_subprocess_execution') is True,'archive source plan scope')
    require(plan['copy_threshold_bytes']==2<<20,'fixed copy threshold')
    for p,value in plan['source_files'].items():require(pin(p)==value,'archive source prerequisite differs '+p)
    required_inputs(plan)
    ctl=import_controller(plan);checked,released=check_authorities(plan,ctl)
    # Only after exact prior grants/releases/static-source qualification do we
    # read actual generated observations. No helper or grader is executed.
    report=analysis(plan,ctl,checked,released);dest=Path(plan['output']);files,external=archive_roots(plan['archive_roots'],dest,plan['copy_threshold_bytes'])
    try:
        for name,path in plan.get('archive_files',{}).items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts and name not in files and name not in external,'unsafe/duplicate archive file')
            value=pin(path)
            if value['bytes']>plan['copy_threshold_bytes']:external[name]=dict(path=path,**value,reason='retained original exceeds fixed compact-copy threshold');continue
            target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);require(pin(target)==value,'single-file copy differs');files[name]=dict(original_path=path,**value)
        put(dest/'paired-analysis.json',report);files['paired-analysis.json']=pin(dest/'paired-analysis.json')
        shutil.copy2(plan_path,dest/'execution-plan.json');files['execution-plan.json']=pin(dest/'execution-plan.json')
        for rel in files:require(pin(dest/rel)=={k:files[rel][k] for k in ['bytes','mode','sha256']},'post-publication file changed')
        physical={str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()};require(physical==set(files),'complete nested physical index differs')
        put(dest/'archive-index.json',dict(status='COMPLETE_FIRST_ATTEMPT_PAIRED_ARCHIVE',files=files,protected_original_references=external,exclude_only_this_actual_top_index=True,plan_sha256=digest,model_or_helper_execution=False,semantic_or_speed_promotion=False))
    except BaseException:
        raise # retain every partial file; never delete, overwrite or retry
    return dict(output=str(dest),archive_index_sha256=sha(dest/'archive-index.json'),copied_files=len(files),protected_original_references=len(external))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--execute',action='store_true');v=p.parse_args()
    require(v.execute,'archive execution requires explicit execute flag; source/CPU fixture tests use pure functions')
    print(json.dumps(execute(v.plan,v.plan_sha256),indent=2))

if __name__=='__main__':main()
