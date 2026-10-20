#!/usr/bin/env python3
"""Private preregistered actual-task controller. No model access on import.

Real run requires the externally pinned prepared manifest and MAIN guard.
CPU fixtures invoke pure validation or an explicitly supplied tiny fake child.
"""
import argparse,errno,fcntl,hashlib,json,math,os,re,selectors,shlex,shutil,signal,struct,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/Volumes/Models/ds4')
PYTHON=Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python')
PYTHON_SHA='b80d260a48404da13a32dd44d7283aadc80f6a6e56ed6e03ed5597ec6f46d48d'
GUARD=ROOT/'speed-bench/ideas/pressure_run.py';GUARD_SHA='461e8c517e82a687b4579fc450c1e7128b1510192d088bab719682b33c946c3c'
MODEL=dict(path=str(ROOT/'gguf/DeepSeek-V4.1-Flash-Q2.gguf'),bytes=365713686528,mtime_ns=1789237692837653163)
TASK_SHA='eb5ff21a8d620e22d67c37d438690e1e8c37b69c7592999d4cf399fc7607e283'
SHADERS={'DS4_METAL_FLASH_ATTN_SOURCE':'flash_attn.metal','DS4_METAL_DENSE_SOURCE':'dense.metal','DS4_METAL_GLM53_BF16_SOURCE':'glm53_bf16.metal','DS4_METAL_GLM53_VISION_SOURCE':'glm53_vision.metal','DS4_METAL_DEEPSEEK4_VISION_SOURCE':'deepseek4_vision.metal','DS4_METAL_GLM53_KDA_SOURCE':'glm53_kda.metal','DS4_METAL_MOE_SOURCE':'moe.metal','DS4_METAL_DSV4_HC_SOURCE':'dsv4_hc.metal','DS4_METAL_UNARY_SOURCE':'unary.metal','DS4_METAL_DSV4_KV_SOURCE':'dsv4_kv.metal','DS4_METAL_DSV41_SOURCE':'dsv41.metal','DS4_METAL_DSV4_ROPE_SOURCE':'dsv4_rope.metal','DS4_METAL_DSV4_MISC_SOURCE':'dsv4_misc.metal','DS4_METAL_ARGSORT_SOURCE':'argsort.metal','DS4_METAL_CPY_SOURCE':'cpy.metal','DS4_METAL_CONCAT_SOURCE':'concat.metal','DS4_METAL_GET_ROWS_SOURCE':'get_rows.metal','DS4_METAL_SUM_ROWS_SOURCE':'sum_rows.metal','DS4_METAL_SOFTMAX_SOURCE':'softmax.metal','DS4_METAL_REPEAT_SOURCE':'repeat.metal','DS4_METAL_GLU_SOURCE':'glu.metal','DS4_METAL_NORM_SOURCE':'norm.metal','DS4_METAL_BIN_SOURCE':'bin.metal','DS4_METAL_SET_ROWS_SOURCE':'set_rows.metal','DS4_METAL_QWEN4_SOURCE':'qwen4.metal','DS4_METAL_QWEN4_VISION_SOURCE':'qwen4_vision.metal'}
POLICY=dict(DS4_AGENT_CONTEXT_SUFFIX_REUSE='1',DS4_AGENT_CONTEXT_SUFFIX_PAIR_TAPE='1',DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL='128',DS4_AGENT_CONTEXT_FRONTIERS='2',DS4_AGENT_CONTEXT_FRONTIER_MB='128',DS4_METAL_V41_SELECTED_TILE='64',DS4_METAL_V41_SELECTED_LIMIT='1024',DS4_METAL_V41_RESIDENT_ROUTE_GAP='0',DS4_METAL_V41_RESIDENT_ROUTE_MODE='native',DS4_AGENT_TEST_TIME='1789319891')
CORRECTION=dict(status='CORRECTIVE_INFRASTRUCTURE_ATTEMPT',attempt=1,
    original_manifest_sha256='1719ce2a865edcd3e8a711353b1226fc61f090a8aa2ef577ab6a1878b45791bb',
    original_command_sha256='a346c428f4f593a8cf4d552fbe91e9799c8c2bcd596b7a8d50fac68b60e5889e',
    original_failed_before_third_evaluation=True,original_structural_selection_created=False,
    correction='Exact immutable third.txt plus one controller protocol LF; no input/task/model/sampling change.')


class ControlFailure(RuntimeError):pass

def json_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ControlFailure('Duplicate private JSON field: '+key)
        result[key]=value
    return result

def decode_json(data):
    return json.loads(data,object_pairs_hook=json_object,parse_constant=lambda value:(_ for _ in ()).throw(ControlFailure('Nonfinite JSON: '+value)))

def load_json(path):return decode_json(Path(path).read_bytes())

def sha(p):
    p=Path(p)
    if p.suffix.lower()=='.gguf':raise ValueError('No model hashing')
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def put(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')

def command(p,digest):return [str(PYTHON),str(GUARD),str(p/'pressure'),'--',str(PYTHON),str(p/'run_agent_suffix_pair_actual.py'),'--run',str(p),'--manifest-sha256',digest]

def require_guard(p,digest):
    expected=command(p,digest);actual=shlex.split(subprocess.check_output(['/bin/ps','-p',str(os.getppid()),'-o','command='],text=True,timeout=2).strip())
    if not actual or actual[1:]!=expected[1:] or not os.path.samefile(actual[0],PYTHON) or sha(PYTHON)!=PYTHON_SHA or sha(GUARD)!=GUARD_SHA:raise RuntimeError('Exact pinned MAIN parent/framework required')
    with (ROOT/'speed-bench/ideas/artifacts/model-job.lock').open('r+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EAGAIN,errno.EWOULDBLOCK):raise
            return
        fcntl.flock(f,fcntl.LOCK_UN)
    raise RuntimeError('MAIN model lock must be held')

def require_root_grant(p,digest):
    proof=load_json(p/'root-launch-proof.json')
    if type(proof) is not dict or proof.get('status')!='GRANTED' or proof.get('sole_model_grant') is not True or proof.get('manifest_sha256')!=digest or proof.get('command')!=command(p,digest):raise ControlFailure('Exact explicit ROOT sole-model launch proof required')
    stamp=proof.get('time')
    if type(stamp) not in (int,float) or not math.isfinite(stamp) or stamp<=0:raise ControlFailure('Finite positive ROOT grant time required')
    return proof

def verify(p,digest):
    if sha(p/'manifest.json')!=digest:raise ValueError('Explicit manifest authority differs')
    plan=load_json(p/'manifest.json')
    if plan['task_sha256']!=TASK_SHA or plan['model']!=MODEL or plan['policy']!=POLICY or plan['turns']!=3 or plan['timeout_seconds']!=900:raise ValueError('Frozen task/policy differs')
    if plan.get('corrective_infrastructure_attempt')!=CORRECTION:raise ValueError('Exact corrective infrastructure provenance required')
    required={'agent','task.json','grader.py','update_context.py','run_grader.py','run_agent_suffix_pair_actual.py','first.txt','second.txt','third.txt','caller-build-manifest.json'}
    required|={'metal/'+name for name in SHADERS.values()}
    if not required<=set(plan['files']):raise ValueError('Required assets missing')
    for name,pin in plan['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or Path(name).suffix.lower()=='.gguf':raise ValueError('Unsafe asset')
        f=p/name
        if f.stat().st_size!=pin['bytes'] or sha(f)!=pin['sha256']:raise ValueError('Asset differs: '+name)
    if sha(p/'task.json')!=TASK_SHA or not os.access(p/'agent',os.X_OK):raise ValueError('Task/agent executable differs')
    if plan['shader_env']!={k:str(p/'metal'/v) for k,v in SHADERS.items()}:raise ValueError('26 shader paths differ')
    if sha(PYTHON)!=PYTHON_SHA or sha(GUARD)!=GUARD_SHA:raise ValueError('Guard/framework differs')
    if load_json(p/'prepared-command.json')!=dict(manifest_sha256=digest,command=command(p,digest),cwd=str(ROOT)):raise ValueError('Prepared command differs')
    return plan

def model_identity():
    p=Path(MODEL['path'])
    if p.resolve()!=p:raise ValueError('Model alias')
    s=p.stat();observed=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    if observed!=MODEL:raise ValueError('Model identity differs')
    return observed

def remaining(deadline):
    v=deadline-time.monotonic()
    if v<=0:raise TimeoutError('Whole-task work deadline including cleanup reserve')
    return v

def stop_process(proc):
    if proc.returncode is not None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except (ProcessLookupError,PermissionError):pass
    # No poll/wait during grace: preserve the unreaped leader birth anchor.
    time.sleep(.5)
    if proc.returncode is None:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except (ProcessLookupError,PermissionError):pass
    proc.wait(timeout=5)

def checked_command(argv,deadline,**kwargs):
    proc=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True,**kwargs)
    try:
        out,err=proc.communicate(timeout=remaining(deadline));return subprocess.CompletedProcess(argv,proc.returncode,out,err)
    except BaseException:stop_process(proc);raise
    finally:
        # communicate may time out while a descendant holds inherited pipes.
        # Cleanup/reap ownership is handled above before closing these handles.
        for stream in (proc.stdout,proc.stderr):
            if stream is not None and not stream.closed:stream.close()

def rounds(path):
    data=Path(path).read_bytes()
    if not data or not data.endswith(b'\n'):raise ControlFailure('Missing/truncated actual round observations')
    rows=[decode_json(line) for line in data.splitlines()]
    if any(type(row) is not dict for row in rows):raise ControlFailure('Actual round must be an object')
    return rows

def validate_rounds(records,completed=None):
    if not records:raise ControlFailure('No actual round observations')
    previous=None
    for r in records:
        for k in ['turn','tool_round','generated','carried','limit','configured_limit']:
            if type(r.get(k)) is not int:raise ControlFailure('Untyped round '+k)
        if not 1<=r['turn']<=3 or r['tool_round']<0 or r['configured_limit']!=512 or not 0<=r['generated']<=r['limit']<=512 or r['carried']<0:raise ControlFailure('Round bounds')
        for k in ['context_limited','model_stopped','got_tool','malformed_tool','early_tool_error','interrupted','at_limit']:
            if type(r.get(k)) is not bool:raise ControlFailure('Untyped flag '+k)
        expected=(1,0) if previous is None else (previous[0],previous[1]+1) if r['turn']==previous[0] else (previous[0]+1,0)
        if (r['turn'],r['tool_round'])!=expected:raise ControlFailure('Duplicate/missing/out-of-order native round')
        previous=(r['turn'],r['tool_round'])
        if r['at_limit']!=(r['generated']==r['limit']):raise ControlFailure('Round limit observation differs')
        if r['at_limit']:raise ControlFailure('First observed generation-limit round retained; stop/cap cause is not inferred')
        if r['context_limited'] or r['malformed_tool'] or r['early_tool_error'] or r['interrupted'] or not (r['model_stopped'] or r['got_tool']):raise ControlFailure('First malformed/interrupted/incomplete round retained')
        if type(r.get('calls')) is not list or bool(r['calls'])!=r['got_tool']:raise ControlFailure('Actual parsed calls differ')
        for call in r['calls']:
            if type(call.get('name')) is not str or type(call.get('args')) is not list:raise ControlFailure('Untyped native call')
            names=[]
            for a in call['args']:
                if type(a.get('name')) is not str or type(a.get('value')) is not str or type(a.get('is_string')) is not bool:raise ControlFailure('Untyped native argument')
                names.append(a['name'])
            if len(names)!=len(set(names)):raise ControlFailure('Duplicate native argument')
    if completed is not None:
        for turn in range(1,completed+1):
            rows=[r for r in records if r['turn']==turn]
            if not rows or rows[-1]['got_tool'] or not rows[-1]['model_stopped']:raise ControlFailure('Completed native turn lacks observed stop')
            if turn<3 and (len(rows)!=1 or rows[0]['calls']):raise ControlFailure('Control turn used tools')
        if previous[0]!=completed:raise ControlFailure('Unexpected completed turn')
    return records

def generated_text(trace):
    turns=[];active=False;pieces=[];current_turn=0
    for line in Path(trace).read_text().splitlines():
        if re.search(r' user="',line):current_turn+=1
        if 'prefill sync done tool_round=' in line:active=True;pieces=[]
        elif active and ' token index=' in line:
            m=re.search(r' token index=(\d+) id=(-?\d+) bytes=(\d+) text=.* hex=([0-9a-f]*)$',line)
            if not m or int(m[1])!=len(pieces)+1 or int(m[2])<0 or int(m[2])>=129280 or len(m[4])!=int(m[3])*2:raise ControlFailure('Malformed generated token trace')
            pieces.append(bytes.fromhex(m[4]))
        elif active and 'generation finished tool_round=' in line:
            m=re.search(r'generated=(\d+)',line)
            if not m or int(m[1])!=len(pieces):raise ControlFailure('Generated trace count differs')
            try:text=b''.join(pieces).decode('utf-8')
            except UnicodeError as error:raise ControlFailure('Incomplete generated UTF-8 bytes') from error
            turns.append(dict(turn=current_turn,text=text,generated=len(pieces)));active=False
    if active:raise ControlFailure('Truncated generation trace')
    return turns

def validate_control(trace,records,turn):
    validate_rounds(records,turn);texts=generated_text(trace);selected=[r for r in texts if r['turn']==turn]
    if len(selected)!=1 or selected[0]['text'].strip()!=['CODE=2718','READY'][turn-1]:raise ControlFailure('First semantic control failed')
    return dict(turn=turn,answer=selected[0]['text'],passed=True)

def read_ids(p):
    data=Path(p).read_bytes()
    if len(data)<16:raise ControlFailure('Truncated selection IDs')
    magic,version,n,user=struct.unpack_from('<4I',data)
    if magic!=0x41535031 or version!=1 or n>8192 or user>=129280 or len(data)!=16+4*n:raise ControlFailure('Selection IDs header/extent')
    ids=list(struct.unpack_from('<'+str(n)+'i',data,16))
    if any(x<0 or x>=129280 for x in ids):raise ControlFailure('Selection token bounds')
    return user,ids

def validate_selection(private,task):
    p=Path(private);s=load_json(p/'selection.json');auser,old=read_ids(p/'selected-old.ids');buser,edited=read_ids(p/'selected-edited.ids')
    ints=['candidate','displacement','source_start','source_end','destination_start','destination_end','reused','fresh','latest_user_start','user_id','old_tokens','edited_tokens','budget','preallocation_peak_with_reserves','current_payload_forecast','old_text_bytes','edited_text_bytes']
    if any(type(s.get(k)) is not int for k in ints) or s['status']!='SELECTED_BEFORE_THIRD_EVAL' or s['actual_commit_rechecks_admission'] is not True:raise ControlFailure('Selection typed authority')
    if not 0<=s['candidate']<16 or s['displacement']<=0 or s['displacement']%2!=1 or s['displacement']!=len(old)-len(edited) or s['old_tokens']!=len(old) or s['edited_tokens']!=len(edited) or auser!=buser or auser!=s['user_id']:raise ControlFailure('Selected odd geometry')
    ss,se,ds,de=(s[k] for k in ['source_start','source_end','destination_start','destination_end'])
    latest=max((i for i,x in enumerate(edited) if x==auser),default=-1)
    if not 0<=ds<de<=latest<len(edited) or not 0<=ss<se<=len(old) or ss-ds!=s['displacement'] or se-de!=s['displacement'] or se-ss!=s['reused'] or s['reused']<64 or s['fresh']!=len(edited)-de or s['fresh']<128 or latest!=s['latest_user_start'] or old[ss:]!=edited[ds:]:raise ControlFailure('Selected source/fresh User range')
    if s['budget']!=128<<20 or not 0<s['preallocation_peak_with_reserves']<=s['budget'] or s['current_payload_forecast']<=0:raise ControlFailure('Selection budget')
    for i in range(s['candidate']+1):
        c=load_json(p/f'candidate-{i:02d}.json')
        if type(c.get('candidate')) is not int or c['candidate']!=i or any(type(c.get(k)) is not bool for k in ['replaced','spliced','eligible']) or type(c.get('displacement')) is not int or c['eligible']!=(i==s['candidate']):raise ControlFailure('First eligible candidate order differs')
        if c['eligible'] and (not c['replaced'] or not c['spliced'] or c['displacement']!=s['displacement']):raise ControlFailure('Selected candidate operation/geometry differs')
    before=(p/'selected-before.txt').read_text();after=(p/'selected-after.txt').read_text()
    from_update=apply_frozen_text(before,task,s['candidate'])
    if from_update!=after or len(before.encode())!=s['old_text_bytes'] or len(after.encode())!=s['edited_text_bytes']:raise ControlFailure('Frozen text operation differs')
    return dict(selection=s,old_ids_sha256=sha(p/'selected-old.ids'),edited_ids_sha256=sha(p/'selected-edited.ids'),before_sha256=sha(p/'selected-before.txt'),after_sha256=sha(p/'selected-after.txt'))

def apply_frozen_text(text,task,index):
    for old,new in [task['edit']['replace_exact_unique_fact'],task['edit']['replace_exact_unique_requirement'],[task['edit']['candidate_order'][index]['delete_padding_prefix'],'']]:
        if not old or text.count(old)!=1:raise ControlFailure('Frozen edit nonunique')
        text=text.replace(old,new,1)
    return text

def audit_calls(records,project):
    third=[r for r in records if r['turn']==3];calls=[c for r in third for c in r['calls']]
    if not calls:raise ControlFailure('No updater tool call')
    first=calls[0];args={a['name']:a['value'] for a in first['args']}
    allowed=[['python3','update_context.py'],['python3','./update_context.py'],['./update_context.py'],[str(PYTHON),'update_context.py']]
    if len(third[0]['calls'])!=1 or first['name']!='bash' or shlex.split(args.get('command','')) not in allowed:raise ControlFailure('First tool was not the single frozen updater invocation')
    forbidden=['update_context.py','task.json','selection.json','grader.py','live-context','selected-before','selected-after','DS4_AGENT_PAIR_ACTUAL_']
    readers={'cat','sed','head','tail','grep','rg','awk','strings','less','more','open','read_text','read_bytes'}
    grader_invocations=0
    for c in calls[1:]:
        args={a['name']:a['value'] for a in c['args']};value=' '.join(args.values())
        # Explicitly permit executing only the wrapper; reading its target or
        # task/update/live files contaminates this one semantic probe.
        if c['name'] in ('read','read_file','glob','grep') and any(x in value for x in forbidden):raise ControlFailure('Golden/live-source read attempt retained as contaminated')
        if c['name']=='bash':
            tokens=shlex.split(args.get('command',''))
            if any(x in value for x in forbidden) and any(t in readers or any(r in t for r in ['read_text','read_bytes','open(']) for t in tokens):raise ControlFailure('Golden/live-source bash read retained as contaminated')
            interpreters=('python3',str(PYTHON))
            grader_invocations+=sum(tokens[i] in interpreters and tokens[i+1] in ('run_grader.py','./run_grader.py') for i in range(len(tokens)-1))
            grader_invocations+=sum(t=='./run_grader.py' and (i==0 or tokens[i-1] not in interpreters) for i,t in enumerate(tokens))
    if grader_invocations!=1:raise ControlFailure('Exactly one explicit immutable grader-wrapper invocation required')
    return dict(first_updater_only=True,explicit_read_audit_passed=True,grader_wrapper_invocations=grader_invocations,security_sandbox_claimed=False)

def structural_trace(trace,selection):
    text=Path(trace).read_text();pat=r'context pair suffix APPROXIMATE reused=(\d+) fresh=(\d+) source=(\d+):(\d+) destination=(\d+):(\d+) rebuilt_pairs=(\d+) boundary_pairs=(\d+) copied_bytes=(\d+) unique_peak=(\d+) budget=(\d+) latest_user_start=(\d+) latest_user_fresh=(\d)'
    matches=re.findall(pat,text)
    if len(matches)!=1:raise ControlFailure('Actual pair reuse absent/duplicate; exact fallback is not reuse success')
    v=list(map(int,matches[0]));s=selection
    if v[0]!=s['reused'] or v[2:6]!=[s['source_start'],s['source_end'],s['destination_start'],s['destination_end']] or v[1]<128 or v[6]<=0 or not 0<v[9]<=128<<20 or v[10]!=128<<20 or v[11]!=s['latest_user_start'] or v[12]!=1:raise ControlFailure('Actual committed geometry/User/budget differs')
    if 'context compaction' in text or 'compaction summary' in text:raise ControlFailure('Unplanned compaction retained')
    return dict(reused=v[0],fresh=v[1],source_start=v[2],source_end=v[3],destination_start=v[4],destination_end=v[5],rebuilt_pairs=v[6],boundary_pairs=v[7],copied_bytes=v[8],unique_peak=v[9],budget=v[10],latest_user_start=v[11],latest_user_fresh=True)

def immutable_project(project,inputs):
    for name,pin in inputs.items():
        f=project/name
        if not f.is_file() or sha(f)!=pin:raise ControlFailure('Fixture mutation: '+name)

def delivery_payload(prompt):
    # Headless production stdin preserves the complete collected buffer;
    # worker_submit then duplicates its C string. NUL is forbidden explicitly,
    # while every legitimate multiline/trailing LF byte remains untouched.
    if type(prompt) is not str or '\0' in prompt:raise ControlFailure('NUL/nontext in preregistered protocol prompt')
    payload=(prompt+'\n').encode('utf-8')
    if len(payload)>4096:raise ControlFailure('Preregistered multiline prompt exceeds one atomic pipe write')
    return payload

def drive(argv,prompts,output,project,private,trace,env,inputs,checker,timeout=900,cleanup_reserve=12):
    output=Path(output);deadline=time.monotonic()+timeout;work_deadline=deadline-cleanup_reserve;proc=None;sel=None;submitted=0;completed=[];status='RUNNING';phase='start';error=None;cleanup=dict(attempted=False,error=None)
    try:
        with (output/'stdout.log').open('xb') as log:
            proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            sel=selectors.DefaultSelector();sel.register(proc.stdout,selectors.EVENT_READ);pending=b''
            while sel.get_map():
                phase='agent_output'
                for key,_ in sel.select(min(.2,remaining(work_deadline))):
                    block=os.read(key.fd,65536)
                    if not block:sel.unregister(key.fileobj);break
                    log.write(block);log.flush();pending+=block
                    if len(pending)>2<<20:raise ControlFailure('Output framing bound exceeded')
                    if b'PAIR_ACTUAL_STATUS_ERROR' in pending:raise ControlFailure('Private observation write failed')
                    if b'PAIR_ACTUAL_ROUND ' in block and (private/'rounds.jsonl').exists():validate_rounds(rounds(private/'rounds.jsonl'))
                    marker=b'+DWARFSTAR_WAITING'
                    while marker in pending:
                        _,pending=pending.split(marker,1)
                        if submitted:
                            if len(completed)>=submitted:raise ControlFailure('Duplicate WAITING completion retained')
                            phase='completed_turn_validation';immutable_project(project,inputs)
                            if (private/'selection-failure.txt').exists():raise ControlFailure('Frozen structural selection failure retained')
                            records=rounds(private/'rounds.jsonl');validate_rounds(records,submitted)
                            if submitted<3:checked=validate_control(trace,records,submitted)
                            else:checked=checker(records,work_deadline)
                            completed.append(checked);put(output/f'turn-{submitted}-check.json',checked)
                        if submitted<3:
                            prompt=prompts[submitted];payload=delivery_payload(prompt)
                            phase='submit_turn';n=os.write(proc.stdin.fileno(),payload)
                            if n!=len(payload):raise ControlFailure('Partial prompt write; no retry')
                            submitted+=1
                        elif not proc.stdin.closed:phase='EOF';proc.stdin.close()
            phase='agent_exit';rc=proc.wait(timeout=remaining(work_deadline))
            if rc!=0:
                if (private/'selection-failure.txt').exists():raise ControlFailure('Frozen structural selection failure retained')
                raise RuntimeError('Agent infrastructure/process exit differs: '+str(rc))
            if submitted!=3 or len(completed)!=3:raise ControlFailure('Agent lacks three completed turns')
        immutable_project(project,inputs);status='SUCCESS';return 0
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc);status='TIMEOUT' if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)) else 'FIRST_FAILURE_RETAINED';(output/'traceback.txt').write_text(traceback.format_exc());return 3 if isinstance(exc,ControlFailure) else 124 if status=='TIMEOUT' else 1
    finally:
        if proc is not None and proc.returncode is None:
            phase_before_cleanup=phase
            cleanup['attempted']=True
            try:stop_process(proc)
            except BaseException as exc:
                cleanup['error']=type(exc).__name__+': '+str(exc);error=(error or '')+'; cleanup '+cleanup['error']
            phase=phase_before_cleanup
        if sel is not None:sel.close()
        if proc is not None:
            for stream in (proc.stdin,proc.stdout):
                if stream is not None and not stream.closed:stream.close()
        cleanup['leader_reaped']=proc is None or proc.returncode is not None
        put(output/'controller-outcome.json',dict(status=status,phase=phase,error=error,cleanup=cleanup,submitted=submitted,completed=[c['turn'] for c in completed],overall_success=status=='SUCCESS',partial_gold_checks=completed,deadline_seconds=timeout,cleanup_reserve_seconds=cleanup_reserve,elapsed_seconds=timeout-(deadline-time.monotonic()),agent_returncode=proc.returncode if proc else None,persistence='NOT_EXERCISED',first_attempt=True,no_retries=True,corrective_infrastructure_attempt=CORRECTION))

def run(p,digest):
    plan=verify(p,digest);require_guard(p,digest);require_root_grant(p,digest);identity=model_identity() # FIRST model path access
    output=p/'run';output.mkdir(exist_ok=False);project=output/'project';project.mkdir();private=output/'private';private.mkdir();cache=output/'cache';cache.mkdir();trace=output/'trace.log';live=output/'live-context.txt'
    for name in ['update_context.py','run_grader.py']:shutil.copy2(p/name,project/name);(project/name).chmod(0o755)
    inputs={name:sha(project/name) for name in ['update_context.py','run_grader.py']};task=json.loads((p/'task.json').read_text());prompts=[(p/name).read_text() for name in ['first.txt','second.txt','third.txt']]
    if prompts!=[task['first_user_text'],task['second_user'],task['third_user_text']] or '3141' in prompts[2]:raise ValueError('Literal preregistered prompts differ/contain gold')
    env={k:v for k,v in os.environ.items() if not k.startswith('DS4')};env.update(POLICY);env.update(plan['shader_env']);env.update(DS4_AGENT_PAIR_ACTUAL_OUTPUT=str(private),DS4_AGENT_PAIR_ACTUAL_LIVE_FILE=str(live),DS4_AGENT_PAIR_ACTUAL_TASK=str(p/'task.json'),DS4_AGENT_PAIR_ACTUAL_GRADER=str(p/'grader.py'),DS4_AGENT_PAIR_ACTUAL_PYTHON=str(PYTHON),DS4_AGENT_CACHE_DIR=str(cache),PYTHONDONTWRITEBYTECODE='1',TZ='America/New_York')
    argv=[str(p/'agent'),'--model',identity['path'],'--ssd-streaming','--ssd-streaming-cache-experts','64GB','--prefill-chunk','256','--non-interactive','--ctx','8192','--seed','41','--temp','0','--nothink','-n','512','--chdir',str(project),'--trace',str(trace),'--live-context-file',str(live)]
    put(output/'command.json',dict(argv=argv,env={k:v for k,v in env.items() if k.startswith('DS4')},model_identity=identity));put(output/'immutable-inputs.json',inputs)
    def checker(records,deadline):
        selection=validate_selection(private,task);audit=audit_calls(records,project);structural=structural_trace(trace,selection['selection']);updated=json.loads((private/'update-result.json').read_text())
        if updated.get('status')!='UPDATED' or updated.get('candidate')!=selection['selection']['candidate']:raise ControlFailure('Actual updater failure/different selection')
        before=(private/'update-before.txt').read_text();after=(private/'update-after.txt').read_text()
        if apply_frozen_text(before,task,updated['candidate'])!=after or sha(private/'update-before.txt')!=updated['before_sha256'] or sha(private/'update-after.txt')!=updated['after_sha256']:raise ControlFailure('Updater operation/history bytes differ')
        if (private/'grader-command-failure.json').exists():raise ControlFailure('First grading timeout/missing/duplicate/failure retained')
        started=load_json(private/'grader-started.json');graded=load_json(private/'grader-command-result.json')
        attempts=[decode_json(line) for line in (private/'grader-attempts.jsonl').read_text().splitlines()]
        if started.get('status')!='FIRST_GRADING_STARTED' or len(attempts)!=1 or attempts[0].get('passed') is not True or graded.get('passed') is not True or type(graded.get('returncode')) is not int or graded['returncode']!=0 or started.get('code_sha256')!=sha(project/'access.py') or attempts[0].get('source_sha256')!=started['code_sha256'] or sha(private/'first-graded-access.py')!=started['code_sha256']:raise ControlFailure('First external gold/file attempt failed/mutated/missing; later repair not accepted')
        result=checked_command([str(PYTHON),str(p/'grader.py'),str(project),'--check-only'],deadline,env=env)
        put(output/'final-grader.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode!=0 or result.stdout.strip()!='PASS':raise ControlFailure('Immutable final unit gold failed')
        return dict(turn=3,passed=True,selection=selection,structural=structural,audit=audit,unit_gold=True,agent_grader_attempts=len(attempts),code_sha256=sha(project/'access.py'),scope='one-case semantic compaction only; no forgetting/state-exact/speed claim',persistence='NOT_EXERCISED')
    require_guard(p,digest);require_root_grant(p,digest) # Recheck the one child grant immediately before launch.
    rc=drive(argv,prompts,output,project,private,trace,env,inputs,checker)
    verify(p,digest);require_guard(p,digest);require_root_grant(p,digest)
    if model_identity()!=identity:raise ValueError('Model changed during task')
    return rc


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--run',type=Path);a.add_argument('--preflight',type=Path);a.add_argument('--manifest-sha256',required=True);v=a.parse_args()
    if bool(v.run)==bool(v.preflight):a.error('Choose run or preflight')
    p=(v.run or v.preflight).resolve()
    if v.preflight:verify(p,v.manifest_sha256);print('PREFLIGHT_PASS; no model metadata or child')
    else:raise SystemExit(run(p,v.manifest_sha256))
