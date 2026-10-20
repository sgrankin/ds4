"""ROOT exclusive five-arm CLM native oracle; retained first outcome, owned process."""
import fcntl,hashlib,importlib.util,json,os,pathlib,shutil,signal,stat,subprocess,time
MAIN=pathlib.Path('/Volumes/Models/ds4');WS=MAIN/'.workspaces/idea-clm-block-oracle-materialization'
P=WS/'speed-bench/ideas/artifacts/clm-block-oracle-native-prepared-v1'
MSHA='5849ca662443d2a1f0ca5b21eb27bf6f1909440e7dd25cab522d976483c0f298'
def sha(p):
    p=pathlib.Path(p);assert p.suffix.lower()!='.gguf'
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):
    p=pathlib.Path(p);s=p.lstat();assert stat.S_ISREG(s.st_mode)
    return dict(bytes=s.st_size,mode=stat.S_IMODE(s.st_mode),sha256=sha(p))
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert sha(P/'manifest.json')==MSHA
static=WS/'speed-bench/ideas/artifacts/clm-block-oracle-materialization-root-v1/root-proof.json'
assert sha(static)=='bddf0d40d9c817f89819a58b375938148357c3ad3e96d3a920b65ac44c99d545' and json.loads(static.read_bytes())['status']=='CLM_ORACLE_STATIC_PREPARATION_PASS'
m=json.loads((P/'manifest.json').read_bytes());c=json.loads((P/'prepared-command.json').read_bytes());command=c['argv']
assert c['manifest_sha256']==MSHA and c['cwd']==str(MAIN) and command[1]==str(MAIN/'speed-bench/ideas/pressure_run.py') and command[2]==str(P/'pressure') and command[3]=='--' and command[0]==command[4]
assert sha(command[0])=='b80d260a48404da13a32dd44d7283aadc80f6a6e56ed6e03ed5597ec6f46d48d' and sha(command[1])=='461e8c517e82a687b4579fc450c1e7128b1510192d088bab719682b33c946c3c'
index=json.loads((P/'publication-index.json').read_bytes());literals={P/n:r for n,r in index['files_excluding_this_index'].items()};literals[P/'publication-index.json']=pin(P/'publication-index.json')
runtimefile=MAIN/'speed-bench/ideas/evidence/reasoning-budget-pilot-root-import-20261003.json';assert sha(runtimefile)=='2f4c1ee85cd8bf6c411366f7907c406de12d7e18aa2a4af2efc28de462538360';runtime=json.loads(runtimefile.read_bytes())['runtime_and_shaders_unchanged'];assert len(runtime)==100
def verify():
    for p,r in literals.items():assert pin(p)==r,str(p)
    for p,r in m['external_nonmodel_files'].items():assert pin(p)==r,p
    for p,d in runtime.items():assert sha(MAIN/p)==d,p
verify();prior=MAIN/'.workspaces/idea-practical-write-native-control-runner-v1/speed-bench/ideas/artifacts/practical-write-native-controls-prepared-v3/root-release-proof.json';assert sha(prior)=='1610427449cd00f8a53c4b4fbd17a86a2f7f7834a58fa09cf49531660a9cd3d6' and json.loads(prior.read_bytes())['MAIN_job_lock_reacquired'] is True
assert not (P/'root-launch-proof.json').exists() and not (P/'run').exists() and not (P/'pressure').exists()
def jj(args):return subprocess.run(['jj','--config','snapshot.auto-track="none()"','--color','never',*args],cwd=WS,capture_output=True,text=True,check=True).stdout
with (MAIN/'speed-bench/ideas/artifacts/jj-mutation.lock').open('r+') as f:
    fcntl.flock(f,fcntl.LOCK_EX);paths=[str(p.relative_to(WS)) for p in literals];jj(['file','track','--include-ignored',*paths]);jj(['describe','-m','CLM native oracle prepared packet: static8 PASS and exact core/build authority']);freeze=jj(['log','--no-graph','-r','@','-T','commit_id']).strip();jj(['new','-m','Protected empty child after CLM oracle packet freeze'])
for p in literals:
    x=subprocess.run(['jj','--ignore-working-copy','--color','never','file','show','-r',freeze,str(p.relative_to(WS))],cwd=WS,capture_output=True,check=True);assert x.stdout==p.read_bytes()
tree=subprocess.run(['jj','--ignore-working-copy','--color','never','debug','tree','-r',freeze,*['file:'+str(p.relative_to(WS)) for p in literals]],cwd=WS,capture_output=True,text=True,check=True).stdout
for p,r in literals.items():
    row=[x for x in tree.splitlines() if x.startswith(str(p.relative_to(WS))+': ')];assert len(row)==1 and ('executable: true' if r['mode']&0o111 else 'executable: false') in row[0]
def inventory():
    ps=subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,command='],text=True,timeout=2);pc=subprocess.check_output(['/bin/ps','-axo','pid=,comm='],text=True,timeout=2)
    rows={int(f[0]):(int(f[1]),f[2]) for l in ps.splitlines() if len(f:=l.strip().split(None,2))==3};names={int(f[0]):f[1] for l in pc.splitlines() if len(f:=l.strip().split(None,1))==2};anc=[];pid=os.getpid()
    while pid:assert pid not in anc and pid in rows;anc.append(pid);pid=rows[pid][0]
    prefix=('ds4','quality-','replay-','budget-','probe-agent-','agent-suffix-','ragged-old','ragged-new','practical-agent','practical-write-agent','functional-','resident-functional-constructor','capture-native','capture-private','scanner-native','probe-old','probe-new')
    live=[pid for pid,(_,c) in rows.items() if pid not in anc and ('gguf/DeepSeek-V4.1-Flash-Q2.gguf' in c or '/speed-bench/ideas/pressure_run.py' in c or pathlib.Path(names.get(pid,'')).name.startswith(prefix))];assert not live,live
    return anc,hashlib.sha256(ps.encode()).hexdigest(),hashlib.sha256(pc.encode()).hexdigest()
anc,pssha,pcsha=inventory();s=importlib.util.spec_from_file_location('root_clm_pressure',MAIN/'speed-bench/ideas/pressure_run.py');pressure=importlib.util.module_from_spec(s);s.loader.exec_module(pressure)
with (MAIN/'speed-bench/ideas/artifacts/model-job.lock').open('r+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);sample=pressure.sample();assert sample['level']==1 and pressure.pressure_reason(sample,sample['swap_gb']) is None and shutil.disk_usage(MAIN).free>40*2**30
    grant=dict(status='GRANTED',time=time.time(),sole_model_grant=True,manifest_sha256=MSHA,command=command,cwd=str(MAIN),prepared_source_freeze=freeze,prepared_literal_files=len(literals),static_proof_sha256=sha(static),ROOT_executor_sha256=sha(__file__),previous_release_sha256=sha(prior),fresh_ps_args_sha256=pssha,fresh_ps_comm_sha256=pcsha,all_ancestors_excluded=anc,live_model_processes=[],fresh_pressure=sample,MAIN_runtime100_before_exact=True,quiet_timing_grant=False,owned_whole_timeout_seconds=900,owned_native_helper_timeout_seconds=600,no_retry=True)
    write(P/'root-launch-proof.json',grant);fcntl.flock(lock,fcntl.LOCK_UN)
scratch=P/'root-scratch';scratch.mkdir();env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))};env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(scratch));start=time.monotonic();timedout=False
with (P/'root-run.stdout').open('xb') as so,(P/'root-run.stderr').open('xb') as se:
    proc=subprocess.Popen(command,cwd=MAIN,env=env,stdout=so,stderr=se,start_new_session=True);write(P/'root-owned-process.json',dict(pid=proc.pid,pgid=proc.pid,start_wall=time.time(),ROOT_pid=os.getpid(),command=command))
    try:rc=proc.wait(timeout=900)
    except subprocess.TimeoutExpired:
        timedout=True;os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        rc=124
verify();proof=dict(status='ROOT_CLM_ORACLE_COMPLETED_PENDING_INDEPENDENT_AUDIT' if rc==0 and not timedout else 'FIRST_CLM_ORACLE_FAILURE_RETAINED',returncode=rc,owned_timeout=timedout,wall_seconds_diagnostic=time.monotonic()-start,manifest_sha256=MSHA,prepared_source_freeze=freeze,launch_proof_sha256=sha(P/'root-launch-proof.json'),stdout_sha256=sha(P/'root-run.stdout'),stderr_sha256=sha(P/'root-run.stderr'),all_packet_and_external_pins_before_after_exact=True,MAIN_runtime100_before_after_exact=True,quiet_timing_grant=False,no_retry=True)
write(P/'root-execution-proof.json',proof);print(json.dumps(proof),flush=True);print('PROOF',sha(P/'root-execution-proof.json'),flush=True)
anc,_,_=inventory()
with (MAIN/'speed-bench/ideas/artifacts/model-job.lock').open('r+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
guard=json.loads((P/'pressure/result.json').read_bytes());samples=[json.loads(x) for x in (P/'pressure/pressure.jsonl').read_bytes().splitlines()];assert samples and all(x['level']==1 and x['reason'] is None and x['consecutive_bad']==0 for x in samples)
initial=json.loads((P/'pressure/command.json').read_bytes())['initial'];swaps=[initial['swap_gb']]+[x['swap_gb'] for x in samples]
release=dict(status='RELEASED',time=time.time(),execution_sha256=sha(P/'root-execution-proof.json'),guard_result=guard,MAIN_job_lock_reacquired=True,live_model_or_guard_processes=[],all_ancestors_excluded=anc,normal_pressure_samples=len(samples),bad_pressure_samples=0,swapgrowth_gib=max(swaps)-swaps[0],peak_wired_plus_compressed_gib=max(x['wired_gb']+x['compressor_gb'] for x in samples),pressure_sha256=sha(P/'pressure/pressure.jsonl'),no_model_or_raw_payload_read_no_retry=True)
write(P/'root-release-proof.json',release);print(json.dumps({k:release[k] for k in ('status','normal_pressure_samples','bad_pressure_samples','swapgrowth_gib','peak_wired_plus_compressed_gib')}),flush=True);print('RELEASE',sha(P/'root-release-proof.json'),flush=True);raise SystemExit(rc)
