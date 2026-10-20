"""ROOT: eight synthetic contracts, exclusive static preparation, static verification."""
import hashlib,json,os,pathlib,stat,subprocess,time
ROOT=pathlib.Path('/Volumes/Models/ds4')
WS=ROOT/'.workspaces/idea-clm-block-oracle-materialization'
B=WS/'speed-bench/clm-block-oracle-materialization-v1'
FREEZE='363978d202d9bcd159fc9b50227d3b5f45414b12'
FW='/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python'
OUT=WS/'speed-bench/ideas/artifacts/clm-block-oracle-materialization-root-v1'
def sha(p):
    p=pathlib.Path(p);assert p.suffix.lower()!='.gguf'
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):
    p=pathlib.Path(p);s=p.lstat();assert stat.S_ISREG(s.st_mode)
    return dict(bytes=s.st_size,mode=stat.S_IMODE(s.st_mode),sha256=sha(p))
def check(r):assert pin(r['path'])=={k:r[k] for k in ('bytes','mode','sha256')},r['path']
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert sha(B/'static-proposal.json')=='71bcf94d7b67641031948a659fd749f6ce56cf80e4bad6199f9c792e58dd1389'
idx=json.loads((B/'source-index.json').read_bytes());plan=json.loads((B/'static-proposal.json').read_bytes())
files={B/n:r for n,r in idx['files_excluding_this_index'].items()};files[B/'source-index.json']=dict(path=str(B/'source-index.json'),**pin(B/'source-index.json'));assert len(files)==8
for p,r in files.items():
    check(r)
    x=subprocess.run(['jj','--ignore-working-copy','--color','never','file','show','-r',FREEZE,str(p.relative_to(WS))],cwd=WS,capture_output=True,check=True);assert x.stdout==p.read_bytes()
tree=subprocess.run(['jj','--ignore-working-copy','--color','never','debug','tree','-r',FREEZE,*['file:'+str(p.relative_to(WS)) for p in files]],cwd=WS,capture_output=True,text=True,check=True).stdout
for p in files:
    row=[x for x in tree.splitlines() if x.startswith(str(p.relative_to(WS))+': ')];assert len(row)==1 and 'executable: false' in row[0]
a=json.loads((B/'input-authorities.json').read_bytes());s=json.loads((B/'shader-authority.json').read_bytes())
pins=list(files.values())+list(a['literal_package_inputs'].values())+list(a['external_source_and_objects'].values())+[a['actual_archive'],s['qualified_manifest'],s['host_object_authority']]+list(s['shader_env_to_literal_source'].values())+[dict(path=FW,bytes=33568,mode=493,sha256='b80d260a48404da13a32dd44d7283aadc80f6a6e56ed6e03ed5597ec6f46d48d')]
def verify():
    for r in pins:check(r)
verify();assert not OUT.exists() and not pathlib.Path(plan['exclusive_prepared_output']).exists();OUT.mkdir(parents=True)
commands=[dict(label='synthetic8',**plan['CPU_command']),dict(label='static-prepare',**plan['static_materialization_command'])]
write(OUT/'root-grant.json',dict(status='ROOT_GRANTS_SYNTHETIC8_AND_STATIC_PREPARATION_ONLY',time=time.time(),source_freeze=FREEZE,plan_sha256=sha(B/'static-proposal.json'),commands=commands,ROOT_executor_sha256=sha(__file__),no_model_metadata_helper_guard_GPU_compiler=True,no_retry=True))
results=[];passed=True
for i in range(3):
    if i==2:
        recipe=json.loads((pathlib.Path(plan['exclusive_prepared_output'])/'static-verify-command.json').read_bytes());assert recipe['no_model_metadata'] is True;commands.append(dict(label='static-verify',**recipe))
    c=commands[i];verify();env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))};env.update(c['env']);env.update(TMPDIR=str(OUT),PYTHONDONTWRITEBYTECODE='1');so=OUT/('command%d.stdout'%i);se=OUT/('command%d.stderr'%i);start=time.monotonic()
    with so.open('xb') as stdout,se.open('xb') as stderr:
        try:r=subprocess.run(c['argv'],cwd=c['cwd'],env=env,stdout=stdout,stderr=stderr,timeout=120);rc=r.returncode
        except subprocess.TimeoutExpired:rc=124
    verify();rr=dict(label=c['label'],argv=c['argv'],cwd=c['cwd'],returncode=rc,wall_seconds_diagnostic=time.monotonic()-start,stdout=pin(so),stderr=pin(se));results.append(rr);print(json.dumps({k:rr[k] for k in ('label','returncode','wall_seconds_diagnostic')}),flush=True)
    if rc!=0:passed=False;break
    if i==0:assert b'Ran 8 tests' in se.read_bytes() and se.read_bytes().rstrip().endswith(b'OK')
proof=dict(status='CLM_ORACLE_STATIC_PREPARATION_PASS' if passed else 'FIRST_STATIC_FAILURE_RETAINED',source_freeze=FREEZE,plan_sha256=sha(B/'static-proposal.json'),source_index_sha256=sha(B/'source-index.json'),ROOT_grant_sha256=sha(OUT/'root-grant.json'),results=results,all_pins_unchanged=True,source_literal_files=8,external_pin_checks=len(pins)-8,no_model_metadata_helper_guard_GPU_compiler=True,no_retry=True)
write(OUT/'root-proof.json',proof);print(json.dumps(dict(status=proof['status'],proof_sha256=sha(OUT/'root-proof.json'))));raise SystemExit(0 if passed else 3)
