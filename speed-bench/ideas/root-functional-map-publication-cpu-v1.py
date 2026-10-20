"""ROOT grants only supplied synthetic DS4FMAP1/DS4FQAL1 qualification CPU16."""
import hashlib,json,os,pathlib,stat,subprocess,time
MAIN=pathlib.Path('/Volumes/Models/ds4');WS=MAIN/'.workspaces/idea-resident-functional-map-publication'
B=WS/'speed-bench/ideas/resident_functional_map_publication_cpu_prepared_v1'
OUT=WS/'speed-bench/ideas/artifacts/resident-functional-map-publication-cpu-v1'
FREEZE='22ca80e1bbeec688d5474069b1d39216d79de6b5'
PSHA='bdb01425b05dc99db34a2659d466f4b2cad17a3d7637b617b0ca0cec491f86c4'
def sha(p):
    p=pathlib.Path(p);assert p.suffix.lower()!='.gguf'
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):
    s=p.lstat();assert stat.S_ISREG(s.st_mode)
    return dict(bytes=s.st_size,mode=stat.S_IMODE(s.st_mode),sha256=sha(p))
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
assert sha(B/'cpu-plan.json')==PSHA
plan=json.loads((B/'cpu-plan.json').read_bytes());idx=json.loads((B/'physical-index.json').read_bytes());files={B/n:r for n,r in idx['indexed_files'].items()};files[B/'physical-index.json']=pin(B/'physical-index.json');assert len(files)==11 and plan['expected_test_methods']==16 and len(plan['commands'])==1
for p,r in files.items():
    assert pin(p)==r,str(p)
    x=subprocess.run(['jj','--ignore-working-copy','--color','never','file','show','-r',FREEZE,str(p.relative_to(WS))],cwd=WS,capture_output=True,check=True);assert x.stdout==p.read_bytes()
tree=subprocess.run(['jj','--ignore-working-copy','--color','never','debug','tree','-r',FREEZE,*['file:'+str(p.relative_to(WS)) for p in files]],cwd=WS,capture_output=True,text=True,check=True).stdout
for p in files:
    row=[x for x in tree.splitlines() if x.startswith(str(p.relative_to(WS))+': ')];assert len(row)==1 and 'executable: false' in row[0]
def verify():
    for p,r in files.items():assert pin(p)==r,str(p)
    for p,r in plan['pins'].items():assert pin(pathlib.Path(p))==r,p
verify();assert not OUT.exists();OUT.mkdir(parents=True)
c=plan['commands'][0];assert c['timeout_seconds']==60
write(OUT/'root-grant.json',dict(status='ROOT_GRANTS_SYNTHETIC_MAP_COMPANION_CPU16_ONLY',time=time.time(),source_freeze=FREEZE,plan_sha256=PSHA,command=c,ROOT_executor_sha256=sha(__file__),no_actual_capture_signature_model_metadata_helper_GPU_compiler_guard=True,diagnostic_CPU_overlap_with_CLM=True,no_retry=True))
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))};env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(OUT));start=time.monotonic();so=OUT/'command.stdout';se=OUT/'command.stderr'
with so.open('xb') as stdout,se.open('xb') as stderr:
    try:p=subprocess.run(c['argv'],cwd=c['cwd'],env=env,stdout=stdout,stderr=stderr,timeout=60);rc=p.returncode
    except subprocess.TimeoutExpired:rc=124
verify();passed=rc==0 and b'Ran 16 tests' in se.read_bytes() and se.read_bytes().rstrip().endswith(b'OK')
proof=dict(status='FUNCTIONAL_MAP_PUBLICATION_SYNTHETIC_CPU16_PASS' if passed else 'FIRST_SYNTHETIC_FAILURE_RETAINED',source_freeze=FREEZE,plan_sha256=PSHA,source_proof_sha256=sha(B/'source-proof.json'),ROOT_grant_sha256=sha(OUT/'root-grant.json'),argv=c['argv'],cwd=c['cwd'],returncode=rc,wall_seconds_diagnostic=time.monotonic()-start,stdout=pin(so),stderr=pin(se),all11_literal_and_plan_pins_before_after_exact=True,no_actual_capture_signature_model_metadata_helper_GPU_compiler_guard=True,no_native_coefficient_bits_or_quality_proof=True,no_retry=True)
write(OUT/'root-proof.json',proof);print(json.dumps({k:proof[k] for k in ('status','returncode','wall_seconds_diagnostic')}));print('PROOF',sha(OUT/'root-proof.json'));raise SystemExit(0 if passed else 3)
