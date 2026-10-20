#!/usr/bin/env python3
"""Fixed untouched confirmation16: fresh native REF, broader3, native repeat."""
import argparse,errno,fcntl,hashlib,importlib.util,json,math,os,re,shlex,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/Volumes/Models/ds4'); GUARD=ROOT/'speed-bench/ideas/pressure_run.py'
GUARD_SHA='461e8c517e82a687b4579fc450c1e7128b1510192d088bab719682b33c946c3c'
GUARD_INTERPRETER=Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python')
GUARD_INTERPRETER_SHA='b80d260a48404da13a32dd44d7283aadc80f6a6e56ed6e03ed5597ec6f46d48d'
FREEZE=ROOT/'speed-bench/ideas/evidence/confirmation16-input-freeze-20261002'
FREEZE_SHA='f46b45e17bc110d6b66fdf125b53935e8b16849f437d8c04bf9fa910c9b3c5de'
TSV_SHA='026134fbeb20786e1736db71ffc85de46515823112dabbaf9859b98a57c296b9'
PRIOR=ROOT/'speed-bench/ideas/artifacts/resident-broad-independent16-20261002/manifest.json'
BUILD=ROOT/'.workspaces/idea-resident-three-confirmation/speed-bench/ideas/artifacts/resident-three-confirmation-build'
BUILD_SHA='4ef05d5c09c9f20d1ea5fcf47b103d4b1f7a8fbcbdbf2f0b14f6ec7a86a8255b'
BINARY_SHA='1ced202ca2ddf3d3394007b1bfafae4cea7ae4c27c8f4173c11b650fdb1dacd5'
RAW=ROOT/'.workspaces/idea-fixture-raw-hardened/speed-bench/ideas/artifacts/fixture-raw-hardened-20261002-v1/run_fixture_raw_matrix.py'
RAW_SHA='dbeae546e41e6443bff714265b5be64bc2ec4b15b78c0ca8e71afb044a5eb132'
SETTINGS=dict(cases=16,tokens=32,rows=512,ctx=4096,cache_gb=64,chunk=512,tile=64,limit=1024)
IO_FIELDS=set('enabled pread_calls requested_bytes returned_bytes cache_loader_bytes cache_hits cache_misses cache_evictions'.split())

def sha(path):
    if str(path).lower().endswith('.gguf'):raise ValueError('model hashing forbidden')
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def raw_helpers(path=RAW):
    if sha(path)!=RAW_SHA:raise ValueError('hardened raw helper changed')
    spec=importlib.util.spec_from_file_location('heldout_raw',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def arms():
    common={'DS4_METAL_V41_SELECTED_TILE':'64','DS4_METAL_V41_SELECTED_LIMIT':'1024','DS4_METAL_STREAMING_EXPERT_IO_METRICS':'1'}
    prefix='DS4_METAL_V41_RESIDENT_ROUTE_'
    native=dict(common,**{prefix+'GAP':'0',prefix+'MODE':'native'})
    candidate=dict(common,**{prefix+k:v for k,v in dict(GAP='.30',MAX_MASS='.20',MAX_TOTAL_MASS='.40',MAX_RANK='48',MAX_SUBS='3',MODE='apply',POLICY='replace').items()})
    return [('native-before',native),('broader3',candidate),('native-after',native)]

def parse_io(text):
    rows=[s[len('PROBE_IO '):] for s in text.splitlines() if s.startswith('PROBE_IO ')]
    if len(rows)!=1:raise ValueError('missing/duplicate exact I/O footer')
    io={}
    for item in rows[0].split():
        key,sep,value=item.partition('=')
        if not sep or key in io or not re.fullmatch('[0-9]+',value) or int(value)>2**64-1:raise ValueError('invalid exact I/O field')
        io[key]=int(value)
    if io.keys()!=IO_FIELDS or io['enabled']!=1 or io['requested_bytes']!=io['returned_bytes']:raise ValueError('I/O fields/enable/complete read mismatch')
    if any(s.startswith('ds4: resident-backfill ') or s.startswith('ds4: resident-backfill-pool ') for s in text.splitlines()):raise ValueError('unexpected background-original work')
    return io

def parse_route(text,name):
    rows=[s[len('ds4: resident-route '):] for s in text.splitlines() if s.startswith('ds4: resident-route ')]
    if name.startswith('native'):
        if rows:raise ValueError('native control unexpectedly routed')
        return None
    if len(rows)!=1:raise ValueError('candidate route footer missing/duplicate')
    integer=set('layers substitutions native_misses skipped_loads skipped_bytes full_snapshots double_layers planned_substitutions drops triple_layers'.split())
    floating=set('mean_mass mean_score_gap mean_affinity_ratio mean_rank'.split());result={}
    for item in rows[0].split():
        key,sep,value=item.partition('=')
        if not sep or key in result:raise ValueError('invalid route field')
        if key in integer:
            if not re.fullmatch('[0-9]+',value) or int(value)>2**64-1:raise ValueError('invalid route integer')
            result[key]=int(value)
        elif key in floating:
            result[key]=float(value)
            if not math.isfinite(result[key]) or result[key]<0:raise ValueError('invalid route float')
        else:raise ValueError('unknown route field')
    if result.keys()!=integer|floating or result['layers']!=20480 or not result['substitutions'] or result['substitutions']!=result['skipped_loads'] or result['substitutions']!=result['planned_substitutions'] or result['drops'] or 2*result['double_layers']+3*result['triple_layers']>result['substitutions']:
        raise ValueError('candidate routing work/accounting differs')
    return result

def command_template(output):
    return [str(GUARD_INTERPRETER),str(GUARD),str(output/'pressure'),'--',str(GUARD_INTERPRETER),str(output/'run_resident_confirmation.py'),'--run',str(output),'--manifest-sha256','<manifest_sha256>']

def guard_command(output,manifest_sha256):
    return command_template(output)[:-1]+[manifest_sha256]

def verified_plan(output,manifest_sha256):
    # The digest is an explicit prepared argv value, not a mutable file's
    # current digest. It is also part of the exact direct-parent guard suffix.
    if not re.fullmatch('[0-9a-f]{64}',manifest_sha256 or '') or sha(output/'manifest.json')!=manifest_sha256:
        raise ValueError('frozen plan changed or missing explicit manifest SHA')
    plan=json.loads((output/'manifest.json').read_text())
    command=json.loads((output/'prepared-command.json').read_text())
    if command.get('manifest_sha256')!=manifest_sha256 or command.get('command')!=guard_command(output,manifest_sha256):
        raise ValueError('prepared command/manifest authority differs')
    if plan.get('guard_command_template')!=command_template(output) or plan.get('guard_interpreter_identity')!=dict(path=str(GUARD_INTERPRETER),sha256=GUARD_INTERPRETER_SHA) or plan.get('guard_sha256')!=GUARD_SHA:
        raise ValueError('prepared guard identity differs')
    if plan['arms']!=[[n,e] for n,e in arms()] or plan['settings']!=SETTINGS or plan['binary_sha256']!=BINARY_SHA or plan['binary_manifest_sha256']!=BUILD_SHA or plan['source_input_freeze_sha256']!=FREEZE_SHA or plan['source_tsv_sha256']!=TSV_SHA:
        raise ValueError('frozen study/policy differs')
    return plan

def verify_required_assets(output,plan):
    files=plan['files']
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in files):raise ValueError('unsafe prepared asset path')
    if files.get('probe')!=BINARY_SHA or files.get('binary-manifest.json')!=BUILD_SHA or files.get('input-freeze.json')!=FREEZE_SHA or files.get('raw_helpers.py')!=RAW_SHA or files.get('run_resident_confirmation.py')!=sha(Path(__file__)):
        raise ValueError('mandatory prepared asset identity differs')
    frozen=json.loads((output/'input-freeze.json').read_text())
    if plan['inputs']!=frozen['cases'] or len(plan['inputs'])!=16:raise ValueError('frozen input hash/order differs')
    required={'probe','binary-manifest.json','input-freeze.json','raw_helpers.py','run_resident_confirmation.py','fixtures.tsv'}
    lines=[]
    for i,case in enumerate(plan['inputs']):
        paths=[]
        for kind in ['prompt','continuation']:
            name=f'inputs/{i:03d}-{kind}.txt';required.add(name);paths.append(str(output/name))
            if files.get(name)!=case[kind+'_sha256']:raise ValueError('mandatory input identity differs')
        lines.append('\t'.join([case['case'],*paths]))
    if (output/'fixtures.tsv').read_bytes()!=('\n'.join(lines)+'\n').encode():raise ValueError('frozen fixture paths/order differ')
    build=json.loads((output/'binary-manifest.json').read_text())
    metal={n:v for n,v in build['files'].items() if n.startswith('metal/')}
    if len(metal)!=26:raise ValueError('frozen shader count differs')
    for name,value in metal.items():
        required.add(name)
        if files.get(name)!=value:raise ValueError('mandatory shader identity differs')
    if set(files)!=required:raise ValueError('mandatory prepared asset set differs')

def require_main_guard(output,manifest_sha256):
    if sha(GUARD)!=GUARD_SHA:raise ValueError('MAIN guard changed')
    args=shlex.split(subprocess.check_output(['ps','-p',str(os.getppid()),'-o','command='],text=True,timeout=2))
    if not args or args[1:]!=guard_command(output,manifest_sha256)[1:]:raise RuntimeError('exact prepared MAIN guard must be direct parent')
    try:same=os.path.samefile(args[0],GUARD_INTERPRETER)
    except OSError:same=False
    if not same or sha(GUARD_INTERPRETER)!=GUARD_INTERPRETER_SHA:raise RuntimeError('direct parent interpreter differs from pinned framework Python')
    with (ROOT/'speed-bench/ideas/artifacts/model-job.lock').open('r+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EAGAIN,errno.EWOULDBLOCK):return
            raise
        fcntl.flock(lock,fcntl.LOCK_UN)
    raise RuntimeError('MAIN guard lock is not held')

def prepare(output):
    if sha(GUARD_INTERPRETER)!=GUARD_INTERPRETER_SHA:raise ValueError('pinned framework Python changed')
    if sha(FREEZE/'input-freeze.json')!=FREEZE_SHA or sha(FREEZE/'confirmation16.tsv')!=TSV_SHA:raise ValueError('untouched input freeze differs')
    frozen=json.loads((FREEZE/'input-freeze.json').read_text())
    if frozen['manifest_sha256']!=TSV_SHA or frozen['settings']!=dict(cases=16,target_tokens=32,ctx=4096,cache_gb=64,chunk=512) or len(frozen['cases'])!=16:raise ValueError('fixed confirmation geometry differs')
    if sha(BUILD/'manifest.json')!=BUILD_SHA or sha(BUILD/'fixture-three-confirmation-io')!=BINARY_SHA or sha(GUARD)!=GUARD_SHA:raise ValueError('frozen build/guard differs')
    build=json.loads((BUILD/'manifest.json').read_text())
    for name,value in build['files'].items():
        if sha(BUILD/name)!=value:raise ValueError('frozen build asset differs: '+name)
    prior=json.loads(PRIOR.read_text());model=prior['model']
    if model['path']!=str(ROOT/'gguf/DeepSeek-V4.1-Flash-Q2.gguf'):raise ValueError('stored explicit model identity differs')
    output.mkdir(parents=True,exist_ok=False)
    raw=raw_helpers();fixtures,inputs=raw.freeze_inputs(FREEZE/'confirmation16.tsv',output,16)
    if inputs!=frozen['cases']:raise ValueError('confirmation input hash/order differs')
    for source,name in [(BUILD/'fixture-three-confirmation-io','probe'),(BUILD/'manifest.json','binary-manifest.json'),(FREEZE/'input-freeze.json','input-freeze.json'),(RAW,'raw_helpers.py'),(Path(__file__),'run_resident_confirmation.py')]:shutil.copy2(source,output/name)
    (output/'metal').mkdir()
    for path in (BUILD/'metal').glob('*.metal'):shutil.copy2(path,output/'metal'/path.name)
    plan=dict(runtime_commit=build['runtime_commit'],binary_sha256=BINARY_SHA,binary_manifest_sha256=BUILD_SHA,settings=SETTINGS,arms=arms(),model_from_prior=model,inputs=inputs,source_input_freeze_sha256=FREEZE_SHA,source_tsv_sha256=TSV_SHA,guard_command_template=command_template(output),guard_interpreter_identity=dict(path=str(GUARD_INTERPRETER),sha256=GUARD_INTERPRETER_SHA),guard_sha256=GUARD_SHA,bootstrap=dict(unit='paired per-prompt mean deltaNLL',prompts=16,resamples=2000,seed=20261002,interval='sorted bootstrap indices49/1949'),model_opened_in_preparation=False,model_metadata_currently_verified=False,input_text_inspected_for_selection=False,reference='Fresh native-before on held-out inputs and identical explicit64 tile/limit runtime; no calibration reference reused',scope='Full512 unmodified logits and paired quality; cumulative whole-engine P+D explicit expert loader preads, no background worker/no physicalSSD or timing claim',files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    write(output/'manifest.json',plan)
    digest=sha(output/'manifest.json');command=guard_command(output,digest)
    write(output/'prepared-command.json',dict(command=command,manifest_sha256=digest,launch_requires_root_grant=True))
    print(json.dumps(dict(prepared=str(output),command=command)),flush=True)

def run(output,manifest_sha256):
    plan=verified_plan(output,manifest_sha256)
    require_main_guard(output,manifest_sha256) # No model metadata before ownership is proved.
    raw=raw_helpers(output/'raw_helpers.py')
    assets={output/n:('prepared '+n,v) for n,v in plan['files'].items()};assets[output/'manifest.json']=('manifest',manifest_sha256);assets[output/'prepared-command.json']=('prepared command',sha(output/'prepared-command.json'));assets[GUARD]=('MAIN guard',GUARD_SHA);assets[GUARD_INTERPRETER]=('framework interpreter',GUARD_INTERPRETER_SHA)
    raw.check_frozen_assets(assets)
    verify_required_assets(output,plan)
    model=Path(plan['model_from_prior']['path']);identity=raw.identity(model)
    if identity!=plan['model_from_prior']:raise ValueError('model identity differs')
    directory=output/'run';directory.mkdir(exist_ok=False);reference=directory/'reference.bin';metadata=None;ref_sha=None
    write(directory/'manifest.json',dict(plan,model_currently_verified=identity,guard_verified=True,driver_sha256=sha(Path(__file__))))
    env={k:v for k,v in os.environ.items() if not k.startswith('DS4')}
    for p in (output/'metal').glob('*.metal'):env['DS4_METAL_'+p.stem.upper()+'_SOURCE']=str(p)
    results=[]
    for name,policy in plan['arms']:
        require_main_guard(output,manifest_sha256)
        raw.check_frozen_assets(assets)
        if raw.identity(model)!=identity:raise ValueError('model identity changed')
        record=directory/(name+'-raw.bin')
        argv=[str(output/'probe'),str(model),str(output/'fixtures.tsv'),str(directory/(name+'.csv')),str(reference),'record' if name=='native-before' else 'compare','--cases','16','--tokens','32','--ctx','4096','--cache-gb','64','--chunk','512','--candidate-record',str(record)]
        childenv=dict(env,**policy)
        write(directory/(name+'-command.json'),dict(command=argv,env={k:v for k,v in childenv.items() if k.startswith('DS4_')}))
        with (directory/(name+'.json')).open('w') as out,(directory/(name+'.log')).open('w') as err:completed=subprocess.run(argv,cwd=output,env=childenv,stdout=out,stderr=err,check=False)
        write(directory/(name+'-exit.json'),dict(returncode=completed.returncode))
        if completed.returncode:raise RuntimeError(name+' failed rc='+str(completed.returncode))
        raw.check_frozen_assets(assets)
        if raw.identity(model)!=identity:raise ValueError('model identity changed during arm')
        row=json.loads((directory/(name+'.json')).read_text())
        if row['cases']!=16 or row['tokens']!=512:raise ValueError('held-out geometry incomplete')
        if name=='native-before':
            metadata=raw.raw_record_metadata(reference,16,32,512);ref_sha=sha(reference);assets[reference]=('fresh held-out reference',ref_sha)
            write(directory/'reference-proof.json',dict(sha256=ref_sha,layout=metadata[2],rows=512,created_before_candidate=True,policy=policy))
        row.update(name=name,**raw.summarize(directory/(name+'.csv')),raw_layout=raw.validate_candidate_record(record,metadata,16,32,512),raw_sha256=sha(record),io=parse_io((directory/(name+'.log')).read_text()),route=parse_route((directory/(name+'.log')).read_text(),name))
        if len(row['per_prompt'])!=16 or any(v['tokens']!=32 for v in row['per_prompt'].values()):raise ValueError('paired prompt completeness differs')
        if name.startswith('native') and (row['raw_sha256']!=ref_sha or row['bit_equal_rows']!=512 or any(row[k]!=0 for k in ['delta_nll','kl_base_to_raw','tv','argmax_flips'])):raise ValueError('native raw/reference control differs')
        results.append(row);write(directory/'summary.json',results);print(json.dumps({k:v for k,v in row.items() if k!='per_prompt'}),flush=True)
    checks=dict(native_before_after_reference_fullraw_exact=True,native_demand_io_equal=results[0]['io']==results[2]['io'],reference_sha256=ref_sha,rows=512,prompts=16,policy_locked_before_confirmation=True,policy=plan['arms'][1],timing_promoted=False)
    write(directory/'checks.json',checks)
    if not checks['native_demand_io_equal']:raise ValueError('native demand cache/read control differs')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',type=Path);p.add_argument('--run',type=Path);p.add_argument('--manifest-sha256');a=p.parse_args()
    if bool(a.prepare)==bool(a.run):p.error('choose prepare or run')
    if a.prepare and a.manifest_sha256:p.error('manifest SHA is supplied only by the frozen run command')
    if a.run and not a.manifest_sha256:p.error('run requires the explicit frozen manifest SHA')
    if a.prepare:prepare(a.prepare.resolve())
    else:run(a.run.resolve(),a.manifest_sha256)
if __name__=='__main__':main()
