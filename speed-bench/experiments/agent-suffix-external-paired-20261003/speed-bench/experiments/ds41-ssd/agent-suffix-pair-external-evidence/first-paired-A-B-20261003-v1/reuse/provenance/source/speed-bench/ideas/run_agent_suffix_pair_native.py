#!/usr/bin/env python3
"""Private native caller controls. Default preflight never accesses a model.

Preparation needs a separately reviewed caller build manifest. Every execution
needs a new explicit ROOT model grant and the exact pinned MAIN guard parent.
"""
import argparse
import csv
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import time

ROOT = Path('/Volumes/Models/ds4')
GUARD = ROOT/'speed-bench/ideas/pressure_run.py'
GUARD_SHA = '461e8c517e82a687b4579fc450c1e7128b1510192d088bab719682b33c946c3c'
PYTHON = Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python')
PYTHON_SHA = 'b80d260a48404da13a32dd44d7283aadc80f6a6e56ed6e03ed5597ec6f46d48d'
MODEL = dict(path=str(ROOT/'gguf/DeepSeek-V4.1-Flash-Q2.gguf'),bytes=365713686528,mtime_ns=1789237692837653163)
POLICY = dict(DS4_METAL_V41_SELECTED_TILE='64',DS4_METAL_V41_SELECTED_LIMIT='1024',DS4_METAL_V41_RESIDENT_ROUTE_GAP='0',DS4_METAL_V41_RESIDENT_ROUTE_MODE='native')
GEOMETRY = dict(prefix=1409,shift=65,span=512,tail=128,decode=16,ctx=4096,chunk=512,cache_gb=64,capacity=1024,app_budget_bytes=128<<20)
ARMS = [('native-before','native'),('tape-native','tape-native'),('agent-reuse','reuse'),('native-after','native')]
INPUT_BUILDER = 'v1:corpus-first2114;User-at1986;deleted-first-piece-sentinel-if-equal;remove65-at1409;teachers-next16'
SHADERS = {'DS4_METAL_FLASH_ATTN_SOURCE': 'flash_attn.metal', 'DS4_METAL_DENSE_SOURCE': 'dense.metal', 'DS4_METAL_GLM53_BF16_SOURCE': 'glm53_bf16.metal', 'DS4_METAL_GLM53_VISION_SOURCE': 'glm53_vision.metal', 'DS4_METAL_DEEPSEEK4_VISION_SOURCE': 'deepseek4_vision.metal', 'DS4_METAL_GLM53_KDA_SOURCE': 'glm53_kda.metal', 'DS4_METAL_MOE_SOURCE': 'moe.metal', 'DS4_METAL_DSV4_HC_SOURCE': 'dsv4_hc.metal', 'DS4_METAL_UNARY_SOURCE': 'unary.metal', 'DS4_METAL_DSV4_KV_SOURCE': 'dsv4_kv.metal', 'DS4_METAL_DSV41_SOURCE': 'dsv41.metal', 'DS4_METAL_DSV4_ROPE_SOURCE': 'dsv4_rope.metal', 'DS4_METAL_DSV4_MISC_SOURCE': 'dsv4_misc.metal', 'DS4_METAL_ARGSORT_SOURCE': 'argsort.metal', 'DS4_METAL_CPY_SOURCE': 'cpy.metal', 'DS4_METAL_CONCAT_SOURCE': 'concat.metal', 'DS4_METAL_GET_ROWS_SOURCE': 'get_rows.metal', 'DS4_METAL_SUM_ROWS_SOURCE': 'sum_rows.metal', 'DS4_METAL_SOFTMAX_SOURCE': 'softmax.metal', 'DS4_METAL_REPEAT_SOURCE': 'repeat.metal', 'DS4_METAL_GLU_SOURCE': 'glu.metal', 'DS4_METAL_NORM_SOURCE': 'norm.metal', 'DS4_METAL_BIN_SOURCE': 'bin.metal', 'DS4_METAL_SET_ROWS_SOURCE': 'set_rows.metal', 'DS4_METAL_QWEN4_SOURCE': 'qwen4.metal', 'DS4_METAL_QWEN4_VISION_SOURCE': 'qwen4_vision.metal'}
SOURCE_END=1986;OLD_N=2114;NEW_N=2049;VOCAB=129280;DECODE=16


def sha(path):
    path=Path(path)
    if path.suffix.lower()=='.gguf':raise ValueError('Model hashing is forbidden')
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def put(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def command(output,digest):
    return [str(PYTHON),str(GUARD),str(output/'pressure'),'--',str(PYTHON),str(output/'run_agent_suffix_pair_native.py'),'--run',str(output),'--manifest-sha256',digest]


def verify(output,digest):
    if sha(output/'manifest.json')!=digest:raise ValueError('Explicit manifest authority differs')
    plan=json.loads((output/'manifest.json').read_text())
    if plan['geometry']!=GEOMETRY or plan['arms']!=[list(x) for x in ARMS] or plan['policy']!=POLICY or plan['model_from_prior']!=MODEL or plan['input_builder']!=INPUT_BUILDER:raise ValueError('Frozen native experiment differs')
    if plan['shader_env']!={k:str(output/'metal'/v) for k,v in SHADERS.items()}:raise ValueError('Explicit per-file shader paths differ')
    required={'probe','caller-build-manifest.json','run_agent_suffix_pair_native.py','corpus.txt','source/ds4_agent.c','source/ds4.h','source/ds4.c','source/ds4_suffix_pair_tape.h','source/speed-bench/agent_suffix_pair_native_fixture.c'}
    if not required<=set(plan['files']) or len([n for n in plan['files'] if n.startswith('metal/') and n.endswith('.metal')])!=26:raise ValueError('Required asset inventory incomplete')
    for name,pin in plan['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or Path(name).suffix.lower()=='.gguf':raise ValueError('Invalid frozen asset path')
        p=output/name
        if p.stat().st_size!=pin['bytes'] or sha(p)!=pin['sha256']:raise ValueError('Frozen asset differs: '+name)
    if not os.access(output/'probe',os.X_OK):raise ValueError('Frozen probe is not executable')
    if sha(GUARD)!=GUARD_SHA or sha(PYTHON)!=PYTHON_SHA:raise ValueError('Guard/interpreter differs')
    if json.loads((output/'prepared-command.json').read_text())!=dict(manifest_sha256=digest,command=command(output,digest),cwd=str(ROOT)):raise ValueError('Frozen direct-parent command differs')
    return plan


def require_guard(output,digest):
    expected=command(output,digest)
    actual=shlex.split(subprocess.check_output(['/bin/ps','-p',str(os.getppid()),'-o','command='],text=True,timeout=2).strip())
    if not actual or actual[1:]!=expected[1:] or not os.path.samefile(actual[0],PYTHON) or sha(PYTHON)!=PYTHON_SHA or sha(GUARD)!=GUARD_SHA:raise RuntimeError('Exact pinned MAIN guard parent required')
    with (ROOT/'speed-bench/ideas/artifacts/model-job.lock').open('r+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EAGAIN,errno.EWOULDBLOCK):raise
            return
        fcntl.flock(lock,fcntl.LOCK_UN)
    raise RuntimeError('MAIN model lock must be held')


def model_identity():
    p=Path(MODEL['path'])
    if p.resolve()!=p:raise ValueError('Model path alias')
    s=p.stat();observed=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    if observed!=MODEL:raise ValueError('Model identity differs')
    return observed


def read_input(path):
    data=Path(path).read_bytes()
    if len(data)!=32+4*(OLD_N+NEW_N+DECODE):raise ValueError('Input extent differs')
    h=struct.unpack_from('<8I',data)
    if h[:3]!=(0x41535431,1,VOCAB) or h[3]>=VOCAB or h[4:]!=(OLD_N,NEW_N,DECODE,0):raise ValueError('Input header differs')
    ids=struct.unpack_from('<'+str(OLD_N+NEW_N+DECODE)+'i',data,32);old=ids[:OLD_N];edited=ids[OLD_N:OLD_N+NEW_N];targets=ids[-DECODE:];user=h[3]
    if any(x<0 or x>=VOCAB for x in ids) or old[SOURCE_END]!=user or edited[SOURCE_END-65]!=user or old[1409]==edited[1409] or any(x==user and i!=SOURCE_END for i,x in enumerate(old)):raise ValueError('Input IDs/fresh User boundary differ')
    if edited!=old[:1409]+old[1474:]:raise ValueError('Input is not the exact frozen odd deletion')
    return dict(sha256=sha(path),bytes=len(data),user_id=user,old_ids=list(old),edited_ids=list(edited),teacher_ids=list(targets),latest_user_start=1921)


def read_raw(path,approximate,targets):
    with Path(path).open('rb') as stream:
        h=struct.unpack('<8I',stream.read(32))
        if h!=(0x41534731,VOCAB,DECODE,1409,65,512,128,int(approximate)):raise ValueError('Raw geometry/header differs')
        actual=struct.unpack('<16i',stream.read(64))
        if list(actual)!=targets:raise ValueError('Teacher IDs differ')
        source=hashlib.sha256();branch=hashlib.sha256()
        for row in range(19):
            block=stream.read(VOCAB*4)
            if len(block)!=VOCAB*4 or any(not math.isfinite(x[0]) for x in struct.iter_unpack('<f',block)):raise ValueError('Truncated/nonfinite raw vector')
            (source if row<2 else branch).update(block)
        if stream.read(1):raise ValueError('Raw trailer differs')
    return dict(full_sha256=sha(path),bytes=Path(path).stat().st_size,source_sha256=source.hexdigest(),branch_sha256=branch.hexdigest(),source_rows=2,branch_rows=17,teacher_ids=targets)


def validate_summary(s,mode,user):
    if any(type(s.get(k)) is not int or s[k]!=v for k,v in dict(prefix=1409,shift=65,span=512,tail=128,decode=16,user_id=user,latest_user_start=1921,app_budget_bytes=128<<20).items()) or s.get('mode')!=mode:raise ValueError('Summary geometry differs')
    if s.get('global_peak_observed') is not False or s.get('performed_phase_admissions_checked') is not True:raise ValueError('Peak scope/admission claims differ')
    flags=['fresh_latest_user','source_move_no_reload','precancel_preserves_state','postmove_cancel_rolls_back','canonical_and_diagnostic_gates','text_only_save_state_unchanged']
    if any(type(s.get(k)) is not bool for k in flags) or not s['fresh_latest_user']:raise ValueError('Summary flags are not typed/true')
    if any(s[k]!=(mode=='reuse') for k in flags[1:]):raise ValueError('Native transaction controls incomplete')
    if type(s.get('transaction_peak_bytes_including_reserves')) is not int or not 0<=s['transaction_peak_bytes_including_reserves']<=128<<20 or (mode=='reuse' and s['transaction_peak_bytes_including_reserves']==0):raise ValueError('Unique app peak differs')
    for k in ['mean_kl','mean_tv','sum_delta_nll','reuse_ms_diagnostic']:
        if type(s.get(k)) not in (int,float) or not math.isfinite(s[k]):raise ValueError('Nonfinite summary')
    if min(s['mean_kl'],s['mean_tv'],s['reuse_ms_diagnostic'])<0:raise ValueError('Negative summary')
    if any(type(s.get(k)) is not int or not 0<=s[k]<=17 for k in ['argmax_flips','bit_equal_rows']):raise ValueError('Raw count bounds differ')


def quality_csv(path,targets,s):
    lines=Path(path).read_text().splitlines();header='step,target,delta_nll,kl,tv,argmax_flip,bit_equal'
    if lines.count(header)!=1:raise ValueError('Quality header count differs')
    rows=list(csv.DictReader(lines[lines.index(header):]))
    if len(rows)!=17:raise ValueError('Quality row completeness differs')
    ks=[];ts=[];ds=[];flips=bits=0
    for i,r in enumerate(rows):
        if None in r or any(v is None for v in r.values()):raise ValueError('Malformed quality row')
        if int(r['step'])!=i or int(r['target'])!=(targets[i] if i<16 else -1):raise ValueError('Quality row/teacher order differs')
        k,t,d=(float(r[n]) for n in ['kl','tv','delta_nll'])
        if not all(map(math.isfinite,[k,t,d])) or min(k,t)<0:raise ValueError('Nonfinite quality row')
        flip,bit=int(r['argmax_flip']),int(r['bit_equal'])
        if flip not in (0,1) or bit not in (0,1):raise ValueError('Invalid equality/argmax flag')
        ks.append(k);ts.append(t);ds.append(d);flips+=flip;bits+=bit
    for value,key in [(math.fsum(ks)/17,'mean_kl'),(math.fsum(ts)/17,'mean_tv'),(math.fsum(ds),'sum_delta_nll')]:
        if not math.isclose(value,s[key],rel_tol=1e-8,abs_tol=1e-10):raise ValueError('Quality summary differs')
    if flips!=s['argmax_flips'] or bits!=s['bit_equal_rows']:raise ValueError('Quality count summary differs')
    return dict(rows=17,mean_kl=math.fsum(ks)/17,mean_tv=math.fsum(ts)/17,sum_delta_nll=math.fsum(ds),argmax_flips=flips,bit_equal_rows=bits,sha256=sha(path))


def native_record(base,raw):
    suffixes=['.prefix.state.bin','.source.endpoint.state.bin','.source.full.state.bin','.fresh.endpoint.state.bin','.fresh.full.state.bin','.fresh.endpoint.logits.bin']
    return dict(states={s:sha(str(base)+s) for s in suffixes},source_sha256=raw['source_sha256'],branch_sha256=raw['branch_sha256'],teacher_ids=raw['teacher_ids'])


def reuse_controls(base,reference,summary,input_record):
    for s in ['.prefix.state.bin','.source.endpoint.state.bin','.source.full.state.bin']:
        if sha(str(base)+s)!=sha(str(reference)+s):raise ValueError('Reuse source state differs')
    for s in ['.precancel.state.bin','.rollback.state.bin']:
        if sha(str(base)+s)!=sha(str(reference)+'.source.full.state.bin'):raise ValueError('Cancellation changed full canonical state')
    with Path(str(base)+'.private.state.bin').open('rb') as f:h=f.read(56)
    with Path(str(reference)+'.fresh.full.state.bin').open('rb') as f:n=f.read(56)
    if len(h)!=56 or len(n)!=56 or h[:4]!=b'DSSD' or n[:4]!=b'DSV4' or h[4:]!=n[4:] or Path(str(base)+'.private.state.bin').stat().st_size!=Path(str(reference)+'.fresh.full.state.bin').stat().st_size:raise ValueError('Diagnostic header/tag/extent differs')
    with Path(str(base)+'.saved.kv').open('rb') as f:kv=f.read(52)
    if len(kv)!=52 or kv[:4]!=b'KVC\x01' or kv[4:7]!=bytes([2,6,8]) or kv[20]!=2 or struct.unpack_from('<I',kv,8)[0]!=NEW_N or struct.unpack_from('<I',kv,16)[0]!=4096 or struct.unpack_from('<Q',kv,40)[0]!=0:raise ValueError('Actual approximate named save header/payload differs')
    trace=Path(str(base)+'.trace').read_text()
    matches=re.findall(r'context pair suffix APPROXIMATE reused=(\d+) fresh=(\d+) source=(\d+):(\d+) destination=(\d+):(\d+) rebuilt_pairs=(\d+) boundary_pairs=(\d+) copied_bytes=(\d+) unique_peak=(\d+) budget=(\d+) latest_user_start=(\d+) latest_user_fresh=(\d)',trace)
    if len(matches)!=1:raise ValueError('Actual agent reuse trace count differs')
    values=list(map(int,matches[0]))
    if values[:8]!=[512,128,1474,1986,1409,1921,768,3] or values[10:]!=[128<<20,1921,1] or not 0<values[9]<=summary['transaction_peak_bytes_including_reserves']:raise ValueError('Actual agent geometry/freshness/budget trace differs')
    return dict(diagnostic_header_tag_and_extent_checked=True,diagnostic_body_exact_claimed=False,text_only_payload_bytes=0,trace=dict(zip(['reused','fresh','source_start','source_end','destination_start','destination_end','pair_rows','boundary_rows','copied_bytes','agent_unique_peak','budget','latest_user_start','latest_user_fresh'],values)),private_state_sha256=sha(str(base)+'.private.state.bin'),saved_kv_sha256=sha(str(base)+'.saved.kv'),input_sha256=input_record['sha256'])


def prepare(output,build,revision):
    output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[2]
    proof=json.loads((build/'manifest.json').read_text())
    if proof['status']!='PRIVATE_CALLERS_BUILT_MODEL_UNRUN' or proof['source_revision']!=revision:raise ValueError('Reviewed caller build proof differs')
    inventory={'probe':Path(proof['fixture_binary']['path']),'caller-build-manifest.json':build/'manifest.json','run_agent_suffix_pair_native.py':Path(__file__),'corpus.txt':source/'speed-bench/promessi_sposi.txt'}
    for name,pin in proof['source_files'].items():
        p=source/name
        if sha(p)!=pin['sha256']:raise ValueError('Build source differs: '+name)
        inventory['source/'+name]=p
    for p in sorted((source/'metal').glob('*.metal')):inventory['metal/'+p.name]=p
    files={}
    for name,p in inventory.items():
        dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);files[name]=dict(sha256=sha(dest),bytes=dest.stat().st_size)
    if sha(output/'probe')!=proof['fixture_binary']['sha256'] or not os.access(output/'probe',os.X_OK):raise ValueError('Probe bytes/execute mode differ')
    plan=dict(source_revision=revision,geometry=GEOMETRY,arms=ARMS,policy=POLICY,model_from_prior=MODEL,input_builder=INPUT_BUILDER,shader_env={k:str(output/'metal'/v) for k,v in SHADERS.items()},files=files,scope='One deterministic structural transaction; all native controls must be exact. Approximate old conditioning can remain. Timings diagnostic; no semantic accuracy, forgetting or speed-promotion claim.')
    put(output/'manifest.json',plan);digest=sha(output/'manifest.json');prepared=dict(manifest_sha256=digest,command=command(output,digest),cwd=str(ROOT));put(output/'prepared-command.json',prepared)
    # These pins live outside manifest authority. ROOT reviews the launcher
    # source/SHA too; it cannot choose another plan/command or inherited DS4 env.
    launcher='''import importlib.util,json,os,subprocess,sys\nfrom pathlib import Path\np=Path(OUTPUT)\nspec=importlib.util.spec_from_file_location("frozen_driver",p/"run_agent_suffix_pair_native.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\nassert m.sha(p/"prepared-command.json")==COMMAND_SHA\nm.verify(p,MANIFEST_SHA)\nif sys.argv[1:]!=["--execute"]:\n assert sys.argv[1:]==[]\n print("PREFLIGHT_PASS; no model metadata or process")\nelse:\n env={k:v for k,v in os.environ.items() if not k.startswith("DS4")}\n raise SystemExit(subprocess.run(json.loads((p/"prepared-command.json").read_text())["command"],cwd=m.ROOT,env=env).returncode)\n'''
    prefix=f'OUTPUT={str(output)!r}\nMANIFEST_SHA={digest!r}\nCOMMAND_SHA={sha(output/"prepared-command.json")!r}\n'
    (output/'launcher.py').write_text(prefix+launcher)
    print(json.dumps(dict(status='PREPARED_NOT_RUN',manifest_sha256=digest,command_sha256=sha(output/'prepared-command.json'),launcher_sha256=sha(output/'launcher.py'),files=len(files),model_metadata_touched=False)))


def run(output,digest):
    plan=verify(output,digest);require_guard(output,digest);identity=model_identity()  # First model access, beneath exact guard.
    directory=output/'run';directory.mkdir(exist_ok=False);put(directory/'run-proof.json',dict(manifest_sha256=digest,model_identity=identity,guarded=True))
    input_path=directory/'frozen-input.bin';input_record=None;input_sha=None;control=None;results=[];env={k:v for k,v in os.environ.items() if not k.startswith('DS4')};env.update(POLICY);env.update(plan['shader_env'])
    try:
        for name,mode in ARMS:
            verify(output,digest);require_guard(output,digest)
            if model_identity()!=identity:raise ValueError('Model changed')
            base=directory/name;argv=[str(output/'probe'),'--model',identity['path'],'--base',str(base),'--input',str(input_path),'--mode',mode]
            if input_record is None:argv+=['--create-input','--corpus',str(output/'corpus.txt')]
            if mode=='reuse':argv+=['--reference-base',str(directory/'native-before')]
            if input_sha is not None and sha(input_path)!=input_sha:raise ValueError('Frozen shared token arrays changed before arm')
            put(directory/(name+'-command.json'),dict(argv=argv,env=env_policy(env)));print(json.dumps(dict(event='arm_start',arm=name)),flush=True)
            start=time.monotonic()
            try:
                with (directory/(name+'.stdout')).open('x') as out,(directory/(name+'.stderr')).open('x') as err:
                    process=subprocess.run(argv,cwd=output,env=env,stdout=out,stderr=err,start_new_session=True,timeout=420)
                code=process.returncode
            except subprocess.TimeoutExpired:
                put(directory/(name+'-exit.json'),dict(returncode=124,timed_out=True,seconds=time.monotonic()-start));raise
            put(directory/(name+'-exit.json'),dict(returncode=code,timed_out=False,seconds=time.monotonic()-start))
            if code:raise RuntimeError(name+' failed rc='+str(code))
            if input_record is None:
                input_record=read_input(input_path);input_sha=input_record['sha256'];put(directory/'complete-input-ids.json',input_record)
            elif sha(input_path)!=input_sha or read_input(input_path)!=input_record:raise ValueError('Shared token array changed')
            if model_identity()!=identity:raise ValueError('Model changed during arm')
            summary=json.loads(Path(str(base)+'.summary.json').read_text());validate_summary(summary,mode,input_record['user_id']);raw=read_raw(str(base)+'.raw.bin',mode=='reuse',input_record['teacher_ids'])
            if mode!='reuse':
                observed=native_record(base,raw)
                if control is None:control=observed
                elif observed!=control:raise ValueError(name+' native raw/state/teacher controls differ')
                checks=dict(native_exact=True)
            else:
                if raw['source_sha256']!=control['source_sha256']:raise ValueError('Reuse source raw vectors differ')
                checks=reuse_controls(base,directory/'native-before',summary,input_record);checks['quality']=quality_csv(directory/(name+'.stdout'),input_record['teacher_ids'],summary)
            result=dict(arm=name,mode=mode,summary=summary,raw=raw,checks=checks);results.append(result);put(directory/(name+'-checked.json'),result);print(json.dumps(result),flush=True)
        put(directory/'summary.json',dict(status='COMPLETE',results=results,input_sha256=input_sha,native_controls_exact=True,scope=plan['scope'],timing_promoted=False))
    except Exception as error:
        put(directory/'failure.json',dict(status='FIRST_FAILURE_RETAINED',error=type(error).__name__+': '+str(error),completed_arms=[r['arm'] for r in results],no_retries=True));raise


def env_policy(env):return {k:v for k,v in env.items() if k.startswith('DS4')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',type=Path);p.add_argument('--build',type=Path);p.add_argument('--source-revision');p.add_argument('--preflight',type=Path);p.add_argument('--run',type=Path);p.add_argument('--manifest-sha256');a=p.parse_args()
    if sum(bool(x) for x in [a.prepare,a.preflight,a.run])!=1:p.error('Choose one preparation/preflight/run mode')
    if a.prepare:
        if not a.build or not a.source_revision or a.manifest_sha256:p.error('Preparation requires build/revision, no manifest override')
        prepare(a.prepare.resolve(),a.build.resolve(),a.source_revision)
    elif not a.manifest_sha256 or a.build or a.source_revision:p.error('Preflight/run needs explicit manifest SHA only')
    elif a.preflight:
        plan=verify(a.preflight.resolve(),a.manifest_sha256);print(json.dumps(dict(status='PREFLIGHT_PASS',files=len(plan['files']),model_metadata_touched=False)))
    else:run(a.run.resolve(),a.manifest_sha256)
