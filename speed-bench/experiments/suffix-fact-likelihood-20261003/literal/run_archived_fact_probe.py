#!/usr/bin/env python3
"""SOURCE-ONLY future runner. No prepared runtime/binary currently exists.

Import/static verification never stats or opens a GGUF. One separately reviewed
ROOT guard grant is required before the sole native process. No automatic retry.
"""
import argparse,errno,fcntl,hashlib,json,math,os,shlex,signal,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/Volumes/Models/ds4')
PYTHON=Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python')
PYTHON_SHA='b80d260a48404da13a32dd44d7283aadc80f6a6e56ed6e03ed5597ec6f46d48d'
GUARD=ROOT/'speed-bench/ideas/pressure_run.py'
GUARD_SHA='461e8c517e82a687b4579fc450c1e7128b1510192d088bab719682b33c946c3c'
MODEL=dict(path=str(ROOT/'gguf/DeepSeek-V4.1-Flash-Q2.gguf'),bytes=365713686528,mtime_ns=1789237692837653163)
TEACHER_SHA='5aa5376bdf7259403fc5e1f5a1cc4e94dd1860bb0b6fd01682f1018c7e411b1c'
POLICY=dict(DS4_METAL_V41_SELECTED_TILE='64',DS4_METAL_V41_SELECTED_LIMIT='1024',DS4_METAL_V41_RESIDENT_ROUTE_GAP='0',DS4_METAL_V41_RESIDENT_ROUTE_MODE='native',DS4_METAL_STREAMING_EXPERT_IO_METRICS='1')
# Exact inventory from the qualified actual A/B controller, never a glob.
SHADERS={'DS4_METAL_FLASH_ATTN_SOURCE':'flash_attn.metal','DS4_METAL_DENSE_SOURCE':'dense.metal','DS4_METAL_GLM53_BF16_SOURCE':'glm53_bf16.metal','DS4_METAL_GLM53_VISION_SOURCE':'glm53_vision.metal','DS4_METAL_DEEPSEEK4_VISION_SOURCE':'deepseek4_vision.metal','DS4_METAL_GLM53_KDA_SOURCE':'glm53_kda.metal','DS4_METAL_MOE_SOURCE':'moe.metal','DS4_METAL_DSV4_HC_SOURCE':'dsv4_hc.metal','DS4_METAL_UNARY_SOURCE':'unary.metal','DS4_METAL_DSV4_KV_SOURCE':'dsv4_kv.metal','DS4_METAL_DSV41_SOURCE':'dsv41.metal','DS4_METAL_DSV4_ROPE_SOURCE':'dsv4_rope.metal','DS4_METAL_DSV4_MISC_SOURCE':'dsv4_misc.metal','DS4_METAL_ARGSORT_SOURCE':'argsort.metal','DS4_METAL_CPY_SOURCE':'cpy.metal','DS4_METAL_CONCAT_SOURCE':'concat.metal','DS4_METAL_GET_ROWS_SOURCE':'get_rows.metal','DS4_METAL_SUM_ROWS_SOURCE':'sum_rows.metal','DS4_METAL_SOFTMAX_SOURCE':'softmax.metal','DS4_METAL_REPEAT_SOURCE':'repeat.metal','DS4_METAL_GLU_SOURCE':'glu.metal','DS4_METAL_NORM_SOURCE':'norm.metal','DS4_METAL_BIN_SOURCE':'bin.metal','DS4_METAL_SET_ROWS_SOURCE':'set_rows.metal','DS4_METAL_QWEN4_SOURCE':'qwen4.metal','DS4_METAL_QWEN4_VISION_SOURCE':'qwen4_vision.metal'}
def object_pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON field: '+key)
        result[key]=value
    return result

def decode_json(data):return json.loads(data,object_pairs_hook=object_pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def sha(path):
    p=Path(path)
    if p.suffix.lower()=='.gguf':raise ValueError('No model hashing')
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check_file(p,pin):
    if p.suffix.lower()=='.gguf':raise ValueError('No model metadata during preflight')
    if type(pin) is not dict or type(pin.get('bytes')) is not int or pin['bytes']<0 or type(pin.get('mode')) is not int or type(pin.get('sha256')) is not str:raise ValueError('Typed file pin required')
    if p.is_symlink() or not p.is_file() or p.stat().st_size!=pin['bytes'] or p.stat().st_mode&0o777!=pin['mode'] or sha(p)!=pin['sha256']:raise ValueError('Frozen nonmodel file differs: '+str(p))
def guard_command(p,digest):return [str(PYTHON),str(GUARD),str(p/'pressure'),'--',str(PYTHON),str(p/'run_archived_fact_probe.py'),'--run',str(p),'--manifest-sha256',digest]
def require_root_grant(p,digest):
    proof=decode_json((p/'root-launch-proof.json').read_bytes())
    if type(proof) is not dict or proof.get('status')!='GRANTED' or proof.get('sole_model_grant') is not True or proof.get('manifest_sha256')!=digest or proof.get('command')!=guard_command(p,digest):raise ValueError('Exact ROOT sole grant required')
    stamp=proof.get('time')
    if type(stamp) not in (int,float) or not math.isfinite(stamp) or stamp<=0:raise ValueError('Finite positive ROOT grant time required')
    return proof
def require_guard(p,digest):
    actual=shlex.split(subprocess.check_output(['/bin/ps','-p',str(os.getppid()),'-o','command='],text=True,timeout=2).strip())
    expected=guard_command(p,digest)
    if not actual or actual[1:]!=expected[1:] or not os.path.samefile(actual[0],PYTHON) or sha(PYTHON)!=PYTHON_SHA or sha(GUARD)!=GUARD_SHA:raise ValueError('Literal MAIN guard/framework parent required')
    with (ROOT/'speed-bench/ideas/artifacts/model-job.lock').open('r+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EAGAIN,errno.EWOULDBLOCK):raise
            return
        fcntl.flock(f,fcntl.LOCK_UN)
    raise ValueError('Held MAIN model lock required')
def verify(p,digest):
    if sha(p/'manifest.json')!=digest:raise ValueError('Explicit manifest authority differs')
    m=decode_json((p/'manifest.json').read_bytes())
    if m.get('status')!='ARCHIVED_FACT_RUNTIME_PREPARED' or m.get('model')!=MODEL or m.get('policy')!=POLICY or m.get('teacher_sha256')!=TEACHER_SHA or m.get('arms')!=['fresh-native-before','identity-suffix','repaired-old-suffix','fresh-native-after'] or m.get('first_attempt') is not True or m.get('no_retry') is not True:raise ValueError('Future qualified runtime authority required')
    if m.get('ctx')!=8192 or m.get('chunk')!=256 or m.get('ssd_cache_gb')!=64 or m.get('staging_cap')!=134217728:raise ValueError('Fixed geometry/configuration differs')
    files=m.get('files')
    required={'probe','proposal.json','fixture-facts.json','caller-build-manifest.json','native-build-plan.json','run_archived_fact_probe.py','source-provenance.json','cpu-proof.json','teacher.txt'}|{'metal/'+v for v in SHADERS.values()}
    if type(files) is not dict or not required<=set(files):raise ValueError('Mandatory source/build/26 shader assets missing')
    for name,pin in files.items():
        if not name or Path(name).is_absolute() or '..' in Path(name).parts or Path(name).suffix.lower()=='.gguf':raise ValueError('Unsafe asset name')
        check_file(p/name,pin)
    if sha(Path(__file__))!=files['run_archived_fact_probe.py']['sha256'] or not os.access(p/'probe',os.X_OK):raise ValueError('Executing runner/probe authority differs')
    expected_env={k:str(p/'metal'/v) for k,v in SHADERS.items()}
    if m.get('shader_env')!=expected_env or len(expected_env)!=26:raise ValueError('Exact26 shader paths required')
    external=m.get('external_nonmodel_files')
    if type(external) is not dict or not external:raise ValueError('Immutable source input authorities required')
    for name,pin in external.items():
        target=Path(name)
        if not target.is_absolute() or target.suffix.lower()=='.gguf':raise ValueError('Nonmodel absolute input authority required')
        check_file(target,pin)
    build=decode_json((p/'caller-build-manifest.json').read_bytes());plan=decode_json((p/'native-build-plan.json').read_bytes())
    if build.get('status')!='ARCHIVED_FACT_CALLER_BUILD_PASS' or build.get('all_pins_unchanged') is not True or build.get('native_core_Metal_built') is not False or build.get('model_or_native_executable_run') is not False or build.get('plan_sha256')!=sha(p/'native-build-plan.json') or build.get('serial_compilers')!=1 or build.get('source_freeze')!=m.get('source_freeze') or build.get('cpu_evidence')!=m.get('cpu_evidence'):raise ValueError('Qualified frontend build proof required')
    records=build.get('results')
    if type(records) is not list or len(records)!=2 or any(type(x.get('returncode')) is not int or x['returncode']!=0 for x in records) or [(x.get('argv'),x.get('cwd')) for x in records]!=[(x['argv'],x['cwd']) for x in plan.get('commands',[])]:raise ValueError('Exact completed compile/link records required')
    if build.get('outputs',{}).get('probe')!=files['probe']:raise ValueError('Built probe identity differs')
    source=decode_json((p/'source-provenance.json').read_bytes());cpu=decode_json((p/'cpu-proof.json').read_bytes())
    if source.get('status')!='ARCHIVED_FACT_SOURCE_FROZEN' or source.get('source_freeze')!=m.get('source_freeze') or cpu.get('status')!='ARCHIVED_FACT_SOURCE_CPU_PASS' or cpu.get('native_model_access') is not False or cpu.get('source_freeze')!=m.get('source_freeze') or cpu.get('source_pins')!=source.get('source_pins'):raise ValueError('Source/CPU prerequisite authority differs')
    if type(plan.get('sources')) is not dict or type(plan.get('objects')) is not dict or set(plan['objects'])!={'ds4.o','ds4_image.o','ds4_distributed.o','ds4_tp.o','ds4_ssd.o','ds4_metal.o','ds4_layer_pack.o','ds4_engram.o'}:raise ValueError('Exact8 native objects/source proof required')
    for pin in plan['sources'].values():check_file(Path(pin['path']),pin)
    for pin in plan['objects'].values():check_file(Path(pin['path']),pin)
    expected=dict(manifest_sha256=digest,command=guard_command(p,digest),cwd=str(ROOT))
    if decode_json((p/'prepared-command.json').read_bytes())!=expected:raise ValueError('Actual frozen guard command authority differs')
    if sha(PYTHON)!=PYTHON_SHA or sha(GUARD)!=GUARD_SHA:raise ValueError('Guard/framework source differs')
    return m

def clean_environment(shader_env):
    env={k:v for k,v in os.environ.items() if not k.startswith('DS4')};env.update(POLICY);env.update(shader_env);return env

def stop_owned(proc):
    # Unreaped leader anchors birth/PGID. Never signal after poll/wait reaps it.
    if proc.returncode is not None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    time.sleep(.5)
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.wait(timeout=5)

def execute(p,digest):
    m=verify(p,digest);require_root_grant(p,digest);require_guard(p,digest)
    actual=Path(MODEL['path']).stat() # FIRST GGUF access, after both authorities.
    if (actual.st_size,actual.st_mtime_ns)!=(MODEL['bytes'],MODEL['mtime_ns']):raise ValueError('Model identity differs')
    out=p/'run';out.mkdir(exist_ok=False);proc=None;result=dict(status='FIRST_FAILURE_RETAINED',native_child_returncode=None,no_retry=True)
    try:
        verify(p,digest);require_root_grant(p,digest);require_guard(p,digest)
        argv=[str(p/'probe'),'--run',str(out)]
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
            proc=subprocess.Popen(argv,cwd=str(ROOT),env=clean_environment(m['shader_env']),stdout=stdout,stderr=stderr,start_new_session=True)
            result['argv']=argv;result['pid']=proc.pid
            with (out/'launch.json').open('x') as f:json.dump(result,f,indent=2)
            rc=proc.wait(timeout=900);result['native_child_returncode']=rc
        result['status']='NATIVE_PROCESS_COMPLETE' if rc==0 else 'FIRST_FAILURE_RETAINED'
        return rc
    except BaseException:
        result['traceback']=traceback.format_exc()
        if proc is not None and proc.returncode is None:stop_owned(proc)
        raise
    finally:
        with (out/'runner-result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path);a.add_argument('--manifest-sha256');a.add_argument('--verify-only',type=Path);n=a.parse_args()
    if not n.manifest_sha256 or (n.run is None)==(n.verify_only is None):a.error('Explicit future runtime manifest and exactly one operation required')
    if n.verify_only is not None:verify(n.verify_only,n.manifest_sha256);print('static preflight only; no model access');return 0
    return execute(n.run,n.manifest_sha256)
if __name__=='__main__':raise SystemExit(main())
