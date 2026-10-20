#!/usr/bin/env python3
"""Model-free preparation of ONE paired external-edit arm, never launch it.

Future explicit ROOT build/CPU/source hashes are required. Native preparation
uses only the new A structural reference plus ROOT archival/release authority;
no grading/generated outcome selects the edit. Each arm needs its own grant.
"""
import argparse,hashlib,importlib.util,json,os,re,shutil
from pathlib import Path
WORK=Path(__file__).resolve().parents[2];IDEAS=WORK/'speed-bench/ideas'
PLAN=IDEAS/'agent_suffix_pair_external_v1/native-build-plan.json'
SOURCE=IDEAS/'agent_suffix_pair_external_v1/source-provenance.json'
CONTROLLER=IDEAS/'run_agent_suffix_pair_external.py'
CPU=IDEAS/'artifacts/agent-suffix-pair-external-cpu-v2/proof.json'
BUILD=IDEAS/'artifacts/agent-suffix-pair-external-caller-build-v1'
OUTPUT=IDEAS/'artifacts/agent-suffix-pair-external-paired-20261003-v1'

def sha(path):
    p=Path(path)
    if p.suffix.lower()=='.gguf':raise ValueError('No model metadata/hash/resolve in preparation')
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(path):
    p=Path(path)
    if p.suffix.lower()=='.gguf':raise ValueError('No model metadata in preparation')
    if p.is_symlink():raise ValueError('Payload symlink forbidden')
    s=p.stat();return dict(bytes=s.st_size,sha256=sha(p),mode=s.st_mode&0o777)
def check(path,value):
    p=pin(path)
    if any(p[k]!=v for k,v in value.items() if k in p):raise ValueError('Pinned prerequisite differs: '+str(path))
    return p
def digest(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('Explicit SHA256 required')
    return value
def module():
    s=importlib.util.spec_from_file_location('external_controller',CONTROLLER);c=importlib.util.module_from_spec(s);s.loader.exec_module(c);return c

def prepare(arm,plan_sha,source_freeze,cpu_freeze,cpu_sha,build_sha,actual=None,actual_sha=None,archive_proof=None):
    c=module();c.policy(arm)
    for value in (plan_sha,cpu_sha,build_sha):digest(value)
    if not re.fullmatch('[0-9a-f]{40}',source_freeze) or not re.fullmatch('[0-9a-f]{40}',cpu_freeze):raise ValueError('Explicit frozen source/CPU commit required')
    check(PLAN,dict(sha256=plan_sha));plan=c.load_json(PLAN);source=c.load_json(SOURCE)
    check(CPU,dict(sha256=cpu_sha));cpu=c.load_json(CPU)
    if cpu.get('status')!='EXTERNAL_PAIRED_SOURCE_CPU_PASS' or cpu.get('model_or_native_access') is not False or cpu.get('runtime_sources')!=source.get('runtime_sources'):raise ValueError('Frozen CPU/source prerequisite differs')
    for name,value in plan['sources'].items():check(WORK/name,value)
    for name,value in source['runtime_sources'].items():check(WORK/name,value)
    for name,value in plan['shaders'].items():check(WORK/name,value)
    for value in plan['objects'].values():check(value['path'],value);check(value['source_path'],value)
    check(plan['compiler_sdk_proof']['path'],plan['compiler_sdk_proof']);compiler=c.load_json(plan['compiler_sdk_proof']['path'])
    for key in ('selected_wrapper','actual_clang','sdk_settings','libSystem_stub'):check(compiler[key]['path'],compiler[key])
    check(BUILD/'build-manifest.json',dict(sha256=build_sha));build=c.load_json(BUILD/'build-manifest.json')
    if build.get('status')!='CALLER_COMPILE_LINK_PASS' or build.get('source_freeze')!=source_freeze or build.get('cpu_evidence')!=cpu_freeze or build.get('plan_sha256')!=plan_sha or build.get('all_pins_unchanged') is not True or build.get('serial_compilers')!=1 or build.get('native_core_Metal_built') is not False or build.get('model_stat_open_or_executable_run') is not False:raise ValueError('ROOT focused build/source prerequisite differs')
    results=build.get('results',[])
    if len(results)!=2 or [(r.get('argv'),r.get('cwd')) for r in results]!=[(r['argv'],r['cwd']) for r in plan['commands']] or any(type(r.get('returncode')) is not int or r['returncode']!=0 or r.get('warning') is not False for r in results):raise ValueError('Exact two warning-free focused commands required')
    for name,value in build['outputs'].items():check(BUILD/name,value)
    if not os.access(BUILD/'ds4-agent-pair-external',os.X_OK):raise ValueError('Caller executable mode required')
    if set(plan['shaders'])!={'metal/'+x for x in c.SHADERS.values()} or len(c.SHADERS)!=26:raise ValueError('Literal26shader inventory differs')
    third=IDEAS/'agent_suffix_pair_external_v1/third.txt';header=(WORK/'speed-bench/agent_suffix_pair_external_literals.h').read_text()
    literal=c.decode_json(re.search(r'pair_actual_third_user = (".*");',header).group(1)).encode()
    if literal!=third.read_bytes()+b'\n' or c.delivery_payload(third.read_text())!=literal:raise ValueError('Exact production delivery differs')
    check(c.PYTHON,dict(sha256=c.PYTHON_SHA));check(c.GUARD,dict(sha256=c.GUARD_SHA))
    if arm=='native' and (actual is None or actual_sha is None or archive_proof is None):raise ValueError('Native requires new A source plus ROOT archived/released authority')
    if arm=='reuse' and any(v is not None for v in (actual,actual_sha,archive_proof)):raise ValueError('A cannot bind previous source or override choice')
    p=OUTPUT/arm;p.mkdir(parents=True,exist_ok=False)
    def copy(src,name):
        dest=p/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        if pin(src)!=pin(dest):raise ValueError('Literal source/copy mode differs: '+name)
    copy(BUILD/'ds4-agent-pair-external','agent');copy(BUILD/'build-manifest.json','caller-build-manifest.json')
    for name in ('task.json','grader.py','run_grader.py','first.txt','second.txt','third.txt'):copy(IDEAS/'agent_suffix_pair_external_v1'/name,name)
    copy(CONTROLLER,'run_agent_suffix_pair_external.py');copy(PLAN,'native-build-plan.json');copy(CPU,'cpu-proof.json');copy(SOURCE,'source-provenance.json');copy(Path(__file__),'provenance/preparer.py')
    for name in plan['sources']:copy(WORK/name,'provenance/source/'+name)
    for name in plan['shaders']:copy(WORK/name,name)
    if arm=='native':
        digest(actual_sha);actual=Path(actual);c.bind_reference(actual,p/'reference',actual_sha)
        for name in ('shared-snapshot.json',):copy(actual/'run'/name,'reference/'+name)
        copy(actual/'root-release-proof.json','reference/root-release-proof.json');copy(Path(archive_proof),'reference/root-archive-proof.json')
    files={str(f.relative_to(p)):pin(f) for f in sorted(p.rglob('*')) if f.is_file()}
    prepared=dict(status='PREPARED_EXTERNAL_PAIR_ARM_NOT_GRANTED_NOT_RUN',arm=arm,caller_source_freeze=source_freeze,cpu_evidence_freeze=cpu_freeze,task_sha256=c.TASK_SHA,external_experiment=c.EXPERIMENT,model=c.MODEL,policy=c.policy(arm),turns=3,timeout_seconds=900,shared_paths=c.SHARED_PATHS,files=files,shader_env={k:str(p/'metal'/v) for k,v in c.SHADERS.items()},first_attempt=True,no_automatic_chain_or_retry=True,sole_root_grant_required=True,native_source_manifest_sha256=actual_sha if arm=='native' else None,primary_persistence='NOT_EXERCISED',scope='One modified task, one preedit native vector control; no fullgenerateddistribution, forgetting, stateexact, speed or promotion claim',worker_budget_scope='Current worker-owned unique caps/transaction forecasts; UI ownership and global observed peak excluded')
    c.put(p/'manifest.json',prepared);authority=sha(p/'manifest.json');c.put(p/'prepared-command.json',dict(manifest_sha256=authority,command=c.command(p,authority),cwd=str(c.ROOT)))
    c.verify(p,authority)
    if any((p/x).exists() for x in ('run','pressure','root-launch-proof.json')):raise ValueError('Preparation cannot grant or run')
    c.put(p/'nonmodel-preflight.json',dict(status='PASS_NOT_GRANTED_NOT_RUN',manifest_sha256=authority,command_sha256=sha(p/'prepared-command.json'),files=len(files),model_stat_or_guard_or_native_executed=False))
    print(json.dumps(dict(prepared=str(p),manifest_sha256=authority,command_sha256=sha(p/'prepared-command.json'),files=len(files))))

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--arm',choices=('reuse','native'),required=True);a.add_argument('--native-plan-sha256',required=True);a.add_argument('--source-freeze',required=True);a.add_argument('--cpu-evidence-freeze',required=True);a.add_argument('--cpu-proof-sha256',required=True);a.add_argument('--build-manifest-sha256',required=True);a.add_argument('--actual',type=Path);a.add_argument('--actual-manifest-sha256');a.add_argument('--root-archive-proof',type=Path);v=a.parse_args()
    prepare(v.arm,v.native_plan_sha256,v.source_freeze,v.cpu_evidence_freeze,v.cpu_proof_sha256,v.build_manifest_sha256,v.actual,v.actual_manifest_sha256,v.root_archive_proof)
