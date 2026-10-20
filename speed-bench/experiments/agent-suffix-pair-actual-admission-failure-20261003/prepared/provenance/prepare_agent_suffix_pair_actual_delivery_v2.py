#!/usr/bin/env python3
"""Prepare only the V2 exact protocol-LF correction with admission observations.

The original model attempt remains a failed infrastructure attempt, with no
selection or third evaluation. This script never runs the caller or guard,
and never reads model metadata. ROOT must supply the future build-manifest SHA.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil

WORK=Path(__file__).resolve().parents[2]
IDEAS=WORK/'speed-bench/ideas'
PLAN=IDEAS/'agent_suffix_pair_actual_delivery_v1/native-build-plan-v2.json'
PLAN_SHA='a885d6497600dc2a6862544f0550e150a52d434aef54ee4fab23dcd0270e6a0a'
SOURCE_FREEZE='cd03499cec7d3db424b4bcb4e4ffb068f580f38b'
CONTROLLER=IDEAS/'run_agent_suffix_pair_actual.py'
CONTROLLER_SHA='ddb9d605b70c6d9259b19c489337a0c2173946174bc688eaee519e2fcd0ec590'
BUILD=IDEAS/'artifacts/agent-suffix-pair-actual-delivery-caller-build-v2'
CPU=IDEAS/'evidence/agent-suffix-pair-actual-delivery-cpu-v2'
CPU_SHA='684fbb383fdf8b6750761ab5b5aa623e88239b67b92b59a24dec3cf1e7a47790'
FAILURE=IDEAS/'evidence/agent-suffix-pair-actual-original-failure-v1'
FAILURE_SHA='fbe7c42ae55efeb327a1932c4b664a88acd7e8393483bee4d78825b2f7f96c18'
OUTPUT=IDEAS/'artifacts/agent-suffix-pair-actual-delivery-prepared-20261003-v2'


def sha(path):
    path=Path(path)
    if path.suffix.lower()=='.gguf':raise ValueError('No model hashing or metadata in preparation')
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def pin(path):
    path=Path(path);stat=path.stat()
    return dict(bytes=stat.st_size,sha256=sha(path),mode=stat.st_mode&0o777)


def check(path,expected):
    actual=pin(path)
    if any(actual[key]!=expected[key] for key in ('bytes','sha256','mode') if key in expected):raise ValueError('Explicit prerequisite differs: '+str(path))
    return actual


def module():
    check(CONTROLLER,dict(sha256=CONTROLLER_SHA))
    spec=importlib.util.spec_from_file_location('corrective_delivery_controller',CONTROLLER)
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);return c


def archive(path,digest,c):
    check(path/'manifest.json',dict(sha256=digest));manifest=c.load_json(path/'manifest.json')
    physical={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p!=path/'manifest.json'}
    if set(manifest['files'])!=physical:raise ValueError('Complete prerequisite archive required: '+str(path))
    for name,expected in manifest['files'].items():check(path/name,expected)
    return manifest


def prepare(build_manifest_sha256):
    if type(build_manifest_sha256) is not str or not re.fullmatch('[0-9a-f]{64}',build_manifest_sha256):raise ValueError('Explicit future ROOT build manifest SHA required')
    c=module();check(PLAN,dict(sha256=PLAN_SHA));plan=c.load_json(PLAN)
    check(BUILD/'build-manifest.json',dict(sha256=build_manifest_sha256));build=c.load_json(BUILD/'build-manifest.json')
    if (build['status']!='CALLER_COMPILE_LINK_PASS' or build['source_freeze']!=SOURCE_FREEZE
        or build['plan_sha256']!=PLAN_SHA or build['all_pins_unchanged'] is not True
        or build['model_stat_open_or_executable_run'] is not False
        or len(build['results'])!=2 or any(type(r['returncode']) is not int or r['returncode']!=0 or r['warning'] is not False for r in build['results'])):
        raise ValueError('Two warning-free ROOT-authorized caller commands required')
    for name,expected in build['outputs'].items():check(BUILD/name,expected)
    if not os.access(BUILD/'ds4-agent-pair-actual',os.X_OK):raise ValueError('Built corrective caller not executable')
    for group in ('sources','shaders'):
        for name,expected in plan[group].items():check(WORK/name,expected)
    for expected in plan['objects'].values():
        check(Path(expected['path']),expected);check(Path(expected['source_path']),expected)
    check(Path(plan['compiler_sdk_proof']['path']),plan['compiler_sdk_proof'])
    compiler=c.load_json(plan['compiler_sdk_proof']['path'])
    for key in ('selected_wrapper','actual_clang','sdk_settings','libSystem_stub'):check(Path(compiler[key]['path']),compiler[key])
    cpu=archive(CPU,CPU_SHA,c);archive(FAILURE,FAILURE_SHA,c)
    if cpu['source_freeze']!=SOURCE_FREEZE or cpu['status']!='CORRECTIVE_PROTOCOL_AND_ADMISSION_CPU_PASS_NATIVE_CALLER_UNBUILT' or cpu['tests']!=dict(delivery=12,authority=8,original36=36):raise ValueError('Exact corrective CPU evidence required')
    check(c.GUARD,dict(sha256=c.GUARD_SHA));check(c.PYTHON,dict(sha256=c.PYTHON_SHA))
    third=IDEAS/'agent_suffix_pair_actual_v1/third.txt'
    text=third.read_bytes();header=(WORK/'speed-bench/agent_suffix_pair_actual_literals.h').read_text()
    literal=c.decode_json(re.search(r'pair_actual_third_user = (".*");',header).group(1)).encode()
    if literal!=text+b'\n' or c.delivery_payload(text.decode())!=literal:raise ValueError('Exact protocol-delivered third binding differs')
    if len(c.SHADERS)!=26 or len(set(c.SHADERS.values()))!=26 or set(plan['shaders'])!={'metal/'+name for name in c.SHADERS.values()}:raise ValueError('Exact26 shader inventory required')

    OUTPUT.mkdir(exist_ok=False)
    def copy(source,name):
        target=OUTPUT/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if pin(source)!=pin(target):raise ValueError('Copied bytes/mode differ: '+str(name))
    copy(BUILD/'ds4-agent-pair-actual','agent');copy(BUILD/'build-manifest.json','caller-build-manifest.json');copy(CONTROLLER,'run_agent_suffix_pair_actual.py')
    for name in ('task.json','grader.py','update_context.py','run_grader.py','first.txt','second.txt','third.txt'):copy(IDEAS/'agent_suffix_pair_actual_v1'/name,name)
    for name in c.SHADERS.values():copy(WORK/'metal'/name,Path('metal')/name)
    for path,label in ((CPU,'corrective-cpu'),(FAILURE,'original-failed-attempt'),(BUILD,'corrective-build')):
        for file in sorted(path.rglob('*')):
            if file.is_file():copy(file,Path('provenance')/label/file.relative_to(path))
    for name in plan['sources']:copy(WORK/name,Path('provenance/source')/name)
    copy(PLAN,'provenance/native-build-plan.json');copy(Path(__file__),'provenance/prepare_agent_suffix_pair_actual_delivery_v2.py')
    files={str(p.relative_to(OUTPUT)):pin(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
    prepared=dict(status='PREPARED_CORRECTIVE_INFRASTRUCTURE_ATTEMPT_NOT_GRANTED_NOT_RUN',corrective_infrastructure_attempt=c.CORRECTION,first_attempt=True,no_retries=True,
        source_freeze=SOURCE_FREEZE,caller_build_manifest_sha256=build_manifest_sha256,cpu_archive_sha256=CPU_SHA,original_failure_archive_sha256=FAILURE_SHA,
        task_sha256=c.TASK_SHA,model=c.MODEL,policy=c.POLICY,turns=3,timeout_seconds=900,files=files,shader_env={key:str(OUTPUT/'metal'/name) for key,name in c.SHADERS.items()},sole_root_grant_required=True,
        selector_admission_ledger_scope='Forecast reservations only; selector does not allocate canonical undo, worker-owned ledger excludes UI-transferred buffers and global observed peak.',
        primary_persistence='NOT_EXERCISED',model_metadata_accessed=False,scope='One corrective infrastructure attempt of unchanged task; future approximate semantic result is one case, no forgetting/default/hidden-state/speed guarantee.',
        native_default_off_baseline='Future separately corrected source/preparation, chosen only from this structural pre-third selection; original failed attempt has no selection.',
        failure_contract='Every wrong/capped/eligibility/fallback/timeout/tool/grading/infrastructure first attempt remains retained; no automatic replay/repair.')
    with (OUTPUT/'manifest.json').open('x') as f:json.dump(prepared,f,indent=2);f.write('\n')
    digest=sha(OUTPUT/'manifest.json')
    with (OUTPUT/'prepared-command.json').open('x') as f:json.dump(dict(manifest_sha256=digest,command=c.command(OUTPUT,digest),cwd=str(c.ROOT)),f,indent=2);f.write('\n')
    c.verify(OUTPUT,digest)
    if any((OUTPUT/name).exists() for name in ('root-launch-proof.json','run','pressure')):raise ValueError('Preparation must not create grant/model run')
    with (OUTPUT/'nonmodel-preflight.json').open('x') as f:json.dump(dict(status='PASS_NOT_GRANTED_NOT_RUN',manifest_sha256=digest,command_sha256=sha(OUTPUT/'prepared-command.json'),files=len(files),exact26_shaders=True,all57sources14objects_compiler_sdk_cpu_originalfailure_verified=True,model_or_guard_or_caller_executed=False),f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared=str(OUTPUT),manifest_sha256=digest,command_sha256=sha(OUTPUT/'prepared-command.json'),files=len(files))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build-manifest-sha256',required=True);a=p.parse_args();prepare(a.build_manifest_sha256)
