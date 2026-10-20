"""Correct the unrun V1 frontend build plan; no compiler or model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

WORK=Path('/Volumes/Models/ds4/.workspaces/idea-reasoning-budget-confirmation')
V1=WORK/'speed-bench/ideas/artifacts/reasoning-budget-confirmation-source-freeze-20261003-v1'
V1_INPUT_SHA='cce60cf7ff19c3ef8d76410c857d0b422c1874d1558c1339551923f3d1a66b83'
V1_PLAN_SHA='8e1cbd7f9b72e14767394bc2235380fd612d42ba37ae4de4890476cb9adf1cc5'
C=WORK/'speed-bench/reasoning_budget_confirmation_probe.c'
C_SHA='dd05bfa3967bc211e9533e4f3108aee2b4bec9b7f44da7bd63a9d6d5b277ea1a'
HEADER=WORK/'speed-bench/reasoning_budget_confirmation_cases.h'
HEADER_SHA='1884857da9e1a9202c7fb938253f4585385d4a7693b2c8f3b4a4e6682cfc53ef'
SDK=Path('/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk')
SELFTEST='--helper-self-test'
TEST=WORK/'tests/test_reasoning_budget_confirmation_build.py'


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    path=Path(path);require(path.suffix.lower()!='.gguf','no model access')
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def put(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def selftest_gate(text,flag):
    require(flag==SELFTEST,'literal accepted model-free selftest argument')
    branch='if (argc==2 && !strcmp(argv[1],"'+flag+'")) return self_test();'
    require(text.count(branch)==1 and text.index(branch)<text.index('ds4_engine_open('),'selftest must return before engine/model open')


def command(old,output):
    require(output.is_absolute() and not output.is_relative_to(V1),'V2 build outputs must be outside immutable V1 inputs')
    require(old['compile_argv'].count('-isysroot')==1 and old['compile_argv'].count('-o')==1,'unique reviewed SDK and output arguments')
    argv=old['compile_argv'][:];argv[argv.index('-isysroot')+1]=str(SDK);argv[argv.index('-o')+1]=str(output/'reasoning-budget-confirmation-probe');return argv


def validate(plan,old,output,text):
    require(plan['compile_argv']==command(old,output),'only SDK/output argv substitutions')
    require(plan['selftest_argv']==[str(output/'reasoning-budget-confirmation-probe'),SELFTEST],'correct model-free selftest command')
    require(plan['sdk_path']==str(SDK) and 'MacOSX.sdk' not in plan['compile_argv'],'explicit versioned SDK')
    require(plan['source_sha256']==C_SHA and plan['header_sha256']==HEADER_SHA and plan['input_freeze_sha256']==V1_INPUT_SHA,'unchanged caller/header/input source authority')
    require(plan['core_or_Metal_rebuild'] is False and plan['compile_or_link_granted'] is False and type(plan['max_concurrent_compilers']) is int and plan['max_concurrent_compilers']==1,'ungranted focused build-only scope')
    selftest_gate(text,plan['selftest_argv'][-1])


def prepare(output,proof_path):
    require(sha(V1/'input-freeze.json')==V1_INPUT_SHA and sha(V1/'build-plan.json')==V1_PLAN_SHA,'V1 unrun input/build plan retained')
    require(sha(C)==C_SHA and sha(HEADER)==HEADER_SHA,'caller and authored cases must remain unchanged')
    old=json.loads((V1/'build-plan.json').read_text());require(old['selftest_argv'][-1]=='--self-test','preserved V1 review refusal')
    selftest_gate(C.read_text(),SELFTEST)
    proof_path=Path(proof_path);proof=json.loads(proof_path.read_text());require(proof['source_sha256']==sha(__file__) and proof['test_sha256']==sha(TEST) and type(proof['returncode']) is int and proof['returncode']==0 and proof['model_or_metadata_opened'] is False,'passing source/plan gate authority')
    for path,pin in old['borrowed_pins'].items():require(sha(path)==pin,'borrowed compiler/object/header/SDK source differs')
    sdk_files={str(SDK/name):sha(SDK/name) for name in ('SDKSettings.json','usr/include/stdio.h','usr/lib/libSystem.tbd')}
    alias=SDK.parent/'MacOSX.sdk'
    require(all(sha(alias/name)==sdk_files[str(SDK/name)] for name in ('SDKSettings.json','usr/include/stdio.h','usr/lib/libSystem.tbd')),'explicit SDK differs from pinned inherited alias')
    output.mkdir(parents=True,exist_ok=False)
    outputs=[output/name for name in ['reasoning-budget-confirmation-probe','reasoning-budget-confirmation-probe.dSYM','compile.log','compile-result.json','selftest.log','selftest-result.json','build-manifest.json']]
    require(all(not p.exists() for p in outputs),'exclusive V2 build outputs already exist')
    plan=dict(old,status='SOURCE_ONLY_UNRUN_V2',compile_argv=command(old,output),selftest_argv=[str(outputs[0]),SELFTEST],all_outputs_must_be_absent=list(map(str,outputs)),sdk_path=str(SDK),explicit_sdk_sha256=sdk_files,borrowed_pins={**old['borrowed_pins'],**sdk_files},input_freeze=dict(path=str(V1/'input-freeze.json'),sha256=V1_INPUT_SHA),input_freeze_sha256=V1_INPUT_SHA,source_freeze_revision='a875c9fa2a609a1cc4fb9c546b5ec8f86c0ac653',preserved_unrun_V1_plan=dict(path=str(V1/'build-plan.json'),sha256=V1_PLAN_SHA,review_refusal='--self-test is unsupported; real model-free main branch is --helper-self-test'),ROOT_compile_grant_required=True,model_or_tokenizer_or_guard_grant=False)
    validate(plan,old,output,C.read_text())
    put(output/'build-plan.json',plan)
    put(output/'review-findings.json',dict(V1_preserved_unrun=True,no_build_or_model_was_run=True,V1_selftest_argument_incorrect=True,V2_argument=SELFTEST,immutable_input_directory_has_no_new_build_outputs=True,ceiling2_choice='Proposal called the path bounded without spelling out the bound. ROOT explicitly accepted the literal y<=x+2 fixture before any build/outcome; it was not claimed fully specified in the original concise proposal.',ROOT_reflection_check=dict(allowed_total=35190,prefix_to_5_4=117,suffix_from_5_4=125,avoiding=20565)))
    for source,name in [(Path(__file__),'build-source.py'),(TEST,'build-tests.py'),(proof_path,'cpu-proof.json'),(proof_path.parent/'tests.log','cpu-tests.log')]:shutil.copy2(source,output/name)
    frozen=dict(files={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},external_source=dict(C=dict(path=str(C),sha256=C_SHA),header=dict(path=str(HEADER),sha256=HEADER_SHA),V1=dict(path=str(V1/'input-freeze.json'),sha256=V1_INPUT_SHA)),ROOT_build_grant_required=True,no_compile_or_model_execution=True)
    put(output/'source-proof.json',frozen)
    print(json.dumps(dict(output=str(output),plan_sha256=sha(output/'build-plan.json'),source_proof_sha256=sha(output/'source-proof.json'),caller_unchanged=True,inputs_unchanged=True,compiled=False,model_opened=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',type=Path,required=True);p.add_argument('--cpu-proof',type=Path,required=True);a=p.parse_args();prepare(a.prepare.resolve(),a.cpu_proof.resolve())
