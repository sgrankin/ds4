#!/usr/bin/env python3
"""Standalone fixed math16 preparation/run; a separate ROOT model grant is mandatory."""
import argparse, errno, fcntl, hashlib, json, os
from pathlib import Path
import re, shlex, shutil, subprocess, sys
import analyze_reasoning_budget_confirmation as analysis

ROOT=Path('/Volumes/Models/ds4')
WORK=ROOT/'.workspaces/idea-reasoning-budget-confirmation-runner'
SOURCE=ROOT/'.workspaces/idea-reasoning-budget-confirmation'
INPUT=SOURCE/'speed-bench/ideas/artifacts/reasoning-budget-confirmation-source-freeze-20261003-v1'
INPUT_SHA='cce60cf7ff19c3ef8d76410c857d0b422c1874d1558c1339551923f3d1a66b83'
BUILD=SOURCE/'speed-bench/ideas/artifacts/reasoning-budget-confirmation-build-prepared-20261003-v2'
BUILD_MANIFEST_SHA='b67874d6642d3c68ea280fa4410fd27235eb654c0b9c3fd8c0b79a5dea072ccf'
BINARY_SHA='1cdeaaa7f7dcb8d4f073b7694f7cce48a9e3cc6ec1253dd63be7bce93bfaca39'
BUILD_PLAN_SHA='f988bacfd6c2dffaf0cdf55bc39cbea53ceef41da7685949549672a7d947987c'
PRIOR=ROOT/'.workspaces/idea-reasoning-budget-runner/speed-bench/ideas/artifacts/reasoning-budget-pilot-prepared-20261003-v2'
PRIOR_PLAN_SHA='d78f72e601fd811280d62adc9d2971bd7612ab10c58b29fdde47abfcb026a286'
GUARD=ROOT/'speed-bench/ideas/pressure_run.py'
GUARD_SHA='461e8c517e82a687b4579fc450c1e7128b1510192d088bab719682b33c946c3c'
PYTHON=Path('/opt/homebrew/opt/python@3.14/bin/python3.14')
PYTHON_TARGET=Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/bin/python3.14')
PYTHON_SHA='7ebde552773a4045532637b114d645a313e045ac07b26f5e520bf039f1e34063'
FRAMEWORK=Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python')
FRAMEWORK_SHA='b80d260a48404da13a32dd44d7283aadc80f6a6e56ed6e03ed5597ec6f46d48d'
SCORER_SHA='73a2263294f2d2257fb3e3c20848c8aa5da99ebcccfc4ec4a6a80aff8bb92d5e'
PLACEHOLDER='<PLAN_SHA256>'
MODEL=dict(path=str(ROOT/'gguf/DeepSeek-V4.1-Flash-Q2.gguf'),bytes=365713686528,mtime_ns=1789237692837653163)
ENV={'DS4_OVERTHINK_PENALTY':'0','DS4_OVERTHINK_SCOPE':'think','DS4_OVERTHINK_AFTER':'0','DS4_OVERTHINK_MARKERS':'hesitation'}
SETTINGS=dict(effort=25,cap=2048,seed=20261002,temperature_float_value=analysis.TEMP,
    top_p_float_value=analysis.TOP_P,top_k=0,min_p=0,cache_gib=64,context_size=8192,
    prefill_chunk=512,threads=2,mtp_draft=1,native_penalty=0)
SOURCES={'reasoning_budget_confirmation_probe.c':'dd05bfa3967bc211e9533e4f3108aee2b4bec9b7f44da7bd63a9d6d5b277ea1a',
    'reasoning_budget_confirmation_cases.h':'1884857da9e1a9202c7fb938253f4585385d4a7693b2c8f3b4a4e6682cfc53ef',
    'reasoning_budget_policy.h':'cce49e24c7a7e538131d5c3ccdb3384128bbef68f19d897b625bf93117924a0c',
    'reasoning_budget_native_api.h':'5f569084ab718b358e3779b50cc0fafacfce1b698fcde4fd13bec4292ae8521b',
    'native_effort_final_score.h':'d7a5d2dd85047ca290365594f12f917ee71a1bb42fb565c639a2f5a4fda6b7b3',
    'overthink_diagnostics.h':'1a4c26f98423cebdbd2268cf354bdcb35b7f9cfe07d2632c58888c7735427a4e'}
SHADER_NAMES=set(('ARGSORT BIN CONCAT CPY DEEPSEEK4_VISION DENSE DSV41 DSV4_HC DSV4_KV DSV4_MISC '
    'DSV4_ROPE FLASH_ATTN GET_ROWS GLM53_BF16 GLM53_KDA GLM53_VISION GLU MOE NORM '
    'QWEN4 QWEN4_VISION REPEAT SET_ROWS SOFTMAX SUM_ROWS UNARY').split())
SCRIPTS=['run_reasoning_budget_confirmation.py','analyze_reasoning_budget_confirmation.py','reasoning_budget_signed64.py']
TEST=WORK/'tests/test_reasoning_budget_confirmation_runner.py'
BUILD_FILES=['build-plan.json','build-manifest.json','reasoning-budget-confirmation-probe','compile.log','compile-result.json','selftest.log','selftest-result.json','source-proof.json','root-build-grant.json','root-preflight-refusal-v1.json','root-preflight-refusal-v2.json','review-findings.json','build-source.py','build-tests.py','cpu-proof.json','cpu-tests.log','reasoning-budget-confirmation-probe.dSYM/Contents/Info.plist','reasoning-budget-confirmation-probe.dSYM/Contents/Resources/DWARF/reasoning-budget-confirmation-probe','reasoning-budget-confirmation-probe.dSYM/Contents/Resources/Relocations/aarch64/reasoning-budget-confirmation-probe.yml']


def put(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def asset(path):
    path=Path(path);analysis.require(path.suffix.lower()!='.gguf','only non-model assets')
    if path==PYTHON:analysis.require(path.resolve()==PYTHON_TARGET,'pinned Python launcher target')
    else:analysis.require(not path.is_symlink(),'no implicit asset aliases')
    return dict(sha256=analysis.sha(path),bytes=path.stat().st_size)


def input_check():
    analysis.require(analysis.sha(INPUT/'input-freeze.json')==INPUT_SHA,'immutable unused16 input authority')
    frozen=analysis.load(INPUT/'input-freeze.json')
    analysis.require(frozen['settings']==SETTINGS and frozen['arms']==[dict(name=n,budget=b) for n,b in zip(analysis.NAMES,[0,1024,0])]
        and frozen['ordered_cases']==list(analysis.GOLDS) and type(frozen['all_first_attempts']) is int and frozen['all_first_attempts']==48,
        'fixed prior authored settings/order/48 first attempts')
    for name,pin in frozen['files'].items():analysis.require(analysis.sha(analysis.local(INPUT,name))==pin,'all original input artifacts unchanged')
    analysis.fixture_check(analysis.load(INPUT/'fixture.json'))
    return frozen


def qualify_build(manifest,expected_sha):
    analysis.require(expected_sha==BUILD_MANIFEST_SHA and analysis.sha(BUILD/'build-manifest.json')==expected_sha,'explicit successful ROOT build-manifest pin')
    analysis.require(analysis.sha(BUILD/'build-plan.json')==BUILD_PLAN_SHA,'unchanged V2 source-only plan')
    bp=analysis.load(BUILD/'build-plan.json')
    analysis.require(analysis.sha(BUILD/'source-proof.json')==manifest['source_proof_sha256'] and analysis.sha(BUILD/'root-build-grant.json')==manifest['root_grant_sha256'],'actual ROOT source proof and compile grant')
    analysis.require(manifest['status']=='NATIVE_PRIVATE_FRONTEND_BUILD_AND_MODEL_FREE_SELFTEST_PASS'
        and manifest['plan_sha256']==BUILD_PLAN_SHA and manifest['input_freeze']==dict(path=str(INPUT/'input-freeze.json'),sha256=INPUT_SHA)
        and manifest['core_or_Metal_rebuild'] is False and manifest['model_or_tokenizer_or_guard_run'] is False,'qualified ROOT frontend-only build')
    for name,command in [('compile_result',bp['compile_argv']),('selftest_result',bp['selftest_argv'])]:
        result=manifest[name]
        analysis.require(result['command']==command and type(result['returncode']) is int and result['returncode']==0
            and analysis.number(result['seconds'],'build diagnostic time')>=0,'reviewed actual one compile/selftest')
        for field in ('stdout_bytes','stderr_bytes'):analysis.integer(result[field],field,1<<30)
        result_path=BUILD/('compile-result.json' if name=='compile_result' else 'selftest-result.json')
        analysis.require(analysis.load(result_path)==result,'actual recorded build result')
        log=BUILD/('compile.log' if name=='compile_result' else 'selftest.log')
        analysis.require(analysis.sha(log)==result['log_sha256'],'retained actual build/selftest log')
    analysis.require(manifest['borrowed_pins']==bp['borrowed_pins'] and manifest['explicit_sdk_sha256']==bp['explicit_sdk_sha256'],'all unchanged borrowed pins and explicit SDK')
    for path,pin in bp['borrowed_pins'].items():analysis.require(analysis.sha(path)==pin,'actual borrowed build asset changed')
    for n,key in [('reasoning_budget_confirmation_probe.c','source'),('reasoning_budget_confirmation_cases.h','header')]:
        analysis.require(manifest[key]==dict(path=str(SOURCE/'speed-bench'/n),sha256=SOURCES[n]),'exact original source/header')
    for name,pin in SOURCES.items():analysis.require(analysis.sha(SOURCE/'speed-bench'/name)==pin,'literal qualified source')
    binary=BUILD/'reasoning-budget-confirmation-probe';info=binary.stat()
    analysis.require(manifest['binary']==dict(path=str(binary),sha256=BINARY_SHA,bytes=info.st_size,mode=info.st_mode&0o777)
        and asset(binary)['sha256']==BINARY_SHA and os.access(binary,os.X_OK),'actual built helper identity')
    return manifest


def rendered_command(plan,pin):return [pin if arg==PLACEHOLDER else arg for arg in plan['guard_command_template']]


def validate_plan(plan,plan_path):
    plan_path=Path(plan_path).resolve();p=plan_path.parent
    analysis.require(plan_path.name=='plan.json' and p.parent==WORK/'speed-bench/ideas/artifacts'
        and re.fullmatch('[A-Za-z0-9_-]+',p.name),'exclusive persistent plan path')
    analysis.require(plan['schema']=='math-budget-confirmation16-v1' and plan['prepared_only'] is True
        and plan['model_grant'] is False and plan['analysis_requires_root_release'] is True
        and plan['no_automatic_retry_or_scheduler'] is True,'prepared-only no-autochain contract')
    analysis.require(plan['settings']==SETTINGS and all(type(plan['settings'][k]) is type(v) for k,v in SETTINGS.items()),'fixed typed settings')
    analysis.require(plan['model']==MODEL and all(type(plan['model'][k]) is type(v) for k,v in MODEL.items()),'retained explicit model identity')
    analysis.require(plan['workspace']==str(WORK) and plan['run_cwd']==str(WORK) and plan['root_cwd']==str(ROOT),'fixed cwd')
    analysis.require(plan['binary']==str(BUILD/'reasoning-budget-confirmation-probe') and plan['build_manifest']==str(BUILD/'build-manifest.json')
        and plan['build_plan_sha256']==BUILD_PLAN_SHA and plan['build_manifest_sha256']==BUILD_MANIFEST_SHA and plan['binary_sha256']==BINARY_SHA and plan['input_freeze_sha256']==INPUT_SHA,'qualified build/input bindings')
    analysis.require(plan['guard']==str(GUARD) and plan['guard_sha256']==GUARD_SHA and plan['launch_python']==str(PYTHON)
        and plan['guard_interpreter_identity']==dict(path=str(FRAMEWORK),sha256=FRAMEWORK_SHA),'literal MAIN guard/framework')
    expected=[str(PYTHON),str(GUARD),str(p/'pressure'),'--',str(PYTHON),str(p/SCRIPTS[0]),'--run',str(plan_path),'--plan-sha256',PLACEHOLDER,'--root-granted']
    analysis.require(plan['guard_command_template']==expected and plan['output']==str(p/'run') and plan['pressure']==str(p/'pressure')
        and plan['fixture']==str(p/'fixture.json'),'exact prepared command/output authority')
    analysis.require(type(plan['arms']) is list and len(plan['arms'])==3,'three arms')
    for arm,name,budget in zip(plan['arms'],analysis.NAMES,[0,1024,0]):
        run_name=p.name+'-'+name
        analysis.require(set(arm)=={'name','budget','run_name','argv','results_dir'} and arm['name']==name and type(arm['budget']) is int
            and arm['budget']==budget and arm['run_name']==run_name and arm['argv']==[plan['binary'],MODEL['path'],run_name,'25',str(budget)]
            and arm['results_dir']==str(WORK/'speed-bench/ideas/artifacts'/run_name),'fixed literal arm argv')
    shaders={k:v for k,v in plan['environment'].items() if k.endswith('_SOURCE')}
    analysis.require(set(plan['environment'])==set(ENV)|{'DS4_METAL_'+n+'_SOURCE' for n in SHADER_NAMES}
        and all(plan['environment'][k]==v for k,v in ENV.items())
        and all(v==str(p/'metal'/(k[len('DS4_METAL_'):-len('_SOURCE')].lower()+'.metal')) for k,v in shaders.items()),'only frozen native environment/26 shaders')
    bp=analysis.load(BUILD/'build-plan.json')
    objects={path:pin for path,pin in bp['borrowed_pins'].items() if path.endswith('.o')}
    source_refs={str(p/'source'/name):pin for name,pin in SOURCES.items()}
    analysis.require(plan['core_objects']==objects and len(objects)==8 and plan['source_refs']==source_refs,'complete core/native-source bindings')
    mandatory={str(p/n) for n in SCRIPTS+['fixture.json','cpu-proof.json','cpu-tests.log','runner-tests.py']}
    mandatory|={str(INPUT/'input-freeze.json')}|{str(INPUT/n) for n in analysis.load(INPUT/'input-freeze.json')['files']}
    mandatory|={str(BUILD/n) for n in BUILD_FILES}
    mandatory|=set(bp['borrowed_pins'])|set(source_refs)|set(shaders.values())|{str(GUARD),str(PYTHON),str(PYTHON_TARGET),str(FRAMEWORK)}
    analysis.require(set(plan['assets'])==mandatory,'exact complete mandatory inventory; no omitted authority')
    for path,pin in plan['assets'].items():
        analysis.require(type(path) is str and Path(path).is_absolute() and type(pin) is dict and set(pin)=={'sha256','bytes'}
            and type(pin['sha256']) is str and re.fullmatch('[0-9a-f]{64}',pin['sha256']),'typed asset')
        analysis.integer(pin['bytes'],'asset bytes',1<<40)
    fixed={str(GUARD):GUARD_SHA,str(PYTHON):PYTHON_SHA,str(PYTHON_TARGET):PYTHON_SHA,str(FRAMEWORK):FRAMEWORK_SHA,
        str(p/'reasoning_budget_signed64.py'):SCORER_SHA,str(INPUT/'input-freeze.json'):INPUT_SHA,str(BUILD/'build-plan.json'):BUILD_PLAN_SHA,
        plan['binary']:plan['binary_sha256'],plan['build_manifest']:plan['build_manifest_sha256'],**objects,**source_refs}
    analysis.require(all(plan['assets'][k]['sha256']==v for k,v in fixed.items()),'independent fixed inventory hashes')
    analysis.fixture_check(analysis.load(plan['fixture']))


def check_assets(plan):
    for path,pin in plan['assets'].items():analysis.require(asset(path)==pin,'frozen asset changed: '+path)
    input_check();qualify_build(analysis.load(plan['build_manifest']),plan['build_manifest_sha256'])


def require_guard(plan,plan_path,pin):
    analysis.require(analysis.sha(GUARD)==GUARD_SHA,'literal MAIN guard unchanged')
    actual=shlex.split(subprocess.check_output(['/bin/ps','-p',str(os.getppid()),'-o','command='],text=True,timeout=2).strip())
    expected=rendered_command(plan,pin)
    analysis.require(bool(actual) and actual[1:]==expected[1:],'exact direct MAIN guard argv suffix')
    identity=plan['guard_interpreter_identity']
    try:same=os.path.samefile(actual[0],identity['path'])
    except OSError:same=False
    analysis.require(same and analysis.sha(identity['path'])==identity['sha256'],'actual pinned framework Python')
    with (ROOT/'speed-bench/ideas/artifacts/model-job.lock').open('r+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EAGAIN,errno.EWOULDBLOCK):raise
        else:
            fcntl.flock(lock,fcntl.LOCK_UN);raise ValueError('MAIN guard must hold model lock')
    return dict(parent_argv=actual,exact_suffix=True,framework_samefile=True,lock_held=True)


def require_grant(plan,plan_path,pin):
    proof=analysis.load(Path(plan_path).parent/'root-launch-proof.json')
    analysis.require(proof['status']=='GRANTED' and proof['sole_model_grant'] is True and proof['manifest_sha256']==pin
        and proof['command']==rendered_command(plan,pin),'separate explicit pinned ROOT sole-model grant')
    analysis.require(analysis.number(proof['time'],'grant time')>0,'ROOT grant timestamp')


def model_identity(plan):
    p=Path(plan['model']['path']);analysis.require(p.resolve()==p,'explicit model no symlink')
    st=p.stat();actual=dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    analysis.require(actual==plan['model'],'guarded model identity');return actual


def run(plan_path,pin,root_granted=False):
    analysis.require(root_granted is True and analysis.sha(plan_path)==pin,'separate explicit grant and frozen plan before guard')
    plan=analysis.load(plan_path);validate_plan(plan,plan_path)
    parent=require_guard(plan,plan_path,pin);require_grant(plan,plan_path,pin);check_assets(plan)
    analysis.require(not Path(plan['output']).exists() and all(not Path(a['results_dir']).exists() for a in plan['arms']),'exclusive first-run paths')
    current=model_identity(plan)  # First model stat: after exact parent, held lock, grant, all assets.
    out=Path(plan['output']);out.mkdir();put(out/'parent-proof.json',parent);put(out/'model-identity.json',current)
    put(out/'launch.json',dict(root_granted=True,plan_sha256=pin,inherited_ds4_removed=[k for k in os.environ if k.startswith('DS4')]))
    env={k:v for k,v in os.environ.items() if not k.startswith('DS4')};env.update(plan['environment'])
    fixture=analysis.load(plan['fixture']);done=[];validated=[]
    try:
        for a in plan['arms']:
            analysis.require(analysis.sha(plan_path)==pin,'frozen plan before each arm');require_guard(plan,plan_path,pin);require_grant(plan,plan_path,pin);check_assets(plan);model_identity(plan)
            analysis.require(not Path(a['results_dir']).exists(),'fresh producer directory')
            put(out/(a['name']+'-command.json'),dict(argv=a['argv'],cwd=plan['run_cwd'],ds4_environment=plan['environment']))
            with (out/(a['name']+'-stdout.log')).open('x') as stdout,(out/(a['name']+'-stderr.log')).open('x') as stderr:
                process=subprocess.Popen(a['argv'],cwd=plan['run_cwd'],env=env,stdout=stdout,stderr=stderr);put(out/(a['name']+'-started.json'),dict(pid=process.pid))
                try:rc=process.wait()
                except BaseException:
                    if process.returncode is None:
                        process.terminate()
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
                    raise
            put(out/(a['name']+'-process.json'),dict(returncode=rc))
            analysis.require(type(rc) is int and rc in (0,3),'first producer infrastructure failure retained')
            value=analysis.validate_arm(a['results_dir'],a['budget'],fixture,rc)
            if a['budget']:analysis.pre_force(validated[0],value)
            elif validated:analysis.exact_native(validated[0],value)
            put(out/(a['name']+'-summary.json'),value[0]);done.append(a['name']);validated.append(value)
            print(json.dumps(dict(arm=a['name'],**value[0]),allow_nan=False),flush=True)
        check_assets(plan);require_guard(plan,plan_path,pin);require_grant(plan,plan_path,pin);model_identity(plan)
        rc=3 if any(v[0]['invalid'] for v in validated) else 0
        put(out/'complete.json',dict(status='COMPLETE',arms=done,returncode=rc,interpretation='All48 first attempts retained. Offline final analysis separately requires actual ROOT release proof. No retries/speed/cross-prefix KL claim.'));return rc
    except BaseException as e:
        put(out/'first-failure.json',dict(type=type(e).__name__,message=str(e),completed_arms=done));raise


def prepare(output,build_sha,cpu_proof):
    output=Path(output).resolve();analysis.require(output.parent==WORK/'speed-bench/ideas/artifacts' and re.fullmatch('[A-Za-z0-9_-]+',output.name),'exclusive persistent prepare path')
    # An absent future build is an explicit dependency; no model/guard/build is attempted.
    input_check();build=qualify_build(analysis.load(BUILD/'build-manifest.json'),build_sha)
    cpu=analysis.load(cpu_proof);analysis.require(type(cpu['returncode']) is int and cpu['returncode']==0 and cpu['model_or_metadata_opened'] is False and cpu['real_helper_or_guard_executed'] is False,'CPU-only source-qualified proof')
    for path,pin in cpu['sources'].items():analysis.require(analysis.sha(path)==pin,'tested source unchanged')
    expected_scripts={str(Path(__file__).parent/n) for n in SCRIPTS}
    analysis.require(set(cpu['sources'])==expected_scripts|{str(TEST)},'complete tested source inventory')
    analysis.require(analysis.sha(PRIOR/'plan.json')==PRIOR_PLAN_SHA,'old validated native shader plan authority')
    prior=analysis.load(PRIOR/'plan.json');bp=analysis.load(BUILD/'build-plan.json')
    output.mkdir(exist_ok=False)
    try:
        for n in SCRIPTS:shutil.copy2(Path(__file__).parent/n,output/n)
        shutil.copy2(TEST,output/'runner-tests.py');shutil.copy2(cpu_proof,output/'cpu-proof.json');shutil.copy2(Path(cpu_proof).parent/'tests.log',output/'cpu-tests.log')
        shutil.copy2(INPUT/'fixture.json',output/'fixture.json');(output/'source').mkdir();(output/'metal').mkdir()
        refs={}
        for n,pin in SOURCES.items():
            src=SOURCE/'speed-bench'/n;analysis.require(analysis.sha(src)==pin,'qualified original source');dst=output/'source'/n;shutil.copy2(src,dst);refs[str(dst)]=pin
        environment=dict(ENV)
        for n in sorted(SHADER_NAMES):
            key='DS4_METAL_'+n+'_SOURCE';src=Path(prior['environment'][key]);analysis.require(asset(src)==prior['assets'][str(src)],'prior frozen shader identity')
            dst=output/'metal'/(n.lower()+'.metal');shutil.copy2(src,dst);environment[key]=str(dst)
        arms=[]
        for name,budget in zip(analysis.NAMES,[0,1024,0]):
            run_name=output.name+'-'+name
            arms.append(dict(name=name,budget=budget,run_name=run_name,argv=[str(BUILD/'reasoning-budget-confirmation-probe'),MODEL['path'],run_name,'25',str(budget)],results_dir=str(WORK/'speed-bench/ideas/artifacts'/run_name)))
        files=[p for p in output.rglob('*') if p.is_file()]+[INPUT/'input-freeze.json']+[INPUT/n for n in analysis.load(INPUT/'input-freeze.json')['files']]
        files+=[BUILD/n for n in BUILD_FILES]+[Path(p) for p in bp['borrowed_pins']]+[GUARD,PYTHON,PYTHON_TARGET,FRAMEWORK]
        assets={str(p):asset(p) for p in files}
        plan=dict(schema='math-budget-confirmation16-v1',prepared_only=True,model_grant=False,analysis_requires_root_release=True,no_automatic_retry_or_scheduler=True,
            workspace=str(WORK),run_cwd=str(WORK),root_cwd=str(ROOT),output=str(output/'run'),pressure=str(output/'pressure'),binary=str(BUILD/'reasoning-budget-confirmation-probe'),binary_sha256=build['binary']['sha256'],
            build_manifest=str(BUILD/'build-manifest.json'),build_manifest_sha256=build_sha,build_plan_sha256=BUILD_PLAN_SHA,input_freeze_sha256=INPUT_SHA,
            core_objects={path:pin for path,pin in bp['borrowed_pins'].items() if path.endswith('.o')},source_refs=refs,fixture=str(output/'fixture.json'),settings=SETTINGS,environment=environment,arms=arms,assets=assets,model=MODEL,
            guard=str(GUARD),guard_sha256=GUARD_SHA,launch_python=str(PYTHON),guard_interpreter_identity=dict(path=str(FRAMEWORK),sha256=FRAMEWORK_SHA),
            guard_command_template=[str(PYTHON),str(GUARD),str(output/'pressure'),'--',str(PYTHON),str(output/SCRIPTS[0]),'--run',str(output/'plan.json'),'--plan-sha256',PLACEHOLDER,'--root-granted'],
            all_first_attempts=48,limitations=['One fixed independently authored math16 set/seed; every wrong, INVALID and capped outcome retained.','No cross-prefix KL/NLL, quantization causality, general equal-quality, latency/SSD or default-promotion claim.','Fresh source-qualified native bracket; no historical generated reference exists for these unused prompts.'])
        put(output/'plan.json',plan);validate_plan(plan,output/'plan.json');check_assets(plan);pin=analysis.sha(output/'plan.json')
        put(output/'prepared-command.json',dict(command=rendered_command(plan,pin),plan_sha256=pin,launch_authorized=False));return plan,pin
    except BaseException as e:put(output/'preparation-first-failure.json',dict(type=type(e).__name__,message=str(e),model_or_guard_or_compile=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',type=Path);g.add_argument('--run',type=Path);g.add_argument('--verify',type=Path)
    p.add_argument('--build-manifest-sha256');p.add_argument('--cpu-proof',type=Path);p.add_argument('--plan-sha256');p.add_argument('--root-granted',action='store_true');a=p.parse_args()
    try:
        if a.prepare:
            analysis.require(a.cpu_proof and a.build_manifest_sha256,'explicit CPU/build authority');_,pin=prepare(a.prepare,a.build_manifest_sha256,a.cpu_proof);print(json.dumps(dict(prepared=str(a.prepare),plan_sha256=pin,model_grant=False)))
        elif a.verify:
            analysis.require(a.plan_sha256 and analysis.sha(a.verify)==a.plan_sha256,'explicit plan verification pin');plan=analysis.load(a.verify);validate_plan(plan,a.verify);check_assets(plan);print(json.dumps(dict(verified=True,assets=len(plan['assets']),model_or_guard_executed=False)))
        else:analysis.require(a.plan_sha256,'explicit run plan pin');sys.exit(run(a.run,a.plan_sha256,a.root_granted))
    except (ValueError,OSError,KeyError,TypeError) as e:p.exit(1,'math16 budget driver: '+str(e)+'\n')
