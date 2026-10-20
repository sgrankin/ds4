#!/usr/bin/env python3
"""Prepare or explicitly ROOT-granted three-arm protocol pilot; no automatic launches."""
import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import analyze_reasoning_budget as analysis

ROOT = Path('/Volumes/Models/ds4')
WORK = ROOT / '.workspaces/idea-reasoning-budget-runner'
BUILD = ROOT / '.workspaces/idea-reasoning-budget/speed-bench/ideas/artifacts/reasoning-budget-native-build-v2'
BINARY_SHA = '6d567272bbf58bb1674e67fa0448cfac9aa8444bab453aa3286943ea28b34d29'
CORE_SHA = 'd77958a390840f3a555f01df64d52964a656c72e6449c2a1cd8c19f497eb8aca'
REFERENCE = ROOT / '.workspaces/idea-quant-closing/speed-bench/ideas/artifacts/closing-effort25-native-before-20261002-v1'
CLOSING = ROOT / '.workspaces/idea-quant-closing/speed-bench/ideas/artifacts/closing-prepared-20261002-v1'
HOLDOUT = ROOT / '.workspaces/idea-native-effort-holdout-signed64/speed-bench/ideas/artifacts/native-effort-holdout-prepared-20261003-v2'
GUARD = ROOT / 'speed-bench/ideas/pressure_run.py'
GUARD_SHA = '461e8c517e82a687b4579fc450c1e7128b1510192d088bab719682b33c946c3c'
PYTHON = Path('/opt/homebrew/opt/python@3.14/bin/python3.14')
PYTHON_TARGET = Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/bin/python3.14')
PYTHON_SHA = '7ebde552773a4045532637b114d645a313e045ac07b26f5e520bf039f1e34063'
FRAMEWORK = Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python')
FRAMEWORK_SHA = 'b80d260a48404da13a32dd44d7283aadc80f6a6e56ed6e03ed5597ec6f46d48d'
PLACEHOLDER = '<PLAN_SHA256>'
MODEL = dict(path=str(ROOT / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf'), bytes=365713686528, mtime_ns=1789237692837653163)
ENV = {'DS4_OVERTHINK_PENALTY': '0', 'DS4_OVERTHINK_SCOPE': 'think',
       'DS4_OVERTHINK_AFTER': '0', 'DS4_OVERTHINK_MARKERS': 'hesitation'}
SETTINGS = dict(effort=25, cap=2048, cases=4, seed=20261002,
                temperature_float_value=analysis.TEMP, top_p_float_value=analysis.TOP_P, top_k=0, min_p=0,
                cache_gib=64, context_size=8192, prefill_chunk=512, n_threads=2, power_percent=100,
                mtp_draft=1, native_penalty=0, selected_tile_override=None, selected_limit_override=None, reasoning_budget=1024)
SCORER_SHA = '73a2263294f2d2257fb3e3c20848c8aa5da99ebcccfc4ec4a6a80aff8bb92d5e'
CPU_PROOF = WORK / 'speed-bench/ideas/evidence/reasoning-budget-runner-cpu-20261003-v2/proof.json'
CPU_TEST = WORK / 'tests/test_reasoning_budget_runner.py'
OBJECTS = {
    'ds4.o': CORE_SHA,
    'ds4_image.o': '752eace0ab62636538b225d4c80011739354649ee395b6648924928079572528',
    'ds4_distributed.o': '3b09b4b252fd1b06894421b88bc62eeaca24869b839cc17cd22effde1d50dc1a',
    'ds4_tp.o': 'd2f2f5ce0b8bfc7ebcfb0fd80f3eb4e434e89dc4518253cc43a7d253a0b2c2a1',
    'ds4_ssd.o': 'a1fbc20ad4102f4a6f126cde822f2995dd4421153c6256dace554ebca26c2b71',
    'ds4_metal.o': 'ec94b88a9a3da8742c77899e370c1b64869d456ce97d9c8fdba4fae1752bd5ff',
    'ds4_layer_pack.o': '174b07bfca784e04cb938ae3803e3b97696afcdb886ccd518117ffa3400135c8',
    'ds4_engram.o': 'd94530e0a20a2c196ef24ea839175152dabbdb9ad8245626f01bd81a97f34a4b'}
SOURCES = {
    'reasoning_budget_probe.c': '48bf48e6ae5b16adffa53129571c9ff3279ed1e5372e3ca8d56738f0aeed519c',
    'reasoning_budget_policy.h': 'cce49e24c7a7e538131d5c3ccdb3384128bbef68f19d897b625bf93117924a0c',
    'reasoning_budget_native_api.h': '5f569084ab718b358e3779b50cc0fafacfce1b698fcde4fd13bec4292ae8521b',
    'native_effort_final_score.h': 'd7a5d2dd85047ca290365594f12f917ee71a1bb42fb565c639a2f5a4fda6b7b3',
    'overthink_diagnostics.h': '1a4c26f98423cebdbd2268cf354bdcb35b7f9cfe07d2632c58888c7735427a4e',
    'reasoning_budget_cases.h': 'b78fdb2e2e95839962e165d0451bfa52f1f72eb5d455669c6dabe9f4e672b173'}
SHADER_NAMES = set(('ARGSORT BIN CONCAT CPY DEEPSEEK4_VISION DENSE DSV41 DSV4_HC DSV4_KV DSV4_MISC '
                    'DSV4_ROPE FLASH_ATTN GET_ROWS GLM53_BF16 GLM53_KDA GLM53_VISION GLU MOE NORM '
                    'QWEN4 QWEN4_VISION REPEAT SET_ROWS SOFTMAX SUM_ROWS UNARY').split())


def put(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def asset(path):
    path = Path(path)
    analysis.require(path.suffix.lower() != '.gguf', 'explicit non-model asset')
    if path == PYTHON:
        analysis.require(path.resolve() == PYTHON_TARGET and analysis.sha(path) == PYTHON_SHA,
                         'exact pinned Homebrew launcher symlink target and bytes')
    else:
        analysis.require(not path.is_symlink(), 'explicit asset must not be a symlink')
    return dict(sha256=analysis.sha(path), bytes=path.stat().st_size)


def check_assets(plan):
    for raw, expected in plan['assets'].items():
        analysis.require(asset(raw) == expected, 'frozen asset changed: ' + raw)
    analysis.require(os.access(plan['binary'], os.X_OK), 'original helper executable mode')


def rendered_command(plan, expected_sha):
    return [expected_sha if arg == PLACEHOLDER else arg for arg in plan['guard_command_template']]


def validate_plan(plan, plan_path):
    plan_path = Path(plan_path).resolve()
    analysis.require(plan['schema'] == 'reasoning-budget-pilot-v1' and plan['prepared_only'] is True
                     and plan['model_grant'] is False and plan['analysis_requires_root_release'] is True
                     and plan['no_automatic_retry_or_scheduler'] is True, 'prepared-only pilot contract')
    analysis.require(plan['settings'] == SETTINGS and all(type(plan['settings'][key]) is type(value)
                     for key, value in SETTINGS.items()), 'fixed typed pilot settings')
    analysis.require(plan['workspace'] == str(WORK) and plan['run_cwd'] == str(WORK)
                     and plan['root_cwd'] == str(ROOT), 'fixed directories')
    analysis.require(plan['model'] == MODEL and all(type(plan['model'][key]) is type(value)
                     for key, value in MODEL.items()), 'explicit retained model identity')
    analysis.require(plan['binary'] == str(BUILD / 'reasoning-budget-probe')
                     and plan['binary_sha256'] == BINARY_SHA, 'reviewed original helper')
    analysis.require(plan['guard'] == str(GUARD) and plan['guard_sha256'] == GUARD_SHA, 'MAIN pressure guard')
    analysis.require(plan['launch_python'] == str(PYTHON) and plan['guard_interpreter_identity']
                     == dict(path=str(FRAMEWORK), sha256=FRAMEWORK_SHA)
                     and plan['launch_python_resolved_identity'] == dict(path=str(PYTHON_TARGET), sha256=PYTHON_SHA),
                     'distinct pinned launcher shim and actual framework interpreter')
    prepared = plan_path.parent
    analysis.require(plan_path.name == 'plan.json' and prepared.parent == WORK / 'speed-bench/ideas/artifacts'
                     and re.fullmatch('[A-Za-z0-9_-]+', prepared.name)
                     and plan['fixture'] == str(prepared / 'fixture.json')
                     and plan['build_manifest'] == str(BUILD / 'build-manifest.json')
                     and plan['cpu_proof'] == str(CPU_PROOF), 'bound preparation/source paths')
    expected = [str(PYTHON), str(GUARD), str(prepared / 'pressure'), '--', str(PYTHON),
                str(prepared / 'run_reasoning_budget.py'), '--run', str(plan_path), '--plan-sha256', PLACEHOLDER,
                '--root-granted']
    analysis.require(plan['guard_command_template'] == expected and expected.count(PLACEHOLDER) == 1,
                     'exact command template and explicit plan authority')
    analysis.require(plan['pressure'] == str(prepared / 'pressure') and plan['output'] == str(prepared / 'run'),
                     'exclusive planned output roots')
    analysis.require(plan['reference_dir'] == str(REFERENCE) and type(plan['reference_returncode']) is int
                     and plan['reference_returncode'] == 3, 'original four-case first native control')
    analysis.require([arm['name'] for arm in plan['arms']] == analysis.NAMES
                     and [arm['budget'] for arm in plan['arms']] == [0, 1024, 0], 'fixed three-arm sequence')
    for arm in plan['arms']:
        analysis.require(set(arm) == {'name', 'budget', 'run_name', 'argv', 'results_dir'}
                         and type(arm['budget']) is int and arm['run_name'] == prepared.name + '-' + arm['name'],
                         'typed budget and fresh safe name')
        analysis.require(arm['argv'] == [plan['binary'], MODEL['path'], arm['run_name'], '25', str(arm['budget'])]
                         and arm['results_dir'] == str(WORK / 'speed-bench/ideas/artifacts' / arm['run_name']),
                         'exact helper arguments and output')
    environment = plan['environment']
    shaders = {key: value for key, value in environment.items() if key.endswith('_SOURCE')}
    analysis.require(set(shaders) == {'DS4_METAL_' + name + '_SOURCE' for name in SHADER_NAMES}
                     and set(environment) == set(ENV) | set(shaders)
                     and all(environment[key] == value for key, value in ENV.items()), 'native profile/environment only')
    analysis.require(all(value == str(prepared / 'metal' / (key[len('DS4_METAL_'):-len('_SOURCE')].lower() + '.metal'))
                     for key, value in shaders.items()), 'frozen shader bindings')
    mandatory = {str(prepared / name) for name in ['run_reasoning_budget.py', 'analyze_reasoning_budget.py',
                                                  'reasoning_budget_signed64.py', 'fixture.json']}
    mandatory |= {plan['binary'], plan['build_manifest'], plan['guard'], str(FRAMEWORK), str(PYTHON), str(PYTHON_TARGET)}
    mandatory |= {str(CPU_PROOF), str(CPU_TEST)}
    mandatory |= set(plan['core_objects']) | set(plan['source_refs']) | set(shaders.values())
    mandatory |= {str(BUILD / name) for name in ['compile.log', 'self-test.log']}
    mandatory |= {str(HOLDOUT / name) for name in ['plan.json', 'analyze_native_effort_holdout.py']}
    mandatory |= {str(CLOSING / name) for name in ['plan.json', 'source/core-proof.json']}
    mandatory |= {str(CLOSING / 'source' / name) for name in ['closing_pilot.c', 'closing_policy.h',
                                                           'overthink_counting_cases.h', 'overthink_diagnostics.h']}
    mandatory.add(str(ROOT / '.workspaces/idea-quant-closing/speed-bench/closing_pilot.c'))
    mandatory |= {str(REFERENCE / name) for name in ['results.jsonl', 'marker-map.json']}
    for case in analysis.GOLDS:
        mandatory |= {str(REFERENCE / (case + '-boost0.0' + suffix)) for suffix in ['.txt', '.tokens.jsonl']}
    analysis.require(mandatory <= set(plan['assets']), 'complete mandatory helper/core/source/reference/shader inventory')
    for path, pin in plan['assets'].items():
        analysis.require(type(path) is str and Path(path).is_absolute() and type(pin) is dict
                         and set(pin) == {'sha256', 'bytes'} and type(pin['sha256']) is str
                         and re.fullmatch('[0-9a-f]{64}', pin['sha256']), 'typed asset identity')
        analysis.integer(pin['bytes'], 'asset bytes', 1 << 40)
    analysis.require(plan['assets'][plan['binary']]['sha256'] == BINARY_SHA
                     and plan['assets'][plan['guard']]['sha256'] == GUARD_SHA
                     and plan['assets'][str(PYTHON)]['sha256'] == PYTHON_SHA
                     and plan['assets'][str(PYTHON_TARGET)]['sha256'] == PYTHON_SHA
                     and plan['assets'][str(FRAMEWORK)]['sha256'] == FRAMEWORK_SHA
                     and plan['assets'][str(prepared / 'reasoning_budget_signed64.py')]['sha256'] == SCORER_SHA,
                     'binary/guard/scorer exact authority')
    analysis.require(plan['core_objects'] == {str(ROOT / '.workspaces/idea-quant-effort' / name): pin
                                            for name, pin in OBJECTS.items()}, 'unchanged eight frozen native objects')
    analysis.require(plan['source_refs'] == {str(prepared / 'source' / name): pin for name, pin in SOURCES.items()},
                     'reviewed helper caller/API/policy/scorer/cases sources')
    analysis.require(all(plan['assets'][path]['sha256'] == value for path, value in plan['core_objects'].items()),
                     'all eight actual object pins')
    analysis.require(all(plan['assets'][path]['sha256'] == value for path, value in plan['source_refs'].items()),
                     'all six reviewed source pins')
    analysis.fixture_check(analysis.load(plan['fixture']))


def require_guard(plan, plan_path, expected_sha):
    analysis.require(analysis.sha(plan['guard']) == GUARD_SHA, 'literal MAIN guard unchanged')
    actual = shlex.split(subprocess.check_output(['/bin/ps', '-p', str(os.getppid()), '-o', 'command='],
                                                 text=True, timeout=2).strip())
    expected = rendered_command(plan, expected_sha)
    analysis.require(bool(actual) and actual[1:] == expected[1:], 'exact direct MAIN guard argv suffix')
    identity = plan['guard_interpreter_identity']
    try:
        same = os.path.samefile(actual[0], identity['path'])
    except OSError:
        same = False
    analysis.require(same and analysis.sha(identity['path']) == identity['sha256'], 'actual pinned framework Python')
    with (ROOT / 'speed-bench/ideas/artifacts/model-job.lock').open('r+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise ValueError('MAIN guard must hold shared model lock')
    return dict(parent_argv=actual, exact_suffix=True, framework_samefile=True, lock_held=True)


def require_grant(plan, plan_path, expected_sha):
    proof = analysis.load(Path(plan_path).parent / 'root-launch-proof.json')
    analysis.require(proof['status'] == 'GRANTED' and proof['sole_model_grant'] is True
                     and proof['manifest_sha256'] == expected_sha
                     and proof['command'] == rendered_command(plan, expected_sha), 'explicit ROOT sole-model grant')
    analysis.require(analysis.number(proof['time'], 'ROOT grant time') > 0, 'ROOT grant timestamp')


def model_identity(plan):
    path = Path(plan['model']['path'])
    analysis.require(path.resolve() == path, 'explicit model path, no alias')
    info = path.stat()
    actual = dict(path=str(path), bytes=info.st_size, mtime_ns=info.st_mtime_ns)
    analysis.require(actual == plan['model'], 'current guarded model identity')
    return actual


def run(plan_path, expected_sha, root_granted=False):
    analysis.require(root_granted is True, 'separate explicit ROOT grant required')
    analysis.require(analysis.sha(plan_path) == expected_sha, 'explicit frozen plan SHA before guard')
    plan = analysis.load(plan_path)
    validate_plan(plan, plan_path)
    parent = require_guard(plan, plan_path, expected_sha)
    require_grant(plan, plan_path, expected_sha)
    check_assets(plan)
    for arm in plan['arms']:
        analysis.require(not Path(arm['results_dir']).exists(), 'all producer outputs must be fresh')
    analysis.require(not Path(plan['output']).exists(), 'run output must be fresh')
    current = model_identity(plan)  # First GGUF stat is after guard, grant and all assets.
    output = Path(plan['output'])
    output.mkdir()
    put(output / 'parent-proof.json', parent)
    put(output / 'model-identity.json', current)
    put(output / 'launch.json', dict(root_granted=True, plan_sha256=expected_sha,
                                   inherited_ds4_removed=[key for key in os.environ if key.startswith('DS4')]))
    env = {key: value for key, value in os.environ.items() if not key.startswith('DS4')}
    env.update(plan['environment'])
    fixture = analysis.load(plan['fixture'])
    completed = []
    validations = []
    try:
        reference = analysis.validate_arm(plan['reference_dir'], 0, fixture, plan['reference_returncode'], historical=True)
        for arm in plan['arms']:
            analysis.require(analysis.sha(plan_path) == expected_sha, 'plan preserved between arms')
            require_guard(plan, plan_path, expected_sha)
            require_grant(plan, plan_path, expected_sha)
            check_assets(plan)
            model_identity(plan)
            analysis.require(not Path(arm['results_dir']).exists(), 'exclusive producer output per arm')
            put(output / (arm['name'] + '-command.json'), dict(argv=arm['argv'], cwd=plan['run_cwd'],
                                                             ds4_environment=plan['environment']))
            with (output / (arm['name'] + '-stdout.log')).open('x') as stdout, \
                    (output / (arm['name'] + '-stderr.log')).open('x') as stderr:
                process = subprocess.Popen(arm['argv'], cwd=plan['run_cwd'], env=env, stdout=stdout, stderr=stderr)
                put(output / (arm['name'] + '-started.json'), dict(pid=process.pid))
                try:
                    rc = process.wait()
                except BaseException:
                    if process.returncode is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                    raise
            put(output / (arm['name'] + '-process.json'), dict(returncode=rc))
            analysis.require(type(rc) is int and rc in (0, 3), 'first producer infrastructure failure retained')
            validated = analysis.validate_arm(arm['results_dir'], arm['budget'], fixture, rc)
            if arm['budget'] == 0:
                analysis.exact_native(reference, validated, historical=True)
                if validations:
                    analysis.exact_native(validations[0], validated)
            else:
                analysis.pre_force(validations[0], validated)
            put(output / (arm['name'] + '-summary.json'), validated[0])
            completed.append(arm['name'])
            validations.append(validated)
            print(json.dumps(dict(arm=arm['name'], **validated[0]), allow_nan=False), flush=True)
        analysis.require(completed == analysis.NAMES, 'all three arms complete')
        check_assets(plan)
        require_guard(plan, plan_path, expected_sha)
        model_identity(plan)
        rc = 3 if any(value[0]['invalid'] for value in validations) else 0
        put(output / 'complete.json', dict(status='COMPLETE', arms=completed, returncode=rc,
            interpretation='First attempts retained; offline final analysis separately requires actual ROOT release proof. No task generalization, KL/NLL or speed claim.'))
        return rc
    except BaseException as error:
        put(output / 'first-failure.json', dict(type=type(error).__name__, message=str(error), completed_arms=completed))
        raise


def prepare(output):
    output = Path(output).resolve()
    analysis.require(output.parent == WORK / 'speed-bench/ideas/artifacts'
                     and re.fullmatch('[A-Za-z0-9_-]+', output.name), 'exclusive persistent preparation path')
    output.mkdir(exist_ok=False)
    # If any check fails, the exclusive directory and refusal remain; never retry over it.
    try:
        files = {}
        for name in ['run_reasoning_budget.py', 'analyze_reasoning_budget.py', 'reasoning_budget_signed64.py']:
            shutil.copy2(Path(__file__).parent / name, output / name)
        build = analysis.load(BUILD / 'build-manifest.json')
        analysis.require(build['binary_sha256'] == BINARY_SHA and build['compile_returncode'] == 0
                         and build['self_test_returncode'] == 0 and not build['core_or_metal_built'], 'reviewed helper build')
        cpu = analysis.load(CPU_PROOF)
        analysis.require(cpu['returncode'] == 0 and cpu['result']['pass_'] is True
                         and cpu['result']['model_opened'] is False
                         and cpu['result']['model_metadata_touched'] is False
                         and cpu['result']['real_helper_executed'] is False, 'actual focused model-free CPU proof')
        for path, pin in cpu['sources'].items():
            analysis.require(analysis.sha(path) == pin, 'CPU-tested source unchanged: ' + path)
        source_refs = {}
        for name, expected in build['source_sha256'].items():
            source = WORK / name
            analysis.require(analysis.sha(source) == expected, 'reviewed helper source: ' + name)
            target = output / 'source' / Path(name).name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
            source_refs[str(target)] = expected
        case_text = (WORK / 'speed-bench/reasoning_budget_cases.h').read_text()
        entries = re.findall(r'\{"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*ANSWER_INTEGER\}', case_text)
        fixture = dict(suite='counting-v1', cases=[dict(case=c, prompt=p, answer=a) for c, p, a in entries])
        analysis.fixture_check(fixture)
        put(output / 'fixture.json', fixture)
        historical = analysis.validate_arm(REFERENCE, 0, fixture, 3, historical=True)
        holdout = analysis.load(HOLDOUT / 'plan.json')
        environment = dict(ENV)
        (output / 'metal').mkdir()
        for key, raw in holdout['environment'].items():
            if key.startswith('DS4_METAL_') and key.endswith('_SOURCE'):
                source = Path(raw)
                target = output / 'metal' / source.name
                shutil.copy2(source, target)
                environment[key] = str(target)
        prefix = output.name
        arms = []
        for name, budget in zip(analysis.NAMES, [0, 1024, 0]):
            run_name = prefix + '-' + name
            arms.append(dict(name=name, budget=budget, run_name=run_name,
                argv=[str(BUILD / 'reasoning-budget-probe'), MODEL['path'], run_name, '25', str(budget)],
                results_dir=str(WORK / 'speed-bench/ideas/artifacts' / run_name)))
        extra = [BUILD / 'reasoning-budget-probe', BUILD / 'build-manifest.json', BUILD / 'compile.log',
                 BUILD / 'self-test.log', GUARD, PYTHON, PYTHON_TARGET, FRAMEWORK, HOLDOUT / 'plan.json',
                 HOLDOUT / 'analyze_native_effort_holdout.py', CLOSING / 'plan.json',
                 CLOSING / 'source/core-proof.json',
                 ROOT / '.workspaces/idea-quant-closing/speed-bench/closing_pilot.c']
        extra += [CPU_PROOF, CPU_TEST]
        extra += [CLOSING / 'source' / name for name in ['closing_pilot.c', 'closing_policy.h',
                                                       'overthink_counting_cases.h', 'overthink_diagnostics.h']]
        extra += [Path(cpu[name]) for name in ['stdout_log', 'first_stdout_log', 'fixture_result']]
        extra += [Path(raw) for raw in build['objects']]
        extra += list(REFERENCE.glob('*'))
        for path in sorted(output.rglob('*')):
            if path.is_file():
                files[str(path)] = asset(path)
        for path in extra:
            analysis.require(path.is_file(), 'mandatory prior asset file')
            files[str(path)] = asset(path)
        analysis.require(files[str(BUILD / 'reasoning-budget-probe')]['sha256'] == BINARY_SHA, 'actual original helper')
        analysis.require(files[str(HOLDOUT / 'analyze_native_effort_holdout.py')]['sha256'] == SCORER_SHA,
                         'signed64 libc-parity checked parser')
        for path, expected in build['objects'].items():
            analysis.require(files[path]['sha256'] == expected, 'unchanged eight native objects')
        template = [str(PYTHON), str(GUARD), str(output / 'pressure'), '--', str(PYTHON),
                    str(output / 'run_reasoning_budget.py'), '--run', str(output / 'plan.json'),
                    '--plan-sha256', PLACEHOLDER, '--root-granted']
        plan = dict(schema='reasoning-budget-pilot-v1', prepared_only=True, model_grant=False,
            workspace=str(WORK), run_cwd=str(WORK), root_cwd=str(ROOT), output=str(output / 'run'),
            pressure=str(output / 'pressure'), binary=str(BUILD / 'reasoning-budget-probe'), binary_sha256=BINARY_SHA,
            build_manifest=str(BUILD / 'build-manifest.json'), core_objects=build['objects'], source_refs=source_refs,
            cpu_proof=str(CPU_PROOF), reviewed_helper_source_revision=build['source_revision'],
            fixture=str(output / 'fixture.json'), reference_dir=str(REFERENCE), reference_returncode=3,
            settings=SETTINGS, environment=environment, arms=arms, assets=files, model=MODEL,
            guard=str(GUARD), guard_sha256=GUARD_SHA, launch_python=str(PYTHON),
            guard_interpreter_identity=dict(path=str(FRAMEWORK), sha256=FRAMEWORK_SHA),
            launch_python_resolved_identity=dict(path=str(PYTHON_TARGET), sha256=PYTHON_SHA),
            guard_command_template=template, original_native_reference_summary=historical[0],
            policy_preregistered='Budget1024 selected before any budget model outcome, original four calibration cases only.',
            counter_scope='API threshold excludes controls; legacy reasoning_tokens includes closing token, so1024 normally records1025 and leaves1023 slots.',
            historical_prompt_control='Old producer exported prompt text/source but no prompt IDs/count. New three arms require exact prompt-ID bytes; historical comparison uses frozen source/text and common token/RNG/raw/result fields.',
            analysis_requires_root_release=True, no_automatic_retry_or_scheduler=True,
            limitations=['Not the six-case holdout. No outcome selection.', 'Protocol-token intervention, not logit penalty.',
                         'No KL/NLL, quantization causality, equal-quality or speed claim.', 'Wrong/capped/INVALID first attempts retained.'])
        put(output / 'plan.json', plan)
        validate_plan(plan, output / 'plan.json')
        check_assets(plan)
        expected_sha = analysis.sha(output / 'plan.json')
        put(output / 'prepared-command.json', dict(command=rendered_command(plan, expected_sha),
                                                  plan_sha256=expected_sha, launch_authorized=False))
        return plan, expected_sha
    except BaseException as error:
        put(output / 'preparation-first-failure.json', dict(type=type(error).__name__, message=str(error),
                                                          model_stat_or_helper_or_build=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare', type=Path)
    choice.add_argument('--run', type=Path)
    choice.add_argument('--verify', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--root-granted', action='store_true')
    args = parser.parse_args()
    try:
        if args.prepare:
            _, pin = prepare(args.prepare)
            print(json.dumps(dict(prepared=str(args.prepare), plan_sha256=pin, model_grant=False)))
        elif args.verify:
            analysis.require(args.plan_sha256 and analysis.sha(args.verify) == args.plan_sha256, 'explicit verification pin')
            plan = analysis.load(args.verify)
            validate_plan(plan, args.verify)
            check_assets(plan)
            print(json.dumps(dict(verified=True, assets=len(plan['assets']), model_or_guard_executed=False)))
        else:
            analysis.require(args.plan_sha256, 'explicit run pin')
            sys.exit(run(args.run, args.plan_sha256, args.root_granted))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, 'reasoning-budget driver: ' + str(error) + '\n')
