#!/usr/bin/env python3
"""Prepare/qualify six fixed ragged arms; separate ROOT model grant is mandatory."""
import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import analyze_ragged_native_tail_qualification as analysis

ROOT = Path('/Volumes/Models/ds4')
WORK = ROOT / '.workspaces/idea-ragged-native-tail-qualification'
SOURCE = ROOT / '.workspaces/idea-long-append-tail-ragged'
SPEC = SOURCE / 'speed-bench/ideas/ragged_native_tail_v1'
BUILD = SOURCE / 'speed-bench/ideas/artifacts/ragged-native-tail-build-v2'
BUILD_SHA = '3d77d22dc1d3da7bfab067911eea7542775e87d93d476c3939ec3b0d3fc22e75'
SPEC_SHA = '84362b3e15bbbe99e9593aee382a4169cf6974a4d180c520a5dffd105e334f97'
REPLAY_SHA = '059d281d116ac8a8df5979443565e1eed95370dbfa232917de6c396073218613'
TARGET_SHA = '27c86007454415311764368cdf0acecaad68bf47fa57797997584082de47cdd1'
ORIGINAL_SHA = 'a859a10f9d74987789aa982a6d061db3991be11cec884e277fc5da4ce9f92adb'
VALIDATOR_SHA = '44c9b424b48bf400c19e6da9d0a12ed1a1b9fb3a67eb57a30d85ddfe07b563fd'
BINARIES = {'old': str(BUILD / 'ragged-old'), 'new': str(BUILD / 'ragged-new')}
BINARY_SHAS = {'old': 'fe50bd40e61f9bda1d84f78387b2ce14921f61b5811f63861eebd4ae886cff6c',
               'new': 'b2cafdd13c5d505be1ded878ec75afd290d1c3551af815a694574de79e5e89ed'}
GUARD = ROOT / 'speed-bench/ideas/pressure_run.py'
GUARD_SHA = '461e8c517e82a687b4579fc450c1e7128b1510192d088bab719682b33c946c3c'
PYTHON = Path('/opt/homebrew/opt/python@3.14/bin/python3.14')
PYTHON_TARGET = Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/bin/python3.14')
PYTHON_SHA = '7ebde552773a4045532637b114d645a313e045ac07b26f5e520bf039f1e34063'
FRAMEWORK = Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python')
FRAMEWORK_SHA = 'b80d260a48404da13a32dd44d7283aadc80f6a6e56ed6e03ed5597ec6f46d48d'
MODEL = dict(path=str(ROOT / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf'), bytes=365713686528, mtime_ns=1789237692837653163)
FLAG = 'DS4_METAL_V41_LONG_APPEND_NATIVE_TAIL'
PLACEHOLDER = '<PLAN_SHA256>'
SHADERS = ('argsort bin concat cpy deepseek4_vision dense dsv41 dsv4_hc dsv4_kv dsv4_misc dsv4_rope flash_attn get_rows glm53_bf16 glm53_kda glm53_vision glu moe norm qwen4 qwen4_vision repeat set_rows softmax sum_rows unary').split()
SCRIPTS = ['run_ragged_native_tail_qualification.py', 'analyze_ragged_native_tail_qualification.py', 'ragged_native_tail_validation.py']
TEST = WORK / 'tests/test_ragged_native_tail_qualification.py'
CPU_ROOTS = ['speed-bench/ideas/evidence/ragged-native-tail-source-cpu-20261003-v1',
             'speed-bench/ideas/evidence/ragged-native-tail-header-correction-20261003-v2']


def put(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def asset(path):
    path = Path(path)
    analysis.require(path.suffix.lower() != '.gguf', 'only static non-model assets')
    if path == PYTHON:
        analysis.require(path.is_symlink() and path.resolve() == PYTHON_TARGET and
                         os.readlink(path) == '../Frameworks/Python.framework/Versions/3.14/bin/python3.14', 'only pinned Homebrew launcher alias')
    else:
        analysis.require(not path.is_symlink(), 'no implicit asset alias')
    return dict(sha256=analysis.sha(path), bytes=path.stat().st_size, mode=path.stat().st_mode & 0o777)


def command(plan, digest):
    return [digest if value == PLACEHOLDER else value for value in plan['guard_command_template']]


def build_check():
    analysis.require(analysis.sha(BUILD / 'build-manifest.json') == BUILD_SHA, 'qualified actual ROOT build')
    build = analysis.load(BUILD / 'build-manifest.json')
    plan_path = SOURCE / 'speed-bench/ideas/ragged_native_tail_v2/build-plan.json'
    prior = analysis.load(plan_path)
    analysis.require(build['status'] == 'ROOT_PRIVATE_RAGGED_FRONTEND_COMPILE_LINK_PASS' and
        build['plan_sha256'] == analysis.sha(plan_path) and build['all_pins_unchanged'] is True and
        build['serial_compilers'] == 1 and type(build['serial_compilers']) is int and
        build['native_core_Metal_built'] is False and build['model_stat_open_or_executable_run'] is False, 'actual frontend-only build scope')
    analysis.require(len(build['results']) == 3, 'exact three build commands')
    for result, expected in zip(build['results'], prior['commands']):
        analysis.require(result['label'] == expected['label'] and result['argv'] == expected['argv'] and
            result['cwd'] == expected['cwd'] and type(result['returncode']) is int and result['returncode'] == 0 and
            result['warning'] is False and analysis.number(result['elapsed'], 'build diagnostic time') >= 0, 'all recorded serial compile/link commands')
    for path, expected in prior['files'].items():
        analysis.require(analysis.sha(path) == expected['sha256'] and Path(path).stat().st_size == expected['bytes'] and
                         oct(Path(path).stat().st_mode & 0o777) == expected['mode'], 'actual original build source/input pin')
    for path, expected in prior['objects'].items():
        analysis.require(analysis.sha(path) == expected['sha256'] and Path(path).stat().st_size == expected['bytes'], 'actual linked object pin')
    for expected in prior['build_authorities']:
        analysis.require(analysis.sha(expected['path']) == expected['sha256'], 'actual original runtime build authority')
    for recorded in build['outputs'].values():
        analysis.require(analysis.sha(recorded['path']) == recorded['sha256'] and Path(recorded['path']).stat().st_size == recorded['bytes'] and
                         oct(Path(recorded['path']).stat().st_mode & 0o777) == recorded['mode'], 'all actual frontend/link outputs')
    analysis.require(analysis.sha(BUILD / 'root-build-grant.json') == build['root_grant_sha256'], 'actual ROOT build grant identity')
    for variant, path in BINARIES.items():
        recorded = build['outputs']['ragged-' + variant]
        analysis.require(recorded['path'] == path and recorded['sha256'] == BINARY_SHAS[variant] and
            recorded['bytes'] == Path(path).stat().st_size and int(recorded['mode'], 8) == Path(path).stat().st_mode & 0o777 and
            analysis.sha(path) == BINARY_SHAS[variant] and os.access(path, os.X_OK), 'built executable identity/mode')
    return build, prior


def authority_paths():
    """Explicit bounded inventory; no artifact glob or model metadata lookup."""
    build, prior = build_check()
    paths = {BUILD / n for n in ['build-manifest.json', 'root-build-grant.json', 'ragged-frontend.o', 'ragged-old', 'ragged-new',
        'frontend-compile.stdout', 'frontend-compile.stderr', 'old-link.stdout', 'old-link.stderr', 'new-link.stdout', 'new-link.stderr']}
    paths |= {Path(x) for x in prior['files']} | {Path(x) for x in prior['objects']}
    paths |= {Path(v['path']) for v in prior['build_authorities']}
    paths |= {SOURCE / 'speed-bench/ideas/ragged_native_tail_v2' / n for n in ['build-plan.json', 'header-identity-proof.json', 'source-before-cpu.json', 'check_build_headers.py']}
    paths |= {SPEC / n for n in ['qualification-spec.json', 'original-growth.txt', 'ragged-growth.txt', 'branch-targets.txt', 'validation.py', 'quiet-proposal.json']}
    for base in CPU_ROOTS:
        paths |= {SOURCE / base / n for n in ('manifest.json', 'proof.json', 'README.md')}
    paths |= {SOURCE / 'speed-bench/ideas/evidence/ragged-native-tail-source-cpu-20261003-v1/original-no-tail-evidence-refs.json'}
    paths |= {SOURCE / 'speed-bench/ideas/ragged_native_tail_v2/refused-build-v1' / n for n in
              ['first-build-refusal.json', 'frontend-compile.stdout', 'frontend-compile.stderr', 'root-build-grant.json']}
    return paths


def shader_pins():
    # The pre-existing native runtime build already pinned all26 shader bytes.
    _, prior = build_check()
    native = analysis.load(prior['build_authorities'][1]['path'])
    values = {}
    def walk(value):
        if type(value) is dict:
            if type(value.get('path')) is str and value['path'].endswith('.metal') and type(value.get('sha256')) is str:
                name = Path(value['path']).name
                analysis.require(name not in values or values[name] == value['sha256'], 'consistent runtime shader pins')
                values[name] = value['sha256']
            for key, item in value.items():
                if key.endswith('.metal') and type(item) is str and re.fullmatch('[0-9a-f]{64}', item):
                    name = Path(key).name
                    analysis.require(name not in values or values[name] == item, 'consistent runtime shader pins')
                    values[name] = item
                walk(item)
        elif type(value) is list:
            for item in value:
                walk(item)
    walk(native)
    analysis.require(set(values) == {name + '.metal' for name in SHADERS}, 'native runtime all26 original shader authority')
    return values


def fixture_check(plan):
    analysis.require(analysis.sha(plan['spec']) == SPEC_SHA and analysis.sha(plan['replay']) == REPLAY_SHA and
        analysis.sha(plan['targets']) == TARGET_SHA and analysis.sha(plan['original']) == ORIGINAL_SHA, 'frozen original/spec/derived teacher authority')
    replay, targets, ids = analysis.contract.derive(Path(plan['original']).read_text())
    analysis.require(replay == Path(plan['replay']).read_text() and targets == Path(plan['targets']).read_text(), 'literal fixed extensions/64 original targets')
    analysis.contract.ragged(replay)
    return ids


def validate_plan(plan, plan_path):
    plan_path = Path(plan_path).resolve()
    parent = plan_path.parent
    analysis.require(parent.parent == WORK / 'speed-bench/ideas/artifacts' and plan_path.name == 'plan.json' and
        re.fullmatch('[A-Za-z0-9_-]+', parent.name), 'exclusive persistent plan path')
    analysis.require(plan['schema'] == 'ragged-native-tail-six-qualifier-v1' and plan['prepared_only'] is True and
        plan['model_grant'] is False and plan['quiet_grant'] is False and plan['no_automatic_retry'] is True, 'preparation-only scope')
    analysis.require(plan['root_cwd'] == str(ROOT) and plan['run_cwd'] == str(WORK) and plan['output'] == str(parent / 'run') and
        plan['pressure'] == str(parent / 'pressure'), 'exact cwd/output authority')
    analysis.require(plan['model'] == MODEL and all(type(plan['model'][k]) is type(v) for k, v in MODEL.items()), 'retained model identity without new stat')
    analysis.require(plan['binaries'] == BINARIES and plan['binary_shas'] == BINARY_SHAS and plan['build_manifest_sha256'] == BUILD_SHA,
        'unchanged qualified binaries')
    analysis.require(plan['spec'] == str(parent / 'qualification-spec.json') and plan['replay'] == str(parent / 'ragged-growth.txt') and
        plan['targets'] == str(parent / 'branch-targets.txt') and plan['original'] == str(parent / 'original-growth.txt'), 'fixed copied inputs')
    analysis.require(plan['guard'] == str(GUARD) and plan['guard_sha256'] == GUARD_SHA and
        plan['framework_identity'] == dict(path=str(FRAMEWORK), sha256=FRAMEWORK_SHA), 'literal MAIN guard/interpreter')
    expected = [str(PYTHON), str(GUARD), str(parent / 'pressure'), '--', str(PYTHON), str(parent / SCRIPTS[0]),
                '--run', str(plan_path), '--plan-sha256', PLACEHOLDER, '--root-granted']
    analysis.require(plan['guard_command_template'] == expected, 'exact frozen command authority')
    spec = analysis.load(plan['spec'])
    analysis.require([a['name'] for a in spec['arms']] == analysis.NAMES and len(plan['arms']) == 6, 'fixed six-arm order')
    for arm, frozen in zip(plan['arms'], spec['arms']):
        name = frozen['name']
        environment = {'DS4_METAL_STREAMING_EXPERT_IO_METRICS': '1', 'DS4_METAL_V41_RESIDENT_ROUTE_GAP': '0',
            'DS4_METAL_V41_RESIDENT_ROUTE_MODE': 'native', FLAG: str(frozen['flag'])}
        environment.update({'DS4_METAL_' + n.upper() + '_SOURCE': str(parent / 'metal' / (n + '.metal')) for n in SHADERS})
        if frozen['policy'] == 'fast':
            environment.update(DS4_METAL_V41_SELECTED_TILE='64', DS4_METAL_V41_SELECTED_LIMIT='1024')
        analysis.require(set(arm) == {'name', 'binary_variant', 'flag', 'policy', 'exact_reference', 'argv', 'output', 'environment'} and
            arm['name'] == name and arm['binary_variant'] == frozen['binary'] and type(arm['flag']) is int and arm['flag'] == frozen['flag'] and
            arm['policy'] == frozen['policy'] and arm['exact_reference'] == frozen['exact_reference'] and
            arm['output'] == str(parent / 'run' / name) and arm['argv'] == [BINARIES[frozen['binary']], MODEL['path'],
                str(parent / 'ragged-growth.txt'), str(parent / 'branch-targets.txt'), str(parent / 'run' / name)] and
            arm['environment'] == environment, 'fixed literal arm/policy/native routing')
    mandatory = authority_paths() | {GUARD, PYTHON, PYTHON_TARGET, FRAMEWORK}
    mandatory |= {parent / n for n in SCRIPTS + ['runner-tests.py', 'cpu-proof.json', 'cpu-tests.log', 'qualification-spec.json',
                                               'ragged-growth.txt', 'branch-targets.txt', 'original-growth.txt']}
    mandatory |= {parent / 'metal' / (n + '.metal') for n in SHADERS}
    analysis.require(set(plan['assets']) == {str(x) for x in mandatory}, 'complete exact mandatory static inventory')
    for path, pin in plan['assets'].items():
        analysis.require(type(path) is str and Path(path).is_absolute() and type(pin) is dict and
            set(pin) == {'sha256', 'bytes', 'mode'} and type(pin['sha256']) is str and re.fullmatch('[0-9a-f]{64}', pin['sha256']), 'typed asset')
        analysis.integer(pin['bytes'], 'asset bytes', 1 << 40)
        analysis.integer(pin['mode'], 'asset mode', 0o777)
    fixed = {str(GUARD): GUARD_SHA, str(PYTHON): PYTHON_SHA, str(PYTHON_TARGET): PYTHON_SHA, str(FRAMEWORK): FRAMEWORK_SHA,
        str(BUILD / 'build-manifest.json'): BUILD_SHA, str(parent / 'qualification-spec.json'): SPEC_SHA,
        str(parent / 'ragged-growth.txt'): REPLAY_SHA, str(parent / 'branch-targets.txt'): TARGET_SHA,
        str(parent / 'original-growth.txt'): ORIGINAL_SHA, str(parent / SCRIPTS[2]): VALIDATOR_SHA,
        **{BINARIES[k]: v for k, v in BINARY_SHAS.items()}}
    fixed.update({str(parent / 'metal' / name): digest for name, digest in shader_pins().items()})
    analysis.require(all(plan['assets'][path]['sha256'] == digest for path, digest in fixed.items()), 'independent static authority pins')
    fixture_check(plan)


def check_assets(plan):
    for path, expected in plan['assets'].items():
        analysis.require(asset(path) == expected, 'frozen static asset drift: ' + path)
    build_check()
    fixture_check(plan)


def require_guard(plan, digest):
    analysis.require(analysis.sha(GUARD) == GUARD_SHA, 'MAIN guard hash')
    actual = shlex.split(subprocess.check_output(['/bin/ps', '-p', str(os.getppid()), '-o', 'command='], text=True, timeout=2).strip())
    expected = command(plan, digest)
    analysis.require(actual and actual[1:] == expected[1:], 'exact direct guard argv suffix')
    try:
        same = os.path.samefile(actual[0], FRAMEWORK)
    except OSError:
        same = False
    analysis.require(same and analysis.sha(FRAMEWORK) == FRAMEWORK_SHA, 'pinned actual framework interpreter')
    # The real MAIN guard inherits ROOT's launch cwd; its child inherits that
    # same cwd until Popen explicitly chooses WORK for the native helper.
    analysis.require(Path.cwd().resolve() == ROOT, 'actual guard-child inherited MAIN launch cwd')
    with (ROOT / 'speed-bench/ideas/artifacts/model-job.lock').open('r+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise ValueError('MAIN guard must hold model lock')
    return dict(parent_argv=actual, framework_samefile=True, lock_held=True, inherited_cwd=str(Path.cwd()))


def require_grant(plan, plan_path, digest):
    proof = analysis.load(Path(plan_path).parent / 'root-launch-proof.json')
    analysis.require(proof['status'] == 'GRANTED' and proof['sole_model_grant'] is True and
        proof['manifest_sha256'] == digest and proof['command'] == command(plan, digest) and
        proof['cwd'] == str(ROOT) and analysis.number(proof['time'], 'ROOT grant time') > 0, 'explicit separate ROOT grant')


def model_identity(plan):
    path = Path(plan['model']['path'])
    analysis.require(path.resolve() == path, 'explicit model path no alias')
    info = path.stat()
    identity = dict(path=str(path), bytes=info.st_size, mtime_ns=info.st_mtime_ns)
    analysis.require(identity == plan['model'], 'guarded model metadata identity')
    return identity


def run(plan_path, digest, root_granted=False):
    analysis.require(root_granted is True and analysis.sha(plan_path) == digest, 'explicit grant flag/frozen plan')
    plan = analysis.load(plan_path)
    validate_plan(plan, plan_path)
    parent = require_guard(plan, digest)
    require_grant(plan, plan_path, digest)
    check_assets(plan)
    analysis.require(not Path(plan['output']).exists() and not Path(plan['pressure']).is_symlink() and
        all(not Path(a['output']).exists() for a in plan['arms']), 'exclusive first-run outputs')
    identity = model_identity(plan)  # FIRST model resolution/stat, after all authority/lock/asset gates.
    out = Path(plan['output'])
    out.mkdir()
    put(out / 'parent-proof.json', parent)
    put(out / 'model-identity.json', identity)
    put(out / 'launch.json', dict(plan_sha256=digest, inherited_DS4_removed=[k for k in os.environ if k.startswith('DS4')]))
    targets = fixture_check(plan)
    done = []
    validated = {}
    try:
        for arm in plan['arms']:
            analysis.require(analysis.sha(plan_path) == digest, 'plan unchanged before each arm')
            require_guard(plan, digest)
            require_grant(plan, plan_path, digest)
            check_assets(plan)
            model_identity(plan)
            analysis.require(not Path(arm['output']).exists(), 'fresh native output')
            environment = {k: v for k, v in os.environ.items() if not k.startswith('DS4')}
            environment.update(arm['environment'])
            put(out / (arm['name'] + '-command.json'), dict(argv=arm['argv'], cwd=plan['run_cwd'], ds4_environment=arm['environment']))
            with (out / (arm['name'] + '-stdout.log')).open('x') as stdout, (out / (arm['name'] + '-stderr.log')).open('x') as stderr:
                process = subprocess.Popen(arm['argv'], cwd=plan['run_cwd'], env=environment, stdout=stdout, stderr=stderr)
                put(out / (arm['name'] + '-started.json'), dict(pid=process.pid))
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
            put(out / (arm['name'] + '-process.json'), dict(returncode=rc))
            report = analysis.validate_arm(arm['output'], out / (arm['name'] + '-stderr.log'), Path(plan['replay']).read_text(), targets, rc)
            if arm['exact_reference']:
                analysis.exact_control(validated[arm['exact_reference']], report)
            if arm['name'] == 'new1-fast':
                analysis.candidate_gate(validated['new0-fast'], report, out / 'new0-fast', out / 'new1-fast')
            put(out / (arm['name'] + '-checked.json'), report)
            validated[arm['name']] = report
            done.append(arm['name'])
            print(json.dumps(dict(arm=arm['name'], checked=True), allow_nan=False), flush=True)
        check_assets(plan)
        require_guard(plan, digest)
        require_grant(plan, plan_path, digest)
        model_identity(plan)
        put(out / 'complete.json', dict(status='COMPLETE', arms=done, returncode=0, quiet_grant=False,
            interpretation='Offline final diagnostic analysis requires separately pinned actual ROOT release; no automatic timing.'))
        return 0
    except BaseException as error:
        put(out / 'first-failure.json', dict(type=type(error).__name__, message=str(error), completed_arms=done))
        raise


def prepare(output, cpu_proof):
    output = Path(output).resolve()
    analysis.require(output.parent == WORK / 'speed-bench/ideas/artifacts' and re.fullmatch('[A-Za-z0-9_-]+', output.name), 'exclusive persistent preparation')
    cpu = analysis.load(cpu_proof)
    analysis.require(type(cpu['returncode']) is int and cpu['returncode'] == 0 and cpu['native_build_helper_guard_model_metadata_executed'] is False, 'source-qualified model-free CPU proof')
    expected_sources = {str(Path(__file__).parent / n) for n in SCRIPTS} | {str(TEST)}
    analysis.require(set(cpu['sources']) == expected_sources and all(analysis.sha(path) == digest for path, digest in cpu['sources'].items()), 'all tested source bytes unchanged')
    build_check()
    output.mkdir(exist_ok=False)
    try:
        for name in SCRIPTS:
            shutil.copy2(Path(__file__).parent / name, output / name)
        shutil.copy2(TEST, output / 'runner-tests.py')
        shutil.copy2(cpu_proof, output / 'cpu-proof.json')
        shutil.copy2(Path(cpu_proof).parent / 'tests.log', output / 'cpu-tests.log')
        for name in ('qualification-spec.json', 'ragged-growth.txt', 'branch-targets.txt', 'original-growth.txt'):
            shutil.copy2(SPEC / name, output / name)
        (output / 'metal').mkdir()
        original_shaders = shader_pins()
        for name in SHADERS:
            analysis.require(analysis.sha(SOURCE / 'metal' / (name + '.metal')) == original_shaders[name + '.metal'], 'original shader before copy')
            shutil.copy2(SOURCE / 'metal' / (name + '.metal'), output / 'metal' / (name + '.metal'))
        spec = analysis.load(output / 'qualification-spec.json')
        arms = []
        for frozen in spec['arms']:
            environment = {'DS4_METAL_STREAMING_EXPERT_IO_METRICS': '1', 'DS4_METAL_V41_RESIDENT_ROUTE_GAP': '0',
                'DS4_METAL_V41_RESIDENT_ROUTE_MODE': 'native', FLAG: str(frozen['flag'])}
            environment.update({'DS4_METAL_' + n.upper() + '_SOURCE': str(output / 'metal' / (n + '.metal')) for n in SHADERS})
            if frozen['policy'] == 'fast':
                environment.update(DS4_METAL_V41_SELECTED_TILE='64', DS4_METAL_V41_SELECTED_LIMIT='1024')
            arms.append(dict(name=frozen['name'], binary_variant=frozen['binary'], flag=frozen['flag'], policy=frozen['policy'],
                exact_reference=frozen['exact_reference'], output=str(output / 'run' / frozen['name']),
                argv=[BINARIES[frozen['binary']], MODEL['path'], str(output / 'ragged-growth.txt'), str(output / 'branch-targets.txt'),
                      str(output / 'run' / frozen['name'])], environment=environment))
        paths = authority_paths() | {GUARD, PYTHON, PYTHON_TARGET, FRAMEWORK}
        paths |= {output / n for n in SCRIPTS + ['runner-tests.py', 'cpu-proof.json', 'cpu-tests.log', 'qualification-spec.json',
                                               'ragged-growth.txt', 'branch-targets.txt', 'original-growth.txt']}
        paths |= {output / 'metal' / (n + '.metal') for n in SHADERS}
        plan = dict(schema='ragged-native-tail-six-qualifier-v1', prepared_only=True, model_grant=False, quiet_grant=False,
            no_automatic_retry=True, root_cwd=str(ROOT), run_cwd=str(WORK), output=str(output / 'run'), pressure=str(output / 'pressure'),
            model=MODEL, binaries=BINARIES, binary_shas=BINARY_SHAS, build_manifest_sha256=BUILD_SHA,
            spec=str(output / 'qualification-spec.json'), replay=str(output / 'ragged-growth.txt'), targets=str(output / 'branch-targets.txt'),
            original=str(output / 'original-growth.txt'), guard=str(GUARD), guard_sha256=GUARD_SHA,
            framework_identity=dict(path=str(FRAMEWORK), sha256=FRAMEWORK_SHA), arms=arms, assets={str(p): asset(p) for p in sorted(paths)},
            guard_command_template=[str(PYTHON), str(GUARD), str(output / 'pressure'), '--', str(PYTHON), str(output / SCRIPTS[0]),
                '--run', str(output / 'plan.json'), '--plan-sha256', PLACEHOLDER, '--root-granted'],
            limitations=['Same frozen nine phases/64 original teacher predictions on synthetic ID extensions.',
                'Observed production progress chunks and IO profiling make times diagnostic; no quiet timing is launched.',
                'No model metadata/tokenizer/helper/compiler/real guard accessed during preparation; old no-tail study is separate.'])
        put(output / 'plan.json', plan)
        validate_plan(plan, output / 'plan.json')
        check_assets(plan)
        digest = analysis.sha(output / 'plan.json')
        put(output / 'prepared-command.json', dict(command=command(plan, digest), cwd=str(ROOT), plan_sha256=digest, launch_authorized=False))
        return plan, digest
    except BaseException as error:
        put(output / 'preparation-first-refusal.json', dict(type=type(error).__name__, message=str(error), native_build_or_model_or_guard=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'verify', 'run'):
        mode.add_argument('--' + name)
    parser.add_argument('--cpu-proof')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--root-granted', action='store_true')
    args = parser.parse_args()
    try:
        if args.prepare:
            analysis.require(args.cpu_proof, 'explicit CPU proof')
            _, digest = prepare(args.prepare, args.cpu_proof)
            print(json.dumps(dict(prepared=args.prepare, plan_sha256=digest, model_grant=False)))
        elif args.verify:
            analysis.require(args.plan_sha256 and analysis.sha(args.verify) == args.plan_sha256, 'explicit plan pin')
            plan = analysis.load(args.verify)
            validate_plan(plan, args.verify)
            check_assets(plan)
            print(json.dumps(dict(verified=True, assets=len(plan['assets']), no_model_stat_or_guard=True)))
        else:
            analysis.require(args.plan_sha256, 'explicit run plan pin')
            raise SystemExit(run(args.run, args.plan_sha256, args.root_granted))
    except (ValueError, OSError, KeyError, TypeError, analysis.contract.Refusal) as error:
        parser.exit(1, 'ragged qualifier: ' + str(error) + '\n')
