#!/usr/bin/env python3
"""Prepare the first actual odd-deletion task; never access a model or run it.

The successful private caller and every copied prerequisite are pinned below.
The copied controller's default preflight is model-free. A future ROOT grant
and the exact MAIN pressure guard remain mandatory for the prepared command.
"""
import hashlib, importlib.util, json, os, shutil
from pathlib import Path

WORK = Path(__file__).resolve().parents[2]
IDEAS = WORK / 'speed-bench/ideas'
BUILD = IDEAS / 'artifacts/agent-suffix-pair-actual-caller-build-v2'
BUILD_ARCHIVE = IDEAS / 'evidence/agent-suffix-pair-actual-caller-build-v2'
CPU_ARCHIVE = IDEAS / 'evidence/agent-suffix-pair-actual-cpu-v3'
PLAN = IDEAS / 'agent_suffix_pair_actual_v1/native-build-plan.json'
OUTPUT = IDEAS / 'artifacts/agent-suffix-pair-actual-prepared-20261003-v1'
BUILD_SHA = 'fbdb61dfcbe6a3ddc55c2f9f1857f18a139f939c0004ea3b173d49238336460e'
BUILD_ARCHIVE_SHA = '08009bb5613a75e420747d1b6dfb64791c7bbd8805dfbd8553235ec16a52e493'
CPU_SHA = 'afc3aab80ac72ba1136f308495913d39b088255f2e6f727c862036c627e0ea08'
PLAN_SHA = 'da9023be6665eb333e42c9719ef4c3df41f5361e497dfe9213e01e6520258f33'
CONTROLLER_SHA = '5bebda8ea489b30b5e17663e7ac106a6cb2e857e26cdb3aac1fc2307fed7d778'


def sha(path):
    path = Path(path)
    if path.suffix.lower() == '.gguf':
        raise ValueError('Model paths are forbidden during preparation')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(path):
    path = Path(path)
    return dict(bytes=path.stat().st_size, sha256=sha(path),
                mode=path.stat().st_mode & 0o777)


def check(path, expected):
    observed = pin(path)
    for key in ('bytes', 'sha256', 'mode'):
        if key in expected and observed[key] != expected[key]:
            raise ValueError('Pinned prerequisite differs: ' + str(path))
    return observed


def load(path):
    return json.loads(Path(path).read_bytes())


def copy(path, name):
    target = OUTPUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    if pin(path) != pin(target):
        raise ValueError('Copied bytes/mode differ: ' + str(name))


def archive_check(path, expected_sha):
    check(path / 'manifest.json', dict(sha256=expected_sha))
    manifest = load(path / 'manifest.json')
    files = manifest['files']
    physical = {str(f.relative_to(path)) for f in path.rglob('*')
                if f.is_file() and f != path / 'manifest.json'}
    if set(files) != physical:
        raise ValueError('Incomplete prerequisite archive: ' + str(path))
    for name, expected in files.items():
        check(path / name, expected)
    return manifest


def prepare():
    # All accesses here are to explicit non-model prerequisites.
    check(PLAN, dict(sha256=PLAN_SHA))
    check(BUILD / 'build-manifest.json', dict(sha256=BUILD_SHA))
    build = load(BUILD / 'build-manifest.json')
    plan = load(PLAN)
    if (build['status'] != 'CALLER_COMPILE_LINK_PASS'
        or build['source_freeze'] != '8eeb54ee8a2bf0d9d15df7d5f5a3c4f65a03df7e'
        or build['plan_sha256'] != PLAN_SHA
        or build['all_pins_unchanged'] is not True
        or build['model_stat_open_or_executable_run'] is not False
        or any(r['returncode'] != 0 or r['warning'] is not False
               for r in build['results'])):
        raise ValueError('Successful reviewed caller build required')
    for name, expected in build['outputs'].items():
        check(BUILD / name, expected)
    if not os.access(BUILD / 'ds4-agent-pair-actual', os.X_OK):
        raise ValueError('Native caller must be executable')
    for group in ('sources', 'shaders'):
        for name, expected in plan[group].items():
            check(WORK / name, expected)
    for expected in plan['objects'].values():
        check(WORK / expected['path'], expected)
        check(Path(expected['source_path']), expected)
    compiler = load(plan['compiler_sdk_proof']['path'])
    check(Path(plan['compiler_sdk_proof']['path']), plan['compiler_sdk_proof'])
    for key in ('selected_wrapper', 'actual_clang', 'sdk_settings', 'libSystem_stub'):
        check(Path(compiler[key]['path']), compiler[key])
    archive_check(BUILD_ARCHIVE, BUILD_ARCHIVE_SHA)
    archive_check(CPU_ARCHIVE, CPU_SHA)
    controller = IDEAS / 'run_agent_suffix_pair_actual.py'
    check(controller, dict(sha256=CONTROLLER_SHA))
    spec = importlib.util.spec_from_file_location('prepared_actual_controller', controller)
    ctl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctl)
    check(ctl.GUARD, dict(sha256=ctl.GUARD_SHA))
    check(ctl.PYTHON, dict(sha256=ctl.PYTHON_SHA))

    OUTPUT.mkdir(exist_ok=False)  # One exclusive first attempt; no overwrite.
    copy(BUILD / 'ds4-agent-pair-actual', 'agent')
    copy(BUILD / 'build-manifest.json', 'caller-build-manifest.json')
    copy(controller, 'run_agent_suffix_pair_actual.py')
    for name in ('task.json', 'grader.py', 'update_context.py', 'run_grader.py',
                 'first.txt', 'second.txt', 'third.txt'):
        copy(IDEAS / 'agent_suffix_pair_actual_v1' / name, name)
    for name in ctl.SHADERS.values():
        if 'metal/' + name not in plan['shaders']:
            raise ValueError('Shader absent from frozen build inventory')
        copy(WORK / 'metal' / name, Path('metal') / name)
    if len(ctl.SHADERS) != 26 or len(set(ctl.SHADERS.values())) != 26:
        raise ValueError('Exactly26 unique explicit shader bindings required')
    for archive, label in ((BUILD_ARCHIVE, 'caller-build'), (CPU_ARCHIVE, 'cpu')):
        for file in sorted(archive.rglob('*')):
            if file.is_file():
                copy(file, Path('provenance') / label / file.relative_to(archive))
    copy(PLAN, 'provenance/native-build-plan.json')
    copy(Path(__file__), 'provenance/prepare_agent_suffix_pair_actual.py')
    files = {str(file.relative_to(OUTPUT)): pin(file)
             for file in sorted(OUTPUT.rglob('*')) if file.is_file()}
    prepared = dict(
        status='PREPARED_NOT_GRANTED_NOT_RUN', first_attempt=True, no_retries=True,
        source_freeze=build['source_freeze'],
        cpu_evidence=build['cpu_evidence'], build_evidence='8614b53d1328d2f5b6d73bb2e0e8742c09ef4d49',
        caller_build_manifest_sha256=BUILD_SHA,
        task_sha256=ctl.TASK_SHA, model=ctl.MODEL, policy=ctl.POLICY,
        turns=3, timeout_seconds=900,
        shader_env={key: str(OUTPUT / 'metal' / name) for key, name in ctl.SHADERS.items()},
        files=files, sole_root_grant_required=True,
        root_launch_proof='Absent until explicitROOT GRANTED/soletrue/exactmanifest+command/finitepositive time',
        scope='One-case approximate semantic compaction; no forgetting, exact-state, global128MiB or speed claim',
        budget='Worker/transaction admission; excludes UI-transferred buffers and global observed peak; actual native admission rechecks forecast',
        primary_persistence='NOT_EXERCISED', model_metadata_accessed=False,
        native_default_off_baseline='Future separate fixedchosenedit plan/grant; no automatic arm or retry',
        failure_contract='Retain all wrong/capped/eligibility/exact-fallback/timeout/tool/grading/infrastructure failures without repair',
    )
    with (OUTPUT / 'manifest.json').open('x') as stream:
        json.dump(prepared, stream, indent=2); stream.write('\n')
    digest = sha(OUTPUT / 'manifest.json')
    with (OUTPUT / 'prepared-command.json').open('x') as stream:
        json.dump(dict(manifest_sha256=digest, command=ctl.command(OUTPUT, digest),
                       cwd=str(ctl.ROOT)), stream, indent=2); stream.write('\n')
    # This verifies only literal non-model bytes and modes. No guard or child.
    ctl.verify(OUTPUT, digest)
    if any((OUTPUT / name).exists() for name in ('root-launch-proof.json', 'run', 'pressure')):
        raise ValueError('Preparation must contain no grant or run outputs')
    with (OUTPUT / 'nonmodel-preflight.json').open('x') as stream:
        json.dump(dict(status='PREFLIGHT_PASS_NOT_GRANTED_NOT_RUN',
                       manifest_sha256=digest,
                       prepared_command_sha256=sha(OUTPUT / 'prepared-command.json'),
                       files=len(files), explicit_shaders=26,
                       all53sources14objects26shaders_compiler_sdk_verified=True,
                       original36tests_unchanged_priorproof_only=True,
                       no_model_metadata_or_engine_or_guard_or_child=True), stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(prepared=str(OUTPUT), manifest_sha256=digest,
                          command_sha256=sha(OUTPUT / 'prepared-command.json'), files=len(files))))


if __name__ == '__main__':
    prepare()
