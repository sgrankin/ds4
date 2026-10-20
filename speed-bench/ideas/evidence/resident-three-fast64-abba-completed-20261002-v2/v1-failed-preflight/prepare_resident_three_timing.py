#!/usr/bin/env python3
"""Prepare only by default; model execution requires an explicit frozen MAIN guard."""
import argparse
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path('/Volumes/Models/ds4')
GUARD = ROOT / 'speed-bench/ideas/pressure_run.py'
WORKSPACE = Path(__file__).resolve().parents[2]
BUILD = WORKSPACE / 'speed-bench/ideas/artifacts/three-timing-build'
THREE_BUILD = ROOT / '.workspaces/idea-resident-three/speed-bench/ideas/artifacts/three-build'
PRIOR = ROOT / 'speed-bench/ideas/evidence/native-prefill64-tile64-limit1024-abba-completed-20261002-v1'
COMMON = ROOT / 'speed-bench/agent-session/resident_replay_abba.py'
SESSION = ROOT / 'speed-bench/ideas/latency-session.txt'
RUNTIME = 'b9408c3b8e3ec5b22ad7efff3e365fd81818c317'
SESSION_SHA = '31e3e00a9f3e17ae16692d2f998f0a2fc63c501552ea01b3de47e11d2a35e928'
NATIVE_LOGITS = 'f91a39503f60233d633b08586229fb1b2e18958e60632efaec9d742ab977e652'
NATIVE_STATE = 'f71f06ea4d72e4329eb7c67a4ba462b284d21352756f8ec50022ec8c9276f2b0'
ROUTE = 'DS4_METAL_V41_RESIDENT_ROUTE_'
COMMON_ENV = {'DS4_REPLAY_CACHE_GB': '64', 'DS4_REPLAY_CONTEXT': '8192',
    'DS4_REPLAY_PREFILL_CHUNK': '512', 'DS4_REPLAY_CACHE_REGIME': '1',
    'DS4_METAL_V41_SELECTED_TILE': '64', 'DS4_METAL_V41_SELECTED_LIMIT': '1024'}
CANDIDATE = {ROUTE + key: value for key, value in dict(GAP='0.30', MAX_MASS='0.20',
    MAX_TOTAL_MASS='0.40', MAX_RANK='48', MAX_SUBS='3', MODE='apply', POLICY='replace').items()}


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_common(path):
    spec = importlib.util.spec_from_file_location('plain_abba_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_assets(output, manifest):
    if digest(output / 'manifest.json') != manifest['_manifest_sha256']:
        raise ValueError('frozen plan changed')
    for name, expected in manifest['files'].items():
        if digest(output / name) != expected:
            raise ValueError('frozen asset changed: ' + name)
    if digest(GUARD) != manifest['guard_sha256']:
        raise ValueError('MAIN pressure guard changed')


def require_main_guard(output, manifest):
    if digest(GUARD) != manifest['guard_sha256']:
        raise ValueError('MAIN pressure guard changed')
    parent = subprocess.check_output(['ps', '-p', str(os.getppid()), '-o', 'command='], text=True)
    if shlex.split(parent.strip()) != manifest['guard_command']:
        raise ValueError('--run requires its exact prepared MAIN guard as direct parent')
    with (ROOT / 'speed-bench/ideas/artifacts/model-job.lock').open('r+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise ValueError('MAIN guard does not hold model-job lock')


def phase_identity(rows):
    return [(r['phase'], r['kind'], r['context'], r['tokens']) for r in rows]


def run_environment(output, variant):
    env = {key: value for key, value in os.environ.items() if not key.startswith('DS4_')}
    env.update(COMMON_ENV)
    env[ROUTE + 'GAP'] = '0'
    if variant == 'B':
        env.update(CANDIDATE)
    for path in sorted((output / 'metal').glob('*.metal')):
        env['DS4_METAL_' + path.stem.upper() + '_SOURCE'] = str(path)
    return env


def audit_result(row, manifest, results):
    if phase_identity(row['phases']) != [tuple(r) for r in manifest['phase_identity']]:
        raise ValueError('replay phase boundaries differ')
    if (row['logits_bytes'], row['snapshot_bytes']) != tuple(manifest['output_bytes']):
        raise ValueError('seven end-vector or final-state size differs')
    cache = row['profiles']['cache_phase']
    if len(cache) != len(row['phases']) or any(r.get('phase') != i or
            r.get('configured_slots') != 6136 for i, r in enumerate(cache)):
        raise ValueError('configured cache slots must remain6136 in every phase')
    if len(row['profiles']['cache']) != 1 or row['profiles']['cache'][0].get('budget') != 6136:
        raise ValueError('final configured cache budget differs')
    if any(row['profiles'][key] for key in ('route_phase', 'io_phase', 'timing')):
        raise ValueError('diagnostic route/IO/timing profiles emitted')
    route = row['profiles']['resident_route']
    if row['variant'] == 'A' and route:
        raise ValueError('native GAP0 emitted resident-route footer')
    if row['variant'] == 'B' and (len(route) != 1 or route[0].get('layers', 0) <= 0 or
            route[0].get('substitutions', 0) <= 0 or 'triple_layers' not in route[0]):
        raise ValueError('candidate resident routing was not observed')
    if row['variant'] == 'A' and (row['logits_sha256'], row['snapshot_sha256']) != (
            manifest['native_control']['logits_sha256'], manifest['native_control']['snapshot_sha256']):
        raise ValueError('native-fast64 differs from prior seven vectors or final state')
    repeats = [r for r in results if r['variant'] == row['variant']]
    if repeats and any(row[k] != repeats[0][k] for k in ('logits_sha256', 'snapshot_sha256')):
        raise ValueError('same-arm repeated vectors or graph state differ')


def prepare(output):
    proof = json.loads((BUILD / 'manifest.json').read_text())
    if proof['runtime_commit'] != RUNTIME or proof['core_or_metal_rebuilt']:
        raise ValueError('reviewed b9408c3b object provenance differs')
    for name, expected in proof['files'].items():
        if digest(BUILD / name) != expected:
            raise ValueError('plain build asset changed: ' + name)
    previous = json.loads((PRIOR / 'manifest.json').read_text())
    rows = json.loads((PRIOR / 'summary.json').read_text())
    native = [r for r in rows if r['variant'] == 'B']
    if len(native) != 2 or any((r['logits_sha256'], r['snapshot_sha256']) !=
            (NATIVE_LOGITS, NATIVE_STATE) for r in native):
        raise ValueError('prior native-fast64 repeated hash records differ')
    if digest(SESSION) != SESSION_SHA or previous['replay_sha256'] != SESSION_SHA:
        raise ValueError('frozen plain replay differs')
    if (previous['cache_gb'], previous['context'], previous['prefill_chunk']) != (64, 8192, 512):
        raise ValueError('prior replay settings differ')
    if len(native[0]['phases']) != 7 or phase_identity(native[0]['phases']) != phase_identity(native[1]['phases']):
        raise ValueError('prior seven phase boundaries differ')
    output.mkdir(parents=True, exist_ok=False)
    copies = {'replay-three-plain': BUILD / 'replay-three-plain', 'build-manifest.json': BUILD / 'manifest.json',
        'reviewed-three-build-manifest.json': BUILD / 'reviewed-three-build-manifest.json',
        'session.txt': SESSION, 'abba_helpers.py': COMMON, 'prepare_resident_three_timing.py': Path(__file__),
        'prior-manifest.json': PRIOR / 'manifest.json', 'prior-summary.json': PRIOR / 'summary.json',
        'prior-audit.json': PRIOR / 'audit.json', 'replay.c': BUILD / 'source/replay.c'}
    for name, source in copies.items():
        shutil.copy2(source, output / name)
    (output / 'replay-three-plain').chmod(0o555)
    (output / 'metal').mkdir()
    reviewed = json.loads((THREE_BUILD / 'manifest.json').read_text())
    for name, expected in reviewed['metal'].items():
        source = THREE_BUILD / 'metal' / name
        key = 'DS4_METAL_' + Path(name).stem.upper() + '_SOURCE'
        if digest(source) != expected or previous['metal'][key]['sha256'] != expected:
            raise ValueError('three-build/prior shader differs: ' + name)
        shutil.copy2(source, output / 'metal' / name)
    command = [sys.executable, str(GUARD), str(output / 'pressure'), '--', sys.executable,
        str(output / 'prepare_resident_three_timing.py'), '--run', str(output),
        '--root-granted', '--cpu-timing-frozen']
    manifest = dict(runtime_commit=RUNTIME, launch_requires_root_grant=True,
        requires_all_cpu_agents_explicitly_paused=True, guard_command=command,
        guard_sha256=digest(GUARD), order='ABBA', configured_slots=6136,
        common_env=COMMON_ENV, candidate_env=CANDIDATE, model_from_prior=previous['model'],
        model_metadata_currently_verified=False, model_metadata_read_in_preparation=False,
        model_opened_in_preparation=False, gpu_initialized_in_preparation=False,
        native_control={k: native[0][k] for k in ('logits_sha256', 'snapshot_sha256')},
        phase_identity=phase_identity(native[0]['phases']),
        output_bytes=[native[0]['logits_bytes'], native[0]['snapshot_bytes']],
        files={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
        plain_build_manifest_sha256=digest(BUILD / 'manifest.json'),
        estimated_seconds=360, estimated_range_seconds=[330, 450],
        timing_scope='C phase wall times exclude sampling, tools, subsequent vector/state export and hashing. Fresh process per arm; one fixed teacher-forced replay.',
        hash_scope='Seven complete phase-end vocabulary vectors concatenated plus final full snapshot; raw343 scored rows and task quality are separate.',
        promotion=False, keep_full_vectors_and_state=True,
        profiling='All inherited DS4 cleared. No diagnostic, IO or route clocks; only post-phase configured-cache observation and ordinary untimed final cache/route footer.')
    write(output / 'manifest.json', manifest)
    write(output / 'prepared-command.json', dict(command=command, launch_authorized=False,
        manifest_sha256=digest(output / 'manifest.json'), requires_root_grant_and_explicit_cpu_freeze=True))
    return manifest


def run(output, root_granted=False, cpu_timing_frozen=False):
    if not root_granted or not cpu_timing_frozen:
        raise ValueError('--run needs root grant and explicit CPU timing freeze assertions')
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['_manifest_sha256'] = json.loads((output / 'prepared-command.json').read_text())['manifest_sha256']
    if digest(output / 'manifest.json') != manifest['_manifest_sha256']:
        raise ValueError('frozen plan changed')
    require_main_guard(output, manifest)  # Before any model metadata or helper/model execution.
    check_assets(output, manifest)
    helpers = load_common(output / 'abba_helpers.py')
    model = Path(manifest['model_from_prior']['path'])
    if helpers.model_identity(model) != manifest['model_from_prior']:
        raise ValueError('model metadata differs from prior native-fast64 run')
    directory = output / 'run'
    directory.mkdir(exist_ok=False)
    results = []
    write(directory / 'launch.json', dict(guard_verified=True, root_granted=True, cpu_timing_frozen=True,
        removed_inherited_ds4_env={k: v for k, v in os.environ.items() if k.startswith('DS4_')}))
    for index, variant in enumerate(manifest['order']):
        require_main_guard(output, manifest)
        check_assets(output, manifest)
        if helpers.model_identity(model) != manifest['model_from_prior']:
            raise ValueError('model metadata changed before arm')
        destination = directory / f'{index}-{variant}'
        destination.mkdir()
        logits, state = destination / 'logits.bin', destination / 'snapshot.bin'
        command = [str(output / 'replay-three-plain'), str(model), str(output / 'session.txt'), str(logits), str(state)]
        env = run_environment(output, variant)
        write(destination / 'run.json', dict(command=command, env={k: v for k, v in env.items() if k.startswith('DS4_')}))
        started = time.monotonic()
        with (destination / 'timings.csv').open('w') as stdout, (destination / 'stderr.log').open('w') as stderr:
            completed = subprocess.run(command, cwd=output, env=env, stdout=stdout, stderr=stderr, check=False)
        process_wall_ms = 1000 * (time.monotonic() - started)
        check_assets(output, manifest)
        if helpers.model_identity(model) != manifest['model_from_prior']:
            raise ValueError('model metadata changed during arm')
        if completed.returncode:
            raise RuntimeError(f'replay arm{index}-{variant} failed rc={completed.returncode}')
        phases = helpers.parse_timings(destination / 'timings.csv')
        row = dict(variant=variant, process_wall_ms=process_wall_ms,
            logits_sha256=digest(logits), snapshot_sha256=digest(state),
            logits_bytes=logits.stat().st_size, snapshot_bytes=state.stat().st_size,
            phases=phases, profiles=helpers.parse_profiles(destination / 'stderr.log'),
            startup_prefill_ms=phases[0]['ms'],
            append_prefill_ms=sum(r['ms'] for r in phases[1:] if r['kind'] == 'P'),
            decode_ms=sum(r['ms'] for r in phases if r['kind'] == 'D'), model_ms=sum(r['ms'] for r in phases))
        row['turn_model_ms'] = row['append_prefill_ms'] + row['decode_ms']
        write(destination / 'result.json', row)  # Preserve failed native/repeat evidence too.
        audit_result(row, manifest, results)
        results.append(row)
        write(directory / 'summary.json', results)
        write(directory / 'comparison.json', helpers.comparison(results))
        print(json.dumps({k: v for k, v in row.items() if k not in ('phases', 'profiles')}), flush=True)
    write(directory / 'checks.json', dict(passed=True, order='ABBA', native_fast64_prior_hashes_exact=True,
        both_arms_repeated_hashes_exact=True, configured_slots_every_phase=6136,
        profile_off=True, timings_are_single_workload_only=True, default_promoted=False))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--root-granted', action='store_true')
    parser.add_argument('--cpu-timing-frozen', action='store_true')
    args = parser.parse_args()
    if bool(args.prepare) == bool(args.run):
        parser.error('choose exactly one of --prepare or --run')
    try:
        if args.prepare:
            manifest = prepare(args.prepare.resolve())
            print(json.dumps(dict(prepared=str(args.prepare.resolve()), command=manifest['guard_command'])), flush=True)
        else:
            run(args.run.resolve(), args.root_granted, args.cpu_timing_frozen)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f'three-slot timing: {error}\n')


if __name__ == '__main__':
    main()
