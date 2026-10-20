#!/usr/bin/env python3
"""Prepare profile-off ABCCBA growth: native default, native fast, resident-three."""
import argparse
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import statistics
import sys
import time

ROOT = Path('/Volumes/Models/ds4')
GUARD = ROOT / 'speed-bench/ideas/pressure_run.py'
GUARD_INTERPRETER = Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python')
GUARD_INTERPRETER_SHA = 'b80d260a48404da13a32dd44d7283aadc80f6a6e56ed6e03ed5597ec6f46d48d'
WORKSPACE = Path(__file__).resolve().parents[2]
BUILD = ROOT / '.workspaces/idea-resident-three-timing/speed-bench/ideas/artifacts/three-timing-build'
SEED = ROOT / '.workspaces/idea-resident-three-timing/speed-bench/ideas/artifacts/resident-three-fast64-abba-plan-20261002-v2'
SEED_SHA = '13adc6b7ce0202caa4f2d25bd5e275b889020ff6275bd5b8d1c427d54f77a317'
SEED_CPU_PROOF_SHA = '20df098634bbb7dcc6597635578dbf45a3b4eacee01c45bc3051eaa9f68427df'
FOUR_DRIVER = ROOT / '.workspaces/idea-resident-three-growth/speed-bench/ideas/prepare_resident_three_growth.py'
FOUR_DRIVER_SHA = 'e553ec43e965a0af2f91438372a2fb20baf62917a8f611f7d25a07fd30f30782'
COMMON = SEED / 'abba_helpers.py'
SESSION = ROOT / 'speed-bench/agent-session/workspace-transition.txt'
RUNTIME = 'b9408c3b8e3ec5b22ad7efff3e365fd81818c317'
SESSION_SHA = 'a859a10f9d74987789aa982a6d061db3991be11cec884e277fc5da4ce9f92adb'
CAPACITY = 32768
VOCAB_FROM_APPROVED_V2 = 129280
EXPECTED_PHASES = [(0, 'P', 1859, 1859), (1, 'P', 1949, 90), (2, 'D', 2021, 72),
    (3, 'P', 10213, 8192), (4, 'D', 10221, 8), (5, 'P', 26605, 16384),
    (6, 'D', 26613, 8), (7, 'P', 26651, 38), (8, 'D', 26659, 8)]
ROUTE = 'DS4_METAL_V41_RESIDENT_ROUTE_'
COMMON_ENV = {'DS4_REPLAY_CACHE_GB': '64', 'DS4_REPLAY_CONTEXT': str(CAPACITY),
    'DS4_REPLAY_PREFILL_CHUNK': '512', 'DS4_REPLAY_CACHE_REGIME': '1'}
FAST_ENV = {'DS4_METAL_V41_SELECTED_TILE': '64', 'DS4_METAL_V41_SELECTED_LIMIT': '1024'}
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
    interpreter = manifest['guard_interpreter_identity']
    if digest(interpreter['path']) != interpreter['sha256']:
        raise ValueError('pinned framework Python changed')


def require_main_guard(output, manifest):
    if digest(GUARD) != manifest['guard_sha256']:
        raise ValueError('MAIN pressure guard changed')
    parent = subprocess.check_output(['ps', '-p', str(os.getppid()), '-o', 'command='], text=True)
    actual = shlex.split(parent.strip())
    expected = manifest['guard_command']
    interpreter = manifest['guard_interpreter_identity']
    if not actual or actual[1:] != expected[1:]:
        raise ValueError('--run requires its exact prepared MAIN guard as direct parent')
    try:
        same_interpreter = os.path.samefile(actual[0], interpreter['path'])
    except OSError:
        same_interpreter = False
    if not same_interpreter or digest(interpreter['path']) != interpreter['sha256']:
        raise ValueError('direct parent interpreter differs from pinned framework Python')
    with (ROOT / 'speed-bench/ideas/artifacts/model-job.lock').open('r+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise ValueError('MAIN guard does not hold model-job lock')
    return dict(parent_argv=actual, exact_suffix_argv=True,
        interpreter_samefile=True, interpreter_sha256=interpreter['sha256'], guard_lock_held=True)


def phase_identity(rows):
    return [(r['phase'], r['kind'], r['context'], r['tokens']) for r in rows]


def replay_contract(raw, capacity=CAPACITY, vocab=VOCAB_FROM_APPROVED_V2):
    """Validate the existing integer recording using replay.c's history rules."""
    rows = []
    history = 0
    for phase, line in enumerate(raw.decode('ascii').splitlines()):
        fields = line.split()
        if len(fields) < 3 or fields[0] not in ('P', 'D') or any(not f.isdigit() for f in fields[1:]):
            raise ValueError('invalid replay header or token integer')
        kind, cached, count = fields[0], int(fields[1]), int(fields[2])
        ids = [int(f) for f in fields[3:]]
        if count < 1 or count > capacity or len(ids) != count or any(i >= vocab for i in ids):
            raise ValueError('replay token count/range mismatch')
        if kind == 'P':
            if cached > history:
                raise ValueError('replay cached prefix exceeds current history')
            history = cached
        elif cached != 0:
            raise ValueError('decode phase requires cached zero')
        history += count
        if history > capacity:
            raise ValueError('replay exceeds requested context capacity')
        rows.append((phase, kind, history, count))
    if len(rows) < 2 or rows[0][1] != 'P':
        raise ValueError('replay needs initial prefill and multiple phases')
    return rows


def run_environment(output, variant):
    if variant not in 'ABC' or len(variant) != 1:
        raise ValueError('unknown growth policy variant')
    env = {key: value for key, value in os.environ.items() if not key.startswith('DS4')}
    env.update(COMMON_ENV)
    env[ROUTE + 'GAP'] = '0'
    if variant in 'BC':
        env.update(FAST_ENV)
    if variant == 'C':
        env.update(CANDIDATE)
    for path in sorted((output / 'metal').glob('*.metal')):
        env['DS4_METAL_' + path.stem.upper() + '_SOURCE'] = str(path)
    return env


def comparison(results):
    keys = ('startup_prefill_ms', 'append_prefill_ms', 'decode_ms', 'model_ms', 'turn_model_ms')
    means = {}
    for variant in 'ABC':
        rows = [row for row in results if row['variant'] == variant]
        if rows:
            means[variant] = {key: statistics.mean(row[key] for row in rows) for key in keys}
            means[variant]['runs'] = len(rows)
    pairs = {}
    for name, baseline, candidate in [('native_fast_vs_default', 'A', 'B'),
            ('resident_three_vs_native_fast', 'B', 'C'), ('resident_three_vs_native_default', 'A', 'C')]:
        if baseline in means and candidate in means:
            pairs[name] = dict(baseline=baseline, candidate=candidate,
                change_percent={key:100*(means[candidate][key]/means[baseline][key]-1)
                    if means[baseline][key] else None for key in keys})
    return dict(arms=means, pairs=pairs, primary='native_fast_vs_default',
        secondary='resident_three_vs_native_fast', cross_variant_state_hashes_are_diagnostic=True,
        qualification='One original-ID teacher-forced growth input, two fresh runs per variant; no correctness, significance or default-promotion claim.')


def audit_result(row, manifest, results):
    if phase_identity(row['phases']) != [tuple(r) for r in manifest['phase_identity']]:
        raise ValueError('replay phase boundaries differ')
    if row['logits_bytes'] != manifest['expected_logits_bytes'] or row['snapshot_bytes'] <= 0:
        raise ValueError('phase end-vector or final-state size differs')
    if any(not math.isfinite(r[key]) or r[key] < 0 for r in row['phases'] for key in ('ms', 'first_decode_ms')):
        raise ValueError('nonfinite or negative replay timings')
    cache = row['profiles']['cache_phase']
    if len(cache) != len(row['phases']) or any(r.get('phase') != i or
            r.get('configured_slots') != 6136 for i, r in enumerate(cache)):
        raise ValueError('configured cache slots must remain6136 in every phase')
    if len(row['profiles']['cache']) != 1 or row['profiles']['cache'][0].get('budget') != 6136:
        raise ValueError('final configured cache budget differs')
    if any(row['profiles'][key] for key in ('route_phase', 'io_phase', 'timing')):
        raise ValueError('diagnostic route/IO/timing profiles emitted')
    route = row['profiles']['resident_route']
    if row['variant'] in 'AB' and route:
        raise ValueError('native GAP0 emitted resident-route footer')
    if row['variant'] == 'C' and (len(route) != 1 or route[0].get('layers', 0) <= 0 or
            route[0].get('substitutions', 0) <= 0 or 'triple_layers' not in route[0]):
        raise ValueError('candidate resident routing was not observed')
    repeats = [r for r in results if r['variant'] == row['variant']]
    if repeats and any(row[k] != repeats[0][k] for k in ('logits_sha256', 'snapshot_sha256', 'logits_bytes', 'snapshot_bytes')):
        raise ValueError('same-arm repeated vectors or graph state differ')


def prepare(output):
    if digest(FOUR_DRIVER) != FOUR_DRIVER_SHA:
        raise ValueError('preserved approved four-arm growth driver changed')
    if digest(GUARD_INTERPRETER) != GUARD_INTERPRETER_SHA:
        raise ValueError('observed MAIN guard framework Python changed')
    proof = json.loads((BUILD / 'manifest.json').read_text())
    if proof['runtime_commit'] != RUNTIME or proof['core_or_metal_rebuilt']:
        raise ValueError('reviewed b9408c3b object provenance differs')
    for name, expected in proof['files'].items():
        if digest(BUILD / name) != expected:
            raise ValueError('plain build asset changed: ' + name)
    if digest(SEED / 'manifest.json') != SEED_SHA:
        raise ValueError('approved V2 seed manifest differs')
    if digest(SEED / 'cpu-proof.json') != SEED_CPU_PROOF_SHA:
        raise ValueError('approved V2 actual guarded CPU proof differs')
    previous = json.loads((SEED / 'manifest.json').read_text())
    if previous['runtime_commit'] != RUNTIME or previous['candidate_env'] != CANDIDATE:
        raise ValueError('approved V2 runtime or candidate differs')
    for name, expected in previous['files'].items():
        if digest(SEED / name) != expected:
            raise ValueError('approved V2 seed asset changed: ' + name)
    if previous['output_bytes'][0] != 7 * VOCAB_FROM_APPROVED_V2 * 4:
        raise ValueError('approved V2 vector-size vocabulary proof differs')
    raw = SESSION.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SESSION_SHA:
        raise ValueError('frozen original growth replay differs')
    phases = replay_contract(raw)
    if phases != EXPECTED_PHASES:
        raise ValueError('original growth replay phase contract differs')
    frontend = (BUILD / 'source/replay.c').read_text()
    if 'getenv("DS4_REPLAY_CONTEXT")' not in frontend or 'rows<=100000' not in frontend:
        raise ValueError('approved frontend context environment support missing')
    output.mkdir(parents=True, exist_ok=False)
    copies = {'replay-three-plain': BUILD / 'replay-three-plain', 'build-manifest.json': BUILD / 'manifest.json',
        'reviewed-three-build-manifest.json': BUILD / 'reviewed-three-build-manifest.json',
        'session.txt': SESSION, 'abba_helpers.py': COMMON, 'prepare_native_resident_growth_six.py': Path(__file__),
        'approved-four-arm-growth-driver.py': FOUR_DRIVER,
        'approved-v2-manifest.json': SEED / 'manifest.json',
        'approved-v2-driver.py': SEED / 'prepare_resident_three_timing.py',
        'approved-v2-cpu-proof.json': SEED / 'cpu-proof.json', 'replay.c': BUILD / 'source/replay.c'}
    for name, source in copies.items():
        shutil.copy2(source, output / name)
    (output / 'replay-three-plain').chmod(0o555)
    (output / 'metal').mkdir()
    for name, expected in previous['files'].items():
        if name.startswith('metal/'):
            source = SEED / name
            shutil.copy2(source, output / name)
    command = [sys.executable, str(GUARD), str(output / 'pressure'), '--', sys.executable,
        str(output / 'prepare_native_resident_growth_six.py'), '--run', str(output),
        '--root-granted', '--cpu-timing-frozen']
    manifest = dict(runtime_commit=RUNTIME, launch_requires_root_grant=True,
        requires_all_cpu_agents_explicitly_paused=True, guard_command=command,
        guard_interpreter_identity=dict(path=str(GUARD_INTERPRETER), sha256=GUARD_INTERPRETER_SHA,
            provenance='actual model-free MAIN guard ps probe; launcher shim is a distinct file'),
        guard_sha256=digest(GUARD), order='ABCCBA', configured_slots=6136,
        common_env=COMMON_ENV, candidate_env=CANDIDATE, model_from_prior=previous['model_from_prior'],
        variants=dict(A=dict(name='native-default', env={ROUTE+'GAP':'0'}, selected_tile_limit_unset=True,
            expected_effective_tile=8, expected_effective_limit=256, large_cache_threshold_slots=7680),
            B=dict(name='native-fast', env=dict(FAST_ENV, **{ROUTE+'GAP':'0'})),
            C=dict(name='resident-three-fast', env=dict(FAST_ENV, **CANDIDATE))),
        primary_comparison='B native-fast versus A native-default', secondary_comparison='C resident-three versus B native-fast',
        model_metadata_currently_verified=False, model_metadata_read_in_preparation=False,
        model_opened_in_preparation=False, gpu_initialized_in_preparation=False,
        no_preexisting_growth_expected_hash=True, native_repeat_hashes_required=True, candidate_repeat_hashes_required=True,
        phase_identity=phases, context_capacity=CAPACITY, final_context=phases[-1][2],
        replay_sha256=SESSION_SHA, original_token_ids_preserved=True, tokenizer_used=False,
        expected_logits_bytes=len(phases) * VOCAB_FROM_APPROVED_V2 * 4,
        vector_vocab_from_approved_v2=VOCAB_FROM_APPROVED_V2, snapshot_size_known_before_run=False,
        files={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
        plain_build_manifest_sha256=digest(BUILD / 'manifest.json'),
        approved_v2_seed_manifest_sha256=SEED_SHA,
        estimated_seconds=4725, estimated_range_seconds=[4050, 5400],
        estimate_basis='Unmeasured four-arm 45–60 minute preliminary estimate scaled by 1.5 for six arms (67.5–90 minutes).',
        timing_scope='C phase wall times exclude sampling, tools, subsequent vector/state export and hashing. Fresh process per arm; one fixed teacher-forced replay.',
        hash_scope='Nine complete phase-end vocabulary vectors concatenated plus final full snapshot. No historical growth reference; exact A/A, B/B and C/C required. Cross-variant differences are diagnostic. Teacher-forced replay only; no generated-task correctness claim.',
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
        raise ValueError('model metadata differs from copied approved V2 identity')
    directory = output / 'run'
    directory.mkdir(exist_ok=False)
    results = []
    write(directory / 'launch.json', dict(guard_verified=True, root_granted=True, cpu_timing_frozen=True,
        removed_inherited_ds4_env={k: v for k, v in os.environ.items() if k.startswith('DS4')}))
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
        write(directory / 'comparison.json', comparison(results))
        print(json.dumps({k: v for k, v in row.items() if k not in ('phases', 'profiles')}), flush=True)
    write(directory / 'checks.json', dict(passed=True, order='ABCCBA', no_historical_growth_reference_used=True,
        native_default_repeated_hashes_exact=True, native_fast_repeated_hashes_exact=True,
        resident_three_repeated_hashes_exact=True, all_variants_repeated_hashes_exact=True,
        cross_variant_hashes_diagnostic=True, configured_slots_every_phase=6136,
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
        parser.exit(1, f'six-arm growth timing: {error}\n')


if __name__ == '__main__':
    main()
