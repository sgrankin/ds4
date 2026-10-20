#!/usr/bin/env python3
"""Offline independent verification/archive; never opens a model or runs a helper."""
import array
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import struct
import subprocess
import sys

ROOT = Path('/Volumes/Models/ds4')
WORKSPACE = ROOT / '.workspaces/idea-native-resident-growth-six'
PREPARED = WORKSPACE / 'speed-bench/ideas/artifacts/native-resident-growth-six-plan-20261002-v1'
BUILD = ROOT / '.workspaces/idea-resident-three-timing/speed-bench/ideas/artifacts/three-timing-build'
ARCHIVE = Path(__file__).resolve().parent
SOURCE_COMMIT = 'aaa62c6bb9024f8ccb41293369963d7a104a9ddc'
RUNTIME_COMMIT = 'b9408c3b8e3ec5b22ad7efff3e365fd81818c317'
BASE_COMMIT = '7c6e7b32a2560e032decea88b7877e330a27c52b'
EXPECTED_MANIFEST = 'a8fd08c721e8ff7f657f4297cc65e112c376c90ec31d7073d2e9369e547049f8'
EXPECTED_COMMAND = '93646f869a1e42475375de3e93647672246c64984995da60694190f23fb68501'
PHASES = [(0, 'P', 1859, 1859), (1, 'P', 1949, 90), (2, 'D', 2021, 72),
          (3, 'P', 10213, 8192), (4, 'D', 10221, 8), (5, 'P', 26605, 16384),
          (6, 'D', 26613, 8), (7, 'P', 26651, 38), (8, 'D', 26659, 8)]
VOCAB = 129280
KEYS = ['startup_prefill_ms', 'append_prefill_ms', 'decode_ms', 'model_ms', 'turn_model_ms']


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8)


def identity(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'mode': oct(path.stat().st_mode & 0o777)}


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    check(not destination.exists(), 'archive refuses overwrite: ' + str(destination))
    shutil.copy2(source, destination)
    check(sha(source) == sha(destination), 'archive copy changed: ' + str(source))


def committed(path, revision):
    result = subprocess.run(['jj', '--ignore-working-copy', 'file', 'show', '-r', revision, path],
                            cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout


def main():
    check(not (ARCHIVE / 'independent-verification.json').exists(), 'completed archive exists')
    check(sys.byteorder == 'little', 'FP32 artifact comparison expects little-endian host')
    manifest = load(PREPARED / 'manifest.json')
    prepared_command = load(PREPARED / 'prepared-command.json')
    launch = load(PREPARED / 'root-launch-proof.json')
    release = load(PREPARED / 'root-release-proof.json')
    guard_result = load(PREPARED / 'pressure/result.json')
    check(sha(PREPARED / 'manifest.json') == EXPECTED_MANIFEST, 'prepared manifest pin')
    check(sha(PREPARED / 'prepared-command.json') == EXPECTED_COMMAND, 'prepared command pin')
    check(prepared_command['manifest_sha256'] == EXPECTED_MANIFEST, 'command authority')
    check(manifest['runtime_commit'] == RUNTIME_COMMIT and manifest['order'] == 'ABCCBA', 'runtime/order')
    check(manifest['phase_identity'] == [list(x) for x in PHASES], 'phase identities')
    check(manifest['promotion'] is False and manifest['configured_slots'] == 6136, 'defaults/capacity')
    check(manifest['context_capacity'] == 32768 and manifest['expected_logits_bytes'] == 9 * VOCAB * 4,
          'context/vector geometry')
    check(launch['command'] == manifest['guard_command'] == prepared_command['command'], 'guard argv')
    check(launch['source_commit'] == SOURCE_COMMIT and launch['manifest_sha256'] == EXPECTED_MANIFEST
          and launch['command_sha256'] == EXPECTED_COMMAND, 'ROOT launch pins')
    check(launch['sole_model_grant'] is True and launch['cpu_timing_frozen'] is True
          and launch['lock_reacquired_before_launch'] is True and launch['live_model_processes_before_launch'] == [],
          'ROOT launch exclusivity')
    check(set(launch['pause_acknowledgements']) == {'clm_design', 'resident_experts', 'checkout_recap'},
          'three CPU pause acknowledgements')
    check(release['guard_result'] == guard_result == {'status': 'complete', 'child_returncode': 0}, 'guard completion')
    check(release['lock_reacquired'] is True and release['live_model_processes'] == [], 'ROOT release')
    for name, expected in manifest['files'].items():
        check(sha(PREPARED / name) == expected, 'prepared asset pin: ' + name)
    wrapper_path = 'speed-bench/ideas/prepare_native_resident_growth_six.py'
    check(committed(wrapper_path, SOURCE_COMMIT) == (PREPARED / 'prepare_native_resident_growth_six.py').read_bytes(),
          'frozen wrapper revision equals executed copied wrapper')
    guard = ROOT / 'speed-bench/ideas/pressure_run.py'
    check(sha(guard) == manifest['guard_sha256'], 'guard pin')
    check(sha(Path(manifest['guard_interpreter_identity']['path'])) == manifest['guard_interpreter_identity']['sha256'],
          'framework interpreter pin')
    build = load(PREPARED / 'build-manifest.json')
    check(build['runtime_commit'] == RUNTIME_COMMIT and build['core_or_metal_rebuilt'] is False, 'reused object proof')
    check(sha(BUILD / 'manifest.json') == manifest['plain_build_manifest_sha256'], 'original build proof pin')
    for name, expected in build['files'].items():
        check(sha(BUILD / name) == expected, 'build source/object/link pin: ' + name)
    for name, expected in build['objects'].items():
        obj = BUILD / 'objects' / name
        check(sha(obj) == expected['sha256'] and obj.stat().st_size == expected['bytes'], 'linked object: ' + name)
    for name in ['ds4.c', 'ds4.h', 'ds4_resident_route.h', 'ds4_gpu.h', 'ds4_metal.m']:
        check(committed(name, RUNTIME_COMMIT) == (BUILD / 'source' / name).read_bytes(), 'runtime source: ' + name)
    tokens = []
    input_phases = []
    for index, line in enumerate((PREPARED / 'session.txt').read_text().splitlines()):
        fields = line.split()
        kind, cached, count = fields[0], int(fields[1]), int(fields[2])
        values = list(map(int, fields[3:]))
        check(len(values) == count and all(0 <= token < VOCAB for token in values), 'input token extent/range')
        if kind == 'P':
            check(0 <= cached <= len(tokens), 'input prefix truncation')
            tokens = tokens[:cached]
        else:
            check(kind == 'D' and cached == 0, 'input decode contract')
        tokens.extend(values)
        check(len(tokens) <= 32768, 'input context bound')
        input_phases.append((index, kind, len(tokens), count))
    check(input_phases == PHASES and len(tokens) == 26659, 'recomputed replay boundaries')
    check(sha(PREPARED / 'session.txt') == manifest['replay_sha256']
          == sha(ROOT / 'speed-bench/agent-session/workspace-transition.txt'), 'original replay identity')
    summary = load(PREPARED / 'run/summary.json')
    check(len(summary) == 6 and ''.join(x['variant'] for x in summary) == 'ABCCBA', 'six-arm completeness')
    results = []
    external = []
    phase_vectors = {}
    artifacts = {}
    phase_rows = []
    shader_env = {'DS4_METAL_' + Path(name).stem.upper() + '_SOURCE': str(PREPARED / name)
                  for name in manifest['files'] if name.startswith('metal/')}
    for index, variant in enumerate('ABCCBA'):
        label = f'{index}-{variant}'
        directory = PREPARED / 'run' / label
        result = load(directory / 'result.json')
        check(result == summary[index] and result['variant'] == variant, 'result/summary identity: ' + label)
        run = load(directory / 'run.json')
        expected_env = dict(manifest['common_env'], **manifest['variants'][variant]['env'], **shader_env)
        check(run['env'] == expected_env, 'complete DS4 environment/policy: ' + label)
        check(run['command'] == [str(PREPARED / 'replay-three-plain'), manifest['model_from_prior']['path'],
              str(PREPARED / 'session.txt'), str(directory / 'logits.bin'), str(directory / 'snapshot.bin')],
              'same binary/input command: ' + label)
        with (directory / 'timings.csv').open(newline='') as stream:
            reader = csv.DictReader(stream)
            check(reader.fieldnames == ['phase', 'kind', 'context', 'tokens', 'ms', 'first_decode_ms'], 'CSV fields')
            rows = [{'phase': int(row['phase']), 'kind': row['kind'], 'context': int(row['context']),
                     'tokens': int(row['tokens']), 'ms': float(row['ms']),
                     'first_decode_ms': float(row['first_decode_ms'])} for row in reader]
        check([(r['phase'], r['kind'], r['context'], r['tokens']) for r in rows] == PHASES, 'nine ordered CSV phases')
        check(rows == result['phases'], 'CSV/result times')
        check(all(math.isfinite(row[key]) and row[key] >= 0 for row in rows for key in ['ms', 'first_decode_ms']),
              'finite timings')
        computed = {'startup_prefill_ms': rows[0]['ms'],
                    'append_prefill_ms': math.fsum(row['ms'] for row in rows[1:] if row['kind'] == 'P'),
                    'decode_ms': math.fsum(row['ms'] for row in rows if row['kind'] == 'D'),
                    'model_ms': math.fsum(row['ms'] for row in rows)}
        computed['turn_model_ms'] = computed['append_prefill_ms'] + computed['decode_ms']
        check(all(close(result[key], value) for key, value in computed.items()), 'aggregate result arithmetic')
        stderr = (directory / 'stderr.log').read_text()
        cache = [(int(a), int(b), int(c)) for a, b, c in re.findall(
            r'^REPLAY_CACHE phase=(\d+) configured_slots=(\d+) resident_entries=(\d+)$', stderr, re.M)]
        check(cache == [(phase, 6136, 6136) for phase in range(9)], 'actual native cache observations')
        check([(r['phase'], r['configured_slots'], r['resident_entries']) for r in result['profiles']['cache_phase']] == cache,
              'cache parsed profile identity')
        check(not re.search(r'^REPLAY_(?:ROUTE|IO)\b|^DS4_(?:V41|TIMING|PROFILE)', stderr, re.M), 'unexpected diagnostic clocks')
        check(all(result['profiles'][key] == [] for key in ['route_phase', 'io_phase', 'timing']), 'profile-off controls')
        check(result['profiles']['cache'][0]['budget'] == 6136, 'final cache slots')
        route = result['profiles']['resident_route']
        check((variant in 'AB' and not route) or (variant == 'C' and len(route) == 1 and route[0]['substitutions'] > 0
              and 'triple_layers' in route[0]), 'native versus active candidate route control')
        logits = directory / 'logits.bin'
        state = directory / 'snapshot.bin'
        for path, key in [(logits, 'logits'), (state, 'snapshot')]:
            record = identity(path)
            check(record['sha256'] == result[key + '_sha256'] and record['bytes'] == result[key + '_bytes'],
                  'actual raw/state identity: ' + label)
            check(record['bytes'] == (9 * VOCAB * 4 if key == 'logits' else 181735616), 'raw/state byte extent')
            record.update(arm=label, kind=key, retention='original path, not duplicated in compact archive')
            external.append(record)
        data = logits.read_bytes()
        floats = array.array('f'); floats.frombytes(data)
        check(all(math.isfinite(value) for value in floats), 'finite complete FP32 vectors')
        vector_hashes = [hashlib.sha256(data[phase*VOCAB*4:(phase+1)*VOCAB*4]).hexdigest() for phase in range(9)]
        with state.open('rb') as stream:
            header = stream.read(52)
            check(struct.unpack('<13I', header) == (0x34565344, 2, 32768, 1, 128, 128, 32769, 26659, 40, 512, 128, VOCAB, 0x413431),
                  'full canonical state header')
            recorded_tokens = struct.unpack('<26659I', stream.read(26659*4))
            check(recorded_tokens == tuple(tokens), 'final snapshot exact original token history')
            check(stream.read(VOCAB*4) == data[-VOCAB*4:], 'snapshot logits equal final full phase vector')
        artifacts[label] = {'logits_sha256': result['logits_sha256'], 'snapshot_sha256': result['snapshot_sha256'],
                            'phase_vector_sha256': vector_hashes, 'state_header_hex': header.hex(),
                            'canonical_context': 32768, 'canonical_position': 26659,
                            'selected_coefficients': 'existing final-affinity normalization' if variant == 'C' else 'native'}
        phase_vectors[label] = floats
        results.append(dict(variant=variant, **computed))
        for row in rows:
            phase_rows.append(dict(arm=label, variant=variant, **row))
    for variant, indices in {'A': [0,5], 'B': [1,4], 'C': [2,3]}.items():
        left, right = (artifacts[f'{i}-{variant}'] for i in indices)
        check(left == right, 'same-variant full vectors/state/header repeat identity: ' + variant)
    means = {variant: {key: statistics.mean(row[key] for row in results if row['variant'] == variant) for key in KEYS}
             for variant in 'ABC'}
    retained_comparison = load(PREPARED / 'run/comparison.json')
    for variant in 'ABC':
        check(retained_comparison['arms'][variant]['runs'] == 2, 'balanced repeat count')
        check(all(close(retained_comparison['arms'][variant][key], value) for key, value in means[variant].items()),
              'retained arm means')
    phase_means = {variant: [statistics.mean(row['ms'] for row in phase_rows
                   if row['variant'] == variant and row['phase'] == phase) for phase in range(9)] for variant in 'ABC'}
    pairs = {}
    for label, a, b in [('native_fast_vs_default','A','B'), ('resident_three_vs_native_fast','B','C'),
                        ('resident_three_vs_native_default','A','C')]:
        changes = {key: 100 * (means[b][key] / means[a][key] - 1) for key in KEYS}
        check(all(close(retained_comparison['pairs'][label]['change_percent'][key], value) for key, value in changes.items()),
              'retained comparison arithmetic')
        phase_changes = [100*(phase_means[b][i]/phase_means[a][i]-1) for i in range(9)]
        pairs[label] = dict(baseline=a, candidate=b, change_percent=changes, phase_change_percent=phase_changes)
    pressure = [json.loads(line) for line in (PREPARED / 'pressure/pressure.jsonl').read_text().splitlines()]
    initial = load(PREPARED / 'pressure/command.json')['initial']
    check(len(pressure) == release['normal_pressure_samples'] == 869, 'pressure sample count')
    check(all(row['level'] == 1 and row['reason'] is None and row['consecutive_bad'] == 0 for row in pressure), 'normal pressure')
    swap_growth = max(row['swap_gb'] for row in pressure) - initial['swap_gb']
    peak = max(row['wired_gb'] + row['compressor_gb'] for row in pressure)
    check(close(swap_growth, release['swap_growth_gb']) and close(peak, release['peak_wired_compressor_gib']), 'pressure summary')
    check(release['bad_pressure_samples'] == 0, 'bad pressure count')
    check(launch['time'] <= initial['time'] <= pressure[0]['time'] < pressure[-1]['time'] < release['time'], 'lifecycle chronology')
    check(load(PREPARED / 'run/checks.json')['passed'] is True, 'retained checks completion')
    original_file_index = [identity(path) for path in sorted(PREPARED.rglob('*')) if path.is_file()]
    for path in sorted(PREPARED.rglob('*')):
        if path.is_file() and not (path.name in ['logits.bin', 'snapshot.bin'] and path.parent.parent == PREPARED / 'run'):
            copy(path, ARCHIVE / 'original-prepared' / path.relative_to(PREPARED))
    for name in build['files']:
        copy(BUILD / name, ARCHIVE / 'runtime-build' / name)
    copy(BUILD / 'manifest.json', ARCHIVE / 'runtime-build/manifest.json')
    copy(guard, ARCHIVE / 'guard/pressure_run.py')
    test_name = 'tests/test_native_resident_growth_six.py'
    test_bytes = committed(test_name, SOURCE_COMMIT)
    check(test_bytes == (WORKSPACE / test_name).read_bytes(), 'frozen CPU test source pin')
    copy(WORKSPACE / test_name, ARCHIVE / 'cpu-tests/test_native_resident_growth_six.py')
    prior_release = Path(launch['prior_release_proof'])
    check(sha(prior_release) == launch['prior_release_sha256'], 'previous job release proof')
    copy(prior_release, ARCHIVE / 'root/prior-wrap-release-proof.json')
    root_log = ROOT / 'speed-bench/ideas/artifacts/native-resident-growth-six-root-launch-20261003-v1.log'
    log_records = [json.loads(line) for line in root_log.read_text().splitlines() if line.strip()]
    check(len(log_records) == 7 and log_records[-1] == guard_result, 'ROOT stdout complete six arm plus guard')
    for index, record in enumerate(log_records[:-1]):
        check(record == {k:v for k,v in summary[index].items() if k not in ['phases','profiles']}, 'ROOT stdout/result identity')
    copy(root_log, ARCHIVE / 'root/launch-stdout.log')
    write(ARCHIVE / 'original-file-index.json', original_file_index)
    write(ARCHIVE / 'external-raw-state-files.json', external)
    write(ARCHIVE / 'independent-arithmetic.json', dict(formula='100*(candidate mean/baseline mean-1); arithmetic means of two complete fresh runs',
          definitions={'startup_prefill_ms':'phase0', 'append_prefill_ms':'P phases1/3/5/7', 'decode_ms':'D phases2/4/6/8',
                       'model_ms':'all nine phases', 'turn_model_ms':'append+decode, excludes phase0'},
          means_ms=means, phase_means_ms=phase_means, comparisons=pairs))
    with (ARCHIVE / 'independent-phase-times.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['arm','variant','phase','kind','context','tokens','ms','first_decode_ms'])
        writer.writeheader(); writer.writerows(phase_rows)
    write(ARCHIVE / 'independent-verification.json', dict(status='PASS', model_or_tokenizer_opened=False,
          gpu_used=False, helper_executed=False, runtime_or_cpu_build_run=False, timing_repeated=False,
          archive_parent=BASE_COMMIT, source_commit=SOURCE_COMMIT, runtime_commit=RUNTIME_COMMIT,
          manifest_sha256=EXPECTED_MANIFEST, command_sha256=EXPECTED_COMMAND,
          prepared_assets_verified=len(manifest['files']), objects_verified=len(build['objects']),
          runtime_sources_revision_equal=['ds4.c','ds4.h','ds4_resident_route.h','ds4_gpu.h','ds4_metal.m'],
          raw_state_files_rehashed=len(external), phase_full_vectors_verified=54, all_full_vectors_finite=True,
          within_variant_full_vector_state_byte_identities_exact=True, cache_observations_verified=54,
          configured_and_resident_slots_every_phase=6136, artifacts=artifacts,
          original_replay_token_ids_exact=True, prefill_tokens=26563, decode_tokens=96,
          preserved_launch_pause_acknowledgements=launch['pause_acknowledgements'],
          cpu_quiet_evidence='ROOT recorded explicit three-agent acknowledgements; no independent OS-wide CPU monitor claim',
          guard_child_returncode=0, pressure_samples=len(pressure), bad_pressure_samples=0,
          swap_growth_gb=swap_growth, peak_wired_compressor_gib=peak,
          cross_variant_state_identity_required=False, cross_variant_hashes_diagnostic_only=True,
          no_generated_task_correctness_or_equal_quality_claim=True, default_promoted=False,
          initial_setup_refusals=['sandbox jj workspace add could not write .git/objects',
                                  'retry original path existed after incomplete initialization',
                                  'fresh workspace add with --ignore-working-copy created root checkout but refused update; explicitly initialized from immutable base'],
          no_run_artifacts_mutated=True))
    print(json.dumps({'status':'PASS','prepared_assets':len(manifest['files']),'linked_objects':len(build['objects']),
          'raw_state_files':len(external),'phase_observations':54,'means_ms':means,
          'comparisons':{key:value['change_percent'] for key,value in pairs.items()}}, indent=2))


if __name__ == '__main__':
    main()
