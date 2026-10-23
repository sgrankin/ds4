#!/usr/bin/env python3
"""Same-binary ABBA replay for approximate resident expert rerouting."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROUTE_PREFIX = 'DS4_METAL_V41_RESIDENT_ROUTE_'
DEFAULT_CANDIDATE = {
    ROUTE_PREFIX + 'GAP': '0.05',
    ROUTE_PREFIX + 'MAX_MASS': '0.12',
    ROUTE_PREFIX + 'MAX_SUBS': '1',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def environment_items(items, candidate=False):
    result = {}
    for item in items or []:
        name, separator, value = item.partition('=')
        if not separator or not re.fullmatch(r'DS4_[A-Z0-9_]+', name):
            raise ValueError(f'expected DS4_NAME=VALUE: {item}')
        result[name] = value
    return result


def freeze_shaders(output, source_root, inherited):
    destination = output / 'metal'
    destination.mkdir()
    overrides, hashes = {}, {}
    sources = sorted(source_root.glob('*.metal'))
    if not sources:
        raise ValueError(f'no Metal sources found in {source_root}')
    for source in sources:
        name = f'DS4_METAL_{source.stem.upper()}_SOURCE'
        actual = Path(inherited.get(name, str(source))).resolve()
        target = destination / source.name
        shutil.copy2(actual, target)
        overrides[name] = str(target)
        hashes[name] = {'source': str(actual), 'sha256': digest(target)}
    write_json(destination / 'sha256.json', hashes)
    return overrides, hashes


def numeric_fields(line):
    result = {}
    for key, value in re.findall(r'([a-zA-Z][a-zA-Z0-9_]*)=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line):
        result[key] = float(value) if any(c in value for c in '.eE') else int(value)
    return result


def parse_profiles(path):
    reports = {'cache': [], 'pending': [], 'timing': [], 'resident_route': [],
               'route_phase': [], 'io_phase': [], 'cache_phase': []}
    for line in path.read_text().splitlines():
        if line.startswith('REPLAY_CACHE '):
            reports['cache_phase'].append(numeric_fields(line))
        elif line.startswith('REPLAY_ROUTE '):
            reports['route_phase'].append(numeric_fields(line))
        elif line.startswith('REPLAY_IO '):
            reports['io_phase'].append(numeric_fields(line))
        elif 'streaming expert cache budget=' in line:
            reports['cache'].append(numeric_fields(line))
        elif 'streaming pending joins=' in line:
            reports['pending'].append(numeric_fields(line))
        elif 'streaming expert timing ' in line:
            reports['timing'].append({'line': line, 'fields': numeric_fields(line)})
        elif 'ds4: resident-route ' in line:
            reports['resident_route'].append(numeric_fields(line))
    return reports


def parse_timings(path):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    if not rows or rows[0]['kind'] != 'P':
        raise ValueError('replay must emit a nonempty initial prefill phase')
    for expected, row in enumerate(rows):
        for key in ('phase', 'context', 'tokens'):
            row[key] = int(row[key])
        for key in ('ms', 'first_decode_ms'):
            row[key] = float(row[key])
        if row['phase'] != expected or row['kind'] not in ('P', 'D') or row['ms'] < 0:
            raise ValueError(f'invalid replay timing row: {row}')
    return rows


def model_identity(path):
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def comparison(results):
    keys = ('startup_prefill_ms', 'append_prefill_ms', 'decode_ms', 'model_ms', 'turn_model_ms')
    grouped = {}
    for arm in ('A', 'B'):
        group = [row for row in results if row['variant'] == arm]
        if group:
            grouped[arm] = {key: statistics.mean(row[key] for row in group) for key in keys}
            grouped[arm]['runs'] = len(group)
    if 'A' in grouped and 'B' in grouped:
        grouped['candidate_change_percent'] = {
            key: 100 * (grouped['B'][key] / grouped['A'][key] - 1)
            if grouped['A'][key] else None for key in keys
        }
    grouped['quality_validation'] = 'separate teacher-forced and held-out evaluations required'
    grouped['state_hashes_are_diagnostic'] = True
    return grouped


def run(args):
    output = args.output.resolve()
    binary_source, replay_source, model = args.binary.resolve(), args.replay.resolve(), args.model.resolve()
    candidate = DEFAULT_CANDIDATE.copy()
    candidate.update(environment_items(args.candidate_env, candidate=True))
    output.mkdir(parents=True, exist_ok=False)
    binary, recording = output / 'replay', output / 'session.txt'
    shutil.copy2(binary_source, binary)
    shutil.copy2(replay_source, recording)
    binary.chmod(0o555)
    env = os.environ.copy()
    removed = {key: env.pop(key) for key in list(env) if key.startswith('DS4_')}
    env.update(environment_items(args.base_env))
    for name in candidate:
        env.pop(name, None)
    env['DS4_REPLAY_CACHE_GB'] = str(args.cache_gb)
    env['DS4_REPLAY_CONTEXT'] = str(args.context)
    env['DS4_REPLAY_PREFILL_CHUNK'] = str(args.prefill_chunk)
    if args.profile_memory:
        env['DS4_REPLAY_PROFILE_MEMORY'] = '1'
    shader_env, shader_hashes = freeze_shaders(output, args.shader_root.resolve(), env)
    env.update(shader_env)
    frozen_model = model_identity(model)
    order = args.order * args.cycles
    manifest = {
        'binary_source': str(binary_source), 'binary_sha256': digest(binary),
        'replay_source': str(replay_source), 'replay_sha256': digest(recording),
        'model': frozen_model, 'model_content_hash': None,
        'order': order, 'candidate': candidate, 'cache_gb': args.cache_gb, 'context': args.context,
        'prefill_chunk': args.prefill_chunk, 'removed_inherited_route_env': removed,
        'env': {key: value for key, value in env.items() if key.startswith('DS4_')},
        'metal': shader_hashes, 'hash_comparison': 'diagnostic only; approximate arms may differ',
        'timing': 'replay phase times exclude sampling, tool work, and state/hash I/O',
    }
    write_json(output / 'manifest.json', manifest)
    results, phase_identity = [], None
    for index, variant in enumerate(order):
        if model_identity(model) != frozen_model:
            raise RuntimeError('model metadata changed during the experiment')
        destination = output / f'{index}-{variant}'
        destination.mkdir()
        runenv = env.copy()
        for name in list(runenv):
            if name.startswith(ROUTE_PREFIX):
                runenv.pop(name)
        runenv[ROUTE_PREFIX + 'GAP'] = '0'
        if variant == 'B':
            runenv.update(candidate)
        logits, snapshot = destination / 'logits.bin', destination / 'snapshot.bin'
        command = [str(binary), str(model), str(recording), str(logits), str(snapshot)]
        write_json(destination / 'run.json', {
            'command': command, 'env': {key: value for key, value in runenv.items() if key.startswith('DS4_')},
        })
        started = time.monotonic()
        with (destination / 'timings.csv').open('w') as stdout, (destination / 'stderr.log').open('w') as stderr:
            completed = subprocess.run(command, cwd=args.shader_root.resolve().parent,
                env=runenv, stdout=stdout, stderr=stderr, check=False)
        if completed.returncode:
            raise RuntimeError(f'replay failed rc={completed.returncode}; see {destination / "stderr.log"}')
        rows = parse_timings(destination / 'timings.csv')
        identity = [(row['phase'], row['kind'], row['context'], row['tokens']) for row in rows]
        if phase_identity is None:
            phase_identity = identity
        elif identity != phase_identity:
            raise RuntimeError('teacher-forced replay phase boundaries changed across arms')
        result = {
            'variant': variant, 'process_wall_ms': 1000 * (time.monotonic() - started),
            'logits_sha256': digest(logits), 'snapshot_sha256': digest(snapshot),
            'logits_bytes': logits.stat().st_size, 'snapshot_bytes': snapshot.stat().st_size,
            'phases': rows, 'profiles': parse_profiles(destination / 'stderr.log'),
            'startup_prefill_ms': rows[0]['ms'],
            'append_prefill_ms': sum(row['ms'] for row in rows[1:] if row['kind'] == 'P'),
            'decode_ms': sum(row['ms'] for row in rows if row['kind'] == 'D'),
            'model_ms': sum(row['ms'] for row in rows),
        }
        result['turn_model_ms'] = result['append_prefill_ms'] + result['decode_ms']
        controls = [row for row in results if row['variant'] == 'A']
        if variant == 'A' and controls and any(
                result[key] != controls[0][key]
                for key in ('logits_sha256', 'snapshot_sha256')):
            raise RuntimeError('repeated native controls differ in logits or graph state')
        results.append(result)
        write_json(output / 'summary.json', results)
        write_json(output / 'comparison.json', comparison(results))
        print(json.dumps({key: value for key, value in result.items() if key not in ('phases', 'profiles')}), flush=True)
        if not args.keep_state:
            logits.unlink()
            snapshot.unlink()
    print(json.dumps(comparison(results)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--binary', type=Path, default=REPO / 'speed-bench/ideas/artifacts/resident-replay-integrated')
    parser.add_argument('--replay', type=Path, default=HERE / 'session-v1.txt')
    parser.add_argument('--model', type=Path, default=REPO / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf')
    parser.add_argument('--shader-root', type=Path, default=REPO / 'src/metal')
    parser.add_argument('--candidate-env', action='append', help='DS4_NAME=VALUE; overrides .05/.12/maxsubs1 defaults')
    parser.add_argument('--base-env', action='append', help='common DS4_NAME=VALUE for both arms')
    parser.add_argument('--cache-gb', type=int, default=64)
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--prefill-chunk', type=int, default=512)
    parser.add_argument('--order', choices=('ABBA', 'BAAB'), default='ABBA')
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--profile-memory', action='store_true', help='untimed cumulative memory report after each phase')
    parser.add_argument('--keep-state', action='store_true', help='retain large output files after hashing')
    args = parser.parse_args()
    if (not 4 <= args.cache_gb <= 96 or not 128 <= args.context <= 100000
            or not 1 <= args.prefill_chunk <= 8192 or not 1 <= args.cycles <= 20):
        parser.error('cache GB must be 4..96; context 128..100000; prefill chunk 1..8192; cycles 1..20')
    try:
        run(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'resident replay ABBA: {error}\n')


if __name__ == '__main__':
    main()
