#!/usr/bin/env python3
"""Actual trained DSpark C/native tensor oracle + runtime; ROOT runs under guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--bin-dir', type=Path, required=True)
p.add_argument('--target', type=Path, required=True)
p.add_argument('--draft', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--debug-native', type=Path)
a = p.parse_args()
a.bin_dir, a.target, a.draft, a.output = [path.resolve() for path in (a.bin_dir, a.target, a.draft, a.output)]
a.output.mkdir(parents=True, exist_ok=False)
env = {key: value for key, value in os.environ.items() if not key.startswith(('DS4', 'PT_'))}
for key, source in re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]', (ROOT / 'ds4_metal.m').read_text()):
    env[key] = str(ROOT / source)
env.update(DS4_V41_COMPRESSED_KV_F16='0', DS4_V41_DECODER_REPLAY_ROWS='0',
    DS4_METAL_V41_RESIDENT_ROUTE_GAP='0', DS4_METAL_V41_RESIDENT_PREFILL_GAP='0',
    DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS='0')

def save(value):
    (a.output / 'result.json').write_text(json.dumps(value, indent=2) + '\n')

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()

binaries = [('c', a.bin_dir / 'dspark-c-oracle'), ('native', a.bin_dir / 'dspark-native-oracle')]
if a.debug_native:
    binaries.append(('native-debug', a.debug_native.resolve()))
result = {'commands': [], 'oracles': {}, 'comparisons': {}, 'pass': False,
    'target': str(a.target), 'draft': str(a.draft), 'runtime_cache': '40GB, fixture-owned',
    'fixture_sha256': digest(ROOT / 'tests/test_deepseek41_dspark.c')}
for mode in ('oracle', 'runtime'):
    for label, binary in binaries:
        if mode == 'runtime' and label == 'native-debug':
            continue
        out = a.output / (mode + '-' + label)
        out.mkdir()
        command = [str(binary), str(a.target), str(a.draft), str(out) if mode == 'oracle' else '--runtime']
        row = {'label': label, 'mode': mode, 'command': command, 'binary_sha256': digest(binary),
            'cwd': str(ROOT), 'environment': {k: v for k, v in env.items() if k.startswith('DS4')}}
        started = time.monotonic()
        with (out / 'stdout.log').open('wb') as stdout, (out / 'stderr.log').open('wb') as stderr:
            try:
                row['returncode'] = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=1200).returncode
            except subprocess.TimeoutExpired:
                row['returncode'] = 'timeout'
        row['seconds'] = time.monotonic() - started
        result['commands'].append(row)
        if mode == 'oracle':
            result['oracles'][label] = {f.name: {'bytes': f.stat().st_size, 'sha256': digest(f)} for f in sorted(out.glob('*.f32'))}
        save(result)
    if mode == 'oracle':
        reference = result['oracles']['c']
        for label, _ in binaries[1:]:
            result['comparisons'][label] = {'tensor_count': len(result['oracles'][label]),
                'expected_tensor_count': 49, 'all_full_tensor_bytes_exact': len(reference) == 49 and reference == result['oracles'][label]}
        save(result)
result['pass'] = all(row['returncode'] == 0 for row in result['commands']) and all(row['all_full_tensor_bytes_exact'] for row in result['comparisons'].values())
save(result)
raise SystemExit(0 if result['pass'] else 3)
