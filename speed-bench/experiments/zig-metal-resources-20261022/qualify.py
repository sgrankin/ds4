#!/usr/bin/env python3
"""Fixed production-linked resource/GPU checks; ROOT runs serially under its guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-bin-dir')
    parser.add_argument('--native-bin-dir')
    parser.add_argument('--debug-bin-dir')
    parser.add_argument('--out', required=True)
    parser.add_argument('--copy-only', action='store_true')
    parser.add_argument('--mode', choices=['default', 'row', 'session', 'batch', 'big'])
    args = parser.parse_args()
    arms = [(name, Path(path).resolve()) for name, path in [
        ('control', args.control_bin_dir), ('native', args.native_bin_dir),
        ('native-debug', args.debug_bin_dir)] if path]
    if not arms:
        parser.error('supply at least one binary directory')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('DS4', 'PT_'))}
    shader_sources = re.findall(
        r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
        (ROOT / 'ds4_metal.m').read_text())
    assert shader_sources, 'Metal source table not found'
    env.update({key: str(ROOT / source) for key, source in shader_sources})
    report = {'status': 'running', 'cwd': str(ROOT), 'rows': [],
              'shader_environment': {key: env[key] for key, _ in shader_sources},
              'first_failure': None}

    def save():
        (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(label, command, settings):
        command = list(map(str, command))
        row = {'label': label, 'command': command, 'environment': settings,
               'stdout': str(out / (label + '.stdout.log')),
               'stderr': str(out / (label + '.stderr.log'))}
        started = time.monotonic()
        try:
            with Path(command[0]).open('rb') as binary:
                row['binary_sha256'] = hashlib.file_digest(binary, 'sha256').hexdigest()
            with Path(row['stdout']).open('wb') as stdout, Path(row['stderr']).open('wb') as stderr:
                result = subprocess.run(command, cwd=ROOT, env=env | settings,
                                        stdout=stdout, stderr=stderr, timeout=600)
            row['returncode'] = result.returncode
            row['pass'] = result.returncode == 0
        except Exception as error:
            row.update(returncode=None, **{'pass': False},
                       error=f'{type(error).__name__}: {error}')
        row['seconds'] = time.monotonic() - started
        report['rows'].append(row)
        if not row['pass']:
            report['status'] = 'failed'
            report['first_failure'] = label
        save()
        print(json.dumps(row), flush=True)
        return row['pass']

    save()
    for arm, directory in arms:
        modes = ['default', 'big'] if arm == 'native-debug' else ['default', 'row', 'session', 'batch', 'big']
        if args.mode:
            modes = [args.mode]
        for mode in modes:
            command = [directory / 'test-metal-command-memory']
            if mode != 'default':
                command.append(mode)
            if not run(f'{arm}-command-memory-{mode}', command, {'MTL_DEBUG_LAYER': '1'}):
                return 1
    if not args.copy_only:
        for arm, directory in arms:
            if not run(arm + '-math-safe-full', [directory / 'test-deepseek41-metal'],
                       {'DS4_METAL_MATH_SAFE': '1'}):
                return 1
    report['status'] = 'complete'
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
