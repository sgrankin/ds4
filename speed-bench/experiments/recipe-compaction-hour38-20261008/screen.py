#!/usr/bin/env python3
"""One fixed suffix-compaction screen; run under pressure_run.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRACTICAL = ROOT / 'speed-bench/experiments/fresh-clm-coding-hour32-20261004/practical'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--case', choices=('ordinary-summary', 'equivalent-instructions',
                   'irrelevant-planning-cleanup', 'changed-fact-stress'), required=True)
    p.add_argument('--fresh-tail', type=int, choices=(64, 128, 256), default=64)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    fixture = HERE / 'fixtures' / a.case
    details = json.loads((fixture / 'fixture.json').read_text())
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                              (ROOT / 'ds4_metal.m').read_text()))
    env.update({k: str(ROOT / v) for k, v in shaders.items()})
    env.update(DS4_METAL_V41_RESIDENT_ROUTE_GAP='0', DS4_METAL_V41_RESIDENT_PREFILL_GAP='0',
               DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS='0',
               DS4_METAL_V41_SELECTED_TILE='64', DS4_METAL_V41_SELECTED_LIMIT='1024')
    command = [str(a.oracle.resolve()), '--model', str(ROOT / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf'),
               '--fixtures', str(fixture), '--case', a.case, '--out', str(a.output.resolve()),
               '--prefix', '256', '--source-edit', '1024', '--edit', '256',
               '--unchanged', '1536', '--fresh-tail', str(a.fresh_tail), '--teacher', '512',
               '--greedy', '512', '--ctx', '8192', '--chunk', '512', '--cache-gb', '64']
    if details['system_edit']:
        command.append('--system-edit')
    save(a.output / 'execution.json', {'command': command, 'cwd': str(ROOT),
         'environment': {k: v for k, v in env.items() if k.startswith('DS4')},
         'oracle_sha256': hashlib.sha256(a.oracle.read_bytes()).hexdigest(),
         'runtime_revision': '9c6918f0ae2e', 'fixture': details})
    with (a.output / 'stdout.log').open('wb') as out, (a.output / 'stderr.log').open('wb') as err:
        rc = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err).returncode
    grades = {}
    if rc == 0:
        for mode in ('fresh', 'reuse'):
            command = [sys.executable, '-B', str(PRACTICAL / 'grade.py'),
                       details['practical_case'], str((a.output / (mode + '-code.py')).resolve())]
            grade = subprocess.run(command, cwd=PRACTICAL, capture_output=True, timeout=4)
            (a.output / (mode + '-grade.stdout')).write_bytes(grade.stdout)
            (a.output / (mode + '-grade.stderr')).write_bytes(grade.stderr)
            grades[mode] = {'returncode': grade.returncode, 'grade': json.loads(grade.stdout)}
        save(a.output / 'grades.json', grades)
    save(a.output / 'result.json', {'returncode': rc, 'grades': grades})
    print(json.dumps({'case': a.case, 'fresh_tail': a.fresh_tail, 'returncode': rc,
                      'grades': grades}), flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
