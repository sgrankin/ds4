#!/usr/bin/env python3
"""Separate-process SSD prefill A/B, preserving each run and checking full logits."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('output', type=Path)
p.add_argument('--candidate-env', required=True, help='NAME=VALUE; unset in control')
p.add_argument('--tokens', type=int, default=2048)
p.add_argument('--ctx', type=int)
p.add_argument('--prompt', default='/tmp/ds41-prefill-input.c')
p.add_argument('--vision')
a = p.parse_args()
name, value = a.candidate_env.split('=', 1)
a.output.mkdir(parents=True, exist_ok=False)
reference = None
results = []
for i, variant in enumerate(('control', 'candidate', 'candidate', 'control')):
    dest = a.output / f'{i}-{variant}'
    dest.mkdir()
    env = os.environ.copy()
    env.pop(name, None)
    if variant == 'candidate':
        env[name] = value
    env['DS4_METAL_GRAPH_PREFILL_PROFILE'] = '1'
    env['DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE'] = '1'
    cmd = ['./ds4-bench', '-m', 'gguf/DeepSeek-V4.1-Flash-Q2.gguf',
           '--ssd-streaming', '--prompt-file', a.prompt,
           '--ctx-start', str(a.tokens), '--ctx-max', str(a.tokens),
           '--gen-tokens', '8', '--teacher-forced-decode',
           '--dump-frontier-logits-dir', str(dest), '--csv', str(dest / 'speed.csv')]
    if a.vision:
        cmd += ['--vision', a.vision]
    if a.ctx:
        cmd += ['--ctx-alloc', str(a.ctx)]
    (dest / 'command.json').write_text(json.dumps({'argv': cmd, 'variant': variant,
                                                 'env': {k: v for k, v in env.items() if k.startswith('DS4_')}}, indent=2))
    with (dest / 'run.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    logits = json.loads(next(dest.glob('*.logits.json')).read_text())['logits']
    if reference is None:
        reference = logits
    if logits != reference:
        raise RuntimeError(f'Full vocabulary logits differ: {dest}')
    with (dest / 'speed.csv').open() as f:
        row = next(csv.DictReader(f))
    results.append({'variant': variant, **row})
    print(json.dumps(results[-1]), flush=True)
(a.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
print('All final full-vocabulary logits match exactly.', flush=True)
