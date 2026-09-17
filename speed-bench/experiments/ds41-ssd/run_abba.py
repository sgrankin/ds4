#!/usr/bin/env python3
"""Separate-process SSD prefill A/B, preserving each run and checking full logits."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
from snapshot_shaders import snapshot_shaders

p = argparse.ArgumentParser()
p.add_argument('output', type=Path)
choice = p.add_mutually_exclusive_group(required=True)
choice.add_argument('--candidate-env', help='NAME=VALUE; unset in control')
choice.add_argument('--candidate-prefill-chunk', type=int)
p.add_argument('--tokens', type=int, default=2048)
p.add_argument('--ctx', type=int)
p.add_argument('--initial-tokens', type=int)
p.add_argument('--gen-tokens', type=int, default=8)
p.add_argument('--prompt', default='/tmp/ds41-prefill-input.c')
p.add_argument('--vision')
p.add_argument('--candidate-vision', help='load this encoder only in candidate runs')
a = p.parse_args()
if a.vision and a.candidate_vision:
    p.error('--vision and --candidate-vision are mutually exclusive')
if a.initial_tokens is not None and not 0 < a.initial_tokens < a.tokens:
    p.error('--initial-tokens must be positive and less than --tokens')
if a.candidate_env and '=' not in a.candidate_env:
    p.error('--candidate-env must be NAME=VALUE')
name, value = a.candidate_env.split('=', 1) if a.candidate_env else (None, None)
if a.candidate_env and not name:
    p.error('--candidate-env requires a nonempty name')
a.output.mkdir(parents=True, exist_ok=False)
# Every subprocess must use the same binary even if development continues.
binary = a.output.resolve() / 'ds4-bench'
shutil.copy2('./ds4-bench', binary)
shaders = snapshot_shaders(a.output)
reference = None
results = []
for i, variant in enumerate(('control', 'candidate', 'candidate', 'control')):
    dest = a.output / f'{i}-{variant}'
    dest.mkdir()
    env = os.environ.copy()
    if name: env.pop(name, None)
    env.update(shaders)
    if variant == 'candidate' and name:
        env[name] = value
    env['DS4_METAL_GRAPH_PREFILL_PROFILE'] = '1'
    env['DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE'] = '1'
    cmd = [str(binary), '-m', 'gguf/DeepSeek-V4.1-Flash-Q2.gguf',
           '--ssd-streaming', '--prompt-file', a.prompt,
           '--ctx-start', str(a.tokens), '--ctx-max', str(a.tokens),
           '--gen-tokens', str(a.gen_tokens), '--teacher-forced-decode',
           '--dump-frontier-logits-dir', str(dest), '--csv', str(dest / 'speed.csv')]
    if a.vision:
        cmd += ['--vision', a.vision]
    if a.candidate_vision and variant == 'candidate':
        cmd += ['--vision', a.candidate_vision]
    if a.ctx:
        cmd += ['--ctx-alloc', str(a.ctx)]
    if variant == 'candidate' and a.candidate_prefill_chunk:
        cmd += ['--prefill-chunk', str(a.candidate_prefill_chunk)]
    if a.initial_tokens:
        cmd += ['--ctx-start', str(a.initial_tokens), '--step-incr',
                str(a.tokens - a.initial_tokens), '--gen-tokens', '0']
    (dest / 'command.json').write_text(json.dumps({'argv': cmd, 'variant': variant,
                                                 'env': {k: v for k, v in env.items() if k.startswith('DS4_')}}, indent=2))
    with (dest / 'run.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    logits = {f.name: json.loads(f.read_text())['logits']
              for f in sorted(dest.glob('*.logits.json'))}
    if reference is None:
        reference = logits
    if logits != reference:
        raise RuntimeError(f'Full vocabulary logits differ: {dest}')
    with (dest / 'speed.csv').open() as f:
        row = list(csv.DictReader(f))[-1]
    results.append({'variant': variant, **row})
    print(json.dumps(results[-1]), flush=True)
(a.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
print('All final full-vocabulary logits match exactly.', flush=True)
