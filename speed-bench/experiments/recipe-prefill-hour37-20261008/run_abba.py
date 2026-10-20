#!/usr/bin/env python3
"""Reused SSD prefill ABBA with fixed cache and descriptive frontier differences."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'speed-bench/experiments/ds41-ssd'))
from snapshot_shaders import snapshot_shaders


def log_distribution(logits):
    maximum = max(logits)
    shifted = [x - maximum for x in logits]
    normalizer = math.log(math.fsum(math.exp(x) for x in shifted))
    return [x - normalizer for x in shifted]


def compare(reference, candidate):
    if len(reference) != len(candidate) or not reference:
        raise ValueError('frontier vocabulary size changed or is empty')
    if not all(math.isfinite(x) for x in (*reference, *candidate)):
        raise ValueError('nonfinite frontier logits')
    before, after = log_distribution(reference), log_distribution(candidate)
    before_p, after_p = [math.exp(x) for x in before], [math.exp(x) for x in after]
    before_top = max(range(len(reference)), key=reference.__getitem__)
    after_top = max(range(len(candidate)), key=candidate.__getitem__)
    return dict(exact=reference == candidate,
        kl_reference_to_candidate=math.fsum(p * (a - b) for p, a, b in zip(before_p, before, after)),
        total_variation=.5 * math.fsum(abs(a - b) for a, b in zip(before_p, after_p)),
        top1_reference=before_top, top1_candidate=after_top,
        top1_agrees=before_top == after_top,
        max_abs_logit_difference=max(abs(a - b) for a, b in zip(reference, candidate)))


p = argparse.ArgumentParser()
p.add_argument('output', type=Path)
p.add_argument('--candidate-env', required=True, help='NAME=VALUE; absent in control')
p.add_argument('--max-subs', type=int, choices=(1, 3), default=1)
p.add_argument('--tokens', type=int, default=2048)
p.add_argument('--ctx', type=int, default=8192)
p.add_argument('--initial-tokens', type=int)
p.add_argument('--gen-tokens', type=int, default=8)
p.add_argument('--prompt', default=str(ROOT / 'ds4.c'))
a = p.parse_args()
if a.initial_tokens is not None and not 0 < a.initial_tokens < a.tokens:
    p.error('--initial-tokens must be positive and less than --tokens')
if '=' not in a.candidate_env:
    p.error('--candidate-env must be NAME=VALUE')
name, value = a.candidate_env.split('=', 1)
if not name:
    p.error('--candidate-env requires a nonempty name')
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
binary = a.output / 'ds4-bench'
shutil.copy2(ROOT / 'ds4-bench', binary)
shaders = snapshot_shaders(a.output)
reference = None
results = []
for i, variant in enumerate(('control', 'candidate', 'candidate', 'control')):
    dest = a.output / f'{i}-{variant}'
    dest.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
    env.update(shaders)
    env.update(DS4_METAL_V41_SELECTED_TILE='64', DS4_METAL_V41_SELECTED_LIMIT='1024',
        DS4_METAL_V41_RESIDENT_ROUTE_GAP='0', DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS='0',
        DS4_METAL_V41_RESIDENT_PREFILL_GAP='0', DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS='.25',
        DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK='64', DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS=str(a.max_subs),
        DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS='.50', DS4_METAL_V41_RESIDENT_ROUTE_POLICY='replace',
        DS4_METAL_GRAPH_PREFILL_PROFILE='1', DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE='1')
    if variant == 'candidate':
        env[name] = value
    cmd = [str(binary), '-m', str(ROOT / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf'),
        '--ssd-streaming', '--ssd-streaming-cache-experts', '64GB', '--prefill-chunk', '512',
        '--prompt-file', str(Path(a.prompt).resolve()), '--ctx-start', str(a.tokens),
        '--ctx-max', str(a.tokens), '--gen-tokens', str(a.gen_tokens), '--teacher-forced-decode',
        '--ctx-alloc', str(a.ctx), '--dump-frontier-logits-dir', str(dest), '--csv', str(dest / 'speed.csv')]
    if a.initial_tokens:
        cmd += ['--ctx-start', str(a.initial_tokens), '--step-incr',
            str(a.tokens - a.initial_tokens), '--gen-tokens', '0']
    (dest / 'command.json').write_text(json.dumps(dict(argv=cmd, variant=variant, max_substitutions=a.max_subs,
        env={k: v for k, v in env.items() if k.startswith('DS4_')}), indent=2) + '\n')
    with (dest / 'run.log').open('w') as log:
        subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    logits = {f.name: json.loads(f.read_text())['logits'] for f in sorted(dest.glob('*.logits.json'))}
    if not logits:
        raise RuntimeError(f'No frontier logits: {dest}')
    if reference is None:
        reference = logits
    if logits.keys() != reference.keys():
        raise RuntimeError(f'Frontier boundaries changed: {dest}')
    differences = {frontier: compare(reference[frontier], row) for frontier, row in logits.items()}
    (dest / 'frontier-differences.json').write_text(json.dumps(differences, indent=2) + '\n')
    with (dest / 'speed.csv').open() as f:
        rows = list(csv.DictReader(f))
    result = dict(variant=variant, max_substitutions=a.max_subs, **rows[-1], phases=rows, frontiers=differences)
    results.append(result)
    (a.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'phases'}), flush=True)
print('ABBA complete; frontier differences retained as measurements.', flush=True)
