#!/usr/bin/env python3
"""Balanced actual-agent trials, including both fresh and restored system KV."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from snapshot_shaders import snapshot_shaders

p = argparse.ArgumentParser()
p.add_argument('output', type=Path)
choice = p.add_mutually_exclusive_group(required=True)
choice.add_argument('--candidate-env', help='NAME=VALUE')
choice.add_argument('--candidate-prefill-chunk', type=int)
a = p.parse_args()
name, value = a.candidate_env.split('=', 1) if a.candidate_env else (None, None)
a.output.mkdir(parents=True, exist_ok=False)
binary = a.output.resolve() / 'ds4-agent'
shutil.copy2('./ds4-agent', binary)
shaders = snapshot_shaders(a.output)
results = []
for i, variant in enumerate(('control', 'candidate', 'candidate', 'control')):
    dest = a.output.resolve() / f'{i}-{variant}'
    env = os.environ.copy()
    if name: env.pop(name, None)
    env.update(shaders)
    cmd = [sys.executable, str(Path(__file__).with_name('agent_probe.py')),
           str(dest), '--binary', str(binary)]
    if variant == 'candidate' and name:
        cmd += ['--env', f'{name}={value}']
    if variant == 'candidate' and a.candidate_prefill_chunk:
        cmd += ['--prefill-chunk', str(a.candidate_prefill_chunk)]
    subprocess.run(cmd, env=env, check=True)
    results.append(dict(variant=variant,
                        runs=json.loads((dest/'summary.json').read_text())))
    (a.output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
