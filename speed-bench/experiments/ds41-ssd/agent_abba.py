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
choice.add_argument('--candidate-env', action='append', help='NAME=VALUE; repeat to test a combination')
choice.add_argument('--candidate-prefill-chunk', type=int)
p.add_argument('--gen-tokens',type=int,default=1)
p.add_argument('--prompts-json',type=Path)
p.add_argument('--cache-gb',type=int)
p.add_argument('--model',type=Path,default=Path('ds4flash.gguf'))
a = p.parse_args()
candidate = dict(item.split('=', 1) for item in (a.candidate_env or []))
a.output.mkdir(parents=True, exist_ok=False)
binary = a.output.resolve() / 'ds4-agent'
shutil.copy2('./ds4-agent', binary)
shaders = snapshot_shaders(a.output)
results = []
reference = None
for i, variant in enumerate(('control', 'candidate', 'candidate', 'control')):
    dest = a.output.resolve() / f'{i}-{variant}'
    env = os.environ.copy()
    for name in candidate: env.pop(name, None)
    env.update(shaders)
    cmd = [sys.executable, str(Path(__file__).with_name('agent_probe.py')),
           str(dest), '--binary', str(binary), '--model', str(a.model.resolve()), '--gen-tokens', str(a.gen_tokens)]
    if a.prompts_json: cmd += ['--prompts-json', str(a.prompts_json.resolve())]
    if a.cache_gb: cmd += ['--cache-gb', str(a.cache_gb)]
    if variant == 'candidate':
        for name, value in candidate.items():
            cmd += ['--env', f'{name}={value}']
    if variant == 'candidate' and a.candidate_prefill_chunk:
        cmd += ['--prefill-chunk', str(a.candidate_prefill_chunk)]
    subprocess.run(cmd, env=env, check=True)
    runs=json.loads((dest/'summary.json').read_text())
    identity={r['mode']:[(g['input_ids_sha256'],g['output_ids_sha256'],g['generated'])
                         for g in r['generations']] for r in runs}
    if reference is None: reference=identity
    if identity != reference:
        raise RuntimeError(f'Agent input or generated tokens differ: {dest}')
    results.append(dict(variant=variant,runs=runs))
    (a.output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
