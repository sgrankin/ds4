#!/usr/bin/env python3
"""Three retained-history cases; ROOT launches once inside the MAIN guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FILES = ('prefix.txt', 'old-register.txt', 'new-register-code.txt',
         'notes.txt', 'question-code.txt')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--helper', type=Path,
                   default=ROOT / '.workspaces/server-agent-day-20261004/curve-build/oracle')
    p.add_argument('--fixtures', type=Path, default=HERE / 'fixtures')
    p.add_argument('--only-case', nargs='+', choices=['ordinary-cleanup', 'helpful-history', 'obsolete-history'])
    p.add_argument('--greedy', type=int, default=16)
    p.add_argument('--shortened-only', action='store_true')
    p.add_argument('--output', type=Path, default=HERE / 'core')
    args = p.parse_args()
    parent = subprocess.check_output(['ps', '-p', str(os.getppid()), '-o', 'command='], text=True)
    if 'speed-bench/ideas/pressure_run.py' not in parent:
        p.error('launch this loop as the direct child of MAIN speed-bench/ideas/pressure_run.py')
    model = ROOT / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf'
    # As in item14, inspect model metadata only after the guard-parent check.
    st = model.stat()
    env = {k:v for k,v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                             (ROOT / 'ds4_metal.m').read_text()))
    env.update({k:str(ROOT/v) for k,v in shaders.items()})
    env.update(DS4_METAL_V41_SELECTED_TILE='64', DS4_METAL_V41_SELECTED_LIMIT='1024',
               DS4_METAL_V41_RESIDENT_ROUTE_GAP='0')
    helper = args.helper.resolve()
    helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
    cases = json.loads((args.fixtures / 'cases.json').read_text())
    if args.only_case:
        cases = [case for case in cases if case['name'] in args.only_case]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        fixture = args.fixtures / case['name']
        for context in (('shortened',) if args.shortened_only else ('shortened', 'archive-rule')):
            output = args.output / (case['name'] + '-' + context)
            output.mkdir(exist_ok=False)
            archive_ms, archive = 0.0, None
            inputs = fixture
            if context == 'archive-rule':
                started = time.monotonic()
                archive = json.loads((fixture / 'policy.json').read_text())
                archive_ms = (time.monotonic() - started) * 1000
                inputs = output / 'fixture'
                inputs.mkdir()
                for name in FILES:
                    shutil.copy2(fixture / name, inputs / name)
                (inputs / 'new-register-code.txt').write_text(
                    'Current approved shipping policy, retrieved from permanent policy.json: '
                    'compute net units as ' + archive['formula'] +
                    ', then round net down to a whole multiple of ' +
                    str(archive['rounding_multiple']) + '. Status: ' + archive['status'] + '.\n')
            command = [str(helper), '--model', str(model), '--fixtures', str(inputs),
                       '--case', 'code', '--out', str(output), '--prefix', '1024',
                       '--source-edit', '256', '--edit', '128', '--unchanged', '512',
                       '--fresh-tail', '128', '--greedy', str(args.greedy), '--ctx', '4096',
                       '--chunk', '512', '--cache-gb', '64', '--gold-answer', case['gold'],
                       '--stale-answer', case['contrast'], '--require-fresh-query']
            record = {'argv':command, 'cwd':str(ROOT), 'case':case, 'context':context,
                      'settings':{k:v for k,v in env.items() if k.startswith('DS4')},
                      'helper_sha256':helper_sha, 'model_bytes':st.st_size,
                      'model_mtime_ns':st.st_mtime_ns, 'first_attempt':True,
                      'source_generated_answers':0, 'archive_path':str(fixture / 'policy.json'),
                      'greedy_cap':args.greedy, 'shortened_only':args.shortened_only,
                      'archive_record':archive, 'archive_read_ms':archive_ms,
                      'primary_modes':['fresh', 'reuse'] if context == 'shortened' else ['fresh'],
                      'timing_scope':'Core edit timing; source setup/capture separate. Archive read is driver I/O, not a model tool call.'}
            save(output / 'execution.json', record)
            print('Running ' + output.name, flush=True)
            started = time.monotonic()
            try:
                with (output / 'helper.stdout').open('x') as stdout, \
                        (output / 'helper.stderr').open('x') as stderr:
                    completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout,
                                               stderr=stderr, timeout=300)
                record['returncode'] = completed.returncode
            except subprocess.TimeoutExpired:
                record['returncode'] = 124
            record['seconds'] = time.monotonic() - started
            save(output / 'execution.json', record)
            if record['returncode']:
                print(json.dumps({'point':output.name, 'returncode':record['returncode']}), flush=True)
                return record['returncode']
            summary = json.loads((output / 'summary.json').read_text())
            for index in ((0, 1) if context == 'shortened' else (0,)):
                result = summary['results'][index]
                row = {'case':case['name'], 'condition':
                       ('fresh-shortened' if index == 0 else 'reuse-shortened')
                       if context == 'shortened' else 'fresh-explicit-archive-rule',
                       'gold':case['gold'], 'answer':result['greedy'],
                       'correct':result['correct_answer'], 'contrast_answer':result['stale_answer'],
                       'stopped':result['stopped'], 'generated_tokens':result['greedy_tokens'],
                       'greedy_cap':args.greedy,
                       'edit_ms':result['edit_ms'], 'archive_read_ms':archive_ms,
                       'gold_logprob':result['gold_logprob'],
                       'mean_delta_nll_vs_same_context_fresh':
                       summary['sum_delta_nll']/summary['rows'] if context == 'shortened' and index else None,
                       'mean_kl_vs_same_context_fresh':
                       summary['mean_kl'] if context == 'shortened' and index else None,
                       'reused_tokens':summary['reused_tokens'] if index else 0,
                       'configured_fresh_tail_tokens':128,
                       'fresh_evaluated_tokens':summary['edit_tokens'] +
                       (summary['fresh_tail_tokens'] if index else summary['unchanged_tokens']),
                       'source_start':summary['source_start'], 'source_end':summary['source_end'],
                       'destination_start':summary['destination_start'], 'destination_end':summary['destination_end']}
                rows.append(row)
            save(args.output / 'results.json', rows)
            print(json.dumps(rows[-(2 if context == 'shortened' else 1):]), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
