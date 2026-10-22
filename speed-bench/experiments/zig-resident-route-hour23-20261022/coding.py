"""C/native/C resident-policy qualification using the unchanged Hour28 caller."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c-server', type=Path, required=True)
    parser.add_argument('--native-server', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for label, binary in [('c-open', args.c_server), ('native', args.native_server),
                          ('c-close', args.c_server)]:
        binary = binary.resolve()
        arm = args.output / label
        command = [sys.executable, '-B', str(HERE/'coding_launch.py'),
            '--arm', 'subs3', '--server', str(binary), '--output', str(arm),
            '--cache', str(args.output/(label+'-cache')),
            '--store', str(args.output/(label+'-store')),
            '--visible', str(args.output/'visible'), '--timeout', '900']
        started = time.monotonic()
        with (args.output/(label+'.stdout')).open('wb') as stdout, \
                (args.output/(label+'.stderr')).open('wb') as stderr:
            process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                     timeout=930)
        row = dict(arm=label, command=command, server=str(binary),
            server_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            returncode=process.returncode, seconds=time.monotonic()-started,
            tasks=[], route_substitutions=0)
        for case in ('ledger', 'lru'):
            result_path = arm/'http'/case/case/'result.json'
            task = json.loads(result_path.read_text()) if result_path.exists() else None
            log_path = arm/(case+'-server.stderr.log')
            lines = log_path.read_text(errors='replace').splitlines() if log_path.exists() else []
            route_lines = [line for line in lines if 'ds4: resident-route ' in line]
            substitutions = sum(int(re.search(r'\bsubstitutions=(\d+)', line)[1])
                                for line in route_lines)
            row['route_substitutions'] += substitutions
            code_path = Path(task['first_code_path']) if task and task.get('first_code_path') else None
            entry = dict(case=case, result=str(result_path), route_substitutions=substitutions,
                route_lines=route_lines,
                native_weight_lines=[line for line in lines if 'resident-route-native-weights' in line],
                prefill_lines=[line for line in lines if 'ds4: resident-prefill ' in line],
                pass_quality=bool(task and task['pass']),
                first_code_sha256=hashlib.sha256(code_path.read_bytes()).hexdigest()
                    if code_path and code_path.exists() else None)
            if task:
                entry.update({key: task[key] for key in ('grade', 'complete', 'cap',
                    'actual_read_count', 'actual_write_count', 'config_unchanged',
                    'final_text', 'http_output_tokens', 'http_reasoning_tokens', 'wall_seconds')})
            row['tasks'].append(entry)
        row['route_active'] = row['route_substitutions'] > 0
        row['pass'] = (row['returncode'] == 0 and row['route_active'] and
                       all(task['pass_quality'] for task in row['tasks']))
        results.append(row)
        save(args.output/'results.json', results)
        print(json.dumps(row), flush=True)
    comparisons = []
    for index, case in enumerate(('ledger', 'lru')):
        first, native, closing = [row['tasks'][index] for row in results]
        comparisons.append(dict(case=case,
            native_first_code_equals_c_open=native['first_code_sha256'] is not None and
                native['first_code_sha256'] == first['first_code_sha256'],
            c_first_code_repeats=first['first_code_sha256'] is not None and
                first['first_code_sha256'] == closing['first_code_sha256'],
            native_final_equals_c_open=native.get('final_text') == first.get('final_text'),
            c_final_repeats=first.get('final_text') == closing.get('final_text')))
    save(args.output/'comparison.json', comparisons)
    return 0 if all(row['pass'] for row in results) else 3


if __name__ == '__main__':
    raise SystemExit(main())
