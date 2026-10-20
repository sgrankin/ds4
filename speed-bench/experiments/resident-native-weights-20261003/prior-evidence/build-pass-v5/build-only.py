"""Execute only the eight granted build/self-test commands, strictly in order."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
EXPECTED = 'f798a1a977f61b7fc7a0f023da452e91f41a09516614e097e361e6bdd925638f'
WORK = Path('/Volumes/Models/ds4/.workspaces/idea-resident-native-weights-next')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify(plan):
    if sha(HERE / 'build-plan.json') != EXPECTED:
        raise RuntimeError('changed plan authority')
    for name, expected in plan['files'].items():
        if sha(name) != expected:
            raise RuntimeError('changed pinned asset: ' + name)
    version = subprocess.check_output(['/usr/bin/cc', '--version'], text=True)
    if version != plan['compiler_version']:
        raise RuntimeError('compiler version changed')
    match = re.search(r'^InstalledDir: (.+)$', version, re.M)
    if not match or str((Path(match[1]) / 'clang').resolve()) != plan['selected_compiler']['path']:
        raise RuntimeError('selected compiler path changed')
    sdk = subprocess.check_output(['/usr/bin/xcrun', '--show-sdk-path'], text=True).strip()
    if sdk != plan['selected_sdk']['path'] or str(Path(sdk).resolve()) != plan['selected_sdk']['resolved_path']:
        raise RuntimeError('selected SDK path changed')
    return {'asset_count': len(plan['files']), 'compiler_version': version,
            'selected_compiler': plan['selected_compiler'], 'selected_sdk': plan['selected_sdk'],
            'verified_at_unix': time.time()}


def main():
    if sha(HERE / 'build-plan.json') != EXPECTED:
        raise RuntimeError('plan SHA mismatch')
    plan = json.loads((HERE / 'build-plan.json').read_text())
    commands = plan['commands_serialized'] + plan['optional_builds']
    if len(commands) != 8 or len(plan['commands_serialized']) != 6:
        raise RuntimeError('unexpected grant shape')
    for i, command in enumerate(commands):
        args = command['argv']
        if i in (2, 5):
            if len(args) != 2 or args[1] != '--self-test' or command['output'] is not None:
                raise RuntimeError('unexpected model-free test command')
        elif args[0] != '/usr/bin/cc' or not command['output'] or args[-2:] != ['-o', command['output']]:
            raise RuntimeError('unexpected compiler command')
    verify(plan)
    if any(Path(c['output']).exists() for c in commands if c['output']):
        raise RuntimeError('compiler output already exists')
    records = []
    for i, command in enumerate(commands):
        for old in records:
            if old['output'] and sha(old['output']) != old['output_sha256']:
                raise RuntimeError('produced dependency changed')
        before = verify(plan)
        if command['output'] and Path(command['output']).exists():
            raise RuntimeError('nonexclusive compiler output')
        print(json.dumps({'step': i, 'state': 'started', 'purpose': command['purpose'], 'argv': command['argv']}), flush=True)
        log = HERE / f'build-step-{i}.log'
        started = time.time()
        with log.open('xb') as f:
            result = subprocess.run(command['argv'], cwd=WORK, stdout=f, stderr=subprocess.STDOUT)
        record = {'step': i, 'command': command['argv'], 'purpose': command['purpose'],
                  'started_unix': started, 'ended_unix': time.time(), 'returncode': result.returncode,
                  'log': str(log), 'log_sha256': sha(log), 'output': command['output'],
                  'output_sha256': sha(command['output']) if command['output'] and Path(command['output']).exists() else None,
                  'before': before, 'model_opened': False, 'metal_compiled': False}
        (HERE / f'build-step-{i}.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'step': i, 'state': 'finished', 'returncode': result.returncode,
                          'output_sha256': record['output_sha256'], 'log': str(log)}), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
        record['after'] = verify(plan)
        (HERE / f'build-step-{i}.json').write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
    (HERE / 'build-result.json').write_text(json.dumps({'status': 'complete', 'steps': records,
        'plan_sha256': EXPECTED, 'execution_source_sha256': sha(__file__), 'concurrency': 1,
        'model_open_or_stat': False, 'tokenizer_called': False, 'metal_compiled': False}, indent=2) + '\n')
    print('All eight granted serial build/self-test steps completed.', flush=True)


if __name__ == '__main__':
    main()
