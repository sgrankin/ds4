from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

root = Path.cwd()
base = root / '.workspaces/zig-cli-chat-20261022'
out = base / 'real-2'
out.mkdir()
model = root / 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
env.pop('LINENOISE_ASSUME_TTY', None)
rows = []


def run(label, binary, arguments, stdin=None, repl=False):
    command = [str(binary), *map(str, arguments)]
    if repl:
        # Keep the ordinary CLI path, but do not add fixture commands to the
        # user's terminal history. Failed history saves are already harmless.
        history = str(Path.home() / '.ds4_history')
        profile = '(version 1)(allow default)(deny file-write* (literal ' + json.dumps(history) + '))'
        command = ['/usr/bin/sandbox-exec', '-p', profile, *command]
    start = time.monotonic()
    process = subprocess.run(command, input=stdin, env=env, capture_output=True, timeout=300)
    (out / (label + '.stdout')).write_bytes(process.stdout)
    (out / (label + '.stderr')).write_bytes(process.stderr)
    text = process.stdout.decode()
    passed = process.returncode == 0
    if repl:
        passed = passed and 'Thinking mode: 0.' in text and 'PINE-251' in text
    else:
        passed = passed and 'PASS' in text
    row = dict(label=label, command=command, seconds=time.monotonic()-start,
               returncode=process.returncode, passed=passed,
               binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    rows.append(row)
    (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(row), flush=True)
    if not passed:
        raise SystemExit('failed: ' + label)


for arm in ['c', 'native']:
    for mode in ['', '-debug']:
        run(arm + mode + '-fixture', base / (arm + mode) / 'bin/test-deepseek41-cli', [model])

conversation = b'/think 0\nRemember the code PINE-251. Reply only OK.\n/ctx 4096\nWhat is the code? Reply with only the code.\n/exit\n'
(out / 'stdin.txt').write_bytes(conversation)
for arm in ['c', 'native']:
    run(arm + '-repl', base / arm / 'bin/ds4',
        ['--model', model, '--ssd-streaming', '--ssd-streaming-cache-experts', '48GB',
         '--ctx', '8192', '--tokens', '32', '--temp', '0', '--seed', '1',
         '--prefill-chunk', '512', '--mtp-draft', '1'], conversation, repl=True)
