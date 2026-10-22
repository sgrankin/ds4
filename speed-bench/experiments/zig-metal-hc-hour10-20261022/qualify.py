from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import time

root = Path.cwd()
base = root / '.workspaces/zig-metal-hc-20261022'
out = base / 'real'
out.mkdir()
env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
rows = []


def run(label, command):
    start = time.monotonic()
    process = subprocess.run(list(map(str, command)), env=env, capture_output=True, timeout=300)
    (out / (label + '.stdout')).write_bytes(process.stdout)
    (out / (label + '.stderr')).write_bytes(process.stderr)
    row = dict(label=label, command=list(map(str, command)), seconds=time.monotonic()-start,
               returncode=process.returncode,
               binary_sha256=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest())
    rows.append(row)
    (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(dict(label=label, seconds=row['seconds'], returncode=process.returncode)), flush=True)
    if process.returncode:
        raise SystemExit('failed: ' + label)
    return row


for arm in ['c', 'native']:
    run(arm + '-hc-gpu', [base / arm / 'bin/test-deepseek41-metal', '--hc-scaled'])

# This changes only the HC controller. Both binaries already contain the
# preceding native helpers and the qualified native F16 publication controller.
binaries = {
    'c-controller': root / '.workspaces/zig-metal-publication-20261022/native/bin/ds4-bench',
    'zig-controller': base / 'native/bin/ds4-bench',
}
model = root / 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
prompt = root / '.workspaces/zig-engram-20261022/bench/prompt.txt'
reference = None
for index, arm in enumerate(['c-controller', 'zig-controller', 'zig-controller', 'c-controller']):
    directory = out / f'{index}-{arm}'
    directory.mkdir()
    (directory / 'logits').mkdir()
    row = run(f'{index}-{arm}', [binaries[arm], '--model', model, '--metal', '--ssd-streaming',
        '--ssd-streaming-cache-experts', '48GB', '--prompt-file', prompt,
        '--ctx-start', '512', '--ctx-max', '512', '--ctx-alloc', '768',
        '--prefill-chunk', '512', '--gen-tokens', '64', '--teacher-forced-decode',
        '--csv', directory / 'metrics.csv', '--dump-frontier-logits-dir', directory / 'logits'])
    logits = json.loads(next((directory / 'logits').glob('*.json')).read_text())['logits']
    if reference is None:
        reference = logits
    row['identical_logits'] = logits == reference
    row['metrics'] = list(csv.DictReader((directory / 'metrics.csv').open()))
    (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    if not row['identical_logits']:
        raise SystemExit('HC model logit mismatch: ' + arm)
