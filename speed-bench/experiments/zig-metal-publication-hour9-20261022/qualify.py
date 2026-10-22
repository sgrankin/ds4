from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import time

root = Path.cwd()
base = root / '.workspaces/zig-metal-publication-20261022'
out = base / 'real'
out.mkdir()
env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
rows = []


def run(label, command, extra=None):
    child_env = env | (extra or {})
    start = time.monotonic()
    process = subprocess.run(list(map(str, command)), env=child_env, capture_output=True, timeout=300)
    (out / (label + '.stdout')).write_bytes(process.stdout)
    (out / (label + '.stderr')).write_bytes(process.stderr)
    row = dict(label=label, command=list(map(str, command)), environment=extra or {},
               seconds=time.monotonic()-start, returncode=process.returncode,
               binary_sha256=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest())
    rows.append(row)
    (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(row), flush=True)
    if process.returncode:
        raise SystemExit('failed: ' + label)
    return row


for arm in ['c', 'native']:
    binary = base / arm / 'bin/test-deepseek41-metal'
    run(arm + '-packed-copy', [binary, '--compressed-kv-f16'])
run('native-generic-copy', [base / 'native/bin/test-deepseek41-metal', '--compressed-kv-f16'],
    {'DS4_METAL_DISABLE_CONTIG_F32_F16_COPY': '1'})
run('native-full-metal', [base / 'native/bin/test-deepseek41-metal'])

# Both model arms already use the preceding native host/tokenizer modules.
# Only the publication controller changes between these two fixed binaries.
model = root / 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
prompt = root / '.workspaces/zig-engram-20261022/bench/prompt.txt'
reference = None
for arm, binary in [
    ('c-controller', root / '.workspaces/zig-tokenizer-20261022/native/bin/ds4-bench'),
    ('zig-controller', base / 'native/bin/ds4-bench'),
]:
    directory = out / arm
    directory.mkdir()
    (directory / 'logits').mkdir()
    command = [binary, '--model', model, '--metal', '--ssd-streaming',
               '--ssd-streaming-cache-experts', '48GB', '--prompt-file', prompt,
               '--ctx-start', '512', '--ctx-max', '512', '--ctx-alloc', '768',
               '--prefill-chunk', '512', '--gen-tokens', '16', '--teacher-forced-decode',
               '--csv', directory / 'metrics.csv', '--dump-frontier-logits-dir', directory / 'logits']
    row = run(arm, command, {'DS4_V41_COMPRESSED_KV_F16': '1'})
    logits = json.loads(next((directory / 'logits').glob('*.json')).read_text())['logits']
    if reference is None:
        reference = logits
    row['identical_logits'] = logits == reference
    row['max_abs_logit_delta'] = max(abs(a-b) for a, b in zip(reference, logits))
    row['metrics'] = list(csv.DictReader((directory / 'metrics.csv').open()))
    stderr = (out / (arm + '.stderr')).read_text()
    row['f16_storage_confirmed'] = 'KV allocated main=f16' in stderr
    (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    if not row['identical_logits'] or not row['f16_storage_confirmed']:
        raise SystemExit('publication mismatch: ' + arm)
