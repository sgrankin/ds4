from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

root = Path.cwd()
base = root / '.workspaces/zig-tokenizer-20261022'
out = base / 'real'
out.mkdir()
corpus = out / 'corpus'
corpus.mkdir()
fixture = json.loads((root / '.workspaces/zig-restart-20261022/c-to-zig-48/fixtures/fixture.json').read_text())
texts = {
    'ordinary-code': 'Explain the boundary and preserve ownership.\n' + Path('zig/cli_chat.zig').read_text(),
    'persistent-context': fixture['prompt'],
    'tool-output': ''.join(f'line {i}: table[{i % 32}] = item.value + offset; /* preserve borrowed keys */\n' for i in range(1200)),
    'long-piece': 'abacabadabacaba' * 128,
    'multilingual': ('Caffè, café, naïve. 日本語と中文。 Ελληνικά русский 한국어 العربية.\n' * 32),
    'rendered-turns': '<｜begin▁of▁sentence｜>System facts remain checkable.\n<｜User｜>Read the file.\n<｜Assistant｜><think>Inspect it.</think>\n<tool_result>int main(void) { return 0; }</tool_result>\n<｜User｜>Keep the code PINE-251.\n',
}
files = []
for label, content in texts.items():
    path = corpus / (label + '.txt')
    path.write_text(content)
    files.append(path)
(corpus / 'manifest.json').write_text(json.dumps([
    dict(label=label, path=str(path), bytes=path.stat().st_size,
         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for label, path in zip(texts, files)
], indent=2) + '\n')
model = root / 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
rows = []
reference = None
env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
for index, arm in enumerate(['c', 'native', 'native', 'c']):
    binary = base / arm / 'bin/test-deepseek41-gguf'
    command = [str(binary), '--tokenizer-bench', str(model), '20', *map(str, files)]
    start = time.monotonic()
    process = subprocess.run(command, env=env, capture_output=True, timeout=300)
    label = f'{index}-{arm}'
    (out / (label + '.stdout')).write_bytes(process.stdout)
    (out / (label + '.stderr')).write_bytes(process.stderr)
    if process.returncode:
        raise SystemExit('failed: ' + label)
    records = [json.loads(line) for line in process.stdout.splitlines()]
    ids = [record['ids'] for record in records[1:]]
    if reference is None:
        reference = ids
    identical = ids == reference
    for record in records[1:]:
        record.pop('ids')
        record['label'] = list(texts)[record['case']]
    row = dict(arm=arm, command=command, seconds=time.monotonic()-start,
               identical_ids=identical, records=records,
               binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    rows.append(row)
    (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(dict(arm=arm, seconds=row['seconds'], identical_ids=identical)), flush=True)
    if not identical:
        raise SystemExit('token ID mismatch: ' + label)
