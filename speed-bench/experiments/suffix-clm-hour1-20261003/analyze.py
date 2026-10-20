"""Summarize completed runs and draw the requested tradeoff curves."""
import csv
import json
import math
import os
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / 'plot-config'))
os.environ.setdefault('XDG_CACHE_HOME', str(HERE / 'plot-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rows = []
histories = {}
for execution in sorted(HERE.glob('*-tail*/execution.json')):
    record = json.loads(execution.read_text())
    if record['returncode'] != 0:
        continue
    directory = execution.parent
    summary = json.loads((directory / 'summary.json').read_text())
    tokens = json.loads((directory / 'tokens.json').read_text())
    family = record['family']
    history = (tokens['old'], tokens['edited'])
    if family in histories:
        assert histories[family] == history, 'Tail sweep changed its final inputs'
    histories[family] = history
    with (directory / 'logits.bin').open('rb') as raw:
        magic, vocabulary, count, length = struct.unpack('<4I', raw.read(16))
        assert magic == 0x53554632 and length == len(tokens['edited'])
        targets = np.frombuffer(raw.read(count * 4), dtype='<i4')
        logits = np.frombuffer(raw.read(), dtype='<f4').astype(np.float64).reshape(2, count, vocabulary)
    assert np.isfinite(logits).all()
    shifted = logits - logits.max(axis=2, keepdims=True)
    log_probability = shifted - np.log(np.exp(shifted).sum(axis=2, keepdims=True))
    delta = log_probability[0, np.arange(count), targets] - log_probability[1, np.arange(count), targets]
    kl = (np.exp(log_probability[0]) * (log_probability[0] - log_probability[1])).sum(axis=1)
    assert math.isclose(float(delta.sum()), summary['sum_delta_nll'], abs_tol=1e-8)
    assert math.isclose(float(kl.mean()), summary['mean_kl'], abs_tol=1e-8)
    fresh, reuse = summary['results']
    rows.append({'family': family, 'fresh_tail': record['fresh_tail'],
                 'fresh_edit_ms': fresh['edit_ms'], 'reuse_edit_ms': reuse['edit_ms'],
                 'speedup': fresh['edit_ms'] / reuse['edit_ms'],
                 'latency_ratio': reuse['edit_ms'] / fresh['edit_ms'],
                 'mean_delta_nll': float(delta.mean()), 'perplexity_ratio': math.exp(float(delta.mean())),
                 'mean_kl': float(kl.mean()), 'max_row_delta_nll': float(delta.max()),
                 'fresh_answer': fresh['greedy'], 'reuse_answer': reuse['greedy'],
                 'fresh_correct': fresh['correct_answer'], 'reuse_correct': reuse['correct_answer'],
                 'fresh_alternative_probability': math.exp(fresh['stale_logprob']),
                 'reuse_alternative_probability': math.exp(reuse['stale_logprob']),
                 'teacher_rows_including_eos': count, 'whole_run_seconds': record['seconds']})

if not rows:
    raise SystemExit('No completed suffix runs yet')
with (HERE / 'results.csv').open('w', newline='') as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

families = ['ordinary-deletion', 'best-self-contained', 'stress-changed-fact',
            'ordinary-arithmetic', 'ordinary-python', 'stress-removed-fact']
fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
for ax, family in zip(axes.flat, families):
    points = sorted((r for r in rows if r['family'] == family), key=lambda r: r['fresh_tail'])
    ax.scatter([1], [0], marker='*', s=100, color='black', label='Fully fresh reference')
    if points:
        ax.plot([r['latency_ratio'] for r in points], [r['mean_delta_nll'] for r in points], color='#2171b5')
        for point in points:
            color = '#238b45' if point['reuse_correct'] else '#cb181d'
            ax.scatter(point['latency_ratio'], point['mean_delta_nll'], color=color, s=45)
            ax.annotate(f"{point['fresh_tail']} fresh", (point['latency_ratio'], point['mean_delta_nll']),
                        xytext=(5, 7), textcoords='offset points', fontsize=9)
    ax.set_title(family.replace('-', ' '))
    if family == 'stress-removed-fact':
        ax.axhline(math.log(1.2), color='#b35806', linestyle='--', linewidth=1,
                   label='20% perplexity target')
        ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('Edit time / paired fresh edit time (lower is faster)')
    ax.set_ylabel('Mean delta-NLL, nats/token (answer + EOS)')
    ax.margins(x=.12, y=.22)
    ax.grid(alpha=.2)
fig.suptitle('Suffix reuse: fixed inputs, fresh-tail sweep\nDiagnostic warm-process timings; green = correct greedy answer')
fig.savefig(HERE / 'pareto.png', dpi=160)
fig.savefig(HERE / 'pareto.svg')
plt.close(fig)
print(json.dumps(rows, indent=2))
