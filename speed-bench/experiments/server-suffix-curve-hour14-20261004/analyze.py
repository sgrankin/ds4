#!/usr/bin/env python3
"""Recompute raw suffix-oracle quality and plot the three fixed-input curves."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import struct

import numpy as np

HERE = Path(__file__).resolve().parent
FAMILIES = ('ordinary-deletion', 'best-self-contained', 'stress-changed-fact')
TAILS = (64, 128, 256)
EOS = 1


def raw_metrics(directory, tokens):
    """KL(fresh || reuse), and reuse NLL minus fresh NLL, without tempering."""
    with (directory / 'logits.bin').open('rb') as raw:
        header = raw.read(16)
        if len(header) != 16:
            raise ValueError(f'{directory}: incomplete logits header')
        magic, vocabulary, count, length = struct.unpack('<4I', header)
        if magic != 0x53554632 or length != len(tokens['edited']):
            raise ValueError(f'{directory}: unexpected logits header or edited length')
        targets = np.frombuffer(raw.read(count * 4), dtype='<i4')
        logits = np.frombuffer(raw.read(), dtype='<f4')
    if count == 0 or vocabulary == 0 or logits.size != 2 * count * vocabulary:
        raise ValueError(f'{directory}: incomplete full-vocabulary logits')
    if targets.tolist() != tokens['gold'] or tokens['gold'][-1] != EOS:
        raise ValueError(f'{directory}: raw targets must equal gold tokens including EOS')
    if np.any(targets < 0) or np.any(targets >= vocabulary):
        raise ValueError(f'{directory}: target outside vocabulary')
    logits = logits.astype(np.float64).reshape(2, count, vocabulary)
    if not np.isfinite(logits).all():
        raise ValueError(f'{directory}: nonfinite logits')
    shifted = logits - logits.max(axis=2, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=2, keepdims=True))
    nll = -logp[:, np.arange(count), targets]
    delta = nll[1] - nll[0]
    kl = (np.exp(logp[0]) * (logp[0] - logp[1])).sum(axis=1)
    return targets, nll, delta, kl, vocabulary


def greedy_status(result, gold_text, gold_tokens, cap):
    generated = result['greedy_tokens']
    stopped = bool(result['stopped'])
    passed = stopped and result['greedy'].strip() == gold_text
    at_cap = len(generated) >= cap
    status = 'pass' if passed else ('wrong-answer' if stopped else
                                  'cap-without-eos' if at_cap else 'not-stopped')
    return dict(answer=result['greedy'], task_pass=passed, status=status,
                stopped=stopped, generated_tokens=len(generated),
                terminal_eos=bool(generated and generated[-1] == EOS),
                at_cap=at_cap, token_ids_equal_gold=generated == gold_tokens)


def read_point(execution, histories):
    record = json.loads(execution.read_text())
    directory = execution.parent
    family, tail = record['family'], record['fresh_tail']
    if family not in FAMILIES or tail not in TAILS:
        raise ValueError(f'{directory}: unexpected family or fresh tail')
    row = dict(family=family, fresh_tail=tail, returncode=record['returncode'],
               oracle_process_seconds=record['seconds'])
    if record['returncode'] != 0:
        row['status'] = 'execution-failed'
        return row, []
    summary = json.loads((directory / 'summary.json').read_text())
    tokens = json.loads((directory / 'tokens.json').read_text())
    history = tuple(tokens[key] for key in ('old', 'edited', 'gold'))
    if family in histories and histories[family] != history:
        raise ValueError(f'{directory}: tail sweep changed old, edited, or gold token IDs')
    histories[family] = history
    targets, nll, delta, kl, vocabulary = raw_metrics(directory, tokens)
    results = {result['mode']: result for result in summary['results']}
    fresh, reuse = results['fresh'], results['reuse']
    argv = record['argv']
    cap = int(argv[argv.index('--greedy') + 1])
    row.update(status='completed', greedy_cap=cap, vocabulary=vocabulary,
               teacher_rows_including_eos=len(targets),
               fresh_edit_ms=fresh['edit_ms'], reuse_edit_ms=reuse['edit_ms'],
               speedup=fresh['edit_ms'] / reuse['edit_ms'],
               latency_ratio=reuse['edit_ms'] / fresh['edit_ms'],
               sum_delta_nll=float(delta.sum()), mean_delta_nll=float(delta.mean()),
               perplexity_ratio=math.exp(float(delta.mean())),
               sum_kl=float(kl.sum()), mean_kl=float(kl.mean()),
               max_row_delta_nll=float(delta.max()),
               fresh_gold_logprob=float(-nll[0].sum()),
               reuse_gold_logprob=float(-nll[1].sum()),
               setup_ms=summary.get('setup_ms'))
    for mode, result in (('fresh', fresh), ('reuse', reuse)):
        row.update({mode + '_' + key: value for key, value in
                    greedy_status(result, summary['gold_answer_text'], tokens['gold'], cap).items()})
        if 'stale_logprob' in result:
            row[mode + '_alternative_probability'] = math.exp(result['stale_logprob'])
    teacher = [dict(family=family, fresh_tail=tail, step=step, target=int(target),
                    row_type='eos' if step == len(targets) - 1 else 'answer',
                    fresh_nll=float(nll[0, step]), reuse_nll=float(nll[1, step]),
                    delta_nll=float(delta[step]), kl=float(kl[step]))
               for step, target in enumerate(targets)]
    return row, teacher


def aggregate(rows):
    completed = [row for row in rows if row['status'] == 'completed']
    count = sum(row['teacher_rows_including_eos'] for row in completed)
    result = dict(points=len(rows), completed_points=len(completed), teacher_rows_including_eos=count)
    for metric in ('delta_nll', 'kl'):
        result['row_weighted_mean_' + metric] = (
            sum(row['sum_' + metric] for row in completed) / count if count else None)
    for mode in ('fresh', 'reuse'):
        result[mode + '_task_passes'] = sum(row[mode + '_task_pass'] for row in completed)
        result[mode + '_task_total'] = len(completed)
        result[mode + '_terminal_eos_count'] = sum(row[mode + '_terminal_eos'] for row in completed)
        result[mode + '_cap_without_eos_count'] = sum(
            row[mode + '_status'] == 'cap-without-eos' for row in completed)
        result[mode + '_statuses'] = {status: sum(row[mode + '_status'] == status for row in completed)
                                     for status in sorted({row[mode + '_status'] for row in completed})}
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plots(rows, output):
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'plot-config'))
    os.environ.setdefault('XDG_CACHE_HOME', str(output / 'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for metric, label, filename in (
            ('mean_delta_nll', 'Mean delta-NLL, nats / token (gold + EOS)', 'pareto'),
            ('mean_kl', 'Mean KL(fresh || reuse), nats (gold + EOS)', 'kl')):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for ax, family in zip(axes, FAMILIES):
            points = sorted((row for row in rows if row['family'] == family and
                             row['status'] == 'completed'), key=lambda row: row['fresh_tail'])
            ax.scatter([1], [0], marker='*', s=100, color='black', label='Fresh reference')
            ax.plot([row['latency_ratio'] for row in points],
                    [row[metric] for row in points], color='#2171b5', linewidth=1)
            for point in points:
                color = '#238b45' if point['reuse_task_pass'] else '#cb181d'
                ax.scatter(point['latency_ratio'], point[metric], color=color, s=45)
                text = (f"tail {point['fresh_tail']}\n"
                        f"F {int(point['fresh_task_pass'])}/1 R {int(point['reuse_task_pass'])}/1")
                if point['reuse_status'] == 'cap-without-eos':
                    text += f"\nR cap {point['greedy_cap']}"
                ax.annotate(text, (point['latency_ratio'], point[metric]),
                            xytext=(5, 7), textcoords='offset points', fontsize=8)
            counts = aggregate(points)
            ax.set_title(f"{family.replace('-', ' ')}\n"
                         f"strict passes F {counts['fresh_task_passes']}/{counts['fresh_task_total']}, "
                         f"R {counts['reuse_task_passes']}/{counts['reuse_task_total']}")
            ax.set_xlabel('Core edit ms / paired fresh core edit ms')
            ax.set_ylabel(label)
            ax.margins(x=.2, y=.35)
            ax.grid(alpha=.2)
        fig.suptitle('Fixed-token suffix-tail sweep: diagnostic core edit timing\n'
                     'Green = reuse task passes; red = fails. HTTP whole-task latency is separate.')
        fig.savefig(output / (filename + '.png'), dpi=160)
        fig.savefig(output / (filename + '.svg'))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=HERE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    rows, teacher, histories = [], [], {}
    for execution in sorted(args.directory.glob('*-tail*/execution.json')):
        row, raw_rows = read_point(execution, histories)
        rows.append(row)
        teacher.extend(raw_rows)
    if not rows:
        raise SystemExit('No suffix execution records yet')
    rows.sort(key=lambda row: (FAMILIES.index(row['family']), row['fresh_tail']))
    expected = {(family, tail) for family in FAMILIES for tail in TAILS}
    seen = {(row['family'], row['fresh_tail']) for row in rows}
    if len(seen) != len(rows):
        raise ValueError('Duplicate family/tail execution records')
    output = args.output or args.directory
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'results.csv', rows)
    write_csv(output / 'teacher-rows.csv', teacher)
    for family in FAMILIES:
        write_csv(output / (family + '.csv'), [row for row in rows if row['family'] == family])
    report = dict(
        scope=dict(quality='Raw untempered full-vocabulary logits; float64 log-softmax and accumulation. '
                           'Delta-NLL = reuse NLL - fresh NLL; KL = KL(fresh || reuse). Gold targets include EOS.',
                   fixed_inputs='Old, edited, and gold token IDs must match within each family across tails.',
                   strict_task_pass='Stopped at EOS and stripped greedy text equals the gold answer text.',
                   timing='Core edit_ms is diagnostic warm-process timing. Oracle process seconds include '
                          'setup and quality diagnostics. Neither is HTTP whole-task latency.'),
        missing_points=[dict(family=family, fresh_tail=tail) for family, tail in sorted(expected - seen)],
        overall=aggregate(rows),
        families={family: aggregate([row for row in rows if row['family'] == family]) for family in FAMILIES},
        results=rows)
    (output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    if not args.no_plot:
        plots(rows, output)
    print(json.dumps({key: report[key] for key in ('overall', 'families', 'missing_points')}, indent=2))


if __name__ == '__main__':
    main()
