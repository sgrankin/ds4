"""Summarize event metadata and first grades; never read raw vectors."""
import csv
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(path, errors):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(errors='replace').splitlines():
        if line.strip():
            try:
                result.append(json.loads(line))
            except ValueError:
                errors.append('incomplete event JSON in ' + path.name)
    return result


def summarize(path):
    record = json.loads(path.read_text())
    errors = []
    events = rows(path.parent / 'audit/events.jsonl', errors)
    samples = [row for row in events if row.get('type') == 'sample']
    ends = [row for row in events if row.get('type') == 'round-end']
    terminal = events[-1] if events else {}
    tools = [row for row in events if row.get('type') == 'tool-before']
    tool_results = [row for row in events if row.get('type') == 'tool-after']
    forced = sum(row.get('intervention') == 'forced-thinking-close'
                 for row in rows(path.parent / 'audit/write-budget.jsonl', errors))
    compliant = [(r.get('name'), r.get('path')) for r in tools] == [('read', 'config.json'), ('write', 'solution.py')]
    stopped = terminal.get('type') == 'finish' and terminal.get('actual_stop') is True
    complete = (record.get('returncode') == 0 and not record.get('timeout') and stopped
                and terminal.get('stage') == 2 and terminal.get('rounds') == 3
                and len(ends) == 3 and len(tool_results) == 2 and compliant
                and not any(r.get(k) for r in ends for k in ('context_limited', 'malformed', 'early_error', 'interrupted'))
                and record.get('config_unchanged') is True
                and record.get('snapshot_written') is True and not errors)
    correct = record.get('grade', {}).get('correct') is True
    refusal = terminal.get('reason', '')
    return dict(arm=record['arm'], task=record['task'], strict_success=complete and correct,
                code_correct=correct, completed=complete, tool_compliant=compliant,
                snapshot_written=record.get('snapshot_written') is True,
                cap=refusal == 'total-output-cap', failure_reason='; '.join(errors) or refusal or record.get('error', '') or
                (record.get('grade', {}).get('reason', '') if not correct else '' if complete else 'protocol incomplete'),
                returncode=record.get('returncode'), timeout=record.get('timeout', False), seconds=record.get('seconds'),
                samples=len(samples), inside_samples=sum(r.get('inside') is True for r in samples),
                admitted_tokens=sum(r.get('stop') is False for r in samples),
                read_samples=sum(r.get('round') == 0 for r in samples),
                write_samples=sum(r.get('round') == 1 for r in samples),
                final_samples=sum(r.get('round') == 2 for r in samples), forced_steps=forced,
                tests=record.get('grade', {}).get('tests', 0), path=str(path.parent))


def write_csv(path, records):
    with path.open('w', newline='') as out:
        if records:
            writer = csv.DictWriter(out, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def aggregate(records):
    result = []
    for arm in sorted({row['arm'] for row in records}):
        owned = [row for row in records if row['arm'] == arm]
        count = len(owned)
        strict = sum(row['strict_success'] for row in owned)
        code = sum(row['code_correct'] for row in owned)
        total = math.fsum(row['seconds'] for row in owned)
        result.append(dict(arm=arm, cases=count, strict_passes=strict, code_passes=code,
                           strict_success_percent=100 * strict / count, code_success_percent=100 * code / count,
                           caps=sum(row['cap'] for row in owned), seconds=total, mean_seconds=total / count,
                           samples=sum(row['samples'] for row in owned),
                           inside_samples=sum(row['inside_samples'] for row in owned),
                           forced_steps=sum(row['forced_steps'] for row in owned)))
    return result


def plots(records, here):
    if not records:
        return
    try:
        os.environ.setdefault('MPLCONFIGDIR', str(here / 'mpl-cache'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Plots skipped: matplotlib unavailable')
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, field, label in zip(axes, ['mean_seconds', 'samples'],
                               ['Mean helper wall time per task (s)', 'Total sampled decisions']):
        for i, row in enumerate(records):
            color = plt.get_cmap('tab10')(i % 10)
            x = row[field]
            ax.plot([x, x], [row['strict_success_percent'], row['code_success_percent']], color=color, alpha=.5)
            ax.scatter(x, row['strict_success_percent'], color=color, marker='o')
            ax.scatter(x, row['code_success_percent'], color=color, marker='x')
            ax.annotate(row['arm'] + ' n=' + str(row['cases']), (x, row['strict_success_percent']),
                        xytext=(5, 8 + (i % 3) * 10), textcoords='offset points', fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel('Task success (%)')
        ax.set_ylim(-7, 125)
        ax.set_yticks([0, 50, 100])
        ax.grid(alpha=.2)
    axes[0].scatter([], [], c='black', marker='o', label='Completed protocol + first code passes')
    axes[0].scatter([], [], c='black', marker='x', label='First code passes, including capped attempts')
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Thinking controls: two fixed tasks / one seed; instrumented timing')
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(here / ('tradeoffs.' + ext), dpi=160)
    plt.close(fig)


def main(here=HERE):
    here = Path(here)
    records = []
    for path in sorted(here.glob('*/**/execution.json')):
        try:
            record = json.loads(path.read_text())
            if 'seconds' not in record or 'grade' not in record:
                continue  # Child or first grader is still running.
            records.append(summarize(path))
        except (OSError, ValueError) as error:
            print('SKIP incomplete record', path, error)
    totals = aggregate(records)
    (here / 'results.json').write_text(json.dumps(dict(scope='First code tests and actual protocol metadata; caps remain failures. Instrumented time; no cross-prefix NLL/KL.',
                                                    rows=records, by_arm=totals), indent=2) + '\n')
    write_csv(here / 'results.csv', records)
    write_csv(here / 'aggregate.csv', totals)
    plots(totals, here)


if __name__ == '__main__':
    main()
