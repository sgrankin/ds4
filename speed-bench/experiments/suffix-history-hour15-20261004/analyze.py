#!/usr/bin/env python3
"""Three-condition retained-history table; run after ROOT releases timing freeze."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDITIONS = {'shortened': [('fresh', 'fresh-shortened'), ('reuse', 'reuse-shortened')],
              'archive-rule': [('fresh', 'fresh-explicit-archive-rule')]}


def load_metrics():
    path = HERE.parent / 'server-suffix-curve-hour14-20261004/analyze.py'
    spec = importlib.util.spec_from_file_location('suffix_curve_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def inspect_case(directory, case, context, metrics, sources):
    record = json.loads((directory / 'execution.json').read_text())
    rows, teacher = [], []
    if record.get('returncode') != 0:
        for mode, condition in CONDITIONS[context]:
            rows.append({'case': case['name'], 'condition': condition, 'gold': case['gold'],
                         'status': 'execution-incomplete' if 'returncode' not in record else 'execution-failed',
                         'returncode': record.get('returncode')})
        return rows, teacher
    summary = json.loads((directory / 'summary.json').read_text())
    tokens = json.loads((directory / 'tokens.json').read_text())
    if summary['gold_answer_text'] != case['gold']:
        raise ValueError(f'{directory}: declared gold differs from fixed fixture')
    source = (tokens['old'], tokens['gold'])
    if case['name'] in sources and sources[case['name']] != source:
        raise ValueError(f'{directory}: explicit archive changed source history or teacher targets')
    sources[case['name']] = source
    targets, nll, delta, kl, vocabulary = metrics.raw_metrics(directory, tokens)
    # Recompute both raw arms, but only shortened-context KL is a primary score.
    if not math.isclose(float(delta.sum()), summary['sum_delta_nll'], abs_tol=1e-9, rel_tol=0):
        raise ValueError(f'{directory}: raw delta-NLL disagrees with summary')
    if not math.isclose(float(kl.mean()), summary['mean_kl'], abs_tol=1e-9, rel_tol=0):
        raise ValueError(f'{directory}: raw KL disagrees with summary')
    results = {row['mode']: row for row in summary['results']}
    argv = record['argv']; cap = int(argv[argv.index('--greedy') + 1])
    for mode, condition in CONDITIONS[context]:
        index = int(mode == 'reuse'); result = results[mode]
        status = metrics.greedy_status(result, case['gold'], tokens['gold'], cap)
        absent = (case['name'] == 'helpful-history' and context == 'shortened'
                  and status['stopped'] and status['answer'].strip() == 'UNKNOWN')
        if absent: status['status'] = 'missing-information-abstention'
        paired = context == 'shortened'
        row = {'case': case['name'], 'condition': condition, 'gold': case['gold'],
               'contrast': case['contrast'], **status, 'missing_information_abstention': absent,
               'contrast_answer': status['stopped'] and status['answer'].strip() == case['contrast'],
               'greedy_cap': cap, 'edit_ms': result['edit_ms'],
               'archive_read_ms': record['archive_read_ms'],
               'edit_plus_archive_read_ms': result['edit_ms'] + record['archive_read_ms'],
               'source_setup_ms': summary['setup_ms'], 'source_capture_ms': summary['source_capture_ms'],
               'oracle_process_seconds': record['seconds'], 'teacher_rows_including_eos': len(targets),
               'vocabulary': vocabulary, 'gold_logprob': float(-nll[index].sum()),
               'mean_gold_nll': float(nll[index].mean()),
               'sum_delta_nll_vs_same_context_fresh': float(delta.sum()) if paired and index else 0.0 if paired else None,
               'mean_delta_nll_vs_same_context_fresh': float(delta.mean()) if paired and index else 0.0 if paired else None,
               'sum_kl_vs_same_context_fresh': float(kl.sum()) if paired and index else 0.0 if paired else None,
               'mean_kl_vs_same_context_fresh': float(kl.mean()) if paired and index else 0.0 if paired else None,
               'contrast_sequence_probability': math.exp(result['stale_logprob']),
               'configured_fresh_tail_tokens': summary['fresh_tail_tokens'],
               'fresh_evaluated_tokens': summary['edit_tokens'] + (summary['fresh_tail_tokens'] if index else summary['unchanged_tokens']),
               'reused_tokens': summary['reused_tokens'] if index else 0,
               'query_tokens': summary['query_tokens_including_assistant_prefix'],
               'query_fully_fresh': summary['query_fully_fresh'],
               'source_generated_answers': record['source_generated_answers']}
        rows.append(row)
        for step, target in enumerate(targets):
            teacher.append({'case': case['name'], 'condition': condition, 'step': step,
                            'target': int(target), 'row_type': 'eos' if step == len(targets)-1 else 'answer',
                            'nll': float(nll[index, step]),
                            'delta_nll_vs_same_context_fresh': float(delta[step]) if paired and index else 0.0 if paired else None,
                            'kl_vs_same_context_fresh': float(kl[step]) if paired and index else 0.0 if paired else None})
    return rows, teacher


def observations(rows):
    by_case = {}
    for row in rows:
        if 'task_pass' in row: by_case.setdefault(row['case'], {})[row['condition']] = row
    observed = {}
    for case, conditions in by_case.items():
        fresh, reuse, archive = [conditions.get(name) for name in
            ('fresh-shortened', 'reuse-shortened', 'fresh-explicit-archive-rule')]
        item = {'answers': {key: value['answer'] for key, value in conditions.items()},
                'task_passes': {key: value['task_pass'] for key, value in conditions.items()}}
        if fresh and reuse:
            item['reuse_changes_correctness'] = fresh['task_pass'] != reuse['task_pass']
            if case == 'helpful-history':
                item['potential_valid_history_benefit'] = reuse['task_pass'] and not fresh['task_pass']
                item['fresh_missing_information_abstention'] = fresh['missing_information_abstention']
            if case == 'obsolete-history':
                item['reuse_introduces_obsolete_answer'] = fresh['task_pass'] and not reuse['task_pass'] and reuse['contrast_answer']
                item['reuse_obsolete_answer_observed'] = reuse['contrast_answer']
        if archive: item['explicit_archive_task_pass'] = archive['task_pass']
        observed[case] = item
    return observed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', '--directory', dest='input', type=Path, default=HERE / 'core')
    p.add_argument('--output', type=Path)
    args = p.parse_args(); metrics = load_metrics()
    plan = args.input.parent / 'fixtures/cases.json'
    if plan.exists():
        cases = json.loads(plan.read_text())
    else:
        recorded = {}
        for execution in sorted(args.input.glob('*/execution.json')):
            case = json.loads(execution.read_text())['case']
            recorded[case['name']] = case
        cases = list(recorded.values())
    rows, teacher, missing, sources = [], [], [], {}
    for case in cases:
        for context in CONDITIONS:
            directory = args.input / (case['name'] + '-' + context)
            if not (directory / 'execution.json').exists():
                missing.append({'case': case['name'], 'context': context}); continue
            scored, raw_rows = inspect_case(directory, case, context, metrics, sources)
            rows.extend(scored); teacher.extend(raw_rows)
    if not rows: raise SystemExit('No execution records yet')
    counts = {}
    for condition in [name for modes in CONDITIONS.values() for _, name in modes]:
        completed = [r for r in rows if r['condition'] == condition and 'task_pass' in r]
        counts[condition] = {'passes': sum(r['task_pass'] for r in completed), 'completed_cases': len(completed),
                             'missing_information_abstentions': sum(r['missing_information_abstention'] for r in completed),
                             'caps': sum(r['status'] == 'cap-without-eos' for r in completed)}
    paired = [r for r in rows if r['condition'] == 'reuse-shortened' and 'task_pass' in r]
    n = sum(r['teacher_rows_including_eos'] for r in paired)
    report = {'scope': {'quality': 'Raw untempered teacher+EOS logits; float64. Same-context delta-NLL and KL compare shortened fresh vs shortened reuse only.',
                       'archive_control': 'Fresh arm of explicit retrieved-rule context; its ancillary reuse arm is not a primary condition. Archive read is driver file I/O, not a model tool call.',
                       'timing': 'Core edit and archive-file-read times exclude previous-source setup/capture/startup and answer generation. HTTP whole-task latency is separate.',
                       'interpretation': 'Helpful UNKNOWN is missing-information abstention. Benefit/harm flags use actual first greedy answers and independent golds, not likelihood alone.'},
              'counts': counts, 'observations': observations(rows), 'missing_runs': missing,
              'same_context_row_weighted_mean_delta_nll': sum(r['sum_delta_nll_vs_same_context_fresh'] for r in paired)/n if n else None,
              'same_context_row_weighted_mean_kl': sum(r['sum_kl_vs_same_context_fresh'] for r in paired)/n if n else None,
              'results': rows}
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'core-results.csv', rows)
    write_csv(output / 'core-teacher-rows.csv', teacher)
    (output / 'core-analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('counts', 'observations', 'missing_runs',
                                                'same_context_row_weighted_mean_delta_nll', 'same_context_row_weighted_mean_kl')}, indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
