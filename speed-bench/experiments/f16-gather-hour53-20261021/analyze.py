#!/usr/bin/env python3
"""Compare packed/unpacked gather with F16 storage fixed in every arm."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('item47_analysis',
    ROOT/'speed-bench/experiments/decoder-replay-hour47-20261020/analyze.py')
item47 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(item47)
PHASES = ('sync_ms', 'teacher_ms', 'greedy_sync_ms', 'greedy_ms', 'practical_ms')


def quality(rows):
    return {'teacher_rows': len(rows),
        'byte_equal_rows': sum(r['byte_equal'] for r in rows),
        'all_rows_byte_equal': all(r['byte_equal'] for r in rows),
        'mean_delta_nll': statistics.mean(r['delta_nll'] for r in rows),
        'max_abs_delta_nll': max(abs(r['delta_nll']) for r in rows),
        'mean_kl': statistics.mean(r['kl'] for r in rows),
        'max_kl': max(r['kl'] for r in rows),
        'argmax_flips': sum(r['argmax_flip'] for r in rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    recorded = json.loads((args.input/'result.json').read_text())
    completed = [r for r in recorded if r['returncode'] == 0]
    assert completed, 'No successful completed arms'
    result = {
        'scope': 'F16 storage fixed; one fixture per case. Ordinary ABBA; favorable single pair.',
        'timing_formula': 'practical_ms = greedy_sync_ms + greedy_ms; teacher run is separate',
        'quality_formula': 'Full-vocabulary fixed-teacher comparison against first packed=0 arm per case',
        'limitations': ['Fresh engine per arm; OS/SSD caches carry over.',
            'Exactness covers captured teacher rows and reported answers, not every intermediate state.',
            'Favorable has one control and candidate; no closing-control drift estimate.'],
        'failed_arms': [r for r in recorded if r['returncode'] != 0],
        'arms': [], 'cases': {}}
    flat = []
    for case in dict.fromkeys(r['case'] for r in completed):
        entries = [r for r in completed if r['case'] == case]
        controls = [r for r in entries if r['packed'] == 0]
        assert controls, f'No control for {case}'
        reference = item47.read_arm(args.input/controls[0]['directory'])
        assert reference['meta']['compressed_kv_f16'] == 1
        reference_answer = next(a for a in reference['answers'] if a['case'] == case)
        case_arms, candidate_rows = [], []
        for entry in entries:
            arm = (reference if entry['directory'] == controls[0]['directory'] else
                   item47.read_arm(args.input/entry['directory']))
            assert arm['meta']['compressed_kv_f16'] == 1
            assert set(arm['timing']) == {case}
            rows = item47.paired(reference, arm)
            assert rows and all(r['case'] == case for r in rows)
            answer = next(a for a in arm['answers'] if a['case'] == case)
            timing = {k: float(arm['timing'][case][k]) for k in PHASES[:-1]}
            timing['practical_ms'] = timing['greedy_sync_ms'] + timing['greedy_ms']
            summary = {'directory': entry['directory'], 'case': case,
                'packed': entry['packed'], 'timing': timing, 'quality': quality(rows),
                'answer': answer,
                'answer_correct': bool(answer['exact'] and answer['stopped']),
                'answer_text_equal': answer['text'] == reference_answer['text'],
                'answer_tokens_equal': answer['tokens'] == reference_answer['tokens'],
                'logits_sha256': hashlib.sha256((arm['path']/'logits.bin').read_bytes()).hexdigest()}
            result['arms'].append(summary)
            case_arms.append(summary)
            if entry['packed']:
                candidate_rows.extend(rows)
            flat.append({'directory': entry['directory'], 'case': case, 'packed': entry['packed'],
                **timing, **summary['quality'],
                'answer_correct': summary['answer_correct'],
                'answer_text_equal': summary['answer_text_equal'],
                'answer_tokens_equal': summary['answer_tokens_equal']})
            with (args.output/(entry['directory']+'-paired.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        native = [r for r in case_arms if r['packed'] == 0]
        candidate = [r for r in case_arms if r['packed'] == 1]
        phases = {}
        for phase in PHASES:
            control_values = [r['timing'][phase] for r in native]
            candidate_values = [r['timing'][phase] for r in candidate]
            control_mean = statistics.mean(control_values)
            candidate_mean = statistics.mean(candidate_values) if candidate_values else None
            phases[phase] = {'control_values': control_values, 'candidate_values': candidate_values,
                'control_mean': control_mean, 'candidate_mean': candidate_mean,
                'saving': 1-candidate_mean/control_mean if candidate_mean is not None else None,
                'control_drift': control_values[-1]/control_values[0]-1 if len(control_values) > 1 else None}
        result['cases'][case] = {
            'reference': controls[0]['directory'],
            'order': [{'directory': r['directory'], 'packed': r['packed']} for r in entries],
            'control_arms': len(native), 'candidate_arms': len(candidate),
            'timing': phases,
            'candidate_quality': quality(candidate_rows) if candidate_rows else None,
            'control_drift_quality': native[-1]['quality'] if len(native) > 1 else None,
            'correct_answers': sum(r['answer_correct'] for r in case_arms),
            'answer_count': len(case_arms),
            'all_answer_text_equal': all(r['answer_text_equal'] for r in case_arms),
            'all_answer_tokens_equal': all(r['answer_tokens_equal'] for r in case_arms)}
    result['all_answers_correct'] = all(r['answer_correct'] for r in result['arms'])
    result['all_teacher_rows_byte_equal'] = all(r['quality']['all_rows_byte_equal'] for r in result['arms'])
    result['complete_planned_order'] = (
        [(r['case'], r['packed']) for r in completed] ==
        [('ordinary', 0), ('ordinary', 1), ('ordinary', 1), ('ordinary', 0),
         ('favorable', 0), ('favorable', 1)] and not result['failed_arms'])
    with (args.output/'arms.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    for case, summary in result['cases'].items():
        print(json.dumps({'case': case, 'practical': summary['timing']['practical_ms'],
            'candidate_quality': summary['candidate_quality'],
            'correct_answers': summary['correct_answers'], 'answer_count': summary['answer_count']}))


if __name__ == '__main__':
    main()
