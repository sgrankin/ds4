#!/usr/bin/env python3
"""Five paired main-FP4 or HC-fusion runs: timing, teacher quality and storage."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics as stats


helper = Path(__file__).resolve().parents[1] / 'decoder-replay-hour47-20261020/analyze.py'
spec = importlib.util.spec_from_file_location('fixed_teacher', helper)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def distribution(values):
    return dict(samples=values, mean=stats.mean(values), median=stats.median(values),
                minimum=min(values), maximum=max(values),
                sample_stdev=stats.stdev(values) if len(values) > 1 else None)


def process_memory(run):
    stderr = (run['path'] / 'stderr.log').read_text()
    allocation = re.findall(r'KV allocated main=(\w+) main_bytes=(\d+) index_bytes=(\d+) raw_bytes=(\d+) index_format=(\w+)', stderr)
    assert allocation, f'Missing actual KV allocation counters: {run["path"]}'
    main_format, main, index, raw, index_format = allocation[-1]
    actual = dict(main=int(main), index=int(index), raw=int(raw))
    actual['total_kv'] = sum(actual.values())
    scratch = re.findall(r'\bscratch ([\d.]+) MiB', stderr)
    peak = re.findall(r'runtime tensors live [\d.]+ MiB peak ([\d.]+) MiB', stderr)
    startup = re.findall(r'^.*memory: .* = ([\d.]+) GiB planned$', stderr, re.MULTILINE)
    buffers = re.findall(r'static context buffers ([\d.]+) MiB', stderr)
    expert = re.findall(r'^.*streaming expert cache budget=.*$', stderr, re.MULTILINE)
    io = dict(re.findall(r'(\w+)=([\d.]+)', expert[-1])) if expert else {}
    expert_summary = {key: int(io[key]) for key in ('hits', 'misses') if key in io}
    expert_summary.update({key + '_gib': float(io[key]) for key in ('live', 'miss_pread') if key in io})
    if 'pread_ms' in io:
        expert_summary['pread_ms'] = float(io['pread_ms'])
    return dict(main_format=main_format, index_format=index_format,
                allocated_kv_bytes=actual,
                planned_bytes={key: value for key, value in run['meta'].items() if key.startswith('planned_')},
                startup_planned_gib=float(startup[-1]) if startup else None,
                static_context_buffers_mib=float(buffers[-1]) if buffers else None,
                scratch_mib=float(scratch[-1]) if scratch else None,
                runtime_tensors_peak_mib=float(peak[-1]) if peak else None,
                expert_cache=expert_summary)


def quality(reference, arm):
    # Reuse the established log-sum-exp/NLL/KL math and common-history checks.
    for key in ('vocab', 'context', 'chunk', 'cases'):
        assert reference['meta'][key] == arm['meta'][key], key
    coordinates = lambda run: [(row['case'], row['step'], row['target']) for row in run['rows']]
    assert coordinates(reference) == coordinates(arm), 'Fixed teacher histories differ'
    exact = reference['logits'].tobytes() == arm['logits'].tobytes()
    rows = [] if exact else probe.paired(reference, arm)
    expected = {answer['case']: answer for answer in reference['answers']}
    result = {}
    for answer in arm['answers']:
        case = answer['case']
        selected = [row for row in rows if row['case'] == case]
        count = sum(row['case'] == case for row in arm['rows'])
        dnll = 0.0 if exact else stats.mean(row['delta_nll'] for row in selected)
        result[case] = dict(teacher_rows=count,
                            mean_nll=stats.mean(float(row['nll']) for row in arm['rows'] if row['case'] == case),
                            reference_mean_nll=stats.mean(float(row['nll']) for row in reference['rows'] if row['case'] == case),
                            byte_equal_rows=count if exact else sum(row['byte_equal'] for row in selected),
                            mean_delta_nll=dnll, teacher_perplexity_ratio=math.exp(dnll),
                            mean_kl=0.0 if exact else stats.mean(row['kl'] for row in selected),
                            max_kl=0.0 if exact else max(row['kl'] for row in selected),
                            argmax_flips=0 if exact else sum(row['argmax_flip'] for row in selected),
                            same_generated_ids=answer['tokens'] == expected[case]['tokens'],
                            same_answer_text=answer['text'] == expected[case]['text'],
                            generated_tokens=len(answer['tokens']),
                            reference_generated_tokens=len(expected[case]['tokens']),
                            answer_text=answer['text'], expected_text=answer['expected'],
                            correct=answer['exact'], stopped=answer['stopped'])
    return result


def times(run):
    output = {}
    for case, source in run['timing'].items():
        values = {name: float(source[name]) for name in ('sync_ms', 'teacher_ms', 'greedy_sync_ms', 'greedy_ms')}
        values['primary_ms'] = values['greedy_sync_ms'] + values['greedy_ms']
        values['teacher_task_ms'] = values['sync_ms'] + values['teacher_ms']
        values['all_measured_phases_ms'] = values['teacher_task_ms'] + values['primary_ms']
        output[case] = values
    cases = list(output)
    output['all_cases'] = {name: sum(output[case][name] for case in cases) for name in output[cases[0]]}
    return output


def checkpoint_sizes(run):
    return {case: {name: int(row[name]) for name in ('payload_bytes', 'suffix_cache_bytes', 'approximate')}
            for case, row in run['timing'].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.workspace
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['pairs'] == 5 and manifest['order'] == ['AB', 'BA', 'AB', 'BA', 'AB']
    assert manifest['A'] == 'control'
    candidate_label = manifest['B']
    if manifest['changed_setting'] == 'DS4_V41_COMPRESSED_KV_FP4':
        assert candidate_label == 'packed'
        control_format = 'f32'
    else:
        assert manifest['changed_setting'] == 'DS4_METAL_V41_FUSED_HC_NORM' and candidate_label == 'fused'
        control_format = 'fp4'
    opening = None
    pairs = []
    for pair in range(1, 6):
        control = probe.read_arm(root / f'pair-{pair}' / 'control')
        candidate = probe.read_arm(root / f'pair-{pair}' / candidate_label)
        assert all(math.isfinite(value) for run in (control, candidate) for value in run['logits']), f'Non-finite logits in pair {pair}'
        if opening is None:
            opening = control
        item = dict(pair=pair, order=manifest['order'][pair - 1],
                    control_quality_against_opening=quality(opening, control))
        for name, run in (('control', control), ('candidate', candidate)):
            memory = process_memory(run)
            assert memory['main_format'] == ('fp4' if name == 'candidate' else control_format), run['path']
            item[name] = dict(timing=times(run), quality=quality(control, run),
                              memory=memory, checkpoints=checkpoint_sizes(run))
        pairs.append(item)
    cases = list(opening['timing'])
    metrics = {}
    for case in cases + ['all_cases']:
        metrics[case] = {}
        for name in pairs[0]['control']['timing'][case]:
            old, new = ([pair[arm]['timing'][case][name] for pair in pairs] for arm in ('control', 'candidate'))
            changes = [candidate / baseline - 1 for baseline, candidate in zip(old, new)]
            metrics[case][name] = dict(control_ms=distribution(old), candidate_ms=distribution(new),
                paired_time_change=distribution(changes),
                candidate_slower_pairs=sum(change > 0 for change in changes),
                aggregate_time_change=stats.mean(new) / stats.mean(old) - 1,
                geometric_time_ratio=math.exp(stats.mean(math.log(candidate / baseline) for baseline, candidate in zip(old, new))),
                by_order={order: distribution([change for pair, change in zip(pairs, changes) if pair['order'] == order])
                          for order in ('AB', 'BA')})
    storage = {}
    for arm in ('control', 'candidate'):
        storage[arm] = dict(
            allocated_kv_bytes={name: distribution([pair[arm]['memory']['allocated_kv_bytes'][name] for pair in pairs])
                                for name in ('main', 'index', 'raw', 'total_kv')},
            planned_bytes={name: distribution([pair[arm]['memory']['planned_bytes'][name] for pair in pairs])
                           for name in pairs[0][arm]['memory']['planned_bytes']},
            canonical_payload_bytes={case: distribution([pair[arm]['checkpoints'][case]['payload_bytes'] for pair in pairs]) for case in cases})
    report = dict(samples_per_arm=5, candidate_label=candidate_label, cases=cases, manifest=manifest,
        timing_phases=dict(sync_ms='Teacher prefill', teacher_ms='Forced teacher continuation',
                           greedy_sync_ms='Answer prefill', greedy_ms='Normal-reasoning answer generation',
                           primary_ms='Answer prefill plus generation'),
        quality_reference='Each candidate arm is compared with its paired control; repeated controls are also compared with the opening control.',
        note='Positive change means candidate slower. All five fresh processes retained. Initialization excluded. '
             'Answer prefill has fresh KV and experts warmed by the forced teacher branch. Fixed teachers are short '
             'answer tokens plus EOS at a reasoning frontier, not broad reasoning-quality scores. Actual row counts come from rows.csv. '
             'Allocated counters describe GPU owners; planned_compressed_bytes includes main plus index KV. Payload sizes are '
             'serializer-reported canonical bytes at each prompt frontier, not measured file allocation or SSD traffic. '
             'Cleanup I/O/peak/scratch counters cover the whole process, not one case or just the primary window.',
        metrics=metrics, storage=storage, pairs=pairs)
    output = args.output or root / 'summary.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(candidate_label=candidate_label,
        primary={case: metrics[case]['primary_ms'] for case in cases + ['all_cases']},
        correct_by_case={case: sum(pair[arm]['quality'][case]['correct'] and pair[arm]['quality'][case]['stopped'] for pair in pairs for arm in ('control', 'candidate')) for case in cases},
        all_paired_logits_equal=all(q['byte_equal_rows'] == q['teacher_rows'] for pair in pairs for q in pair['candidate']['quality'].values()),
        all_generated_ids_equal=all(q['same_generated_ids'] for pair in pairs for q in pair['candidate']['quality'].values())), indent=2))


if __name__ == '__main__':
    main()
