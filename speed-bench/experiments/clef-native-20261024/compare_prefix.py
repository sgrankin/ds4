#!/usr/bin/env python3
"""Compare saved uncached/cached Clef fixture outputs, timings and probabilities."""
import argparse
import json
import math
from pathlib import Path
import re


def load(directory):
    replies = [json.loads(line) for line in (directory / 'answers.jsonl').read_text().splitlines() if line.strip()]
    replies = [reply for reply in replies if 'answers' in reply]
    times, cache, pending = [], [], None
    for line in (directory / 'run.log').read_text().splitlines():
        match = re.search(r'clef-cache: tokens=(\d+) reused=(\d+) retained=(\d+) bytes=(\d+)', line)
        if match:
            pending = dict(zip(('tokens', 'reused', 'retained', 'bytes'), map(int, match.groups())))
        match = re.search(r'clef-smoke: request (\d+), ([\d.]+) ms', line)
        if match:
            assert int(match[1]) == len(times), directory
            times.append(float(match[2]))
            cache.append(pending)
            pending = None
    assert len(replies) == len(times), directory
    return replies, times, cache


def probabilities(answer):
    if 'probabilities' in answer:
        return answer['probabilities']
    return {'true': answer['noul'], 'false': 1-answer['noul']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control', type=Path)
    parser.add_argument('cached', type=Path)
    parser.add_argument('expected', type=Path)
    args = parser.parse_args()
    reference, control_ms, _ = load(args.control)
    candidate, cached_ms, cache = load(args.cached)
    expected = json.loads(args.expected.read_text())
    assert len(reference) == len(candidate) == len(expected)
    rows, gaps, kl, dnll = [], [], [], []
    correct_reference = correct_candidate = same_top = fields = 0
    nll_reference = nll_candidate = 0.0
    for index, (before, after, case) in enumerate(zip(reference, candidate, expected)):
        assert case['index'] == index
        assert before['usage']['input_tokens'] == after['usage']['input_tokens']
        assert before['answers'].keys() == after['answers'].keys() == case['expected_top'].keys()
        for key, gold in case['expected_top'].items():
            p, q = probabilities(before['answers'][key]), probabilities(after['answers'][key])
            assert p.keys() == q.keys() and gold in p, (index, key)
            top_p, top_q = max(p, key=p.get), max(q, key=q.get)
            fields += 1
            same_top += top_p == top_q
            correct_reference += top_p == gold
            correct_candidate += top_q == gold
            gaps.extend(abs(p[option]-q[option]) for option in p)
            kl.append(sum(value*math.log(value/max(q[option], 1e-300)) for option, value in p.items() if value > 0))
            loss_p, loss_q = -math.log(max(p[gold], 1e-300)), -math.log(max(q[gold], 1e-300))
            nll_reference += loss_p
            nll_candidate += loss_q
            dnll.append(loss_q-loss_p)
        rows.append({'index': index, 'name': case['name'], 'tokens': before['usage']['input_tokens'],
                     'control_ms': control_ms[index], 'cached_ms': cached_ms[index],
                     'speed_ratio': control_ms[index]/cached_ms[index], 'cache': cache[index]})
    summary = {'requests': len(rows), 'decisions': fields, 'same_top': same_top,
               'control_correct': correct_reference, 'cached_correct': correct_candidate,
               'control_accuracy': correct_reference/fields, 'cached_accuracy': correct_candidate/fields,
               'max_probability_gap': max(gaps), 'mean_kl_reference_to_cached': sum(kl)/fields,
               'control_nll': nll_reference/fields, 'cached_nll': nll_candidate/fields,
               'mean_delta_nll_gold': sum(dnll)/fields,
               'control_total_ms': sum(control_ms), 'cached_total_ms': sum(cached_ms),
               'total_speed_ratio': sum(control_ms)/sum(cached_ms),
               'cache_hit_requests': sum(bool(row and row['reused']) for row in cache),
               'reused_tokens': sum(row['reused'] for row in cache if row),
               'peak_retained_bytes': max((row['bytes'] for row in cache if row), default=0),
               'rows': rows, 'note': 'NLL and KL are nats per decision. Timings include first calls; inspect individual rows.'}
    (args.cached / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
