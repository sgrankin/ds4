#!/usr/bin/env python3
"""Five paired HTTP workflows; retain initial capture cost and every edited reply."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics as stats


def distribution(values):
    return dict(mean=stats.mean(values), median=stats.median(values),
                minimum=min(values), maximum=max(values),
                sample_sd=stats.stdev(values) if len(values) > 1 else 0.0)


CAPTURE = re.compile(
    r'context_capture slot=(?P<slot>\d+) tokens=(?P<tokens>\d+) transient=(?P<transient>\d+) '
    r'state_bytes=(?P<state_bytes>\d+) pre_bytes=(?P<pre_bytes>\d+) '
    r'capture_ms=(?P<capture_ms>[\d.]+) publish_ms=(?P<publish_ms>[\d.]+) '
    r'hot_bytes=(?P<hot_bytes>\d+) ssd_bytes=(?P<ssd_bytes>\d+) '
    r'endpoints=(?P<endpoints>\d+) status=(?P<status>\d+)')
SUFFIX = re.compile(
    r'suffix APPROXIMATE slot=(?P<slot>\d+) reused=(?P<reused>\d+) fresh=(?P<fresh>\d+) '
    r'graft_fresh=(?P<graft_fresh>\d+) source=(?P<source_start>\d+):(?P<source_end>\d+) '
    r'destination=(?P<destination_start>\d+):(?P<destination_end>\d+) '
    r'latest_user_start=(?P<latest_user_start>-?\d+) pre_rope=1 source_ssd=(?P<source_ssd>\d+) '
    r'source_approximate=(?P<source_approximate>\d+) ongoing_user=(?P<ongoing_user>\d+) '
    r'prefix_pad=(?P<prefix_pad>\d+) right_shift=(?P<right_shift>\d+) (?P<elapsed_ms>[\d.]+) ms')
BOUNDARY = re.compile(r'context_boundaries slot=\d+ selected=(?P<selected>\d+)')
FALLBACK = re.compile(r'suffix exact fallback slot=\d+ reason=(?P<reason>.*)')


def numeric(match, line_number):
    return dict(line=line_number, **{
        name: float(value) if name.endswith('_ms') else int(value)
        for name, value in match.groupdict().items()
    })


def process_report(processes):
    samples = []
    for pair in range(1, 6):
        for arm in ('control', 'candidate'):
            directory = processes / f'pair-{pair}' / arm
            command = json.loads((directory / 'command.json').read_text())
            outcome = json.loads((directory / 'result.json').read_text())
            assert outcome['ok'], directory
            captures, suffixes, selected = [], [], []
            fallbacks = Counter()
            log = directory / 'server.stderr'
            for line_number, line in enumerate(log.read_text().splitlines(), 1):
                if 'context_capture ' in line:
                    match = CAPTURE.search(line)
                    assert match, f'Unrecognized capture record: {log}:{line_number}'
                    captures.append(numeric(match, line_number))
                elif 'suffix APPROXIMATE ' in line:
                    match = SUFFIX.search(line)
                    assert match, f'Unrecognized suffix record: {log}:{line_number}'
                    suffixes.append(numeric(match, line_number))
                elif match := BOUNDARY.search(line):
                    selected.append(dict(line=line_number, selected=int(match['selected'])))
                elif match := FALLBACK.search(line):
                    fallbacks[match['reason']] += 1
            assert captures, f'Completed workflows have no capture records: {log}'
            transient = [row for row in captures if row['transient']]
            assert not selected or transient, f'Selected boundaries have no transient captures: {log}'
            row = dict(pair=pair, arm=arm, log=str(log),
                       capture_count=len(captures), transient_count=len(transient),
                       failed_captures=sum(item['status'] != 0 for item in captures),
                       capture_ms_total=sum(item['capture_ms'] for item in captures),
                       publish_ms_total=sum(item['publish_ms'] for item in captures),
                       hot_bytes_peak=max(item['hot_bytes'] for item in captures),
                       ssd_bytes_peak=max(item['ssd_bytes'] for item in captures),
                       state_bytes_peak=max(item['state_bytes'] for item in captures),
                       pre_bytes_peak=max(item['pre_bytes'] for item in captures),
                       suffix_count=len(suffixes),
                       reused_tokens_sum=sum(item['reused'] for item in suffixes),
                       fresh_tokens_sum=sum(item['fresh'] for item in suffixes),
                       graft_fresh_tokens_sum=sum(item['graft_fresh'] for item in suffixes),
                       suffix_ms_total=sum(item['elapsed_ms'] for item in suffixes),
                       ram_only_sources=sum(not item['source_ssd'] for item in suffixes),
                       durable_sources=sum(item['source_ssd'] for item in suffixes),
                       ongoing_source_count=sum(item['ongoing_user'] for item in suffixes),
                       selected_boundaries_sum=sum(item['selected'] for item in selected),
                       fallbacks=dict(fallbacks), captures=captures, suffixes=suffixes,
                       selections=selected, settings=command['settings'])
            samples.append(row)
    keys = ('capture_count', 'transient_count', 'failed_captures', 'capture_ms_total',
            'publish_ms_total', 'hot_bytes_peak', 'ssd_bytes_peak', 'state_bytes_peak',
            'pre_bytes_peak', 'suffix_count', 'reused_tokens_sum', 'fresh_tokens_sum',
            'graft_fresh_tokens_sum', 'suffix_ms_total', 'ram_only_sources',
            'durable_sources', 'ongoing_source_count', 'selected_boundaries_sum')
    metrics = {name: {arm: distribution([row[name] for row in samples if row['arm'] == arm])
                      for arm in ('control', 'candidate')} for name in keys}
    order_path = processes.parent / 'execution-order.json'
    execution = json.loads(order_path.read_text())
    starts = [event for event in execution if event['event'] == 'started']
    assert len(starts) == 10, order_path
    order = [''.join('A' if event['arm'] == 'control' else 'B'
                     for event in starts if event['pair'] == pair) for pair in range(1, 6)]
    return dict(samples_per_arm=5, order=order, execution=execution,
                metrics=metrics, samples=samples,
                note='Process totals combine both workflows; they are not per-case estimates. '
                     'Hot/SSD peaks are observed cache-page counters at captures, not OS allocation or IO bytes. '
                     'Source events retain actual positions and fresh/imported counts. '
                     'source_ssd is endpoint durability, not the actual materialization read tier: '
                     'durable endpoints may be read from RAM; zero identifies a RAM-only transient. '
                     'These logs do not identify per-endpoint eviction or attribute cache-read counters to grafts.')


parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--processes', type=Path)
args = parser.parse_args()
root = args.root
pairs = []
for pair in range(1, 6):
    result = dict(pair=pair)
    for arm in ('control', 'candidate'):
        directory = root / f'pair-{pair}' / arm
        manifest = json.loads((directory / 'manifest.json').read_text())
        rows = json.loads((directory / 'results.json').read_text())
        assert [row['edit'] for row in rows] == list(range(manifest['edits'] + 1)), directory
        result[arm] = dict(initial_seconds=rows[0]['seconds'],
                           edited_seconds=sum(row['seconds'] for row in rows[1:]),
                           whole_seconds=sum(row['seconds'] for row in rows),
                           correct=sum(row['correct'] for row in rows), replies=len(rows),
                           cached_tokens=[row['usage']['input_tokens_details']['cached_tokens'] for row in rows],
                           output_tokens=[row['usage']['output_tokens'] for row in rows],
                           statuses=[row['status'] for row in rows], samples=rows)
    pairs.append(result)

metrics = {}
for name in ('initial_seconds', 'edited_seconds', 'whole_seconds'):
    control = [pair['control'][name] for pair in pairs]
    candidate = [pair['candidate'][name] for pair in pairs]
    changes = [b / a - 1 for a, b in zip(control, candidate)]
    metrics[name] = dict(control=distribution(control), candidate=distribution(candidate),
                         paired_time_change=distribution(changes),
                         aggregate_time_change=stats.mean(candidate) / stats.mean(control) - 1,
                         candidate_slower_pairs=sum(value > 0 for value in changes))
summary = dict(samples_per_arm=5, metrics=metrics, pairs=pairs,
               note='Positive time change means candidate slower. All initial/edited requests retained. '
                    'Caller records actual alternating order and server settings. '
                    'HTTP answers/cache usage do not replace fixed-teacher delta-NLL/KL.')
if args.processes:
    summary['processes'] = process_report(args.processes)
    summary['order'] = summary['processes']['order']
    (args.processes / 'summary.json').write_text(json.dumps(summary['processes'], indent=2) + '\n')
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(metrics, indent=2))
