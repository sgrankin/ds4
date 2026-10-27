#!/usr/bin/env python3
"""Compare the existing probe's fixed teachers and greedy token trajectories."""
import importlib.util
import json
from pathlib import Path
import statistics
import sys

helper = Path(__file__).parents[1] / 'decoder-replay-hour47-20261020/analyze.py'
spec = importlib.util.spec_from_file_location('replay_analysis', helper)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
root = Path(sys.argv[1])
arms = [analysis.read_arm(root / name) for name in ('dense', 'candidates', 'dense-repeat')]
reference = arms[0]
result = []
for arm in arms:
    rows = analysis.paired(reference, arm)
    cases = []
    for answer in arm['answers']:
        name = answer['case']
        timing = arm['timing'][name]
        controls = [x['timing'][name] for x in (arms[0], arms[2])]
        decode_per_token = float(timing['greedy_ms']) / int(timing['generated_tokens'])
        control_per_token = statistics.mean(float(x['greedy_ms']) / int(x['generated_tokens']) for x in controls)
        cases.append(dict(case=name, decode_ms=float(timing['greedy_ms']),
            generated_tokens=int(timing['generated_tokens']), decode_ms_per_token=decode_per_token,
            decode_time_saving=1 - decode_per_token / control_per_token,
            prefill_ms=float(timing['greedy_sync_ms']), exact=answer['exact'], stopped=answer['stopped'],
            same_generated_tokens=answer['tokens'] == next(x for x in reference['answers'] if x['case'] == name)['tokens'],
            answer=answer['text']))
    result.append(dict(arm=arm['path'].name, teacher_rows=len(rows),
        byte_equal_rows=sum(x['byte_equal'] for x in rows),
        mean_delta_nll=statistics.mean(x['delta_nll'] for x in rows),
        mean_kl=statistics.mean(x['kl'] for x in rows),
        argmax_flips=sum(x['argmax_flip'] for x in rows), cases=cases))
(root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
