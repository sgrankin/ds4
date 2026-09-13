"""Held-out checks, executed by the harness outside the editable project."""
import csv
from decimal import Decimal
import json
from pathlib import Path
import sys

project = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(project))
from ledger.money import cents
from ledger.report import summarize

stage = int(sys.argv[2])
for i in range(10000):
    assert cents(f'{i//100}.{i%100:02}') == i, f'money: {i}'
rows = [dict(event_id='a', account='x', kind='charge', amount='0.29', date='2026-08-01'),
        dict(event_id='a', account='x', kind='charge', amount='9.99', date='2026-08-20'),
        dict(event_id='b', account='x', kind='refund', amount='0.29', date='2026-08-15'),
        dict(event_id='c', account='v', kind='void', amount='99.00', date='2026-08-15')]
assert summarize(iter(rows)) == {'x': 0}
if stage >= 1:
    assert summarize(iter(rows), since='2026-08-15') == {'x': -29}
    assert summarize(rows, since='2026-09-01') == {}
    assert summarize(rows, since=None) == {'x': 0}
if stage >= 2:
    with (project/'data.csv').open() as f:
        data=list(csv.DictReader(f))
    for filename, since in [('result.json', None), ('recent.json','2026-08-15')]:
        expected={}; seen=set()
        for row in data:
            if row['event_id'] in seen: continue
            seen.add(row['event_id'])
            if since and row['date'] < since or row['kind']=='void': continue
            value=int(Decimal(row['amount'])*100) * (-1 if row['kind']=='refund' else 1)
            expected[row['account']]=expected.get(row['account'],0)+value
        actual=json.loads((project/filename).read_text())
        assert actual == expected and all(type(x) is int for x in actual.values()), filename
print(f'PASS held-out stage {stage}')
