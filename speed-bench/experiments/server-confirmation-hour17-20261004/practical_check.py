#!/usr/bin/env python3
"""Small CPU check of independent gold/reference methods before outcomes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PRIVATE=HERE/'practical'
sys.path.insert(0,str(PRIVATE))
spec=importlib.util.spec_from_file_location('hour17_grade',PRIVATE/'grade.py')
grade=importlib.util.module_from_spec(spec);spec.loader.exec_module(grade)


def main():
    tasks=json.loads((PRIVATE/'spec.json').read_text())
    checks=[]
    for task in tasks['tasks']:
        key=task['key'];cases=grade.fixtures(key)
        assert cases[0][0]==task['config']['sample']
        assert all(cases[0][1][k]==v for k,v in task['config'].items() if k!='sample')
        result=grade.grade(key,(PRIVATE/('reference-'+key+'.py')).read_text())
        checks.append({'task':key,'inputs':len(cases),'reference_grade':result,
                       'cases_sha256':hashlib.sha256(json.dumps(cases,sort_keys=True,ensure_ascii=False).encode()).hexdigest()})
    sources={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(PRIVATE.glob('*')) if p.is_file()}
    result={'checks':checks,'sources':sources,'decisions':'Frozen before model outcomes: new ledger/LRU contracts, 161 inputs each, budgets128 then256 thennative, no repairs.',
            'gold_methods':'Ledger per-key exhaustive max(revision,position) vs reference incremental winners; LRU timestamps vs reference queue.'}
    (PRIVATE/'check-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
