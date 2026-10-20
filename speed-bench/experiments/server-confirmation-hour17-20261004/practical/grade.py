"""Existing safe task language, new independent private golds."""
import builtins
import copy
import importlib.util
import json
from pathlib import Path
import sys
from fixtures import fixtures

ORIGINAL = Path('/Volumes/Models/ds4/.workspaces/idea-resident-practical-tools/speed-bench/practical-tools/grade.py')
sys.path.insert(0,str(ORIGINAL.parent))
try:
    spec=importlib.util.spec_from_file_location('existing_practical_grade',ORIGINAL)
    existing=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(existing)
finally:
    sys.path.pop(0)


def grade(key, source):
    tree=existing.language(source)
    scope={'__builtins__':{name:getattr(builtins,name) for name in (*existing.SAFE_NAMES, 'iter')}}
    exec(compile(tree,'<first-candidate>','exec'),scope)
    solve=scope.get('solve')
    existing.require(callable(solve),'missing solve')
    cases=fixtures(key)
    for i,(payload,config,gold) in enumerate(cases):
        data,options=copy.deepcopy(payload),copy.deepcopy(config)
        result=solve(data,options)
        json.dumps(result,allow_nan=False)
        existing.require(existing.same_json(result,gold),'unit '+str(i))
        existing.require(existing.same_json(data,payload) and existing.same_json(options,config),'input mutation '+str(i))
    return {'correct':True,'tests':len(cases)}


if __name__=='__main__':
    try:
        result=grade(sys.argv[1],Path(sys.argv[2]).read_text())
    except BaseException as exc:
        result={'correct':False,'reason':type(exc).__name__,'message':str(exc),'tests_complete':False}
    print(json.dumps(result,allow_nan=False))
    sys.exit(0 if result['correct'] else 3)
