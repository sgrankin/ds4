#!/usr/bin/env python3
"""Immutable external unit gold. Agent stdout is only PASS/FAIL, never gold."""
import argparse,ast,hashlib,json,traceback
from pathlib import Path


def evaluate(project):
    p=Path(project)/'access.py';checks=dict(source_exists=p.is_file(),pure_module=False,callable_exists=False,current_value=False,obsolete_absent=False)
    result=None;error=None
    try:
        text=p.read_text();tree=ast.parse(text)
        # No imports, dynamic evaluation/file/process APIs or dunder escape.
        # Pure functions/constants/string arithmetic remain allowed.
        if any(isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.Attribute) and n.attr.startswith('__') or isinstance(n,ast.Name) and n.id.startswith('__') for n in ast.walk(tree)):raise ValueError('External-capability syntax')
        allowed=(ast.FunctionDef,ast.Assign,ast.AnnAssign)
        if any(not isinstance(n,allowed) and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)) for n in tree.body):raise ValueError('Executable module top level')
        checks['pure_module']=True
        builtins={'str':str,'int':int,'len':len,'dict':dict,'list':list,'tuple':tuple,'range':range,'sum':sum,'min':min,'max':max,'bool':bool}
        scope={'__builtins__':builtins};exec(compile(tree,str(p),'exec'),scope)
        fn=scope.get('current_access_line');checks['callable_exists']=callable(fn)
        if checks['callable_exists']:result=fn()
        checks['current_value']=result=='CODE=3141';checks['obsolete_absent']=isinstance(result,str) and 'LEGACY' not in result and '2718' not in result
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    return dict(passed=all(checks.values()),checks=checks,error=error,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None)


def main():
    a=argparse.ArgumentParser();a.add_argument('project',type=Path);a.add_argument('--attempt-dir',type=Path);a.add_argument('--check-only',action='store_true');v=a.parse_args()
    if bool(v.attempt_dir)==v.check_only:a.error('Choose an agent attempt or external final check')
    result=evaluate(v.project)
    if v.attempt_dir:
        with (v.attempt_dir/'grader-attempts.jsonl').open('a') as stream:json.dump(result,stream);stream.write('\n')
    print('PASS' if result['passed'] else 'FAIL')
    return 0 if result['passed'] else 3


if __name__=='__main__':raise SystemExit(main())
