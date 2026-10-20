"""First-attempt grader, reusing the original task-language restrictions."""
import builtins, copy, importlib.util, json, sys
from pathlib import Path
from fixtures import fixtures

ORIGINAL = Path('/Volumes/Models/ds4/.workspaces/idea-resident-practical-tools/speed-bench/practical-tools/grade.py')
sys.path.insert(0, str(ORIGINAL.parent))
try:
    spec = importlib.util.spec_from_file_location('borrowed_practical_grade', ORIGINAL)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
finally:
    sys.path.pop(0)

def grade(key, source):
    tree = original.language(source)
    scope = {'__builtins__': {key: getattr(builtins, key) for key in original.SAFE_NAMES}}
    exec(compile(tree, '<first-candidate>', 'exec'), scope)
    solve = scope.get('solve')
    original.require(callable(solve), 'missing solve')
    cases = fixtures(key)
    for index, (payload, config, gold) in enumerate(cases):
        inputs, options = copy.deepcopy(payload), copy.deepcopy(config)
        result = solve(inputs, options)
        json.dumps(result, allow_nan=False)
        original.require(original.same_json(result, gold), f'unit {index}')
        original.require(original.same_json(inputs, payload) and original.same_json(options, config),
                         f'input mutation {index}')
    return {'correct': True, 'tests': len(cases)}

if __name__ == '__main__':
    try:
        result = grade(sys.argv[1], Path(sys.argv[2]).read_text())
    except BaseException as error:
        result = {'correct': False, 'reason': type(error).__name__, 'message': str(error),
                  'tests_complete': False}
    print(json.dumps(result, allow_nan=False))
    sys.exit(0 if result['correct'] else 3)
