"""Build the small strong-policy extension after the first campaign exits."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = Path('/Volumes/Models/ds4')
plan = json.loads((HERE / 'build-plan.json').read_text())
old_dir = str(Path(plan[0]['argv'][-1]).parent)
new_dir = old_dir.replace('original-deadline-native-adapter-build-v1', 'original-deadline-strong-direct-v1')
Path(new_dir).mkdir(exist_ok=True)
plan = [{**spec, 'argv': [arg.replace(old_dir, new_dir) for arg in spec['argv']]} for spec in plan]
commands = [spec['argv'] for spec in plan]
metal, link = plan[1:]
old_object, old_binary = metal['argv'][-1], link['argv'][-1]
for age in (1, 8):
    obj, binary = old_object.removesuffix('.o') + f'-age{age}.o', old_binary + f'-age{age}'
    command = metal['argv'][:-1] + [obj]
    command.insert(1, f'-DDS4_OD_READY_AGE={age}')
    commands += [command, [obj if arg == old_object else binary if arg == old_binary else arg for arg in link['argv']]]
cli_dir = ROOT / '.workspaces/idea-resident-functional-runtime-native/speed-bench/ideas/artifacts/resident-functional-native-direct-v1'
cli_objects = [str(cli_dir / (name + '.o')) for name in ('ds4_cli','ds4_help','ds4_prompt_prefix','linenoise','ds4_gpu_args')]
helper_source = link['argv'][link['argv'].index('-I') + 2]
for age in (80, 1, 8):
    suffix = '' if age == 80 else f'-age{age}'
    command = [arg for arg in link['argv'] if arg != helper_source]
    command[command.index(old_object)] = old_object.removesuffix('.o') + suffix + '.o'
    command[-1] = str(Path(new_dir) / ('ds4-original-deadline' + suffix))
    command[command.index('-lm'):command.index('-lm')] = cli_objects
    commands.append(command)
(HERE / 'strong-build-commands.json').write_text(json.dumps(commands, indent=2) + '\n')
with (HERE / 'build-strong.log').open('w') as log:
    for command in commands:
        print('Building', command[-1], flush=True)
        log.write(json.dumps(command) + '\n'); log.flush()
        subprocess.run(command, cwd=metal['cwd'], stdout=log, stderr=subprocess.STDOUT, check=True)
