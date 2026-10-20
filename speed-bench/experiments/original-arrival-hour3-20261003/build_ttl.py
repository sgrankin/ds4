"""Rebuild only the worker-containing Metal TU and link for retention ages."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
plan = json.loads((HERE / 'build-plan.json').read_text())
metal, link = plan[1:]
old_object = metal['argv'][-1]
old_binary = link['argv'][-1]
with (HERE / 'build-ttl.log').open('w') as log:
    for age in (1, 8):
        obj = old_object.removesuffix('.o') + f'-age{age}.o'
        binary = old_binary + f'-age{age}'
        commands = [metal['argv'][:-1] + [obj],
                    [obj if arg == old_object else binary if arg == old_binary else arg
                     for arg in link['argv']]]
        commands[0].insert(1, f'-DDS4_OD_READY_AGE={age}')
        for command in commands:
            log.write(json.dumps(command) + '\n'); log.flush()
            subprocess.run(command, cwd=metal['cwd'], stdout=log, stderr=subprocess.STDOUT, check=True)
        print('built retention age', age, flush=True)
