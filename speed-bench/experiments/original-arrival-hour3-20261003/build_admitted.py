"""Keep the strong core; build only the macro-guarded join experiment."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
commands = json.loads((HERE / 'strong-build-commands.json').read_text())
old_dir = str(Path(commands[0][-1]).parent)
new_dir = old_dir.replace('original-deadline-strong-direct-v1', 'original-deadline-admitted-direct-v1')
Path(new_dir).mkdir(exist_ok=True)
old_core = str(Path(old_dir) / 'ds4-original-deadline.o')
rebuild = []
for original in commands[1:]:
    command = [arg if arg == old_core else arg.replace(old_dir, new_dir) for arg in original]
    if '-c' in command:
        command.insert(1, '-DDS4_OD_JOIN_ONLY_ADMITTED')
    rebuild.append(command)
(HERE / 'admitted-build-commands.json').write_text(json.dumps(rebuild, indent=2) + '\n')
with (HERE / 'build-admitted.log').open('w') as log:
    for command in rebuild:
        print('Building', command[-1], flush=True)
        log.write(json.dumps(command) + '\n'); log.flush()
        subprocess.run(command, cwd=str(HERE.parents[2] / '.workspaces/idea-resident-original-deadline-native'),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
