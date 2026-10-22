from pathlib import Path
import json,subprocess
w=Path('.workspaces/zig-engine-startup-memory-20261022')
with (w/'debug-auto.stdout').open('wb') as so,(w/'debug-auto.stderr').open('wb') as se:
 subprocess.run(json.loads((w/'debug-auto-command.json').read_text()),stdout=so,stderr=se,check=True,timeout=300)
assert json.loads((w/'debug-auto.json').read_text()) == json.loads((w/'model/auto-control.json').read_text())
print('Debug automatic-cache full frontier: PASS')
