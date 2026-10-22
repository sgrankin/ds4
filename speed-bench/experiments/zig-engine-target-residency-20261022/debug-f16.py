from pathlib import Path
import json,subprocess
w=Path('.workspaces/zig-engine-target-residency-20261022')
with (w/'debug-f16.stdout').open('wb') as so,(w/'debug-f16.stderr').open('wb') as se:
 subprocess.run(json.loads((w/'debug-f16-command.json').read_text()),stdout=so,stderr=se,check=True,timeout=300)
assert json.loads((w/'debug-f16.json').read_text()) == json.loads((w/'model/f16-bytes-control.json').read_text())
print('Debug F16 target-residency full frontier: PASS')
