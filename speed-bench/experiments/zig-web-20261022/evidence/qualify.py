from pathlib import Path
import subprocess,json
w=Path('.workspaces/zig-web-20261022');rec=['python3','-B','.workspaces/zig-server-input-20261022/record.py']
for arm,args in [('control',['-Dnative=false']),('native',[]),('debug',['-Doptimize=Debug'])]:
 if arm!='native':subprocess.run([*rec,str(w/(arm+'-build')),'tools/zig-build','web-test','ds4-agent',*args,'--prefix',str(w/arm)],check=True)
 subprocess.run([*rec,str(w/(arm+'-recovery')),'python3','-B','tests/test_web_recovery.py',str(w/arm/'bin/test-web-recovery')],check=True)
print('three-way actual local HTTP/WebSocket recovery passes',flush=True)
