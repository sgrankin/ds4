from pathlib import Path
import subprocess,json,shutil
r=Path.cwd();w=r/'.workspaces/zig-metal-integration-20261022';stage=w/'stage';records=[]
def apply(owner,name):
 f=r/'.workspaces'/('zig-metal-'+owner+'-20261022')/name
 x=subprocess.run(['patch','--forward','--batch','--no-backup-if-mismatch','-p1','-i',str(f)],cwd=stage,text=True,capture_output=True)
 records.append(dict(owner=owner,patch=name,returncode=x.returncode,output=x.stdout+x.stderr));(w/'extend-patch-check.json').write_text(json.dumps(records,indent=2)+'\n')
 print(records[-1],flush=True)
 if x.returncode:raise SystemExit(1)
for owner,names in [('rope',['activation.patch','bridge.patch']),('compressor',['records.patch','activation.patch','bridge.patch']),('scheduling',['records.patch','activation.patch']),('embedding',['records.patch','activation.patch','bridge.patch'])]:
 for name in names:apply(owner,name)
