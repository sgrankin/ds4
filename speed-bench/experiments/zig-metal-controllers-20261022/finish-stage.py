from pathlib import Path
import subprocess,re,json
r=Path.cwd();w=r/'.workspaces/zig-metal-integration-20261022';stage=w/'stage';records=[]
def resolve(path,label):
 text=path.read_text();removed=[x[1:] for x in text.splitlines() if x.startswith('-') and not x.startswith('---')];added=[x[1:] for x in text.splitlines() if x.startswith('+') and not x.startswith('+++')]
 target=Path(str(path)[:-4]);s=target.read_text()
 if removed:
  assert all(x.startswith('#if defined(DS4_NATIVE_METAL_') for x in removed),text
  conditions=[x for x in added if x.startswith("#if defined(")]
  tail=[x for x in added if not x.startswith("#if defined(")]
  assert all(not x or x.startswith(("#include","#ifdef","#endif")) for x in tail),text
  for x in conditions:
   assert x.startswith('#if defined(DS4_NATIVE_METAL_'),text
   flags=re.findall(r'defined\(([^)]+)\)',x)
   anchor='#include "zig/metal/bridge.inc.m"';pos=s.index(anchor);a=s.rfind('#if ',0,pos);b=s.index('\n',a)
   for flag in flags:
    if 'defined('+flag+')' not in s[a:b]:s=s[:b]+' || defined('+flag+')'+s[b:];b=s.index('\n',a)
  s+="\n"+"\n".join(tail)+"\n"
 else:
  assert all(not x or x.startswith(('#include','#ifdef','#endif','#if ')) for x in added),text
  if all(not x or x.startswith('#include') for x in added) and target.name=='ds4_metal.m':
   a=s.index('#include "ds4_gpu.h"')+len('#include "ds4_gpu.h"');s=s[:a]+'\n'+'\n'.join(added)+s[a:]
  else:s+='\n'+'\n'.join(added)+'\n'
 target.write_text(s);path.rename(w/(label+'-'+path.name));print('resolved include context',label,flush=True)
resolve(stage/'ds4_metal.m.rej','embedding-activation')
def apply(owner,name):
 f=r/'.workspaces'/('zig-metal-'+owner+'-20261022')/name
 x=subprocess.run(['patch','--forward','--batch','--no-backup-if-mismatch','-p1','-i',str(f)],cwd=stage,text=True,capture_output=True)
 records.append(dict(owner=owner,patch=name,returncode=x.returncode,output=x.stdout+x.stderr));(w/'finish-patch-check.json').write_text(json.dumps(records,indent=2)+'\n');print(owner,name,x.returncode,flush=True)
 if x.returncode:
  rejs=list(stage.rglob('*.rej'));assert rejs,x.stdout+x.stderr
  for rej in rejs:resolve(rej,owner+'-'+name.replace('.patch',''))
for owner,names in [('embedding',['bridge.patch']),('quant',['activation.patch'])]:
 for name in names:apply(owner,name)
