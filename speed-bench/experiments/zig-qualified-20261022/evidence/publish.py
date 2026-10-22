from pathlib import Path
import json,gzip,shutil,hashlib
r=Path.cwd();w=r/'.workspaces/zig-qualified-20261022';d=r/'speed-bench/experiments/zig-qualified-20261022';e=d/'evidence';e.mkdir(parents=True,exist_ok=True)
def copy(p,rel):
 dst=e/rel;dst.parent.mkdir(parents=True,exist_ok=True)
 if p.stat().st_size>100000:dst.with_name(dst.name+'.gz').write_bytes(gzip.compress(p.read_bytes(),mtime=0))
 else:shutil.copy2(p,dst)
for p in w.iterdir():
 if p.is_file() and p.suffix in ['.json','.txt','.py','.sh','.stdout','.stderr']:copy(p,p.name)
for name in ['fixtures','gpu','diagnostics']:
 for p in (w/name).iterdir():
  if p.is_file():copy(p,Path(name)/p.name)
for p in w.glob('guard-*'):
 for f in p.iterdir():
  if f.is_file():copy(f,Path(p.name)/f.name)
for p in (w/'ordinary').rglob('*'):
 if p.is_file() and 'logits' not in p.parts:copy(p,p.relative_to(w))
frontiers=list((w/'ordinary').glob('*/logits/*.json'))
(e/'ordinary/frontier-files.json').write_text(json.dumps({str(p.relative_to(w)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frontiers},indent=2)+'\n')
copy(frontiers[0],Path('ordinary/frontier.logits.json'))
for arm in ['c','native']:
 for p in (w/'agent'/arm/'run').iterdir():
  if p.is_file():copy(p,p.relative_to(w))
 for p in (w/'agent'/arm/'shared/work').glob('*.json'):copy(p,p.relative_to(w))
copy(w/'agent/results.json',Path('agent/results.json'))
for name in ['http','trained','vision','image-cache']:
 for p in (w/name).rglob('*'):
  if p.is_file() and not any(x in p.relative_to(w/name).parts for x in ['cache','responses']):copy(p,p.relative_to(w))
builds=r/'.workspaces/zig-qualified-builds-20261022'
for p in builds.iterdir():
 if p.is_file():copy(p,Path('builds')/p.name)
expected=json.loads((w/'source-freeze.json').read_text());actual={v:hashlib.sha256((r/v).read_bytes()).hexdigest() for v in expected};assert actual==expected,'active source changed during qualification'
(e/'source-after.json').write_text(json.dumps(actual,indent=2)+'\n')
(e/'binary-hashes.json').write_text(json.dumps({arm+'/'+name:hashlib.sha256((w/arm/'bin'/name).read_bytes()).hexdigest() for arm in ['c','native','native-debug'] for name in ['ds4','ds4-agent','ds4-server','ds4-bench','ds4-eval']},indent=2)+'\n')
print('published final evidence; source unchanged')
