#!/usr/bin/env python3
"""CPU-only previous-preparation inventory; no engine/model/guard execution."""
import hashlib,json,os,shutil
from pathlib import Path
SOURCE=Path('/Volumes/Models/ds4/.workspaces/idea-suffix-pair-tape')
OUT=Path(__file__).parent

def sha(path):
 path=Path(path);assert path.suffix.lower()!='.gguf'
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def identity(path):
 path=Path(path);assert path.suffix.lower()!='.gguf'
 return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def copy(path,relative):
 target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
 shutil.copy2(path,target);assert sha(path)==sha(target)

def main():
 records={}
 for version in ['v1','v2']:
  folder=SOURCE/'speed-bench/ideas/artifacts'/('suffix-pair-tape-odd65-prepared-20261002-'+version)
  manifest=json.loads((folder/'manifest.json').read_text());assets={}
  for name,pin in manifest['files'].items():
   assert not Path(name).is_absolute() and '..' not in Path(name).parts
   observed=identity(folder/name);assert observed['bytes']==pin['bytes'] and observed['sha256']==pin['sha256']
   assets[name]=observed
  for name in ['manifest.json','prepared-command.json']:
   copy(folder/name,'history/'+version+'/'+name)
  extra={}
  for name in ['root-preflight-proof.json','root-release-proof.json','pressure/result.json','pressure/command.json','pressure/pressure.jsonl','run/run-proof.json','run/old-native-before-command.json','run/old-native-before.csv','run/old-native-before.log']:
   path=folder/name
   if path.is_file():extra[name]=identity(path);copy(path,'history/'+version+'/'+name)
  records[version]=dict(directory=str(folder),manifest=identity(folder/'manifest.json'),command=identity(folder/'prepared-command.json'),declared_assets_verified=assets,extra_failure_proofs=extra,
   original_probe_execute_modes={n:dict(mode=oct((folder/n).stat().st_mode&0o777),executable=os.access(folder/n,os.X_OK)) for n in ['baseline-probe','new-probe']},
   run_scope='Prepared only, model unrun' if version=='v1' else 'Guarded launch failed execute bit before baseline process; metadata checked only inside guard',originals_unchanged=True)
 proof=SOURCE/'speed-bench/ideas/artifacts/suffix-pair-tape-header-cpu-v1/proof.json'
 assert sha(proof)=='08eaa89263c4ac239c69369a094c3dc087f0ee412acb94cb2611e7dd4796aecb'
 copy(proof,'cpu/tape-header-proof.json')
 cpu_folder=proof.parent
 for name in ['production-test.log','sanitized-test.log','first-failure.log','coverage-fix.diff']:
  path=cpu_folder/name
  if path.is_file():copy(path,'cpu/'+name)
 with (OUT/'historical-preparations.json').open('x') as f:
  json.dump(dict(preparations=records,tape_cpu_proof=identity(proof),model_assets_accessed=False,no_originals_changed=True),f,indent=2);f.write('\n')
 print(json.dumps(dict(status='HISTORICAL_PREPARATIONS_VERIFIED',versions=2,no_originals_changed=True,model_assets_accessed=False)))
if __name__=='__main__':main()
