from pathlib import Path
import hashlib,json,subprocess,time
root=Path.cwd();base=root/'.workspaces/zig-qualified-20261022';out=base/'fixtures';out.mkdir();rows=[]
common=['web-test','graph-test','session-state-gpu-test','metal-test','command-memory-test','shared-bf16-test','ssd-experts-test','dense-mpp-test','agent-test','server-test','state-cache-test','session-state-test','tokenizer-test','pretokenizer-test','gguf-test','ssd-test','prompt-prefix-test','engram-test','image-test','vision-image-test','sampling-test','resident-route-test','response-store-test']
extra=['vocab-test','native-f16-publication-test','native-hc-projection-test','native-norm-test','native-activation-test']
executables=['ds4_agent_test','test-session-state','test-tokenizer-piece','test-pretokenizer','test-gguf-cursor','test-ssd-cache','test-prompt-prefix','test-engram','test-image-decode','test-deepseek4-vision-image','test-sampling','test-resident-route','test-resident-three','test-resident-drop']
extra_executables=['test-vocab','test-f16-publication','test-hc-projection','test-norm-cpu','test-activation']
def run(label,command):
 command=list(map(str,command));start=time.monotonic();p=subprocess.run(command,cwd=root,capture_output=True,timeout=180)
 (out/(label+'.stdout')).write_bytes(p.stdout);(out/(label+'.stderr')).write_bytes(p.stderr)
 row=dict(label=label,command=command,returncode=p.returncode,seconds=time.monotonic()-start);rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True);assert p.returncode==0,label
for arm in ['c','native']:
 for mode in ['ReleaseFast','Debug']:
  label=arm+('-debug' if mode=='Debug' else '');prefix=base/label
  run(label+'-build',[root/'tools/zig-build','-Dnative='+str(arm=='native').lower(),'-Doptimize='+mode,'--prefix',prefix,'install',*common,*(extra if arm=='native' else []),*(['dspark-c-oracle','dspark-native-oracle'] if arm=='native' else [])])
  for exe in executables+(extra_executables if arm=='native' else []):run(label+'-'+exe,[prefix/'bin'/exe])
  run(label+'-web',['python3','-B','tests/test_web_recovery.py',prefix/'bin/test-web-recovery'])
  run(label+'-eval',[prefix/'bin/ds4-eval','--self-test-extractors'])
  run(label+'-eval-data',[prefix/'bin/ds4-eval','--validate-cases'])
  run(label+'-admission',[prefix/'bin/test-deepseek41-graph','--batch-admission'])
  run(label+'-vision-routing',[prefix/'bin/test-deepseek41-graph','--vision-routing'])
  run(label+'-server',[prefix/'bin/ds4_test','--server'])
  run(label+'-cache',[prefix/'bin/test-state-cache',out/(label+'-cache-files')])
  run(label+'-response-store',[prefix/'bin/test-response-store',out/(label+'-response-files')])
identities={}
for arm in ['c','native']:
 for program in ['ds4','ds4-agent','ds4-server','ds4-bench','ds4-eval']:
  path=base/arm/'bin'/program;identities[f'{arm}/{program}']=hashlib.sha256(path.read_bytes()).hexdigest()
(out/'binaries.json').write_text(json.dumps(identities,indent=2)+'\n')
