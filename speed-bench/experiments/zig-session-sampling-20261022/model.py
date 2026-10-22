#!/usr/bin/env python3
"""Three fixed real-model Session sampling/diagnostic comparisons, serially."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,time
w=Path('.workspaces/zig-session-sampling-20261022');out=w/'model';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=['ds4.c','build.zig','ds4_session_sampling.h','zig/engine/session_sampling.zig','zig/engine/session_sampling_entry.zig','zig/engine/session_sampling_api.h','tests/test_session_state.c']
report={'source_before':{p:sha(p) for p in files},'runs':[],'comparisons':{}}
model=str(Path('gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf').resolve())
for case,settings,diagnostic in [
 ('ordinary',{},False),
 ('penalty',{'DS4_OVERTHINK_PENALTY':'0.75','DS4_OVERTHINK_SCOPE':'all','DS4_OVERTHINK_AFTER':'0'},False),
 ('penalty-logprobs',{'DS4_OVERTHINK_PENALTY':'0.75','DS4_OVERTHINK_SCOPE':'all','DS4_OVERTHINK_AFTER':'0'},True)]:
 for arm in ['control','native']:
  label=case+'-'+arm;binary=(w/arm/'bin/ds4').resolve()
  cmd=[sys.executable,'-B','speed-bench/experiments/zig-session-frontier-20261022/clean_exec.py','--settings',json.dumps(settings),str(binary),'--model',model,'--metal','--ssd-streaming','--ssd-streaming-cache-experts','32GB','--ctx','1024','--prefill-chunk','512','--tokens','24','--temp','0.7','--top-p','0.9','--min-p','0.05','--seed','17','--nothink','--prompt','Write exactly: perhaps maybe wait instead however.']
  if diagnostic:cmd += ['--dump-logprobs',str((out/(label+'.json')).resolve()),'--logprobs-top-k','6']
  start=time.monotonic()
  with (out/(label+'.stdout')).open('wb') as so,(out/(label+'.stderr')).open('wb') as se:r=subprocess.run(cmd,stdout=so,stderr=se,timeout=180)
  row={'label':label,'command':cmd,'settings':settings,'seconds':time.monotonic()-start,'returncode':r.returncode,'binary_sha256':sha(binary)}
  report['runs'].append(row);(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
  assert r.returncode==0,label
  if settings:assert re.search(r'overthinking logit penalty 0.75, scope=all.*mapped=[1-9]',(out/(label+'.stderr')).read_text()),label
 comparison={'stdout_identical':(out/(case+'-control.stdout')).read_bytes()==(out/(case+'-native.stdout')).read_bytes()}
 assert comparison['stdout_identical'],case
 if diagnostic:
  left=json.loads((out/(case+'-control.json')).read_text());right=json.loads((out/(case+'-native.json')).read_text())
  comparison['logprobs_json_identical']=left==right
  assert comparison['logprobs_json_identical'],case
 report['comparisons'][case]=comparison
report['source_after']={p:sha(p) for p in files};assert report['source_before']==report['source_after'];report['all_pass']=True
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
