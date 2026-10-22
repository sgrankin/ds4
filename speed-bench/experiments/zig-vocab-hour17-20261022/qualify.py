from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path.cwd();base=root/'.workspaces/zig-vocab-20261022';out=base/'real';out.mkdir();rows=[];env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))}
models=['DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf','GLM-5.3-Flash-Q2.gguf','Qwen3.8-Flash-Next-Q4.gguf']
for model in models:
 reference=None
 for arm in ['c','native']:
  binary=base/arm/'bin/test-deepseek41-gguf';command=[str(binary),'--token-text',str(root/'gguf'/model)];start=time.monotonic();p=subprocess.run(command,env=env,capture_output=True,timeout=180)
  label=model.split('-')[0]+'-'+arm;(out/(label+'.stdout')).write_bytes(p.stdout);(out/(label+'.stderr')).write_bytes(p.stderr);assert p.returncode==0,label
  records=[json.loads(line) for line in p.stdout.splitlines()];assert records[0]=={'id':-1,'bytes':[]} and records[-1]['bytes']==[];assert len(records)==records[-1]['id']+2
  if reference is None:reference=p.stdout
  row=dict(label=label,command=command,seconds=time.monotonic()-start,returncode=p.returncode,vocab=records[-1]['id'],identical_all_decoded_bytes=p.stdout==reference,decoded_sha256=hashlib.sha256(p.stdout).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest());rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True);assert row['identical_all_decoded_bytes'],label
# Same real-vocabulary tokenizer vectors, including borrowed-map use after load.
files=sorted((root/'speed-bench/experiments/zig-chat-hour16-20261022/corpus').glob('*.txt'))
for model in models:
 reference=None
 for arm in ['c','native']:
  binary=base/arm/'bin/test-deepseek41-gguf';command=[str(binary),'--tokenizer-bench',str(root/'gguf'/model),'20',*map(str,files)];start=time.monotonic();p=subprocess.run(command,env=env,capture_output=True,timeout=180)
  label=model.split('-')[0]+'-'+arm+'-tokenizer';(out/(label+'.stdout')).write_bytes(p.stdout);(out/(label+'.stderr')).write_bytes(p.stderr);assert p.returncode==0,label
  records=[json.loads(line) for line in p.stdout.splitlines()];ids=[r.pop('ids') for r in records[1:]]
  if reference is None:reference=ids
  row=dict(label=label,command=command,seconds=time.monotonic()-start,returncode=p.returncode,identical_ids=ids==reference,records=records,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest());rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({k:row[k] for k in ['label','seconds','identical_ids']}),flush=True);assert row['identical_ids'],label
