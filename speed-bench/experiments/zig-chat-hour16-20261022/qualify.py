from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
root=Path.cwd();base=root/'.workspaces/zig-chat-20261022';out=base/'real';out.mkdir();corpus=out/'corpus';corpus.mkdir()
for f in sorted((root/'speed-bench/experiments/zig-tokenizer-hour8-20261022/corpus').glob('*.txt')):shutil.copy2(f,corpus/f.name)
(corpus/'rendered-glm.txt').write_text('[gMASK]<sop><|system|>Keep facts in checkable storage.\n<|user|>Read the code PINE-251.\n<|assistant|><think>Inspect it.</think>\n<|observation|>PINE-251 is current.\n<|user|>Return only the code.\n')
(corpus/'rendered-qwen.txt').write_text('<|im_start|>system\nKeep facts in checkable storage.<|im_end|>\n<|im_start|>user\nRead the code PINE-251.<|im_end|>\n<|im_start|>assistant\n<think>Inspect it.</think>\n<tool_call>{"name":"read","arguments":{"path":"note"}}</tool_call><|im_end|>\n<|im_start|>tool\nPINE-251 is current.<|im_end|>\n')
files=sorted(corpus.glob('*.txt'))
(corpus/'manifest.json').write_text(json.dumps([dict(label=p.stem,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))};rows=[]
binaries={'c-chat':root/'.workspaces/zig-pretokenizer-20261022/native/bin/test-deepseek41-gguf','zig-chat':base/'native/bin/test-deepseek41-gguf'}
models={'deepseek':root/'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf','glm':root/'gguf/GLM-5.3-Flash-Q2.gguf','qwen':root/'gguf/Qwen3.8-Flash-Next-Q4.gguf'}
for family,model in models.items():
 reference=None
 for i,arm in enumerate(['c-chat','zig-chat','zig-chat','c-chat']):
  label=f'{family}-{i}-{arm}';command=[str(binaries[arm]),'--tokenizer-bench',str(model),'20',*map(str,files)]
  start=time.monotonic();proc=subprocess.run(command,env=env,capture_output=True,timeout=240)
  (out/(label+'.stdout')).write_bytes(proc.stdout);(out/(label+'.stderr')).write_bytes(proc.stderr)
  row=dict(label=label,family=family,arm=arm,command=command,seconds=time.monotonic()-start,returncode=proc.returncode,binary_sha256=hashlib.sha256(binaries[arm].read_bytes()).hexdigest());rows.append(row)
  (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');assert proc.returncode==0,label
  records=[json.loads(line) for line in proc.stdout.splitlines()];ids=[r['ids'] for r in records[1:]]
  if reference is None:reference=ids
  row['identical_ids']=ids==reference
  for r in records[1:]:r.pop('ids');r['label']=files[r['case']].stem
  row['records']=records
  (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({k:row[k] for k in ['label','seconds','identical_ids']}),flush=True)
  assert row['identical_ids'],label
