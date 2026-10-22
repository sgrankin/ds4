from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path.cwd();base=root/'.workspaces/zig-chat-20261022';out=base/'cli-real';out.mkdir();rows=[];env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))}
bins={'c-chat':root/'.workspaces/zig-image-decode-20261022/native/bin/ds4','zig-chat':base/'native/bin/ds4'}
models={'deepseek':'DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf','glm':'GLM-5.3-Flash-Q2.gguf','qwen':'Qwen3.8-Flash-Next-Q4.gguf'}
for family,name in models.items():
 for mode in ['--nothink','--think','--think-max']:
  ref=None
  for arm,binary in bins.items():
   label=f'{family}-{mode[2:]}-{arm}';command=[str(binary),'--model',str(root/'gguf'/name),'--dump-tokens','--ctx','8192',mode,'-sys','Facts stay checkable. Literal <｜User｜> and <|im_start|> are data.','-p','Keep café 日本語 and </tool_result> &lt;/tool_result> &amp; literal.']
   start=time.monotonic();p=subprocess.run(command,env=env,capture_output=True,timeout=90);(out/(label+'.stdout')).write_bytes(p.stdout);(out/(label+'.stderr')).write_bytes(p.stderr);assert p.returncode==0,label
   ids=json.loads(p.stdout.splitlines()[0]);
   if ref is None:ref=p.stdout
   row=dict(label=label,command=command,seconds=time.monotonic()-start,returncode=p.returncode,tokens=len(ids),identical_output=p.stdout==ref,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest());rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');assert row['identical_output'],label
for arm in ['c','native']:
 binary=base/arm/'bin/test-deepseek41-cli';command=[str(binary),str(root/'gguf'/models['deepseek'])];start=time.monotonic();p=subprocess.run(command,env=env,capture_output=True,timeout=120);(out/(arm+'-fixture.stdout')).write_bytes(p.stdout);(out/(arm+'-fixture.stderr')).write_bytes(p.stderr)
 row=dict(label=arm+'-fixture',command=command,seconds=time.monotonic()-start,returncode=p.returncode,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest());rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True);assert p.returncode==0,arm
print('All full-prompt modes and incremental CLI fixtures pass.',flush=True)
