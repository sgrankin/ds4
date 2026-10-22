from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path.cwd();base=root/'.workspaces/zig-sampling-20261022';out=base/'real-2';out.mkdir();rows=[]
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))}
bins={'c-sampler':root/'.workspaces/zig-combined-20261022/native/bin/ds4','native-sampler':base/'native/bin/ds4'}
common=['--model',str(root/'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'),'--ssd-streaming','--ssd-streaming-cache-experts','48GB','--ctx','1024','--prefill-chunk','512','--mtp-draft','1','--seed','73','--nothink','--tokens','48']
cases=[('greedy',['--temp','0','-p','Reply with exactly ORCHID42.'],{}),('nucleus',['--temp','0.7','--top-p','0.9','--min-p','0'],{}),('default-min-p',['--temp','1','--top-p','1','--min-p','0.05'],{}),('top-p-min-p',['--temp','1','--top-p','0.9','--min-p','0.05'],{}),('full-softmax',['--temp','0.7','--top-p','1','--min-p','0'],{}),('penalty',['--temp','0.7','--top-p','0.9','--min-p','0.05'],{'DS4_OVERTHINK_PENALTY':'1.5','DS4_OVERTHINK_SCOPE':'all'})]
for name,options,extra in cases:
 reference=None
 for arm,binary in bins.items():
  args=common+options
  if '-p' not in options:args+=['-p','Write one short sentence describing a quiet garden.']
  command=[str(binary),*args];start=time.monotonic();p=subprocess.run(command,capture_output=True,env=env|extra,timeout=180)
  label=name+'-'+arm;(out/(label+'.stdout')).write_bytes(p.stdout);(out/(label+'.stderr')).write_bytes(p.stderr)
  if reference is None:reference=p.stdout
  row=dict(label=label,command=command,env=extra,returncode=p.returncode,seconds=time.monotonic()-start,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),identical_output=p.stdout==reference,nonempty=bool(p.stdout.strip()),penalty_enabled=b'overthinking logit penalty' in p.stderr)
  rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
  assert p.returncode==0 and row['nonempty'],label
  if name=='greedy':assert b'ORCHID42' in p.stdout,label
  if name=='penalty':assert row['penalty_enabled'],label
# Output differences are recorded for inspection, not automatic quality rejection.
