from pathlib import Path
import csv,hashlib,json,os,subprocess,time
root=Path.cwd();base=root/'.workspaces/zig-metal-norm-20261022';out=base/'real-2';out.mkdir();env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))};rows=[]
def run(label,command):
 start=time.monotonic();p=subprocess.run(list(map(str,command)),env=env,capture_output=True,timeout=300);(out/(label+'.stdout')).write_bytes(p.stdout);(out/(label+'.stderr')).write_bytes(p.stderr)
 row=dict(label=label,command=list(map(str,command)),seconds=time.monotonic()-start,returncode=p.returncode,binary_sha256=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest());rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(label=label,returncode=p.returncode,seconds=row['seconds'])),flush=True);assert p.returncode==0,label;return row
for arm in ['c','native']:run(arm+'-norm-gpu',[base/arm/'bin/test-metal-norm-bf16'])
bins={'c-controller':root/'.workspaces/zig-cli-frontend-20261022/native/bin/ds4-bench','zig-controller':base/'native/bin/ds4-bench'};reference=None
for i,arm in enumerate(['c-controller','zig-controller','zig-controller','c-controller']):
 label=f'{i}-{arm}';p=out/label;p.mkdir();(p/'logits').mkdir()
 row=run(label,[bins[arm],'--model',root/'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf','--metal','--ssd-streaming','--ssd-streaming-cache-experts','48GB','--prompt-file',root/'.workspaces/zig-engram-20261022/bench/prompt.txt','--ctx-start','512','--ctx-max','512','--ctx-alloc','768','--prefill-chunk','512','--gen-tokens','32','--teacher-forced-decode','--csv',p/'metrics.csv','--dump-frontier-logits-dir',p/'logits'])
 logits=json.loads(next((p/'logits').glob('*.json')).read_text())['logits']
 if reference is None:reference=logits
 row['identical_logits']=logits==reference;row['metrics']=list(csv.DictReader((p/'metrics.csv').open()));(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');assert row['identical_logits'],label
