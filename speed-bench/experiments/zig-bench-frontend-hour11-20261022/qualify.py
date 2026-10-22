from pathlib import Path
import csv, hashlib, json, os, re, subprocess, time
root=Path.cwd();base=root/'.workspaces/zig-bench-frontend-20261022';out=base/'real';out.mkdir()
clean={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))}
rows=[]
bins={'c':root/'.workspaces/zig-metal-hc-20261022/native/bin/ds4-bench','zig':base/'native/bin/ds4-bench'}
model=root/'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
prompt=root/'.workspaces/zig-engram-20261022/bench/prompt.txt'
for profile,settings,extra in [
 ('snapshot',{},['--ctx-start','64','--ctx-max','128','--step-incr','64','--gen-tokens','8','--teacher-forced-decode']),
 ('replay',{'DS4_BENCH_DISABLE_SNAPSHOT':'1'},['--ctx-start','64','--ctx-max','128','--step-incr','64','--gen-tokens','8','--teacher-forced-decode']),
 ('multiplicative',{},['--ctx-start','32','--ctx-max','128','--step-mul','2','--gen-tokens','2'])]:
 reference=None
 for arm in ['c','zig']:
  label=profile+'-'+arm;p=out/label;p.mkdir();(p/'logits').mkdir()
  command=[str(bins[arm]),'--model',str(model),'--metal','--ssd-streaming','--ssd-streaming-cache-experts','48GB',
    '--prompt-file',str(prompt),'--ctx-alloc','256','--prefill-chunk','512','--show-output',
    '--csv',str(p/'metrics.csv'),'--dump-frontier-logits-dir',str(p/'logits')]+extra
  start=time.monotonic();proc=subprocess.run(command,env=clean|settings,capture_output=True,timeout=240)
  (p/'stdout').write_bytes(proc.stdout);(p/'stderr').write_bytes(proc.stderr)
  row=dict(label=label,command=command,environment=settings,seconds=time.monotonic()-start,returncode=proc.returncode,binary_sha256=hashlib.sha256(bins[arm].read_bytes()).hexdigest());rows.append(row)
  (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
  assert proc.returncode==0,label
  metrics=list(csv.DictReader((p/'metrics.csv').open()));row['metrics']=metrics
  frontiers={f.name:json.loads(f.read_text()) for f in (p/'logits').glob('*.json')}
  # Timing fields live in CSV; serialized full frontier diagnostics must match.
  decoded=re.findall(rb'ds4-bench: gen\[ctx=(\d+)\] decoded text: \"(.*?)\"\n',proc.stderr,re.S)
  assert len(decoded)==len(metrics),label+' decoded outputs'
  if reference is None:reference=(frontiers,decoded)
  row['identical_frontiers']=frontiers==reference[0];row['identical_decoded_output']=decoded==reference[1]
  assert row['identical_frontiers'],label+' logits'
  assert row['identical_decoded_output'],label+' output'
  assert [int(x['prefill_tokens']) for x in metrics]==([64,64] if profile!='multiplicative' else [32,32,64])
  assert all(int(x['gen_tokens'])==(8 if profile!='multiplicative' else 2) for x in metrics)
  assert (int(metrics[0]['kvcache_bytes'])>0)==(profile!='replay')
  (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
