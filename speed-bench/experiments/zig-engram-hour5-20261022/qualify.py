from pathlib import Path
import csv,hashlib,json,os,subprocess,time
root=Path.cwd();base=root/'.workspaces/zig-engram-20261022';out=base/'bench';out.mkdir()
source=json.loads((root/'.workspaces/zig-restart-20261022/c-to-zig-48/fixtures/fixture.json').read_text())
prompt=out/'prompt.txt';prompt.write_text(source['prompt'])
rows=[]; reference=None
for i,arm in enumerate(['c','native','native','c']):
    run=out/f'{i}-{arm}';run.mkdir();(run/'logits').mkdir()
    binary=base/arm/'bin/ds4-bench'
    command=[str(binary),'--model',str(root/'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'),'--metal','--ssd-streaming','--ssd-streaming-cache-experts','48GB','--prompt-file',str(prompt),'--ctx-start','2048','--ctx-max','2048','--ctx-alloc','2304','--prefill-chunk','512','--gen-tokens','128','--teacher-forced-decode','--csv',str(run/'metrics.csv'),'--dump-frontier-logits-dir',str(run/'logits')]
    env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))}
    t=time.monotonic()
    with (run/'stdout').open('wb') as stdout,(run/'stderr').open('wb') as stderr:p=subprocess.run(command,env=env,stdout=stdout,stderr=stderr,timeout=600)
    row=dict(arm=arm,command=command,seconds=time.monotonic()-t,returncode=p.returncode,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    if p.returncode:raise SystemExit('failed '+str(run))
    row['metrics']=list(csv.DictReader((run/'metrics.csv').open()))
    logits=json.loads(next((run/'logits').glob('*.json')).read_text())['logits']
    if reference is None:reference=logits
    row['max_abs_logit_delta']=max(abs(a-b) for a,b in zip(reference,logits));row['identical_logits']=reference==logits
    rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(arm=arm,seconds=row['seconds'],identical_logits=row['identical_logits'],metrics=row['metrics'])),flush=True)
