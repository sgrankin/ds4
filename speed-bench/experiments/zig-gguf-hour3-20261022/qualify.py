from pathlib import Path
import hashlib,json,subprocess,time
root=Path.cwd(); out=root/'.workspaces/zig-gguf-20261022/real'
out.mkdir(exist_ok=False); rows=[]
def run(label,command):
    t=time.monotonic(); p=subprocess.run(list(map(str,command)),capture_output=True,timeout=120)
    row=dict(label=label,command=list(map(str,command)),seconds=time.monotonic()-t,returncode=p.returncode,stdout_sha256=hashlib.sha256(p.stdout).hexdigest())
    (out/(label+'.stdout')).write_bytes(p.stdout);(out/(label+'.stderr')).write_bytes(p.stderr)
    rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    if p.returncode: raise SystemExit('failed: '+label)
    return p.stdout
for mode in ['c','native','c-debug','native-debug']:
    run(mode+'-sparse',[root/'.workspaces/zig-gguf-20261022'/mode/'bin/test-deepseek41-gguf'])
model=root/'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
servers={'c':root/'.workspaces/zig-kvstore-20261022/c/bin/ds4','native':root/'.workspaces/zig-gguf-20261022/native/bin/ds4'}
reference=None
for index,mode in enumerate(['c','native','native','c']):
    result=run(f'{index}-{mode}-tokenize',[servers[mode],'--model',model,'--raw','--dump-tokens','-p','Small, readable modules preserve exact tokens: MAPLE-4826. Unicode: café 日本語.'])
    if reference is None: reference=result
    elif result != reference: raise SystemExit('token output mismatch')
print(json.dumps(dict(pass_=True,runs=len(rows),identical_token_output=True)),flush=True)
