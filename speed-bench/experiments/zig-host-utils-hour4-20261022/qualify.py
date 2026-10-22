from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path.cwd();base=root/'.workspaces/zig-host-utils-20261022';out=base/'cli';out.mkdir()
prefix=out/'prefix.txt';prefix.write_text('USER: Remember the archive word ORCHID42.\nASSISTANT: The archive word is ORCHID42.\n')
rows=[]
for arm in ['c','native']:
    binary=base/arm/'bin/ds4'
    command=[str(binary),'--model',str(root/'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'),'--ssd-streaming','--ssd-streaming-cache-experts','48GB','--ctx','2048','--prefill-chunk','512','--nothink','--temp','0','-n','16','--prefix-file',str(prefix),'-p','Reply with only the archive word, without punctuation.']
    env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))}
    t=time.monotonic();p=subprocess.run(command,env=env,capture_output=True,timeout=300)
    (out/(arm+'.stdout')).write_bytes(p.stdout);(out/(arm+'.stderr')).write_bytes(p.stderr)
    row=dict(arm=arm,command=command,seconds=time.monotonic()-t,returncode=p.returncode,answer=p.stdout.decode().strip(),sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    row['pass']=p.returncode==0 and row['answer']=='ORCHID42';rows.append(row)
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
    if not row['pass']:raise SystemExit(3)
