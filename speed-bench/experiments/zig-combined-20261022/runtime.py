from pathlib import Path
import json,subprocess,time
root=Path.cwd();base=root/'.workspaces/zig-combined-20261022';out=base/'runtime';out.mkdir();rows=[]
model=root/'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
def run(label,command):
 command=list(map(str,command));start=time.monotonic()
 with (out/(label+'.stdout')).open('wb') as stdout,(out/(label+'.stderr')).open('wb') as stderr:p=subprocess.run(command,cwd=root,stdout=stdout,stderr=stderr,timeout=1200)
 row=dict(label=label,command=command,returncode=p.returncode,seconds=time.monotonic()-start);rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True);assert p.returncode==0,label
for first,second in [('c','native'),('native','c')]:
 label=first+'-to-'+second
 run(label,['python3','-B',root/'speed-bench/experiments/zig-combined-20261022/serve.py','--model',model,'--server',base/first/'bin/ds4-server','--restart-server',base/second/'bin/ds4-server','--output',out/label,'--expert-cache','48GB','--timeout','600'])
run('agent',['python3','-B',root/'speed-bench/experiments/zig-combined-20261022/run_agent.py','--model',model,'--agent',base/'native/bin/ds4-agent','--output',out/'agent/run','--shared',out/'agent/shared','--mode','suffix','--prerope','1','--tail','128','--notes','24','--blob-units','128','--timeout','600'])
run('vision',['python3','-B',root/'speed-bench/experiments/zig-combined-20261022/vision.py'])
