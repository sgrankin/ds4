from pathlib import Path
import subprocess,json,hashlib
r=Path.cwd();w=r/'.workspaces/zig-qualified-20261022';rows=[]
for name in ['ds4','ds4-agent','ds4-server','ds4-bench','ds4-eval']:
 b=r/'zig-out/bin'/name;p=subprocess.run([str(b),'--help'],capture_output=True);rows.append(dict(program=name,returncode=p.returncode,sha256=hashlib.sha256(b.read_bytes()).hexdigest()));assert p.returncode==0,(name,p.stderr)
for server in [False,True]:
 p=subprocess.run(['./ds4-local','--bin-dir',str(r/'zig-out/bin'),*(['--server'] if server else []),'persistent-fast','--help'],capture_output=True);rows.append(dict(preset_server=server,returncode=p.returncode));assert p.returncode==0,p.stderr
(w/'install-checks.json').write_text(json.dumps(rows,indent=2)+'\n')
rec=['python3','-B','.workspaces/zig-server-input-20261022/record.py'];clean=['python3','-B','speed-bench/experiments/zig-session-frontier-20261022/clean_exec.py']
for arm,binary in [('qualified',w/'native/bin/ds4-agent'),('installed',r/'zig-out/bin/ds4-agent')]:
 settings={'DS4_AGENT_CACHE_DIR':str(w/('install-'+arm+'-agent-cache')),'DS4_STATE_CACHE_DIR':str(w/('install-'+arm+'-state-cache'))}
 subprocess.run([*rec,str(w/('install-'+arm+'-raw')),*clean,'--settings',json.dumps(settings),str(binary),'--model','gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf','--ssd-streaming','--ssd-streaming-cache-experts','32GB','--ctx','512','--prefill-chunk','128','--tokens','8','--temp','0','--seed','1','--non-interactive','--raw-prompt','--prompt','Complete the sequence: one, two, three,'],check=True)
a=(w/'install-qualified-raw.stdout').read_bytes();b=(w/'install-installed-raw.stdout').read_bytes();assert a and a==b,(a,b)
print('five installed help checks, both preset selections and actual paired continuation pass')
