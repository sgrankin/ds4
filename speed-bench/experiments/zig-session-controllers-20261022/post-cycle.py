from pathlib import Path
import subprocess,json
w=Path('.workspaces/zig-session-integration-20261022');rec=['python3','-B','.workspaces/zig-server-input-20261022/record.py'];clean=['python3','-B','speed-bench/experiments/zig-session-frontier-20261022/clean_exec.py']
for arm in ['native','debug']:
 b=w/arm/'bin/test-deepseek41-graph'
 for name,args in [('admission',['--batch-admission']),('vision-routing',['--vision-routing'])]:subprocess.run([*rec,str(w/(arm+'-'+name)),*clean,str(b),*args],check=True)
# Production image observation also checks the public encode/layout transaction.
b=w/'debug/bin/ds4_agent_test'
subprocess.run([*rec,str(w/'debug-image'),*clean,str(b),'--image-observation','gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf','gguf/DeepSeek-V4.1-Flash-1017-vision-BF16.gguf','speed-bench/experiments/zig-engine-vision-20261022/red.png'],check=True)
subprocess.run(['python3','-B','speed-bench/experiments/zig-server-protocols-http-20261022/qualify_http.py','--server',str(w/'native/bin/ds4-server'),'--restart-server',str(w/'debug/bin/ds4-server'),'--output',str(w/'http')],check=True)
