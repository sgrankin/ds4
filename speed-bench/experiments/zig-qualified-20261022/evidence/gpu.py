from pathlib import Path
import subprocess,json,shutil
w=Path('.workspaces/zig-qualified-20261022');rec=['python3','-B','.workspaces/zig-server-input-20261022/record.py'];clean=['python3','-B','speed-bench/experiments/zig-session-frontier-20261022/clean_exec.py','--settings','{"DS4_METAL_MATH_SAFE":"1"}'];out=w/'gpu';out.mkdir(exist_ok=True)
checks=[('ds4_test',['--metal-kernels']),('test-deepseek41-metal',[]),('test-metal-shared-bf16',[]),('test-metal-dense-mpp',[]),*[('test-metal-command-memory',[x]) for x in ['parallel-ffn','deferred-kv','row','session','batch','big','exact-cache-evict','exact-cache-unlimited']],*[('test-metal-ssd-experts',x) for x in [[],['--q4'],['--mxfp4']]]]
for arm in ['c','native','native-debug']:
 for n,(binary,args) in enumerate(checks):
  if arm=='c' and n<10:continue
  settings={'DS4_METAL_MATH_SAFE':'1'}
  if args and args[0].startswith('exact-cache-'):settings['DS4_METAL_EXACT_VIEW_CACHE_MIB']='1' if args[0]=='exact-cache-evict' else '0'
  current_clean=[*clean[:-1],json.dumps(settings)]
  subprocess.run([*rec,str(out/(arm+'-'+str(n))),*current_clean,str(w/arm/'bin'/binary),*args],check=True)
print('final C/native/Debug focused and full GPU fixtures pass',flush=True)
