from pathlib import Path
import subprocess,json
w=Path('.workspaces/zig-metal-integration-20261022');b=w/'control/bin';rec=['python3','-B','.workspaces/zig-server-input-20261022/record.py'];clean=['python3','-B','speed-bench/experiments/zig-session-frontier-20261022/clean_exec.py','--settings','{"DS4_METAL_MATH_SAFE":"1"}']
for label,binary,args in [('kernels','ds4_test',['--metal-kernels']),('metal','test-deepseek41-metal',[]),('index','test-deepseek41-metal',['--index-scores']),('mpp','test-metal-dense-mpp',[]),('ssd','test-metal-ssd-experts',[]),('ssd-q4','test-metal-ssd-experts',['--q4']),('ssd-mx','test-metal-ssd-experts',['--mxfp4'])]:
 subprocess.run([*rec,str(w/('control-'+label)),*clean,str(b/binary),*args],check=True)
