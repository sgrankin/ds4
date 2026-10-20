#!/usr/bin/env python3
import json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
results=[]
for name,mode in [('repeat-native-before','native'),('repeat-tail128','suffix'),('repeat-native-after','native')]:
 if time.time()>1791044450: break
 argv=[sys.executable,'-B',str(HERE/'run_repeat.py'),'--model','/Volumes/Models/ds4/gguf/DeepSeek-V4.1-Flash-Q2.gguf','--mode',mode,'--tail','128','--output',str(HERE/'runs'/name)]
 print(json.dumps(dict(event='start',arm=name)),flush=True)
 with (HERE/(name+'.log')).open('w') as out: p=subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT)
 results.append(dict(arm=name,returncode=p.returncode))
 (HERE/'repeat-campaign-result.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(dict(event='done',arm=name,returncode=p.returncode)),flush=True)
