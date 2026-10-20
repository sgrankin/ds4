#!/usr/bin/env python3
import json, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
arms=[('short-native-before','native',128,24),('short-tail128','suffix',128,24),('short-native-after','native',128,24),('long-native-before','native',128,64),('long-tail256','suffix',256,64),('long-tail128','suffix',128,64),('long-native-after','native',128,64)]
results=[]
for name,mode,tail,notes in arms:
 if time.time() > 1791044450:  #16:20:50 UTC, reserve15min for inspection and commit
  print('STOP: analysis reserve reached',flush=True);break
 argv=[sys.executable,'-B',str(HERE/'run.py'),'--model','/Volumes/Models/ds4/gguf/DeepSeek-V4.1-Flash-Q2.gguf','--mode',mode,'--tail',str(tail),'--notes',str(notes),'--output',str(HERE/'runs'/name)]
 print(json.dumps(dict(event='start',arm=name,argv=argv)),flush=True)
 with (HERE/(name+'.log')).open('w') as out:
  p=subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT)
 results.append(dict(arm=name,returncode=p.returncode))
 (HERE/'campaign-result.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(dict(event='done',arm=name,returncode=p.returncode)),flush=True)
