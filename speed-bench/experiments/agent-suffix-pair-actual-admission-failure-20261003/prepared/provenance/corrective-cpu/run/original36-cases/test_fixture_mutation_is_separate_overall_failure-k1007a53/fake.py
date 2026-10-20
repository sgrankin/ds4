
import hashlib,json,os,sys,time
from pathlib import Path
p=Path(os.environ['FAKE_PRIVATE']);trace=Path(os.environ['FAKE_TRACE']);project=Path(os.environ['FAKE_PROJECT']);mode=os.environ['FAKE_MODE']
def marker():
 os.write(1,b'+DWARFSTAR_');time.sleep(.003);os.write(1,b'WAITING\n')
def record(turn,answer):
 r=dict(turn=turn,tool_round=0,generated=1,carried=0,limit=512,configured_limit=512,context_limited=False,model_stopped=True,got_tool=False,malformed_tool=False,early_tool_error=False,interrupted=False,at_limit=False,calls=[])
 if mode=='cap' and turn==2:r.update(generated=512,at_limit=True)
 if not(mode=='missinground' and turn==2):
  with (p/'rounds.jsonl').open('a') as f:json.dump(r,f);f.write('\n')
 with trace.open('a') as f:
  f.write('trace user="literal"\n');f.write('prefill sync done tool_round=0\n')
  raw=answer.encode();f.write(' token index=1 id=3 bytes='+str(len(raw))+' text="literal" hex='+raw.hex()+'\n')
  if not(mode=='truncatedtrace' and turn==1):f.write('generation finished tool_round=0 generated=1\n')
 os.write(1,b'PAIR_ACTUAL_ROUND native fake protocol fixture\n')
marker()
for turn in range(1,4):
 payload=os.read(0,4096)
 if not payload:sys.exit(11)
 (p/('prompt-'+str(turn)+'.bin')).write_bytes(payload)
 answer='CODE=9999' if mode=='wrong' and turn==1 else 'CODE=2718' if turn==1 else 'READY' if turn==2 else 'PASS'
 if mode=='mutation' and turn==2:(project/'run_grader.py').write_text('modified')
 if mode=='eligiblefail' and turn==3:
  (p/'selection-failure.txt').write_text('No frozen candidate eligible')
  sys.exit(1)
 if mode=='infraexit' and turn==2:sys.exit(9)
 record(turn,answer);marker()
 if mode=='duplicate' and turn==3:marker()
if mode=='hangeof':time.sleep(30)
else:
 while os.read(0,4096):pass
