from pathlib import Path
import subprocess,json,shutil
w=Path('.workspaces/zig-qualified-20261022');out=w/'diagnostics';out.mkdir(exist_ok=True);rec=['python3','-B','.workspaces/zig-server-input-20261022/record.py'];clean=['python3','-B','speed-bench/experiments/zig-session-frontier-20261022/clean_exec.py']
for stage in ['record','merge-replay']:
 for arm in ['c','native','native-debug']:
  trace=out/(arm+'.ids');hot=out/(arm+'.hotlist')
  settings={'DS4_METAL_MATH_SAFE':'1','DS4_MOE_RECORD_SELECTED_HOTLIST':str(hot)}
  if stage=='record':settings['DS4_MOE_RECORD_SELECTED_IDS']=str(trace)
  else:settings.update(DS4_MOE_REPLAY_SELECTED_IDS=str(out/'c.ids'),DS4_MOE_RECORD_SELECTED_HOTLIST_MERGE='1')
  subprocess.run([*rec,str(out/(arm+'-'+stage)),*clean,'--settings',json.dumps(settings),str(w/arm/'bin/test-metal-ssd-experts'),'--q4'],check=True)
  if stage=='record':
   assert trace.stat().st_size>0 and trace.stat().st_size%24==0
   if arm!='c':assert trace.read_bytes()==(out/'c.ids').read_bytes()
  if arm!='c':assert hot.read_bytes()==(out/'c.hotlist').read_bytes()
  shutil.copy2(hot,out/(arm+'-'+stage+'.hotlist'))
print('C/native/Debug diagnostic record/replay and hotlist merge bytes match',flush=True)
