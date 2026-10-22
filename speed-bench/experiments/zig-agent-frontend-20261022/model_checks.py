from pathlib import Path
import json,subprocess,sys,hashlib
w=Path(__file__).resolve().parent
rec=['python3','-B','.workspaces/zig-server-input-20261022/record.py']
clean=['python3','-B','speed-bench/experiments/zig-session-frontier-20261022/clean_exec.py']
model='gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
vision='gguf/DeepSeek-V4.1-Flash-1017-vision-BF16.gguf'
for arm in ('native','debug'):
 binary=w/arm/'bin/ds4_agent_test'; vocab=w/(arm+'-vocab-output');vocab.mkdir()
 for label,command in [
  ('vocab',[str(w/arm/'bin/test-agent-live-context'),model,str(vocab)]),
  ('thinking',[str(binary),'--think-fixture',model]),
  ('image',[str(binary),'--image-observation',model,vision,'speed-bench/experiments/zig-engine-vision-20261022/red.png']),
  ('saved-model',[str(binary),'--full-context-save',model]),
 ]:
  subprocess.run([*rec,str(w/(arm+'-'+label)),*clean,*command],check=True)
for arm in ('fallback','native'):
 subprocess.run([*rec,str(w/(arm+'-raw')),*clean,'--settings',json.dumps({'DS4_AGENT_CACHE_DIR':str(w/(arm+'-raw-cache')),'DS4_STATE_CACHE_DIR':str(w/(arm+'-raw-state-cache'))}),str(w/arm/'bin/ds4-agent'),'--model',model,'--ssd-streaming','--ssd-streaming-cache-experts','32GB','--ctx','512','--chunk','128','--tokens','8','--temp','0','--seed','1','--non-interactive','--raw-prompt','--prompt','Complete the sequence: one, two, three,'],check=True)
assert (w/'native-image.stdout').read_bytes()==(w/'debug-image.stdout').read_bytes()
assert (w/'native-raw.stdout').read_bytes()==(w/'fallback-raw.stdout').read_bytes()
for arm in ('native','debug'):
 text=(w/(arm+'-saved-model.stdout')).read_text();assert 'saved-session-cycle tokens=10 payload=1400412 first=70935 history=MAPLE' in text,text
for p,h in json.loads((w/'active-source.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
print('actual vocabulary, thinking, image, nonzero KV reload/strip/history and raw generation pass; source unchanged',flush=True)
