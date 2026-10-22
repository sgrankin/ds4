from pathlib import Path
import json,subprocess,hashlib
w=Path(__file__).resolve().parent
for arm in ('fallback','native'):
 subprocess.run(['python3','-B','.workspaces/zig-server-input-20261022/record.py',str(w/(arm+'-raw-valid')),'python3','-B','speed-bench/experiments/zig-session-frontier-20261022/clean_exec.py','--settings',json.dumps({'DS4_AGENT_CACHE_DIR':str(w/(arm+'-raw-cache')),'DS4_STATE_CACHE_DIR':str(w/(arm+'-raw-state-cache'))}),str(w/arm/'bin/ds4-agent'),'--model','gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf','--ssd-streaming','--ssd-streaming-cache-experts','32GB','--ctx','512','--prefill-chunk','128','--tokens','8','--temp','0','--seed','1','--non-interactive','--raw-prompt','--prompt','Complete the sequence: one, two, three,'],check=True)
assert (w/'native-image.stdout').read_bytes()==(w/'debug-image.stdout').read_bytes()
assert (w/'native-raw-valid.stdout').read_bytes()==(w/'fallback-raw-valid.stdout').read_bytes()
for arm in ('native','debug'):
 assert 'saved-session-cycle tokens=10 payload=1400412 first=70935 history=MAPLE' in (w/(arm+'-saved-model.stdout')).read_text()
for p,h in json.loads((w/'active-source.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
print('actual image/KV continuation and C/native raw generation match; source unchanged',flush=True)
