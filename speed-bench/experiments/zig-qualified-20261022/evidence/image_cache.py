from pathlib import Path
import json,os,subprocess,socket,time,urllib.request,base64
r=Path.cwd();w=r/'.workspaces/zig-qualified-20261022/image-cache';w.mkdir(exist_ok=True)
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))};env.update(DS4_STATE_CACHE_DIR=str(w/'cache'),DS4_STATE_CACHE_HOT_MB='0',DS4_STATE_CACHE_SSD_MB='128',DS4_SERVER_RESPONSE_STORE_DIR=str(w/'responses'))
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
base=f'http://127.0.0.1:{port}';command=[str(r/'zig-out/bin/ds4-server'),'--model','gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf','--vision','gguf/DeepSeek-V4.1-Flash-1017-vision-BF16.gguf','--ssd-streaming','--ssd-streaming-cache-experts','32GB','--ctx','8192','--prefill-chunk','512','--host','127.0.0.1','--port',str(port),'--tokens','32'];(w/'execution.json').write_text(json.dumps(dict(command=command,env={k:v for k,v in env.items() if k.startswith('DS4')}),indent=2)+'\n')
with (w/'server.stdout').open('wb') as stdout,(w/'server.stderr').open('wb') as stderr:
 p=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env);started=time.monotonic()
 try:
  while True:
   assert p.poll() is None,p.returncode
   try:
    with urllib.request.urlopen(base+'/v1/models',timeout=1) as response:model=json.load(response)['data'][0]['id']
    break
   except OSError:
    if time.monotonic()-started>120:raise
    time.sleep(.2)
  image='data:image/png;base64,'+base64.b64encode((r/'speed-bench/experiments/zig-engine-vision-20261022/red.png').read_bytes()).decode()
  history=[{'role':'user','content':[{'type':'text','text':'Name the main color in this image. Reply with one word.'},{'type':'image_url','image_url':{'url':image}}]}];rows=[]
  for n in range(2):
   body=dict(model=model,messages=history,temperature=0,reasoning_effort='none',max_tokens=16);(w/f'{n}.request.json').write_text(json.dumps(body,indent=2)+'\n');req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=180) as response:reply=json.load(response)
   (w/f'{n}.response.json').write_text(json.dumps(reply,indent=2)+'\n');text=reply['choices'][0]['message']['content'];cached=reply['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0);assert 'red' in text.lower(),text;rows.append(dict(turn=n,answer=text,cached_tokens=cached,usage=reply['usage']))
   history.extend([{'role':'assistant','content':text},{'role':'user','content':'Again, name that color with one word.'}])
  assert rows[1]['cached_tokens']>0,rows
  (w/'result.json').write_text(json.dumps(dict(pass_image_cache=True,rows=rows),indent=2)+'\n');print(rows,flush=True)
 finally:
  p.terminate();p.wait(timeout=10)
