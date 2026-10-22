from pathlib import Path
import hashlib,json,os,re,selectors,signal,subprocess,time
root=Path.cwd();base=root/'.workspaces/zig-cli-frontend-20261022';out=base/'real-2';out.mkdir()
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))};env.pop('LINENOISE_ASSUME_TTY',None)
model=root/'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
bins={'c':root/'.workspaces/zig-pretokenizer-20261022/native/bin/ds4','zig':base/'native/bin/ds4'}
common=['--model',str(model),'--ssd-streaming','--ssd-streaming-cache-experts','48GB','--ctx','1024','--prefill-chunk','512','--mtp-draft','1','--seed','1','--nothink']
rows=[]
def save(): (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
def command(binary,args,history=False):
 cmd=[str(binary),*map(str,args)]
 if history:
  profile='(version 1)(allow default)(deny file-write* (literal '+json.dumps(str(Path.home()/'.ds4_history'))+'))'
  cmd=['/usr/bin/sandbox-exec','-p',profile,*cmd]
 return cmd
def run(label,binary,args,stdin=None,history=False):
 cmd=command(binary,args,history);start=time.monotonic();proc=subprocess.run(cmd,input=stdin,env=env,capture_output=True,timeout=240)
 (out/(label+'.stdout')).write_bytes(proc.stdout);(out/(label+'.stderr')).write_bytes(proc.stderr)
 row=dict(label=label,command=cmd,seconds=time.monotonic()-start,returncode=proc.returncode,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest());rows.append(row);save();print(json.dumps(row),flush=True)
 assert proc.returncode==0,label
 return proc,row
# CLI diagnostic shortcut opens only readonly metadata/vocabulary, no GPU.
for family,name in [('deepseek',model.name),('glm','GLM-5.3-Flash-Q2.gguf'),('qwen','Qwen3.8-Flash-Next-Q4.gguf')]:
 for raw in [False,True]:
  ref=None
  for arm,binary in bins.items():
   args=['--model',root/'gguf'/name,'--dump-tokens','--ctx','1024','--nothink','-p','Café 日本語 1234567\nKeep PINE-251 checkable.']+(['--raw'] if raw else [])
   proc,row=run(f'{family}-{raw}-{arm}-tokens',binary,args)
   value=(json.loads(proc.stdout.splitlines()[0]),proc.stdout)
   if ref is None:ref=value
   row['identical_tokens']=value==ref;save();assert row['identical_tokens'],row['label']
# Exercise both ordinary engine generation and the frontend-owned sampled loop.
for mode,extra in [('greedy',['--temp','0']),('sampled',['--temp','0.6'])]:
 ref=None
 for arm,binary in bins.items():
  proc,row=run(mode+'-'+arm,binary,common+['-p','Reply with exactly ORCHID42.','--tokens','16']+extra)
  if ref is None:ref=proc.stdout
  row['identical_output']=proc.stdout==ref;save();assert row['identical_output'] and proc.stdout.strip(),row['label']
# Public JSON diagnostics and fixed-prefix perplexity scoring.
for mode in ['logits','logprobs','consistency','perplexity']:
 ref=None
 for arm,binary in bins.items():
  target=out/f'{mode}-{arm}.json'
  args=common+['--temp','0','--tokens','4']
  if mode=='perplexity':args+=['--perplexity-file',root/'.workspaces/zig-engram-20261022/bench/prompt.txt']
  else:
   args+=['-p','Reply with exactly ORCHID42.']
   args+=['--decode-consistency','2'] if mode=='consistency' else ['--dump-'+mode,target]
   if mode=='logprobs':args+=['--logprobs-top-k','128']
  proc,row=run(mode+'-'+arm,binary,args)
  if mode in ['logits','logprobs']:value=json.loads(target.read_bytes())
  elif mode=='consistency':value=re.findall(rb'^ds4: (?:decode-consistency|live_top|fresh_top).*$',proc.stderr,re.M)
  else:value=proc.stdout
  if ref is None:ref=value
  row['identical_diagnostic']=value==ref;save();assert row['identical_diagnostic'],row['label']
# Session replacement must retain the transcript and /read must submit file text.
read_file=out/'read.txt';read_file.write_text('Reply with exactly READ42.')
conversation=f'/think 0\nRemember the code PINE-251. Reply only OK.\n/ctx 1024\nWhat is the code? Reply with only the code.\n/read {read_file}\n/exit\n'.encode();(out/'repl-input.txt').write_bytes(conversation)
ref=None
for arm,binary in bins.items():
 proc,row=run('repl-'+arm,binary,common+['--temp','0','--tokens','24'],conversation,True)
 if ref is None:ref=proc.stdout
 row['identical_output']=proc.stdout==ref;row['recall_and_read']=b'PINE-251' in proc.stdout and b'READ42' in proc.stdout
 save();assert row['identical_output'] and row['recall_and_read'],row['label']
# Interrupt before any decoded token, then reuse that REPL for a fresh turn.
for arm,binary in bins.items():
 label='cancel-'+arm;cmd=command(binary,common+['--temp','0','--tokens','24'],True)
 first='Ignore this canceled request. '+('This text is being canceled. '*20)+' Reply CANCELED42.'
 stdin=(first+'\nReply with exactly RESUMED42.\n/exit\n').encode();(out/(label+'.stdin')).write_bytes(stdin)
 start=time.monotonic()
 with (out/(label+'.stdout')).open('wb') as stdout:
  proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=stdout,stderr=subprocess.PIPE,env=env,bufsize=0)
  proc.stdin.write(stdin);proc.stdin.close();proc.stdin=None
  selector=selectors.DefaultSelector();selector.register(proc.stderr,selectors.EVENT_READ);prefix=b''
  try:
   while b'processing ' not in prefix:
    assert time.monotonic()-start<90,label+' no progress'
    if selector.select(1):
     data=os.read(proc.stderr.fileno(),4096);assert data,label+' premature exit';prefix+=data
   proc.send_signal(signal.SIGINT)
   _,rest=proc.communicate(timeout=180)
  finally:
   selector.close()
   if proc.poll() is None:proc.kill();proc.wait()
 stderr=prefix+rest;(out/(label+'.stderr')).write_bytes(stderr);text=(out/(label+'.stdout')).read_bytes()
 sizes=[int(x) for x in re.findall(rb'processing (\d+) input tokens:',stderr)]
 unique=list(dict.fromkeys(sizes))
 row=dict(label=label,command=cmd,seconds=time.monotonic()-start,returncode=proc.returncode,prefill_lengths=unique,resumed=b'RESUMED42' in text,no_canceled_output=b'CANCELED42' not in text,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest());rows.append(row);save();print(json.dumps(row),flush=True)
 assert proc.returncode==0 and row['resumed'] and row['no_canceled_output'] and len(unique)==2 and unique[1]<unique[0],label
assert rows[-1]['prefill_lengths']==rows[-2]['prefill_lengths'],'cancellation transcript mismatch'
