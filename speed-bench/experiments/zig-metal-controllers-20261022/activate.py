from pathlib import Path
import sys,subprocess,json
w=Path('.workspaces/zig-metal-integration-20261022')
specs=[('index_selection','INDEX_SELECTION'),('dense_matmul','DENSE_MATMUL'),('attention_decode','ATTENTION_DECODE'),('attention_batch','ATTENTION_BATCH'),('routed_batch','ROUTED_BATCH'),('routed_single','ROUTED_SINGLE'),('expert_cache','EXPERT_CACHE'),('expert_tables','EXPERT_TABLES'),('expert_load','EXPERT_LOAD'),('attention_output','ATTENTION_OUTPUT'),('rope','ROPE'),('compressor','COMPRESSOR'),('scheduling','SCHEDULING'),('embedding','EMBEDDING'),('quant','QUANT')]
def preserve(stem):
 f=w/(stem+'.json')
 if f.exists() and json.loads(f.read_text()).get('returncode') != 0:
  index=1
  while (w/(stem+'-failed'+str(index)+'.json')).exists():index+=1
  for old in w.glob(stem+'.*'):old.rename(old.with_name(old.name.replace(stem,stem+'-failed'+str(index),1)))
for owner,flag in specs:
 if (w/(owner+'-qualified.json')).exists():continue
 preserve(owner+'-build')
 if sys.argv[1:] and owner not in sys.argv[1:]:continue
 p=Path('build.zig');s=p.read_text();define='"-DDS4_NATIVE_METAL_'+flag+'=1"'
 if define not in s:
  a=s.index('const native_objc_flags');b=s.index(' } })',a);s=s[:b]+', '+define+s[b:]
 entry='zig/metal/'+owner+'_entry.zig'
 if '"'+entry+'"' not in s:
  anchor='        const activation_test_module =';pos=s.index(anchor);s=s[:pos]+'        core_module.addObjectFile(nativeObject(b, target, sdk, optimize, "'+entry+'", "metal_c", metal_c));\n'+s[pos:]
 p.write_text(s);subprocess.run(['zig','fmt','build.zig'],check=True)
 subprocess.run(['python3','-B','.workspaces/zig-server-input-20261022/record.py',str(w/(owner+'-build')),'tools/zig-build','metal-test','server-test','ssd-experts-test','dense-mpp-test','--prefix',str(w/owner)],check=True)
 checks={
  'index_selection':[('test-deepseek41-metal',['--index-scores']),('test-deepseek41-metal',['--index-topk']),('test-deepseek41-metal',['--general-topk'])],
  'dense_matmul':[('test-metal-dense-mpp',[]),('test-deepseek41-metal',[])],
  'attention_decode':[('ds4_test',['--metal-kernels']),('test-deepseek41-metal',['--compressed-kv-f16'])],
  'attention_batch':[('ds4_test',['--metal-kernels']),('test-deepseek41-metal',['--tp-attention'])],
  'attention_output':[('test-deepseek41-metal',['--attention-output']),('test-deepseek41-metal',['--attention-output-large'])],
  'rope':[('ds4_test',['--metal-kernels'])],
  'compressor':[('ds4_test',['--metal-kernels']),('test-deepseek41-metal',['--compact-carry'])],
  'scheduling':[('ds4_test',['--metal-kernels']),('test-metal-dense-mpp',[]),('test-metal-ssd-experts',[])],
  'embedding':[('test-deepseek41-metal',['--embedding']),('test-deepseek41-metal',[])],
  'quant':[('test-deepseek41-metal',[]),('test-deepseek41-metal',['--attention-output'])]}
 if owner in ['routed_batch','routed_single','expert_cache','expert_tables','expert_load']:checks[owner]=[('test-metal-ssd-experts',x) for x in [[],['--q4'],['--mxfp4']]]
 for index,(binary,args) in enumerate(checks[owner]):
  preserve(owner+'-gpu-'+str(index))
  command=['python3','-B','.workspaces/zig-server-input-20261022/record.py',str(w/(owner+'-gpu-'+str(index))),'python3','-B','speed-bench/experiments/zig-session-frontier-20261022/clean_exec.py','--settings','{"DS4_METAL_MATH_SAFE":"1"}',str(w/owner/'bin'/binary),*args]
  subprocess.run(command,check=True)
 (w/(owner+'-qualified.json')).write_text(json.dumps({'owner':owner,'status':'existing focused GPU fixtures pass'})+'\n')
 print('qualified '+owner,flush=True)
