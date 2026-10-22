from pathlib import Path
import shutil,difflib,subprocess,json
r=Path.cwd();w=r/'.workspaces/zig-metal-integration-20261022';stage=w/'stage'
specs={
'index-selection':['index_selection'], 'dense-matmul':['dense_matmul'], 'attention':['attention_decode'], 'attention-batch':['attention_batch'], 'routed-batch':['routed_batch'], 'routed-single':['routed_single','routed_single_policy','routed_single_selected','routed_single_dispatch'], 'expert-cache':['expert_cache'], 'expert-tables':['expert_tables'], 'expert-load':['expert_io','expert_storage','expert_load','expert_selected_load','expert_prefill_load','expert_load_policy'], 'attention-output':['attention_output'], 'rope':['rope'], 'compressor':['compressor','compressor_ops','compressor_projection'], 'scheduling':['command_schedule','parallel_ffn','deferred_kv','scheduling'], 'embedding':['embedding'], 'quant':['quant']}
files=[]
for owner,prefixes in specs.items():
 source=r/'.workspaces'/('zig-metal-'+owner+'-20261022')
 for prefix in prefixes:
  for ending in ['.zig','_entry.zig','.h','.inc.m']:
   f=source/(prefix+ending)
   if not f.exists():continue
   dest=r/'zig/metal'/f.name
   # Preserve reviewed shared-visibility/resource edits already staged.
   staged=stage/'zig/metal'/f.name
   selected=staged if staged.exists() and owner in ['attention','routed-batch','expert-cache'] else f
   shutil.copy2(selected,dest);files.append(str(dest.relative_to(r)))
patch=''
for name in ['ds4_metal.m','zig/metal/bridge.h','tests/test_deepseek41_metal.c','tests/ds4_test.c']:
 target=stage/name
 if not target.exists():continue
 before=(r/name).read_text();after=target.read_text()
 patch+=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
(w/'declaration-guards.patch').write_text(patch)
subprocess.run(['patch','--forward','--batch','--no-backup-if-mismatch','-p1'],input=patch,text=True,check=True)
(w/'installed-files.json').write_text(json.dumps(files,indent=2)+'\n')
