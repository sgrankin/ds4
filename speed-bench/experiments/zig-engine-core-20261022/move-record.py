"""Mechanical move of the actual Core after independently activated owners."""
from pathlib import Path
import difflib,json,re
w=Path('.workspaces/zig-engine-core-20261022');evidence=w/'integration'
p=Path('ds4.c');before=p.read_text();source=before
for pattern in [r'typedef struct \{\n    uint32_t stages;.*?\} ds4_dspark_summary;',r'typedef enum \{\n    DS4_SUPPORT_NONE = 0,.*?\} ds4_support_kind;',r'typedef struct \{\n    ds4_tensor \*e_proj;.*?\} ds4_mtp_weights;',r'typedef enum \{\n    DS4_VISION_NONE = 0,.*?\} ds4_vision_kind;']:
 value=re.search(pattern,source,re.S).group();assert source.count(value)==1;source=source.replace(value,'/* Actual common-core declaration is in ds4_engine_core.h. */',1)
start=source.index('struct ds4_engine {')+len('struct ds4_engine {\n');end=source.index('#ifndef DS4_NO_GPU\n    bool shared_prefill_workspace_ready;',start)
old_fields=source[start:end];fields=old_fields
for line in ['    ds4_engine_tp_state tp;\n','    bool cuda_tp_expert_split;  /* Resident expert layout, independent of binding. */\n']:
 assert fields.count(line)==1;fields=fields.replace(line,'')
header=(w/'ds4_engine_core.h').read_text();assert ('typedef struct {\n'+fields+'} ds4_engine_core;') in header
Path('ds4_engine_core.h').write_text(header)
source=source[:start]+'''    ds4_engine_core core;
    /* Select native cleanup only after the post-family startup dispatch.
     * Earlier option failures and explicit placement retain C fallback. */
    bool native_core;
    ds4_engine_tp_state tp;
    bool cuda_tp_expert_split;  /* Resident expert layout, independent of binding. */
'''+source[end:]
source=source.replace('#include "ds4_engine_memory.h"','#include "ds4_engine_memory.h"\n#include "ds4_engine_core.h"',1)
names=[]
for line in fields.splitlines():
 line=re.sub(r'/\*.*','',line).strip()
 if line.endswith(';'):names.append(re.search(r'(\w+)(?:\[[^\]]*\])?;',line).group(1))
count=0
for name in names:
 source,n=re.subn(r'(\b(?:e|eng|engine)|->engine)->'+re.escape(name)+r'\b',lambda m:m[1]+'->core.'+name,source);count+=n
(evidence/'ds4-before.c').write_text(before);(evidence/'record.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),source.splitlines(True),fromfile='a/ds4.c',tofile='b/ds4.c')));p.write_text(source)
fixture_counts={}
for p in [*Path('tests').glob('*.c'),Path('speed-bench/experiments/zig-engine-dspark-20261022/verify_commit.c')]:
 original=p.read_text();edited=original
 variables=set(re.findall(r'\bds4_engine\s*\**\s*(\w+)\s*(?:[=;,)]|\[)',original))
 for name in names:
  for var in variables:edited=re.sub(r'\b'+re.escape(var)+r'(->|\.)'+re.escape(name)+r'\b',lambda m:var+m[1]+'core.'+name,edited)
  edited=re.sub(r'->engine->'+re.escape(name)+r'\b','->engine->core.'+name,edited)
 for declaration in reversed(list(re.finditer(r'\bds4_engine\s+\w+\s*=\s*\{',edited))):
  first=edited.index('{',declaration.start());position=first+1;depth=1;replacements=[]
  while depth:
   if edited[position]=='{':depth+=1
   elif edited[position]=='}':depth-=1
   elif edited[position]=='.' and depth==1:
    match=re.match(r'\.(\w+)',edited[position:])
    if match and match[1] in names:replacements.append((position,position+len(match[0]),'.core.'+match[1]))
   position+=1
  for first,last,replacement in reversed(replacements):edited=edited[:first]+replacement+edited[last:]
 if edited!=original:
  delta=''.join(difflib.unified_diff(original.splitlines(True),edited.splitlines(True),fromfile='a/'+str(p),tofile='b/'+str(p)))
  (evidence/(p.name+'.patch')).write_text(delta);p.write_text(edited);fixture_counts[str(p)]=len(delta.splitlines())
(evidence/'record-move.json').write_text(json.dumps({'actual_fields':names,'production_reference_changes':count,'fixture_diff_lines':fixture_counts},indent=2)+'\n')
print(count,fixture_counts)
