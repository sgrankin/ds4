from pathlib import Path
from typing import NamedTuple
import array,ast,ctypes as C,hashlib,json,math,subprocess,time
root=Path.cwd();base=root/'.workspaces/zig-image-preprocess-20261022';out=base/'real';out.mkdir();(out/'patches').mkdir();rows=[]
class Image(C.Structure):_fields_=[('width',C.c_uint32),('height',C.c_uint32),('rgb',C.POINTER(C.c_uint8)),('fingerprint',C.c_uint8*32)]
class Patches(C.Structure):_fields_=[(n,C.c_uint32) for n in ['content_width','content_height','padded_width','padded_height','grid_width','grid_height','patch_count','image_token_count']]+[('patches',C.POINTER(C.c_float))]
class DeepPatches(C.Structure):_fields_=[(n,C.c_uint32) for n in ['content_width','content_height','padded_width','padded_height','grid_width','grid_height','llm_grid_width','llm_grid_height','patch_count']]+[('patches',C.POINTER(C.c_float))]
class Layout(C.Structure):_fields_=[('token_count',C.c_uint32),('image_count',C.c_uint32),('types',C.POINTER(C.c_uint8)),('perm',C.POINTER(C.c_uint32))]
libs={a:C.CDLL(str(base/a/'lib/libds4-image.dylib')) for a in ['c','native','native-debug']}
for lib in libs.values():
 for family in ['glm53','qwen4','deepseek4','deepseek41']:
  f=getattr(lib,'ds4_image_preprocess_'+family);f.restype=C.c_int;f.argtypes=[C.POINTER(Patches if family in ['glm53','qwen4'] else DeepPatches),C.POINTER(Image)]+([C.c_uint32,C.c_uint32] if family in ['glm53','qwen4'] else [])+[C.c_char_p,C.c_size_t]
 lib.ds4_image_patches_free.argtypes=[C.POINTER(Patches)];lib.ds4_deepseek4_image_patches_free.argtypes=[C.POINTER(DeepPatches)]
 lib.ds4_deepseek41_image_layout_build.argtypes=[C.POINTER(Layout),C.c_uint32,C.c_uint32,C.c_char_p,C.c_size_t];lib.ds4_deepseek4_image_layout_free.argtypes=[C.POINTER(Layout)]
# Primary official geometry, extracted without importing PyTorch or PIL.
source=Path('/Volumes/Models/DeepInfer/deepinfer/ve/processor/image_processor.py');names={'ImageLayout','expanded_image_token_len','_layout_from_pixels','solve_resize_ratio','safe_resize','compute_image_layout'}
tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert {n.name for n in nodes}==names
namespace={'math':math,'NamedTuple':NamedTuple};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
(out/'reference.json').write_text(json.dumps(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),parameters=dict(patch_size=14,downsample_ratio=3,max_n_token=1024,min_pixels=295936,max_wh_ratio=None)),indent=2)+'\n')
sizes=[(1,1),(17,9),(9,17),(543,545),(544,544),(777,391),(1920,1080),(1080,1920),(2048,2048),(4096,64),(64,4096)]
for family in ['deepseek41','deepseek4','glm53','qwen4']:
 for width,height in sizes:
  data=bytes((i*37+i//32)%256 for i in range(width*height*3));rgb=(C.c_uint8*len(data)).from_buffer_copy(data);image=Image(width,height,rgb);reference=None
  for arm in ['c','native']+(['native-debug'] if (width,height) in [(17,9),(543,545)] else []):
   lib=libs[arm];kind=Patches if family in ['glm53','qwen4'] else DeepPatches;p=kind();error=C.create_string_buffer(256);args=[C.byref(p),C.byref(image)]+([16,128] if family in ['glm53','qwen4'] else [])+[error,len(error)]
   start=time.monotonic();ok=getattr(lib,'ds4_image_preprocess_'+family)(*args);elapsed=time.monotonic()-start;assert ok,(family,width,height,arm,error.value)
   fields={name:getattr(p,name) for name,_ in kind._fields_ if name!='patches'};count=p.patch_count*(1176 if family=='glm53' else 768 if family=='qwen4' else 588);raw=C.string_at(p.patches,count*4)
   label=f'{family}-{width}x{height}-{arm}';(out/'patches'/(label+'.f32')).write_bytes(raw)
   if reference is None:reference=(fields,raw)
   assert fields==reference[0],label+' geometry'
   row=dict(label=label,family=family,arm=arm,width=width,height=height,fields=fields,seconds=elapsed,values=count,identical_patch_bytes=raw==reference[1],sha256=hashlib.sha256(raw).hexdigest())
   if raw!=reference[1]:
    a=array.array('f');a.frombytes(reference[1]);b=array.array('f');b.frombytes(raw);errors=[abs(x-y) for x,y in zip(a,b)];assert all(math.isfinite(x) for x in b);row.update(max_abs=max(errors),rms=math.sqrt(sum(x*x for x in errors)/count),unequal=sum(x!=0 for x in errors))
   if family=='deepseek41':
    plan=namespace['compute_image_layout'](width,height,patch_size=14,downsample_ratio=3,max_n_token=1024,min_pixels=295936,max_wh_ratio=None)
    actual=(p.llm_grid_height,p.llm_grid_width,p.padded_width,p.padded_height,p.grid_width,p.grid_height)
    assert actual==plan[:6],(label,actual,plan)
    layout=Layout();assert lib.ds4_deepseek41_image_layout_build(C.byref(layout),p.llm_grid_height,p.llm_grid_width,error,len(error))
    assert layout.token_count==plan.num_tokens and layout.image_count==p.llm_grid_height*p.llm_grid_width
    assert list(layout.types[:layout.token_count])==[0]+([2]*p.llm_grid_width+[3])*p.llm_grid_height+[4]
    assert list(layout.perm[:layout.image_count])==list(range(layout.image_count));lib.ds4_deepseek4_image_layout_free(C.byref(layout));row['official_geometry']=True
   getattr(lib,'ds4_image_patches_free' if kind is Patches else 'ds4_deepseek4_image_patches_free')(C.byref(p));assert not p.patches and p.patch_count==0
   rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k not in ['fields','sha256']}),flush=True)
print('Image geometry, patch buffers and official V41 layouts measured.',flush=True)
