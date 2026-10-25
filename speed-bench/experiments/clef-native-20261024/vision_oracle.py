#!/usr/bin/env python3
"""Offline numerical/timing oracle for the BF16 Clef vision tower only."""
import argparse,json,statistics,time
from pathlib import Path

def main():
 parser=argparse.ArgumentParser();parser.add_argument('model',type=Path);parser.add_argument('patches',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--grid',type=int,nargs=3,default=[1,16,16]);parser.add_argument('--fp32',action='store_true');args=parser.parse_args()
 import mlx.core as mx
 import numpy as np
 from mlx_vlm.utils import load_model
 model=load_model(args.model)
 thinker=getattr(model,'thinker',model)
 tower=getattr(thinker,'vision_tower',getattr(thinker,'visual',None))
 if tower is None:raise ValueError('No vision tower found')
 if args.fp32:tower.set_dtype(mx.float32)
 dtype=tower.patch_embed.proj.weight.dtype
 pixels=mx.array(np.fromfile(args.patches,dtype=np.float32).reshape(-1,1536)).astype(dtype)
 grid=mx.array([args.grid],dtype=mx.int32)
 result,deep=tower(pixels,grid);mx.eval(result,deep)
 args.output.mkdir(parents=True,exist_ok=True)
 np.asarray(result.astype(mx.float32)).tofile(args.output/'features.f32')
 for index,features in enumerate(deep):np.asarray(features.astype(mx.float32)).tofile(args.output/f'deepstack-{index}.f32')
 times=[]
 for _ in range(4):
  start=time.perf_counter();y,d=tower(pixels,grid);mx.eval(y,d);times.append(time.perf_counter()-start)
 info={'grid':args.grid,'shape':list(result.shape),'deepstack_shapes':[list(x.shape) for x in deep],'dtype':str(dtype),'median_seconds':statistics.median(times),'seconds':times}
 (args.output/'summary.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
