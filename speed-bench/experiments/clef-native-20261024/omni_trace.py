#!/usr/bin/env python3
"""Offline Omni layer trace for locating the first native numerical divergence."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    import mlx.core as mx
    import numpy as np
    spec=importlib.util.spec_from_file_location('clef_omni_reference',args.model/'clef_mlx.py')
    ref=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=ref
    spec.loader.exec_module(ref)
    model=ref.load(args.model,head_dtype=mx.float32)
    ids=mx.array(list(range(1,17)),dtype=mx.int32)[None]
    positions=mx.broadcast_to(mx.arange(16)[None,None],(3,1,16))
    final,states=model._text_model(ids,position_ids=positions,output_hidden_states=True)
    mx.eval(final,states)
    args.output.mkdir(parents=True,exist_ok=True)
    np.stack([np.asarray(x[0].astype(mx.float32)) for x in [*states,final]]).tofile(args.output/'layers.f32')
    layer=model._text_model.layers[0]
    embed=states[0]
    normalized=layer.input_layernorm(embed)
    attention=layer.self_attn(normalized,mask='causal',position_ids=positions)
    post_attention=embed+attention
    ffn_input=layer.post_attention_layernorm(post_attention)
    gates=layer.mlp.gate(ffn_input)
    probabilities=mx.softmax(gates,axis=-1,precise=True)
    selected=mx.argpartition(probabilities,kth=-8,axis=-1)[...,-8:]
    mx.eval(normalized,attention,post_attention,ffn_input,gates,probabilities,selected)
    for name,value in {'embedding':embed,'input_norm':normalized,'attention':attention,
                       'post_attention':post_attention,'ffn_input':ffn_input,'gates':gates,
                       'gate_probabilities':probabilities}.items():
        np.asarray(value[0].astype(mx.float32)).tofile(args.output/f'first-{name}.f32')
    (args.output/'router-ids.json').write_text(json.dumps(selected[0].tolist(),indent=2)+'\n')
    (args.output/'layout.json').write_text(json.dumps({'shape':[len(states)+1,16,2048],
        'stages':['embedding',*[f'block-{i}' for i in range(48)],'final_norm'],
        'input_ids':list(range(1,17))},indent=2)+'\n')
    print('Wrote50×16×2048 layer trace and first-layer router intermediates.',flush=True)


if __name__=='__main__':main()
