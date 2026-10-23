#!/usr/bin/env python3
"""Phase attribution and native/no-op parity; never a production timing claim."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'speed-bench/agent-session'))
from resident_replay_abba import (digest,freeze_shaders,model_identity,
                                  parse_profiles,parse_timings,write_json)


def run(args):
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    binary=output/'replay'
    replay=output/'session.txt'
    shutil.copy2(args.binary.resolve(),binary)
    shutil.copy2(args.replay.resolve(),replay)
    binary.chmod(0o555)
    env={k:v for k,v in os.environ.items() if not k.startswith('DS4_')}
    shaders,hashes=freeze_shaders(output,ROOT/'src/metal',env)
    env.update(shaders)
    env.update(DS4_REPLAY_CONTEXT=str(args.ctx),DS4_REPLAY_CACHE_GB=str(args.cache_gb),
        DS4_REPLAY_PREFILL_CHUNK=str(args.chunk),DS4_REPLAY_ROUTE_METRICS='1',
        DS4_METAL_V41_RESIDENT_ROUTE_PROFILE='1',DS4_METAL_STREAMING_EXPERT_IO_METRICS='1',
        DS4_METAL_V41_RESIDENT_ROUTE_GAP='0.05',DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS='0.12',
        DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK='12',DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS='1')
    model=args.model.resolve()
    manifest=dict(binary_source=str(args.binary.resolve()),binary_sha256=digest(binary),
        replay_sha256=digest(replay),model=model_identity(model),metal=hashes,
        settings=dict(ctx=args.ctx,cache_gb=args.cache_gb,chunk=args.chunk),
        env={k:v for k,v in env.items() if k.startswith('DS4_')},
        scope='Instrumented stage attribution. Explicit pread bytes may be page-cache served; '
              'process disk-read bytes include all process I/O. Use separate profile-off ABBA for latency.')
    write_json(output/'manifest.json',manifest)
    results=[]
    expected=None
    if args.native_hashes:
        first=json.loads(args.native_hashes.read_text())[0]
        expected={key:first[key] for key in ('logits_sha256','snapshot_sha256')}
    for mode in ('native','lookup','noop','apply'):
        if model_identity(model)!=manifest['model']:
            raise RuntimeError('model metadata changed')
        destination=output/mode
        destination.mkdir()
        runenv=env.copy()
        runenv['DS4_METAL_V41_RESIDENT_ROUTE_MODE']=mode
        logits,snapshot=destination/'logits.bin',destination/'snapshot.bin'
        command=[str(binary),str(model),str(replay),str(logits),str(snapshot)]
        write_json(destination/'command.json',dict(command=command,
            env={k:v for k,v in runenv.items() if k.startswith('DS4_')}))
        with (destination/'timings.csv').open('w') as stdout,(destination/'stderr.log').open('w') as stderr:
            result=subprocess.run(command,cwd=ROOT,env=runenv,stdout=stdout,stderr=stderr)
        if result.returncode:
            raise RuntimeError(f'{mode} failed rc={result.returncode}')
        profiles=parse_profiles(destination/'stderr.log')
        phases=parse_timings(destination/'timings.csv')
        row=dict(mode=mode,logits_sha256=digest(logits),snapshot_sha256=digest(snapshot),
            phases=phases,profiles=profiles,
            decode_ms=sum(p['ms'] for p in phases if p['kind']=='D'),
            append_ms=sum(p['ms'] for p in phases[1:] if p['kind']=='P'))
        if mode!='apply':
            actual={key:row[key] for key in ('logits_sha256','snapshot_sha256')}
            if expected is None: expected=actual
            if actual!=expected:
                raise RuntimeError(f'{mode} modified native logits or serialized graph state')
        route=profiles['route_phase']
        io=profiles['io_phase']
        row['route_totals']={key:sum(r[key] for r in route)
            for key in route[0] if key not in ('phase','layer')} if route else {}
        row['io_totals']={key:sum(r[key] for r in io)
            for key in io[0] if key not in ('phase','explicit_enabled','process_valid')} if io else {}
        if not route or len(io)!=len(phases):
            raise RuntimeError('expected route and I/O snapshots were not emitted')
        results.append(row)
        write_json(output/'summary.json',results)
        print(json.dumps({k:v for k,v in row.items() if k not in ('profiles','phases')}),flush=True)
        if not args.keep_state:
            logits.unlink()
            snapshot.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=ROOT/'gguf/DeepSeek-V4.1-Flash-Q2.gguf')
    parser.add_argument('--replay',type=Path,default=ROOT/'speed-bench/ideas/latency-session.txt')
    parser.add_argument('--native-hashes',type=Path)
    parser.add_argument('--ctx',type=int,default=8192)
    parser.add_argument('--cache-gb',type=int,default=64)
    parser.add_argument('--chunk',type=int,default=512)
    parser.add_argument('--keep-state',action='store_true')
    args=parser.parse_args()
    if not (128<=args.ctx<=100000 and 4<=args.cache_gb<=96 and 1<=args.chunk<=8192):
        parser.error('context, cache or chunk outside runtime bounds')
    run(args)


if __name__=='__main__':
    main()
