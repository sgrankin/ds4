#!/usr/bin/env python3
"""Screen multiple resident policies against one frozen real-agent reference.

Run this command through the MAIN pressure_run.py; this program starts model
processes sequentially, but deliberately does not create a second model lock.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from run_screen import ROOT, sha, summarize


def variant(spec):
    name, separator, assignments = spec.partition(':')
    if not separator or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', name):
        raise argparse.ArgumentTypeError('expected NAME:DS4_NAME=value,...')
    if name in ('reference', 'native-before', 'native-after'):
        raise argparse.ArgumentTypeError('variant name is reserved for native controls')
    env = {}
    for item in assignments.split(','):
        key, separator, value = item.partition('=')
        if not separator or not re.fullmatch(r'DS4_[A-Z0-9_]+', key) or key in env:
            raise argparse.ArgumentTypeError('invalid or duplicate DS4 environment assignment')
        if key.startswith('DS4_METAL_') and key.endswith('_SOURCE'):
            raise argparse.ArgumentTypeError('shader sources must be frozen for all variants')
        env[key] = value
    return name, env


def identity(model):
    stat = model.stat()
    return dict(path=str(model.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run(args):
    output=args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary=output/'probe'
    replay=output/'session.txt'
    shutil.copy2(args.binary.resolve(), binary)
    shutil.copy2(args.replay.resolve(), replay)
    binary.chmod(0o555)
    env={k:v for k,v in os.environ.items() if not k.startswith('DS4_')}
    metal=output/'metal'
    metal.mkdir()
    for source in sorted((args.source_root.resolve()/'metal').glob('*.metal')):
        target=metal/source.name
        shutil.copy2(source, target)
        env['DS4_METAL_'+source.stem.upper()+'_SOURCE']=str(target)
    if not list(metal.glob('*.metal')):
        raise ValueError('source root contains no Metal shaders')
    model=args.model.resolve()
    settings=dict(ctx=args.ctx, cache_gb=args.cache_gb, prefill_chunk=args.chunk)
    manifest=dict(binary_source=str(args.binary.resolve()), binary_sha256=sha(binary),
        session_sha256=sha(replay), model=identity(model), settings=settings,
        metal={p.name:sha(p) for p in metal.glob('*.metal')},
        variants=dict(args.variant),
        driver_sha256=sha(Path(__file__)),
        guard_sha256=sha(ROOT/'speed-bench/ideas/pressure_run.py'),
        quality_scope='Full raw distributions on one teacher-forced tool trajectory; '
                      'phases are correlated. Timings are instrumented, not a throughput claim.')
    ref=output/'reference.bin'
    if args.reference:
        prior=json.loads(args.reference_manifest.read_text())
        for key in ('session_sha256','model','settings','metal'):
            if prior.get(key)!=manifest[key]:
                raise ValueError(f'reused reference {key} differs from this experiment')
        if prior.get('reference_sha256')!=sha(args.reference):
            raise ValueError('reused reference hash differs from its manifest')
        shutil.copy2(args.reference.resolve(),ref)
        manifest['reference_source']=str(args.reference.resolve())
        manifest['reference_sha256']=sha(ref)
    write_json(output/'manifest.json',manifest)
    arms=([] if args.reference else [('reference',{},True)])
    arms += [('native-before',{},False)]
    arms += [(name, extra, False) for name,extra in args.variant]
    arms += [('native-after',{},False)]
    results=[]
    for name, extra, record in arms:
        if identity(model)!=manifest['model']:
            raise RuntimeError('model metadata changed during this experiment')
        command=[str(binary),str(model),str(replay),str(output/(name+'.csv')),
                 str(ref),'record' if record else 'compare',
                 '--ctx',str(args.ctx),'--cache-gb',str(args.cache_gb),'--chunk',str(args.chunk)]
        runenv=env.copy()
        runenv.update(extra)
        write_json(output/(name+'-command.json'),dict(command=command,
            env={k:v for k,v in runenv.items() if k.startswith('DS4_')}))
        with (output/(name+'.json')).open('w') as stdout,(output/(name+'.log')).open('w') as stderr:
            completed=subprocess.run(command,cwd=args.source_root.resolve(),env=runenv,
                stdout=stdout,stderr=stderr,check=False)
        if completed.returncode:
            raise RuntimeError(f'{name} failed rc={completed.returncode}; see {name}.log')
        result=json.loads((output/(name+'.json')).read_text())
        result.update(name=name,**summarize(output/(name+'.csv')))
        # A single evolving trajectory cannot supply independent prompt trials.
        for key in ('prompt_bootstrap_95','prompt_mean_delta_nll','worst_prompt_delta_nll','per_prompt'):
            value=result.pop(key)
            if key!='prompt_bootstrap_95':
                result[key.replace('prompt','phase')]=value
        if not extra and result['bit_equal_rows']!=result['tokens']:
            raise RuntimeError(f'{name} does not reproduce all reference logits byte-for-byte')
        results.append(result)
        write_json(output/'summary.json',results)
        print(json.dumps({k:v for k,v in result.items() if k!='per_phase'}),flush=True)
    manifest['reference_sha256']=sha(ref)
    write_json(output/'manifest.json',manifest)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--replay',type=Path,default=ROOT/'speed-bench/ideas/latency-session.txt')
    parser.add_argument('--model',type=Path,default=ROOT/'gguf/DeepSeek-V4.1-Flash-Q2.gguf')
    parser.add_argument('--cache-gb',type=int,default=64)
    parser.add_argument('--ctx',type=int,default=8192)
    parser.add_argument('--chunk',type=int,default=512)
    parser.add_argument('--variant',type=variant,action='append',default=[])
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--reference-manifest',type=Path)
    args=parser.parse_args()
    if bool(args.reference)!=bool(args.reference_manifest):
        parser.error('reusing a reference requires its manifest')
    if not (1<=args.cache_gb<=96 and 1<=args.ctx<=131072 and 1<=args.chunk<=8192):
        parser.error('context, cache or chunk size is out of range')
    names=[name for name,_ in args.variant]
    if len(names)!=len(set(names)):
        parser.error('variant names must be unique')
    run(args)


if __name__=='__main__':
    main()
