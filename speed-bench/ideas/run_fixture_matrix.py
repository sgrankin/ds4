#!/usr/bin/env python3
"""Compare named policies on frozen independent prompts with one raw reference.

Run through the MAIN pressure_run.py, which owns the single model-job lock.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from run_replay_matrix import identity, variant, write_json
from run_screen import ROOT, sha, summarize


def freeze_inputs(manifest, output, cases):
    destination=output/'inputs'
    destination.mkdir()
    records=[]
    lines=[]
    names=set()
    for line in manifest.read_text().splitlines():
        if not line or line.startswith('#'): continue
        fields=line.split('\t')
        if len(fields)!=3 or fields[0] in names:
            raise ValueError('invalid or duplicate input case')
        name,prompt,continuation=fields
        names.add(name)
        copied=[]
        record=dict(case=name)
        for kind,source in [('prompt',prompt),('continuation',continuation)]:
            path=Path(source)
            if not path.is_absolute(): path=ROOT/path
            target=destination/f'{len(records):03d}-{kind}.txt'
            shutil.copy2(path,target)
            record[kind+'_sha256']=sha(target)
            copied.append(str(target))
        records.append(record)
        lines.append('\t'.join([name,*copied]))
        if len(records)==cases: break
    if len(records)!=cases: raise ValueError('not enough input cases')
    frozen=output/'fixtures.tsv'
    frozen.write_text('\n'.join(lines)+'\n')
    return frozen,records


def check_frozen_assets(assets):
    for path,(kind,digest) in assets.items():
        if sha(path)!=digest:
            raise RuntimeError(f'frozen {kind} changed')


def run(args):
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    binary=output/'probe'
    shutil.copy2(args.binary.resolve(),binary)
    binary.chmod(0o555)
    fixtures,inputs=freeze_inputs(args.manifest,output,args.cases)
    env={key:value for key,value in os.environ.items() if not key.startswith('DS4_')}
    metal=output/'metal'
    metal.mkdir()
    for source in sorted((args.source_root.resolve()/'metal').glob('*.metal')):
        target=metal/source.name
        shutil.copy2(source,target)
        env['DS4_METAL_'+source.stem.upper()+'_SOURCE']=str(target)
    if not list(metal.glob('*.metal')): raise ValueError('no Metal sources')
    model=args.model.resolve()
    manifest=dict(binary_source=str(args.binary.resolve()),binary_sha256=sha(binary),
        source_manifest_sha256=sha(args.manifest),fixtures_sha256=sha(fixtures),
        inputs=inputs,model=identity(model),
        settings=dict(cases=args.cases,tokens=args.tokens,ctx=args.ctx,
                      cache_gb=args.cache_gb,prefill_chunk=args.chunk),
        metal={path.name:sha(path) for path in metal.glob('*.metal')},
        variants=dict(args.variant),driver_sha256=sha(Path(__file__)),
        guard_sha256=sha(ROOT/'speed-bench/ideas/pressure_run.py'),
        quality_scope='Full temperature-one distributions on frozen continuations; '
                      'prompt intervals are descriptive, timings instrumented.')
    frozen_assets={binary:('binary',manifest['binary_sha256']),
                   fixtures:('fixture manifest',manifest['fixtures_sha256'])}
    frozen_assets.update({metal/name:('Metal source',digest)
                          for name,digest in manifest['metal'].items()})
    reference=output/'reference.bin'
    if args.reference:
        prior=json.loads(args.reference_manifest.read_text())
        for key in ('inputs','model','settings','metal'):
            if prior.get(key)!=manifest[key]:
                raise ValueError(f'reused reference {key} differs')
        if prior.get('reference_sha256')!=sha(args.reference):
            raise ValueError('reused reference hash differs')
        shutil.copy2(args.reference,reference)
        manifest['reference_source']=str(args.reference.resolve())
        manifest['reference_sha256']=sha(reference)
    write_json(output/'manifest.json',manifest)
    arms=[] if args.reference else [('reference',{},True)]
    arms += [('native-before',{},False)]
    arms += [(name,extra,False) for name,extra in args.variant]
    arms += [('native-after',{},False)]
    results=[]
    for name,extra,record in arms:
        if identity(model)!=manifest['model']: raise RuntimeError('model metadata changed')
        check_frozen_assets(frozen_assets)
        for index,row in enumerate(inputs):
            for kind in ('prompt','continuation'):
                if sha(output/'inputs'/f'{index:03d}-{kind}.txt')!=row[kind+'_sha256']:
                    raise RuntimeError('frozen input changed')
        command=[str(binary),str(model),str(fixtures),str(output/(name+'.csv')),
                 str(reference),'record' if record else 'compare',
                 '--cases',str(args.cases),'--tokens',str(args.tokens),
                 '--ctx',str(args.ctx),'--cache-gb',str(args.cache_gb),'--chunk',str(args.chunk)]
        runenv=env.copy()
        runenv.update(extra)
        write_json(output/(name+'-command.json'),dict(command=command,
            env={key:value for key,value in runenv.items() if key.startswith('DS4_')}))
        with (output/(name+'.json')).open('w') as stdout,(output/(name+'.log')).open('w') as stderr:
            result=subprocess.run(command,cwd=args.source_root.resolve(),env=runenv,
                                  stdout=stdout,stderr=stderr,check=False)
        if result.returncode: raise RuntimeError(f'{name} failed rc={result.returncode}')
        check_frozen_assets(frozen_assets)
        row=json.loads((output/(name+'.json')).read_text())
        row.update(name=name,**summarize(output/(name+'.csv')))
        if not extra and row['bit_equal_rows']!=row['tokens']:
            raise RuntimeError(f'{name} differs from reference bytes')
        if record:
            manifest['reference_sha256']=sha(reference)
            write_json(output/'manifest.json',manifest)
        results.append(row)
        write_json(output/'summary.json',results)
        print(json.dumps({key:value for key,value in row.items() if key!='per_prompt'}),flush=True)
    manifest['reference_sha256']=sha(reference)
    write_json(output/'manifest.json',manifest)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--model',type=Path,default=ROOT/'gguf/DeepSeek-V4.1-Flash-Q2.gguf')
    parser.add_argument('--cases',type=int,default=16)
    parser.add_argument('--tokens',type=int,default=32)
    parser.add_argument('--ctx',type=int,default=4096)
    parser.add_argument('--cache-gb',type=int,default=64)
    parser.add_argument('--chunk',type=int,default=512)
    parser.add_argument('--variant',type=variant,action='append',default=[])
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--reference-manifest',type=Path)
    args=parser.parse_args()
    if bool(args.reference)!=bool(args.reference_manifest):
        parser.error('reference and its manifest must be supplied together')
    if not (1<=args.cases<=512 and 1<=args.tokens<=512 and 128<=args.ctx<=131072
            and 4<=args.cache_gb<=96 and 1<=args.chunk<=8192):
        parser.error('case, token, context, cache or chunk bound exceeded')
    names=[name for name,extra in args.variant]
    if len(set(names))!=len(names): parser.error('duplicate policy name')
    run(args)


if __name__=='__main__': main()
