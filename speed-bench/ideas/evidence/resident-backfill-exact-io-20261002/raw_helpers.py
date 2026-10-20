#!/usr/bin/env python3
"""Compare named policies on frozen independent prompts with one raw reference.

Run through the MAIN pressure_run.py, which owns the single model-job lock.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import struct
import subprocess

# The serialized model job always uses the MAIN guard, regardless of the
# persistent experiment workspace from which this copied runner is invoked.
ROOT = Path('/Volumes/Models/ds4')
RAW_HEADER = struct.Struct('=4I')
RAW_ROW_META = struct.Struct('=4IQ')
RAW_MAGIC = 0x44535131

def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

def identity(model):
    stat = model.stat()
    return dict(path=str(model.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

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

def summarize(path):
    with Path(path).open() as source:
        rows = list(csv.DictReader(source))
    groups = {}
    for row in rows:
        groups.setdefault(row['case'], []).append(row)
    case_delta = [statistics.mean(float(r['delta_raw_nll']) for r in rs)
                  for rs in groups.values()]
    rng = random.Random(20261002)
    boots = sorted(statistics.mean(rng.choices(case_delta, k=len(case_delta)))
                   for _ in range(2000))
    kl = sorted(float(r['raw_kl']) for r in rows)
    return dict(prompt_mean_delta_nll=statistics.mean(case_delta),
        prompt_bootstrap_95=[boots[49], boots[1949]], worst_prompt_delta_nll=max(case_delta),
        kl_p99=kl[min(len(kl)-1, int(len(kl)*.99))], kl_max=max(kl),
        per_prompt={name: dict(tokens=len(rs),
            delta_nll=statistics.mean(float(r['delta_raw_nll']) for r in rs),
            kl=statistics.mean(float(r['raw_kl']) for r in rs))
            for name, rs in groups.items()})


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


def raw_record_metadata(path, cases, limit, expected_rows=None):
    """Read headers/24-byte row keys once, seeking past every float payload."""
    size=Path(path).stat().st_size
    with Path(path).open('rb') as source:
        header=source.read(RAW_HEADER.size)
        if len(header)!=RAW_HEADER.size:
            raise ValueError('raw record header is truncated')
        magic,vocab,got_cases,got_limit=RAW_HEADER.unpack(header)
        if magic!=RAW_MAGIC or not 1<=vocab<=10000000 or (got_cases,got_limit)!=(cases,limit):
            raise ValueError('raw record header/configuration differs')
        row_bytes=RAW_ROW_META.size+4*vocab
        payload=size-RAW_HEADER.size
        rows,remainder=divmod(payload,row_bytes)
        if remainder or not cases<=rows<=cases*limit or (expected_rows is not None and rows!=expected_rows):
            raise ValueError('raw record row count/size differs')
        metadata=[]
        expected_case=expected_step=0
        case_prefix=None
        for _ in range(rows):
            record=source.read(RAW_ROW_META.size)
            if len(record)!=RAW_ROW_META.size:
                raise ValueError('raw record metadata is truncated')
            case,step,target,prompt_tokens,fingerprint=RAW_ROW_META.unpack(record)
            if case==expected_case+1 and step==0 and expected_step:
                expected_case+=1
                expected_step=0
                case_prefix=None
            if (case,step)!=(expected_case,expected_step) or case>=cases or step>=limit:
                raise ValueError('raw record case/step order differs')
            if target>=vocab or not prompt_tokens:
                raise ValueError('raw record target/prompt metadata is invalid')
            prefix=(prompt_tokens,fingerprint)
            if case_prefix is not None and prefix!=case_prefix:
                raise ValueError('raw record case fingerprint/prompt length differs')
            case_prefix=prefix
            metadata.append(record)
            expected_step+=1
            source.seek(4*vocab,1)
        if expected_case!=cases-1 or source.tell()!=size:
            raise ValueError('raw record case coverage/end offset differs')
    layout=dict(magic=magic,vocab=vocab,cases=cases,token_limit=limit,
                header_bytes=RAW_HEADER.size,row_metadata_bytes=RAW_ROW_META.size,
                float_bytes=4,row_bytes=row_bytes,rows=rows,bytes=size,
                metadata_sha256=hashlib.sha256(b''.join(metadata)).hexdigest())
    return header,tuple(metadata),layout


def validate_candidate_record(path, reference_metadata, cases, limit, expected_rows):
    header,metadata,layout=raw_record_metadata(path,cases,limit,expected_rows)
    reference_header,reference_rows,reference_layout=reference_metadata
    if header!=reference_header or layout['bytes']!=reference_layout['bytes'] or len(metadata)!=len(reference_rows):
        raise ValueError('candidate raw header/count/size differs from reference')
    for row,(actual,expected) in enumerate(zip(metadata,reference_rows)):
        if actual!=expected:
            raise ValueError(f'candidate raw row metadata differs from reference at row {row}')
    return layout


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
        candidate_records=args.candidate_record,
        guard_sha256=sha(ROOT/'speed-bench/ideas/pressure_run.py'),
        reference_integrity='Pinned SHA256 before/after every compare arm; copied reused REF verified; '
                            'candidate header, row order/count/size and 24-byte metadata match REF.',
        quality_scope='Full temperature-one distributions on frozen continuations; '
                      'prompt intervals are descriptive, timings instrumented.')
    frozen_assets={binary:('binary',manifest['binary_sha256']),
                   fixtures:('fixture manifest',manifest['fixtures_sha256'])}
    frozen_assets.update({metal/name:('Metal source',digest)
                          for name,digest in manifest['metal'].items()})
    reference=output/'reference.bin'
    reference_metadata=None
    if args.reference:
        prior=json.loads(args.reference_manifest.read_text())
        for key in ('inputs','model','settings','metal'):
            if prior.get(key)!=manifest[key]:
                raise ValueError(f'reused reference {key} differs')
        if prior.get('reference_sha256')!=sha(args.reference):
            raise ValueError('reused reference hash differs')
        shutil.copy2(args.reference,reference)
        copied_sha=sha(reference)
        if copied_sha!=prior['reference_sha256']:
            raise ValueError('reused reference copy hash differs')
        reference_metadata=raw_record_metadata(reference,args.cases,args.tokens)
        manifest['reference_source']=str(args.reference.resolve())
        manifest['reference_sha256']=copied_sha
        manifest['reference_layout']=reference_metadata[2]
        frozen_assets[reference]=('reference',copied_sha)
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
        candidate_record=output/(name+'-raw.bin') if args.candidate_record else None
        if candidate_record:
            command += ['--candidate-record',str(candidate_record)]
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
        if record:
            reference_metadata=raw_record_metadata(reference,args.cases,args.tokens,row['tokens'])
            manifest['reference_sha256']=sha(reference)
            manifest['reference_layout']=reference_metadata[2]
            frozen_assets[reference]=('reference',manifest['reference_sha256'])
            write_json(output/'manifest.json',manifest)
        if candidate_record:
            row['candidate_record_layout']=validate_candidate_record(candidate_record,reference_metadata,
                args.cases,args.tokens,row['tokens'])
            row['candidate_record']=str(candidate_record)
            row['candidate_record_sha256']=sha(candidate_record)
            row['candidate_record_bytes']=candidate_record.stat().st_size
        if not extra and row['bit_equal_rows']!=row['tokens']:
            raise RuntimeError(f'{name} differs from reference bytes')
        if candidate_record and not extra and row['candidate_record_sha256']!=manifest['reference_sha256']:
            raise RuntimeError(f'{name} raw export differs from reference bytes')
        results.append(row)
        write_json(output/'summary.json',results)
        print(json.dumps({key:value for key,value in row.items() if key!='per_prompt'}),flush=True)
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
    parser.add_argument('--candidate-record',action='store_true',
                        help='Export all scored raw rows per arm in reference format; instrumented only')
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
