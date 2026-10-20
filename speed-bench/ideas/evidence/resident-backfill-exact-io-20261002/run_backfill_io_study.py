#!/usr/bin/env python3
"""Prepare/run the fixed five-arm background-original whole-engine I/O screen."""
import argparse
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import sys

ROOT=Path('/Volumes/Models/ds4')
GUARD=ROOT/'speed-bench/ideas/pressure_run.py'
PRIOR=ROOT/'speed-bench/ideas/artifacts/resident-broad-independent16-20261002'
BUILD=ROOT/'.workspaces/idea-resident-fixture-io/speed-bench/ideas/artifacts/fixture-io-build'
DRIVER=ROOT/'.workspaces/idea-fixture-raw-hardened/speed-bench/ideas/artifacts/fixture-raw-hardened-20261002-v1/run_fixture_raw_matrix.py'
BINARY_SHA='554fe1c706fdedea9c3062650cc353fb361edc670b32252ea2275e5d3954e9cf'
DRIVER_SHA='dbeae546e41e6443bff714265b5be64bc2ec4b15b78c0ca8e71afb044a5eb132'
REF_SHA='1993aa5ff886b3f51401c1d1015e3882550120777809c31ea263936e9d63dda8'
BF0_SHA='3cc8f88de88258e79f89cfb211ce49372af3b5ea9956e365e07d158573947eb1'
SETTINGS=dict(cases=16,tokens=32,ctx=4096,cache_gb=64,prefill_chunk=512)
IO_FIELDS=set('enabled pread_calls requested_bytes returned_bytes cache_loader_bytes cache_hits cache_misses cache_evictions'.split())
STAGE_FIELDS=set('attempts started busy_skips ready_skips completed failed consumed expired resets pread_calls requested_bytes returned_bytes copied_bytes peak_bytes live_bytes state'.split())

def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def arms():
    base={'DS4_METAL_STREAMING_EXPERT_IO_METRICS':'1'}
    prefix='DS4_METAL_V41_RESIDENT_ROUTE_'
    policy={prefix+k:v for k,v in dict(GAP='0.30',MAX_MASS='0.20',MAX_TOTAL_MASS='0.35',MAX_RANK='48',MAX_SUBS='2').items()}
    return [('native-before',base),('bf0',dict(base,**policy,**{prefix+'BACKFILL':'0'})),
            ('bf1',dict(base,**policy,**{prefix+'BACKFILL':'1'})),
            ('bf1-repeat',dict(base,**policy,**{prefix+'BACKFILL':'1'})),('native-after',base)]

def prepare(output):
    prior=json.loads((PRIOR/'manifest.json').read_text())
    if prior['reference_sha256']!=REF_SHA or prior['settings']!=SETTINGS:
        raise ValueError('fixed reference hash/settings differ')
    if sha(BUILD/'fixture-backfill-raw-io-quality')!=BINARY_SHA or sha(DRIVER)!=DRIVER_SHA:
        raise ValueError('frozen helper/driver differs')
    reference=PRIOR/'reference.bin'
    with reference.open('rb') as source: header=source.read(16)
    magic,vocab,cases,tokens=struct.unpack('=4I',header)
    if (magic,cases,tokens)!=(0x44535131,16,32) or reference.stat().st_size!=16+512*(24+4*vocab):
        raise ValueError('fixed raw reference geometry differs')
    output.mkdir(parents=True,exist_ok=False)
    for source,name in [(BUILD/'fixture-backfill-raw-io-quality','probe'),(BUILD/'manifest.json','binary-manifest.json'),
                        (DRIVER,'raw_helpers.py'),(PRIOR/'manifest.json','reference-manifest.json'),
                        (Path(__file__),'run_backfill_io_study.py')]:
        shutil.copy2(source,output/name)
    (output/'metal').mkdir(); (output/'inputs').mkdir()
    for name,expected in prior['metal'].items():
        source=BUILD/'metal'/name
        if sha(source)!=expected: raise ValueError('reference/frozen Metal differs')
        shutil.copy2(source,output/'metal'/name)
    rows=[]
    for index,record in enumerate(prior['inputs']):
        paths=[]
        for kind in ['prompt','continuation']:
            name=f'{index:03d}-{kind}.txt'; source=PRIOR/'inputs'/name
            if sha(source)!=record[kind+'_sha256']: raise ValueError('frozen input differs')
            target=output/'inputs'/name; shutil.copy2(source,target); paths.append(str(target))
        rows.append('\t'.join([record['case'],*paths]))
    (output/'fixtures.tsv').write_text('\n'.join(rows)+'\n')
    manifest=dict(parent_commit='5383e437f098850402a737daa2f0a49915e17134',inference_commit='4c5845c282bb2ad9ba460b13695507eaffd3e234',
        binary_sha256=BINARY_SHA,hardened_driver_sha256=DRIVER_SHA,
        settings=SETTINGS,arms=arms(),model_from_prior=prior['model'],reference_path=str(reference),
        reference_sha256=REF_SHA,expected_bf0_sha256=BF0_SHA,guard_sha256=sha(GUARD),
        model_opened_in_preparation=False,model_metadata_currently_verified=False,gpu_initialized_in_preparation=False,
        reference_full_sha_verified_only_under_guard=True,
        files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        scope='Whole-engine prefill+decode explicit logical I/O and fullraw quality; instrumented, no latency/physical-SSD claim')
    write(output/'manifest.json',manifest)
    argv=[sys.executable,str(GUARD),str(output/'pressure'),'--',sys.executable,str(output/'run_backfill_io_study.py'),'--run',str(output)]
    write(output/'prepared-command.json',dict(command=argv,launch_requires_root_grant=True))
    print(json.dumps(dict(prepared=str(output),guard_command=argv)),flush=True)

def require_main_guard(output,expected_sha):
    if sha(GUARD)!=expected_sha: raise RuntimeError('MAIN guard differs')
    parent=subprocess.check_output(['ps','-p',str(os.getppid()),'-o','command='],text=True)
    if str(GUARD) not in shlex.split(parent) or str(output/'pressure') not in shlex.split(parent):
        raise RuntimeError('prepared MAIN guard must be direct parent')
    with (ROOT/'speed-bench/ideas/artifacts/model-job.lock').open('r+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EAGAIN,errno.EWOULDBLOCK): return
            raise
        fcntl.flock(lock,fcntl.LOCK_UN)
    raise RuntimeError('MAIN guard lock is not held')

def load_raw(path):
    if sha(path)!=DRIVER_SHA: raise ValueError('hardened dbe helpers differ')
    spec=importlib.util.spec_from_file_location('bf_io_raw_helpers',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def footer(text,prefix,fields,required=True):
    lines=[line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
    if not lines and not required: return None
    if len(lines)!=1: raise ValueError('missing/duplicate '+prefix.strip())
    result={}
    for field in lines[0].split():
        key,separator,value=field.partition('=')
        if not separator or key in result or not re.fullmatch('[0-9]+',value):
            raise ValueError('invalid/duplicate counter field')
        result[key]=int(value)
        if result[key]>2**64-1: raise ValueError('counter exceeds uint64')
    if result.keys()!=fields: raise ValueError('counter fields differ: '+prefix.strip())
    return result

def parse_io(text,backfill):
    io=footer(text,'PROBE_IO ',IO_FIELDS)
    if io['enabled']!=1 or io['returned_bytes']>io['requested_bytes']:
        raise ValueError('disabled/impossible demand counters')
    stage=footer(text,'ds4: resident-backfill ',STAGE_FIELDS,required=backfill)
    if not backfill and stage is not None: raise ValueError('disabled backfill emitted stage footer')
    if stage:
        if stage['live_bytes'] or stage['state'] or stage['failed']:
            raise ValueError('backfill cleanup/failure control failed')
        if stage['started']!=stage['completed']+stage['failed'] or stage['consumed']+stage['expired']>stage['completed']:
            raise ValueError('backfill completion/use accounting differs')
        if stage['returned_bytes']>stage['requested_bytes'] or stage['copied_bytes']>stage['returned_bytes']:
            raise ValueError('impossible stage byte accounting')
    stage_requested=stage['requested_bytes'] if stage else 0
    stage_returned=stage['returned_bytes'] if stage else 0
    return dict(demand=io,stage=stage,total_requested_logical_bytes=io['requested_bytes']+stage_requested,
                total_returned_logical_bytes=io['returned_bytes']+stage_returned)

def run(output):
    prepared=json.loads((output/'manifest.json').read_text())
    require_main_guard(output,prepared['guard_sha256'])
    if (prepared['arms']!=[[name,env] for name,env in arms()] or prepared['settings']!=SETTINGS or
        prepared['binary_sha256']!=BINARY_SHA or prepared['reference_sha256']!=REF_SHA or
        prepared['expected_bf0_sha256']!=BF0_SHA):
        raise ValueError('fixed prepared policy/provenance differs')
    assets={output/name:('prepared '+name,expected) for name,expected in prepared['files'].items()}
    assets[output/'manifest.json']=('prepared manifest',sha(output/'manifest.json'))
    assets[GUARD]=('MAIN guard',prepared['guard_sha256'])
    raw=load_raw(output/'raw_helpers.py'); raw.check_frozen_assets(assets)
    prior=json.loads((output/'reference-manifest.json').read_text())
    metal={p.name:sha(p) for p in (output/'metal').glob('*.metal')}
    if metal!=prior['metal']: raise ValueError('prepared/reference Metal set differs')
    model=Path(prepared['model_from_prior']['path'])
    model_identity=raw.identity(model)  # First model stat, after MAIN ownership.
    if model_identity!=prepared['model_from_prior']: raise ValueError('model metadata differs')
    directory=output/'run'; directory.mkdir(exist_ok=False)
    fixtures,inputs=raw.freeze_inputs(output/'fixtures.tsv',directory,16)
    if inputs!=prior['inputs']: raise ValueError('fixed input identity differs')
    reference=directory/'reference.bin'; original=Path(prepared['reference_path'])
    if sha(original)!=prepared['reference_sha256']: raise ValueError('original reference differs')
    shutil.copy2(original,reference)
    if sha(reference)!=prepared['reference_sha256']: raise ValueError('copied reference differs')
    metadata=raw.raw_record_metadata(reference,16,32,512)
    assets[reference]=('reference',prepared['reference_sha256'])
    assets[fixtures]=('run fixtures',sha(fixtures))
    for index,record in enumerate(inputs):
        for kind in ['prompt','continuation']:
            assets[directory/'inputs'/f'{index:03d}-{kind}.txt']=(kind,record[kind+'_sha256'])
    env={key:value for key,value in os.environ.items() if not key.startswith('DS4_')}
    for path in (output/'metal').glob('*.metal'):
        env['DS4_METAL_'+path.stem.upper()+'_SOURCE']=str(path)
    manifest=dict(prepared,model_currently_verified=model_identity,reference_layout=metadata[2],
                  guard_verified=True,driver_sha256=sha(Path(__file__)))
    write(directory/'manifest.json',manifest)
    results=[]
    for name,extra in prepared['arms']:
        raw.check_frozen_assets(assets)
        if raw.identity(model)!=model_identity: raise ValueError('model metadata changed')
        candidate=directory/(name+'-raw.bin')
        command=[str(output/'probe'),str(model),str(fixtures),str(directory/(name+'.csv')),str(reference),'compare',
            '--cases','16','--tokens','32','--ctx','4096','--cache-gb','64','--chunk','512','--candidate-record',str(candidate)]
        runenv=dict(env,**extra)
        write(directory/(name+'-command.json'),dict(command=command,env={key:value for key,value in runenv.items() if key.startswith('DS4_')}))
        with (directory/(name+'.json')).open('w') as stdout,(directory/(name+'.log')).open('w') as stderr:
            completed=subprocess.run(command,cwd=output,env=runenv,stdout=stdout,stderr=stderr,check=False)
        if completed.returncode: raise RuntimeError(name+' failed rc='+str(completed.returncode))
        raw.check_frozen_assets(assets)
        if raw.identity(model)!=model_identity: raise ValueError('model metadata changed during arm')
        row=json.loads((directory/(name+'.json')).read_text())
        if row['cases']!=16 or row['tokens']!=512: raise ValueError('incomplete quality fixture')
        row.update(name=name,**raw.summarize(directory/(name+'.csv')),
            candidate_record_layout=raw.validate_candidate_record(candidate,metadata,16,32,512),
            candidate_record_sha256=sha(candidate),candidate_record=str(candidate),
            io=parse_io((directory/(name+'.log')).read_text(),name.startswith('bf1')))
        if name.startswith('native') and (row['candidate_record_sha256']!=prepared['reference_sha256'] or row['bit_equal_rows']!=512):
            raise RuntimeError(name+' raw bytes differ from native reference')
        if name=='bf0' and row['candidate_record_sha256']!=prepared['expected_bf0_sha256']:
            raise RuntimeError('bf0 raw bytes differ from pinned prior candidate')
        results.append(row); write(directory/'summary.json',results)
        print(json.dumps({key:value for key,value in row.items() if key!='per_prompt'}),flush=True)
    by_name={row['name']:row for row in results}
    checks=dict(native_reference_exact=True,bf0_prior_raw_exact=True,
        native_demand_io_equal=by_name['native-before']['io']['demand']==by_name['native-after']['io']['demand'],
        bf1_repeat_raw_equal=by_name['bf1']['candidate_record_sha256']==by_name['bf1-repeat']['candidate_record_sha256'],
        reference_sha256=prepared['reference_sha256'],bf0_sha256=prepared['expected_bf0_sha256'],
        earlier_async_bf1_raw_equality_required=False,scope=prepared['scope'])
    write(directory/'checks.json',checks)
    if not checks['native_demand_io_equal']:
        raise RuntimeError('native I/O repetition control differs')
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path); parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if bool(args.prepare)==bool(args.run): parser.error('choose prepare or run')
    if args.prepare: prepare(args.prepare.resolve())
    else: run(args.run.resolve())

if __name__=='__main__': main()
