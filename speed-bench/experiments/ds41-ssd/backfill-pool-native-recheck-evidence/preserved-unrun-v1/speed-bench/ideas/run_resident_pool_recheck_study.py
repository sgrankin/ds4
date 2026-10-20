#!/usr/bin/env python3
"""Prepare/run the fixed six-arm owner native-resident admission recheck quality and I/O screen."""
import argparse
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
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
GUARD_SHA='461e8c517e82a687b4579fc450c1e7128b1510192d088bab719682b33c946c3c'
GUARD_INTERPRETER=Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python')
GUARD_INTERPRETER_SHA='b80d260a48404da13a32dd44d7283aadc80f6a6e56ed6e03ed5597ec6f46d48d'
PRIOR=ROOT/'speed-bench/ideas/artifacts/resident-broad-independent16-20261002'
PRIOR_SHA='d5c373e7a4ddc7338ef08c1615b519d4283bd651804f8c5ee9910c90b2012f95'
BUILD=ROOT/'.workspaces/idea-resident-backfill-pool-recheck/speed-bench/ideas/artifacts/resident-pool-recheck-build'
DRIVER=ROOT/'.workspaces/idea-fixture-raw-hardened/speed-bench/ideas/artifacts/fixture-raw-hardened-20261002-v1/run_fixture_raw_matrix.py'
BINARY_SHA='f1125df66bd795868a334143012aaaff854d10053e66f8f7045204fa74e96d9f'
BINARY_MANIFEST_SHA='569c712f36a2f11940a1bc5bdbf6959fbdf6fe2a2be8509783f5f9d9184eaaa8'
DRIVER_SHA='dbeae546e41e6443bff714265b5be64bc2ec4b15b78c0ca8e71afb044a5eb132'
REF_SHA='1993aa5ff886b3f51401c1d1015e3882550120777809c31ea263936e9d63dda8'
INFERENCE_COMMIT='f4a96514ac10db85f9bddc085a1995f0dbe27f3d'
SETTINGS=dict(cases=16,tokens=32,ctx=4096,cache_gb=64,prefill_chunk=512)
IO_FIELDS=set('enabled pread_calls requested_bytes returned_bytes cache_loader_bytes cache_hits cache_misses cache_evictions'.split())
STAGE_FIELDS=set('attempts started busy_skips ready_skips completed failed consumed expired resets pread_calls requested_bytes returned_bytes copied_bytes peak_bytes live_bytes state'.split())

def sha(path):
    if str(path).lower().endswith('.gguf'):raise ValueError('model hashing forbidden')
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def arms():
    base={'DS4_METAL_STREAMING_EXPERT_IO_METRICS':'1'}
    prefix='DS4_METAL_V41_RESIDENT_ROUTE_'
    policy={prefix+k:v for k,v in dict(GAP='0.30',MAX_MASS='0.20',MAX_TOTAL_MASS='0.35',MAX_RANK='48',MAX_SUBS='1').items()}
    def variant(**kw): return dict(base,**policy,**{prefix+k:v for k,v in kw.items()})
    return [('native-before',base),
            ('bf0',variant(BACKFILL='0',BACKFILL_LATE='0',BACKFILL_POOL='0')),
            ('pool1600-off',variant(BACKFILL='1',BACKFILL_LATE='1',BACKFILL_POOL='1',BACKFILL_POOL_READY_AGE='1600',BACKFILL_POOL_NATIVE_RECHECK='0')),
            ('pool1600-recheck',variant(BACKFILL='1',BACKFILL_LATE='1',BACKFILL_POOL='1',BACKFILL_POOL_READY_AGE='1600',BACKFILL_POOL_NATIVE_RECHECK='1')),
            ('pool1600-recheck-repeat',variant(BACKFILL='1',BACKFILL_LATE='1',BACKFILL_POOL='1',BACKFILL_POOL_READY_AGE='1600',BACKFILL_POOL_NATIVE_RECHECK='1')),
            ('native-after',base)]

def prepare(output):
    if sha(GUARD)!=GUARD_SHA or sha(GUARD_INTERPRETER)!=GUARD_INTERPRETER_SHA:raise ValueError('pinned MAIN guard/framework differs')
    if sha(PRIOR/'manifest.json')!=PRIOR_SHA:raise ValueError('fixed calibration reference/input authority differs')
    prior=json.loads((PRIOR/'manifest.json').read_text())
    if prior['reference_sha256']!=REF_SHA or prior['settings']!=SETTINGS:
        raise ValueError('fixed reference hash/settings differ')
    if (sha(BUILD/'fixture-pool-recheck-io-quality')!=BINARY_SHA or sha(DRIVER)!=DRIVER_SHA or
        sha(BUILD/'manifest.json')!=BINARY_MANIFEST_SHA or
        json.loads((BUILD/'manifest.json').read_text())['source_commit']!=INFERENCE_COMMIT):
        raise ValueError('frozen helper/driver differs')
    reference=PRIOR/'reference.bin'
    with reference.open('rb') as source: header=source.read(16)
    magic,vocab,cases,tokens=struct.unpack('=4I',header)
    if (magic,cases,tokens)!=(0x44535131,16,32) or reference.stat().st_size!=16+512*(24+4*vocab):
        raise ValueError('fixed raw reference geometry differs')
    output.mkdir(parents=True,exist_ok=False)
    for source,name in [(BUILD/'fixture-pool-recheck-io-quality','probe'),(BUILD/'manifest.json','binary-manifest.json'),
                        (DRIVER,'raw_helpers.py'),(PRIOR/'manifest.json','reference-manifest.json'),
                        (Path(__file__),'run_resident_pool_recheck_study.py')]:
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
    manifest=dict(inference_commit=INFERENCE_COMMIT,
        binary_sha256=BINARY_SHA,binary_manifest_sha256=BINARY_MANIFEST_SHA,hardened_driver_sha256=DRIVER_SHA,
        settings=SETTINGS,arms=arms(),model_from_prior=prior['model'],reference_path=str(reference),
        reference_manifest_sha256=PRIOR_SHA,
        reference_sha256=REF_SHA,guard_sha256=GUARD_SHA,
        guard_command_template=command_template(output),guard_interpreter_identity=dict(path=str(GUARD_INTERPRETER),sha256=GUARD_INTERPRETER_SHA),
        model_opened_in_preparation=False,model_metadata_currently_verified=False,gpu_initialized_in_preparation=False,
        reference_full_sha_verified_only_under_guard=True,
        files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        scope='Whole-engine prefill+decode explicit logical I/O and fullraw quality; instrumented, no latency/physical-SSD claim; default-off native and deterministic BF0 fullraw controls; pool payload age1600 and queueTTL80 unchanged, native-resident admission recheck off/on; background teardown bytes included once')
    write(output/'manifest.json',manifest)
    digest=sha(output/'manifest.json');argv=guard_command(output,digest)
    write(output/'prepared-command.json',dict(command=argv,manifest_sha256=digest,launch_requires_root_grant=True))
    print(json.dumps(dict(prepared=str(output),guard_command=argv)),flush=True)

def command_template(output):
    return [str(GUARD_INTERPRETER),str(GUARD),str(output/'pressure'),'--',str(GUARD_INTERPRETER),str(output/'run_resident_pool_recheck_study.py'),'--run',str(output),'--manifest-sha256','<manifest_sha256>']

def guard_command(output,digest):return command_template(output)[:-1]+[digest]

def verified_plan(output,digest):
    if not re.fullmatch('[0-9a-f]{64}',digest or '') or sha(output/'manifest.json')!=digest:raise ValueError('frozen plan changed or explicit SHA missing')
    prepared=json.loads((output/'manifest.json').read_text());command=json.loads((output/'prepared-command.json').read_text())
    if command.get('manifest_sha256')!=digest or command.get('command')!=guard_command(output,digest):raise ValueError('prepared command/manifest authority differs')
    if prepared.get('guard_command_template')!=command_template(output) or prepared.get('guard_interpreter_identity')!=dict(path=str(GUARD_INTERPRETER),sha256=GUARD_INTERPRETER_SHA) or prepared.get('guard_sha256')!=GUARD_SHA:raise ValueError('prepared guard identity differs')
    if (prepared['arms']!=[[name,env] for name,env in arms()] or prepared['settings']!=SETTINGS or
        prepared['binary_sha256']!=BINARY_SHA or prepared['binary_manifest_sha256']!=BINARY_MANIFEST_SHA or prepared['reference_sha256']!=REF_SHA or
        prepared['inference_commit']!=INFERENCE_COMMIT or prepared['reference_manifest_sha256']!=PRIOR_SHA or prepared['reference_path']!=str(PRIOR/'reference.bin')):raise ValueError('fixed prepared policy/provenance differs')
    return prepared

def verify_required_assets(output,prepared):
    files=prepared['files']
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in files):raise ValueError('unsafe prepared asset path')
    identities={'probe':BINARY_SHA,'binary-manifest.json':BINARY_MANIFEST_SHA,'raw_helpers.py':DRIVER_SHA,'reference-manifest.json':PRIOR_SHA,'run_resident_pool_recheck_study.py':sha(Path(__file__))}
    if any(files.get(n)!=v for n,v in identities.items()):raise ValueError('mandatory asset identity differs')
    prior=json.loads((output/'reference-manifest.json').read_text())
    if prepared['model_from_prior']!=prior['model'] or len(prior['inputs'])!=16 or len(prior['metal'])!=26:raise ValueError('copied reference authority geometry/model differs')
    required=set(identities)|{'fixtures.tsv'};lines=[]
    for i,case in enumerate(prior['inputs']):
        paths=[]
        for kind in ['prompt','continuation']:
            name=f'inputs/{i:03d}-{kind}.txt';required.add(name);paths.append(str(output/name))
            if files.get(name)!=case[kind+'_sha256']:raise ValueError('mandatory input identity differs')
        lines.append('\t'.join([case['case'],*paths]))
    if (output/'fixtures.tsv').read_bytes()!=('\n'.join(lines)+'\n').encode():raise ValueError('frozen fixture paths/order differ')
    for name,value in prior['metal'].items():
        key='metal/'+name;required.add(key)
        if files.get(key)!=value:raise ValueError('mandatory shader identity differs')
    if set(files)!=required:raise ValueError('mandatory prepared asset set differs')

def require_main_guard(output,digest):
    if sha(GUARD)!=GUARD_SHA: raise RuntimeError('MAIN guard differs')
    parent=subprocess.check_output(['ps','-p',str(os.getppid()),'-o','command='],text=True)
    actual=shlex.split(parent)
    if not actual or actual[1:]!=guard_command(output,digest)[1:]:raise RuntimeError('exact prepared MAIN guard must be direct parent')
    try:same=os.path.samefile(actual[0],GUARD_INTERPRETER)
    except OSError:same=False
    if not same or sha(GUARD_INTERPRETER)!=GUARD_INTERPRETER_SHA:raise RuntimeError('direct parent interpreter differs from pinned framework')
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

POOL_FIELDS=set("requests invalid_requests dedup_ready dedup_reading dedup_queued queued queue_updates queue_expired queue_ready_drops queue_reading_drops admissions busy_defers handoff_defers budget_defers worker_starts worker_start_failures read_started completed failed pread_calls requested_bytes returned_bytes allocations allocation_failures frees free_failures allocated_bytes freed_bytes dup_opens dup_closes dup_failures consumed_future consumed_late copied_bytes ready_replaced ready_expired ready_superseded ready_superseded_bytes reset_ready_discards reset_result_discards reset_queue_discards resets joins limit_bytes metadata_bytes stack_bytes guard_bytes payload_live_bytes payload_peak_bytes owned_live_bytes owned_peak_bytes step enabled ready_count ready_peak queue_count queue_peak worker_active worker_state worker_present ready_age completion_harvest_expired stored_ready_expired consumed_future_age_sum consumed_future_age_max consumed_future_older80 consumed_late_age_sum consumed_late_age_max consumed_late_older80".split())
BF0_SHA='ceb6df37c311a1b294353c58865510581e41cc086c5f89c826ac0460a8647b86'
NATIVE_IO=dict(enabled=1,pread_calls=67908,requested_bytes=225302446080,returned_bytes=225302446080,cache_loader_bytes=225302446080,cache_hits=191684,cache_misses=22636,cache_evictions=16500)
BF0_IO=dict(enabled=1,pread_calls=59277,requested_bytes=196666859520,returned_bytes=196666859520,cache_loader_bytes=196666859520,cache_hits=194561,cache_misses=19759,cache_evictions=13623)
RECHECK_FIELDS=set('admission_rechecks queue_native_resident_drops not_started_bytes owner_rejections enabled'.split())

def parse_io(text,backfill,pool_enabled=False,recheck_enabled=False):
    io=footer(text,'PROBE_IO ',IO_FIELDS)
    if io['enabled']!=1 or io['returned_bytes']>io['requested_bytes']:
        raise ValueError('disabled/impossible demand counters')
    stage=footer(text,'ds4: resident-backfill ',STAGE_FIELDS,required=backfill)
    pool=footer(text,'ds4: resident-backfill-pool ',POOL_FIELDS,required=pool_enabled)
    recheck=footer(text,'ds4: resident-pool-recheck ',RECHECK_FIELDS,required=recheck_enabled)
    if recheck_enabled and not pool_enabled:raise ValueError('recheck requires pool')
    if not recheck_enabled and recheck is not None:raise ValueError('disabled recheck emitted footer')
    if recheck:
        if recheck['enabled']!=1 or recheck['owner_rejections'] or recheck['queue_native_resident_drops']>recheck['admission_rechecks']:
            raise ValueError('recheck gate/owner/count control failed')
        drops=recheck['queue_native_resident_drops'];size=recheck['not_started_bytes']
        if (not drops and size) or (drops and (size<3*drops or size>16*1024*1024*drops)):
            raise ValueError('unstarted logical byte bounds differ')
    if not backfill and stage is not None: raise ValueError('disabled backfill emitted global footer')
    if not pool_enabled and pool is not None: raise ValueError('disabled pool emitted footer')
    if stage:
        if stage['live_bytes'] or stage['state'] or stage['failed']:
            raise ValueError('global stage cleanup/failure control failed')
        if stage['started']!=stage['completed']+stage['failed'] or stage['consumed']+stage['expired']>stage['completed']:
            raise ValueError('global stage completion/use accounting differs')
        if stage['returned_bytes']>stage['requested_bytes'] or stage['copied_bytes']>stage['returned_bytes']:
            raise ValueError('impossible global stage byte accounting')
    if pool:
        if pool['enabled']!=1 or pool['limit_bytes']!=400*1024*1024:
            raise ValueError('pool opt-in/budget differs')
        for key in ('invalid_requests','worker_start_failures','failed','allocation_failures','free_failures','dup_failures',
                    'payload_live_bytes','ready_count','queue_count','worker_active','worker_state','worker_present','stack_bytes','guard_bytes'):
            if pool[key]: raise ValueError('pool failed/unclean '+key)
        if pool['metadata_bytes']!=16384 or pool['owned_live_bytes']!=pool['metadata_bytes']:
            raise ValueError('pool fixed metadata charge/cleanup differs')
        if (pool['owned_peak_bytes']>pool['limit_bytes'] or pool['payload_peak_bytes']>pool['owned_peak_bytes'] or
            pool['ready_peak']>40 or pool['queue_peak']>40): raise ValueError('pool cap/queue violation')
        if not (pool['admissions']==pool['read_started']==pool['completed']): raise ValueError('pool complete read accounting')
        if pool['worker_starts']!=pool['joins']: raise ValueError('persistent pool worker not joined')
        if pool['dup_opens']!=pool['dup_closes'] or pool['dup_opens']!=pool['admissions']: raise ValueError('pool duplicate descriptor leak')
        if pool['allocations']!=pool['frees'] or pool['allocated_bytes']!=pool['freed_bytes']: raise ValueError('pool allocation leak')
        if pool['ready_age'] not in (80,1600): raise ValueError('fixed payload lifetime differs')
        if pool['ready_expired']!=pool['completion_harvest_expired']+pool['stored_ready_expired']:
            raise ValueError('payload expiry split double-counted or missing')
        for kind in ('future','late'):
            count=pool['consumed_'+kind]; total=pool['consumed_'+kind+'_age_sum']
            maximum=pool['consumed_'+kind+'_age_max']; older=pool['consumed_'+kind+'_older80']
            if (older>count or maximum>pool['ready_age'] or total>count*maximum or maximum>total or
                older*81>total or (not older and maximum>80) or (older and maximum<=80)):
                raise ValueError('consumed payload age observations impossible')
        uses=pool['consumed_future']+pool['consumed_late']
        discards=sum(pool[k] for k in ('ready_expired','ready_replaced','ready_superseded','reset_ready_discards','reset_result_discards'))
        if pool['completed']!=uses+discards: raise ValueError('pool completed ownership balance')
        if pool['queued']!=pool['admissions']+sum(pool[k] for k in ('queue_expired','queue_ready_drops','queue_reading_drops','reset_queue_discards'))+(recheck['queue_native_resident_drops'] if recheck else 0):
            raise ValueError('pool queue ownership balance')
        if pool['requests']!=sum(pool[k] for k in ('dedup_ready','dedup_reading','dedup_queued','queued','queue_updates')):
            raise ValueError('pool request classification balance')
        if pool['returned_bytes']>pool['requested_bytes'] or pool['copied_bytes']+pool['ready_superseded_bytes']>pool['returned_bytes']:
            raise ValueError('pool logical bytes impossible')
        if any(stage[k] for k in STAGE_FIELDS-{'resets'}): raise ValueError('pool also used legacy global staging')
    stage_requested=stage['requested_bytes'] if stage else 0
    stage_returned=stage['returned_bytes'] if stage else 0
    pool_requested=pool['requested_bytes'] if pool else 0
    pool_returned=pool['returned_bytes'] if pool else 0
    return dict(demand=io,stage=stage,pool=pool,recheck=recheck,
        total_requested_logical_bytes=io['requested_bytes']+stage_requested+pool_requested,
        total_returned_logical_bytes=io['returned_bytes']+stage_returned+pool_returned,
        scope='whole-engine P+D plus background teardown reads; logical only; cache_loader_bytes not added again')

LATE_FIELDS=set('tentative rollbacks final_substitutions cancelled candidate_finish candidate_finish_failures ready_checks not_ready no_victim installs install_failures'.split())

def parse_late(text,enabled):
    late=footer(text,'ds4: resident-backfill-late ',LATE_FIELDS,required=enabled)
    if not enabled and late is not None: raise ValueError('disabled late path emitted footer')
    if late:
        if late['cancelled'] or late['candidate_finish_failures'] or late['install_failures']:
            raise ValueError('late candidate/install/cleanup failed')
        if (late['tentative']!=late['rollbacks']+late['final_substitutions'] or
            late['candidate_finish']!=late['tentative'] or late['ready_checks']!=late['tentative'] or
            late['installs']!=late['rollbacks'] or
            late['ready_checks']!=late['not_ready']+late['no_victim']+late['installs']):
            raise ValueError('late transaction accounting differs')
    return late

ROUTE_INTEGER_FIELDS=set('layers substitutions native_misses skipped_loads skipped_bytes full_snapshots double_layers planned_substitutions'.split())
ROUTE_FLOAT_FIELDS=set('mean_mass mean_score_gap mean_affinity_ratio mean_rank'.split())

def parse_route(text,name):
    prefix='ds4: resident-route '
    lines=[line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
    if name in ('native-before','native-after'):
        if lines: raise ValueError('default-off control emitted route footer')
        return None
    if len(lines)!=1: raise ValueError('missing/duplicate route footer')
    result={}
    for field in lines[0].split():
        key,separator,value=field.partition('=')
        if not separator or key in result: raise ValueError('invalid/duplicate route counter')
        if key in ROUTE_INTEGER_FIELDS:
            if not re.fullmatch('[0-9]+',value) or int(value)>2**64-1:
                raise ValueError('invalid route integer')
            result[key]=int(value)
        elif key in ROUTE_FLOAT_FIELDS:
            result[key]=float(value)
            if not math.isfinite(result[key]) or result[key]<0: raise ValueError('invalid route float')
        else: raise ValueError('unknown route counter')
    if result.keys()!=ROUTE_INTEGER_FIELDS|ROUTE_FLOAT_FIELDS: raise ValueError('route fields differ')
    if result['double_layers'] or result['substitutions']!=result['skipped_loads'] or result['substitutions']!=result['planned_substitutions']:
        raise ValueError('one-slot route accounting differs')
    if name=='native-positive' and (result['layers'] or result['substitutions'] or result['native_misses']):
        raise ValueError('positive-gap native control performed routing work')
    return result

def check_control(row):
    name=row['name'];expected=NATIVE_IO if name.startswith('native') else BF0_IO if name=='bf0' else None
    if expected is not None and row['io']['demand']!=expected:raise ValueError('prior exact native/BF0 demand cache/read control differs')
    if name.startswith('native') and (row['candidate_record_sha256']!=REF_SHA or row['bit_equal_rows']!=512 or any(row[k]!=0 for k in ['delta_nll','kl_base_to_raw','tv','argmax_flips'])):
        raise ValueError('prior exact native raw/control differs')
    if name=='bf0' and row['candidate_record_sha256']!=BF0_SHA:raise ValueError('prior deterministic BF0 raw differs')

def run(output,digest):
    prepared=verified_plan(output,digest)
    require_main_guard(output,digest)
    assets={output/name:('prepared '+name,expected) for name,expected in prepared['files'].items()}
    assets[output/'manifest.json']=('prepared manifest',digest)
    assets[output/'prepared-command.json']=('prepared command',sha(output/'prepared-command.json'))
    assets[GUARD]=('MAIN guard',prepared['guard_sha256'])
    assets[GUARD_INTERPRETER]=('framework interpreter',GUARD_INTERPRETER_SHA)
    raw=load_raw(output/'raw_helpers.py'); raw.check_frozen_assets(assets)
    verify_required_assets(output,prepared)
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
    env={key:value for key,value in os.environ.items() if not key.startswith('DS4')}
    for path in (output/'metal').glob('*.metal'):
        env['DS4_METAL_'+path.stem.upper()+'_SOURCE']=str(path)
    manifest=dict(prepared,model_currently_verified=model_identity,reference_layout=metadata[2],
                  guard_verified=True,driver_sha256=sha(Path(__file__)))
    write(directory/'manifest.json',manifest)
    results=[]
    for name,extra in prepared['arms']:
        require_main_guard(output,digest)
        raw.check_frozen_assets(assets)
        if raw.identity(model)!=model_identity: raise ValueError('model metadata changed')
        candidate=directory/(name+'-raw.bin')
        command=[str(output/'probe'),str(model),str(fixtures),str(directory/(name+'.csv')),str(reference),'compare',
            '--cases','16','--tokens','32','--ctx','4096','--cache-gb','64','--chunk','512','--candidate-record',str(candidate)]
        runenv=dict(env,**extra)
        write(directory/(name+'-command.json'),dict(command=command,env={key:value for key,value in runenv.items() if key.startswith('DS4_')}))
        with (directory/(name+'.json')).open('w') as stdout,(directory/(name+'.log')).open('w') as stderr:
            completed=subprocess.run(command,cwd=output,env=runenv,stdout=stdout,stderr=stderr,check=False)
        write(directory/(name+'-exit.json'),dict(returncode=completed.returncode))
        if completed.returncode: raise RuntimeError(name+' failed rc='+str(completed.returncode))
        raw.check_frozen_assets(assets)
        if raw.identity(model)!=model_identity: raise ValueError('model metadata changed during arm')
        row=json.loads((directory/(name+'.json')).read_text())
        if row['cases']!=16 or row['tokens']!=512: raise ValueError('incomplete quality fixture')
        text=(directory/(name+'.log')).read_text()
        row.update(name=name,**raw.summarize(directory/(name+'.csv')),
            candidate_record_layout=raw.validate_candidate_record(candidate,metadata,16,32,512),
            candidate_record_sha256=sha(candidate),candidate_record=str(candidate),
            io=parse_io(text,name.startswith('pool'),name.startswith('pool'),'recheck' in name),route=parse_route(text,name),
            late=parse_late(text,name.startswith('pool')))
        check_control(row)
        if row['late'] and not row['io']['pool'] and row['io']['stage']['consumed']<row['late']['installs']:
            raise ValueError('late installs exceed total staged consumption')
        if row['late'] and row['late']['tentative']!=row['route']['substitutions']:
            raise ValueError('late tentative differs from committed resident plan')
        if name.startswith('native') and (row['candidate_record_sha256']!=prepared['reference_sha256'] or row['bit_equal_rows']!=512 or any(row[k]!=0 for k in ['delta_nll','kl_base_to_raw','tv','argmax_flips'])):
            raise RuntimeError(name+' raw bytes differ from native reference')
        if row['io']['pool']:
            pool=row['io']['pool']
            if pool['ready_age']!=1600: raise ValueError('configured arm payload lifetime differs')
            if pool['consumed_late']!=row['late']['installs'] or pool['requests']!=row['late']['tentative']:
                raise ValueError('pool actual late-copy/admission scope differs')
        if name=='bf0' and row['candidate_record_sha256']!=BF0_SHA:
            raise RuntimeError('POOL0 deterministic BF0 raw differs from earlier c3 one-slot control')
        results.append(row); write(directory/'summary.json',results)
        print(json.dumps({key:value for key,value in row.items() if key!='per_prompt'}),flush=True)
    by_name={row['name']:row for row in results}
    checks=dict(native_reference_exact=True,
        default_off_native_reference_exact=True,bf0_previous_fullraw_exact=True,
        native_demand_io_equal=by_name['native-before']['io']['demand']==by_name['native-after']['io']['demand'],
        pool_repeat_raw_equal=by_name['pool1600-recheck']['candidate_record_sha256']==by_name['pool1600-recheck-repeat']['candidate_record_sha256'],
        pool_repeat_demand_io_equal=by_name['pool1600-recheck']['io']['demand']==by_name['pool1600-recheck-repeat']['io']['demand'],
        pool_repeat_counters_equal=by_name['pool1600-recheck']['io']['pool']==by_name['pool1600-recheck-repeat']['io']['pool'],
        pool_repeat_late_equal=by_name['pool1600-recheck']['late']==by_name['pool1600-recheck-repeat']['late'],
        reference_sha256=prepared['reference_sha256'],
        earlier_async_raw_equality_required=False,approximate_repeat_equality_required=False,
        scope=prepared['scope'])
    write(directory/'checks.json',checks)
    if not checks['native_demand_io_equal']:
        raise RuntimeError('native I/O repetition control differs')
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path); parser.add_argument('--run',type=Path);parser.add_argument('--manifest-sha256')
    args=parser.parse_args()
    if bool(args.prepare)==bool(args.run): parser.error('choose prepare or run')
    if args.prepare and args.manifest_sha256:parser.error('manifest SHA is only a frozen run argument')
    if args.run and not args.manifest_sha256:parser.error('run requires explicit frozen manifest SHA')
    if args.prepare: prepare(args.prepare.resolve())
    else: run(args.run.resolve(),args.manifest_sha256)

if __name__=='__main__': main()
