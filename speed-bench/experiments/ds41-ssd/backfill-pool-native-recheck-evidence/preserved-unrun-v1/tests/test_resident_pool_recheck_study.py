"""Parser/guard failures without model metadata, model opens or GPU access."""
from pathlib import Path
import importlib.util,json,subprocess,sys,tempfile,errno,shlex
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pool_study',root/'speed-bench/ideas/run_resident_pool_recheck_study.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def line(prefix,d): return prefix+' '.join(f'{k}={v}' for k,v in d.items())+'\n'
io=dict.fromkeys(m.IO_FIELDS,0);io.update(enabled=1,requested_bytes=1000,returned_bytes=1000,cache_loader_bytes=1000)
stage=dict.fromkeys(m.STAGE_FIELDS,0);stage['resets']=3
pool=dict.fromkeys(m.POOL_FIELDS,0);pool.update(enabled=1,limit_bytes=400*1024*1024,metadata_bytes=16384,
    owned_live_bytes=16384,owned_peak_bytes=310000,payload_peak_bytes=16384,ready_peak=1,queue_peak=1,
    requests=1,queued=1,admissions=1,read_started=1,completed=1,worker_starts=1,joins=1,
    dup_opens=1,dup_closes=1,allocations=1,frees=1,allocated_bytes=16384,freed_bytes=16384,
    consumed_late=1,ready_age=80,copied_bytes=320,requested_bytes=320,returned_bytes=320,pread_calls=3)
def text(p=pool,s=stage): return line('PROBE_IO ',io)+line('ds4: resident-backfill ',s)+line('ds4: resident-backfill-pool ',p)
value=m.parse_io(text(),True,True);assert value['total_returned_logical_bytes']==1320
trials=1
for key,new in [('enabled',0),('limit_bytes',1),('owned_peak_bytes',400*1024*1024+1),('metadata_bytes',1),
                ('failed',1),('free_failures',1),('worker_present',1),('ready_count',1),('ready_age',79),('completion_harvest_expired',1),('stored_ready_expired',1),('consumed_late_older80',1),('consumed_late_age_max',81),('dup_closes',0),
                ('frees',0),('joins',0),('requests',2),('queued',2),('consumed_late',0),('returned_bytes',321),
                ('copied_bytes',321),('ready_superseded_bytes',321)]:
    bad=dict(pool);bad[key]=new
    try:m.parse_io(text(bad),True,True)
    except ValueError:trials+=1
    else:raise AssertionError(key)
badstage=dict(stage);badstage.update(started=1,completed=1,requested_bytes=100,returned_bytes=100)
try:m.parse_io(text(s=badstage),True,True)
except ValueError:trials+=1
else:raise AssertionError('double global+pool source')
try:m.parse_io(text(),True,False)
except ValueError:trials+=1
else:raise AssertionError('unexpected pool footer')
try:m.parse_io(text()+line('ds4: resident-backfill-pool ',pool),True,True)
except ValueError:trials+=1
else:raise AssertionError('duplicate pool footer')
# Superseded and reset dispositions each count one completed ownership and
# retain all logical reads in the denominator; no served-byte double count.
for disposition in ('ready_superseded','ready_expired','ready_replaced','reset_ready_discards','reset_result_discards'):
    p=dict(pool);p.update(consumed_late=0,copied_bytes=0);p[disposition]=1
    if disposition=='ready_superseded':p['ready_superseded_bytes']=320
    if disposition=='ready_expired':p['stored_ready_expired']=1
    assert m.parse_io(text(p),True,True)['total_returned_logical_bytes']==1320;trials+=1
assert len(m.arms())==6 and len({n for n,e in m.arms()})==6
assert m.arms()[0][1]==m.arms()[-1][1]=={'DS4_METAL_STREAMING_EXPERT_IO_METRICS':'1'}
assert m.parse_route('', 'native-before') is None and m.parse_route('', 'native-after') is None
# Parent rejection is explicit before any model stat. ps itself is denied by
# the tool sandbox here, so test the actual guard branch with a mocked parent.
prepared=root/'speed-bench/ideas/artifacts/resident-pool-recheck-prepared-20261002'
original=m.subprocess.check_output
m.subprocess.check_output=lambda *args,**kw:'python3 tests/test_resident_pool_study.py'
try:
    try:m.require_main_guard(prepared,'a'*64)
    except RuntimeError as error:assert 'direct parent' in str(error)
    else:raise AssertionError('unguarded launch accepted')
finally:m.subprocess.check_output=original
assert not (prepared/'run').exists();trials+=1
# A genuine older future use is admissible only with age1600; split losses
# retain the existing aggregate so ownership is counted exactly once.
p=dict(pool);p.update(ready_age=1600,consumed_late_age_sum=121,consumed_late_age_max=121,consumed_late_older80=1)
assert m.parse_io(text(p),True,True)['pool']['consumed_late_older80']==1;trials+=1
p.update(consumed_late=0,consumed_late_age_sum=0,consumed_late_age_max=0,consumed_late_older80=0,copied_bytes=0,ready_expired=1,completion_harvest_expired=1)
assert m.parse_io(text(p),True,True)['pool']['ready_expired']==1;trials+=1
# Already-resident queued reads are removed without background byte credits.
r=dict.fromkeys(m.RECHECK_FIELDS,0);r.update(enabled=1,admission_rechecks=3,queue_native_resident_drops=2,not_started_bytes=640)
p=dict(pool,requests=3,queued=3)
newtext=text(p)+line('ds4: resident-pool-recheck ',r)
value=m.parse_io(newtext,True,True,True);assert value['total_requested_logical_bytes']==1320 and value['recheck']==r;trials+=1
for key,v in [('enabled',0),('owner_rejections',1),('admission_rechecks',1),('not_started_bytes',0),('not_started_bytes',2**64)]:
    bad=dict(r,**{key:v})
    try:m.parse_io(text(p)+line('ds4: resident-pool-recheck ',bad),True,True,True)
    except ValueError:trials+=1
    else:raise AssertionError(key)
for payload,pool_on,recheck_on in [(newtext,True,False),(newtext,False,True),(text(p),True,True),(newtext+line('ds4: resident-pool-recheck ',r),True,True)]:
    try:m.parse_io(payload,True,pool_on,recheck_on)
    except ValueError:trials+=1
    else:raise AssertionError('missing/duplicate/incompatible recheck')
assert m.arms()[2][1]['DS4_METAL_V41_RESIDENT_ROUTE_BACKFILL_POOL_NATIVE_RECHECK']=='0'
assert m.arms()[3][1]==m.arms()[4][1] and m.arms()[3][1]['DS4_METAL_V41_RESIDENT_ROUTE_BACKFILL_POOL_NATIVE_RECHECK']=='1'
assert {k:v for k,v in m.arms()[2][1].items() if not k.endswith('NATIVE_RECHECK')}=={k:v for k,v in m.arms()[3][1].items() if not k.endswith('NATIVE_RECHECK')}
for name,expected,digest in [('native-before',m.NATIVE_IO,m.REF_SHA),('native-after',m.NATIVE_IO,m.REF_SHA),('bf0',m.BF0_IO,m.BF0_SHA)]:
    control=dict(name=name,io=dict(demand=expected),candidate_record_sha256=digest,bit_equal_rows=512,delta_nll=0,kl_base_to_raw=0,tv=0,argmax_flips=0)
    m.check_control(control);trials+=1
    altered=dict(control,io=dict(demand=dict(expected,cache_hits=expected['cache_hits']+1)))
    try:m.check_control(altered)
    except ValueError:trials+=1
    else:raise AssertionError('old I/O control changed')
    altered=dict(control,candidate_record_sha256='0'*64)
    try:m.check_control(altered)
    except ValueError:trials+=1
    else:raise AssertionError('old raw control changed')
with tempfile.TemporaryDirectory(dir=root/'speed-bench/ideas/artifacts') as folder:
    output=Path(folder)
    plan=dict(arms=m.arms(),settings=m.SETTINGS,binary_sha256=m.BINARY_SHA,binary_manifest_sha256=m.BINARY_MANIFEST_SHA,reference_sha256=m.REF_SHA,inference_commit=m.INFERENCE_COMMIT,reference_manifest_sha256=m.PRIOR_SHA,reference_path=str(m.PRIOR/'reference.bin'),guard_command_template=m.command_template(output),guard_interpreter_identity=dict(path=str(m.GUARD_INTERPRETER),sha256=m.GUARD_INTERPRETER_SHA),guard_sha256=m.GUARD_SHA,files={})
    def pin(value):
        m.write(output/'manifest.json',value);digest=m.sha(output/'manifest.json')
        m.write(output/'prepared-command.json',dict(command=m.guard_command(output,digest),manifest_sha256=digest));return digest
    digest=pin(plan);m.verified_plan(output,digest);trials+=1
    pin(dict(plan,files={'probe':'0'*64}))
    try:m.verified_plan(output,digest)
    except ValueError as e:assert 'frozen plan changed' in str(e);trials+=1
    else:raise AssertionError('mutable manifest+command accepted old argv digest')
    for key,value in [('arms',[]),('guard_command_template',[]),('guard_interpreter_identity',{}),('reference_manifest_sha256','0'*64),('reference_path','/foreign-reference')]:
        d=pin(dict(plan,**{key:value}))
        try:m.verified_plan(output,d)
        except ValueError:trials+=1
        else:raise AssertionError(key)
    digest=pin(plan);command=json.loads((output/'prepared-command.json').read_text());command['command'].append('foreign');m.write(output/'prepared-command.json',command)
    try:m.verified_plan(output,digest)
    except ValueError:trials+=1
    else:raise AssertionError('mutated prepared command')
    digest=pin(plan);expected=m.guard_command(output,digest);oldps=m.subprocess.check_output;oldlock=m.fcntl.flock
    def held(*args):raise BlockingIOError(errno.EAGAIN,'held')
    try:
        m.fcntl.flock=held
        for argv in [['python3','unrelated.py',str(m.GUARD),str(output/'pressure')],expected+['foreign'],[str(root/'fake-python'),*expected[1:]],[str(m.GUARD_INTERPRETER),*expected[1:-1],'0'*64]]:
            m.subprocess.check_output=lambda *a,_argv=argv,**k:shlex.join(_argv)
            try:m.require_main_guard(output,digest)
            except RuntimeError:trials+=1
            else:raise AssertionError('parent alias/suffix accepted')
        m.subprocess.check_output=lambda *a,**k:shlex.join(expected);m.require_main_guard(output,digest);trials+=1
        alias=output/'python-alias';alias.symlink_to(m.GUARD_INTERPRETER)
        m.subprocess.check_output=lambda *a,**k:shlex.join([str(alias),*expected[1:]]);m.require_main_guard(output,digest);trials+=1
        m.fcntl.flock=lambda *args:None
        try:m.require_main_guard(output,digest)
        except RuntimeError:trials+=1
        else:raise AssertionError('unheld MAIN lock accepted')
        m.fcntl.flock=held;m.subprocess.check_output=lambda *a,**k:'python3 foreign.py';oldraw=m.load_raw
        m.load_raw=lambda *a:(_ for _ in ()).throw(AssertionError('helper/model reached before guard'))
        try:
            try:m.run(output,digest)
            except RuntimeError:trials+=1
            else:raise AssertionError('unguarded run accepted')
        finally:m.load_raw=oldraw
    finally:m.subprocess.check_output=oldps;m.fcntl.flock=oldlock
    # Metadata-only inventory/TSV path check; no source fixture text read.
    prior=json.loads((m.PRIOR/'manifest.json').read_text());m.write(output/'reference-manifest.json',prior)
    files={'probe':m.BINARY_SHA,'binary-manifest.json':m.BINARY_MANIFEST_SHA,'raw_helpers.py':m.DRIVER_SHA,'reference-manifest.json':m.PRIOR_SHA,'run_resident_pool_recheck_study.py':m.sha(root/'speed-bench/ideas/run_resident_pool_recheck_study.py'),'fixtures.tsv':'0'*64}
    lines=[]
    for i,c in enumerate(prior['inputs']):
        paths=[]
        for kind in ['prompt','continuation']:
            name=f'inputs/{i:03d}-{kind}.txt';files[name]=c[kind+'_sha256'];paths.append(str(output/name))
        lines.append('\t'.join([c['case'],*paths]))
    for name,v in prior['metal'].items():files['metal/'+name]=v
    (output/'fixtures.tsv').write_text('\n'.join(lines)+'\n');assetplan=dict(files=files,model_from_prior=prior['model'])
    m.verify_required_assets(output,assetplan);trials+=1
    for name in ['probe','inputs/000-prompt.txt','metal/dsv41.metal']:
        bad=dict(files);bad.pop(name)
        try:m.verify_required_assets(output,dict(assetplan,files=bad))
        except ValueError:trials+=1
        else:raise AssertionError('omitted '+name)
    bad=dict(files,**{'../foreign':'0'*64})
    try:m.verify_required_assets(output,dict(assetplan,files=bad))
    except ValueError:trials+=1
    else:raise AssertionError('foreign path')
    (output/'fixtures.tsv').write_text('foreign\t/foreign\t/foreign\n')
    try:m.verify_required_assets(output,assetplan)
    except ValueError:trials+=1
    else:raise AssertionError('foreign fixture')
print(json.dumps({'pass':True,'parser_failure_and_balance_trials':trials,'guard_rejected_before_model_stat':True,'model_opened':False,'gpu_initialized':False}))
