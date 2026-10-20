"""Held-out wrapper controls/paired statistics without touching model metadata."""
from pathlib import Path
import hashlib,importlib.util,json,subprocess,tempfile,errno,os,shlex
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('confirmation',root/'speed-bench/ideas/run_resident_confirmation.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def line(prefix,d):return prefix+' '.join(f'{k}={v}' for k,v in d.items())+'\n'
io=dict.fromkeys(m.IO_FIELDS,0);io.update(enabled=1,requested_bytes=9007199254740993,returned_bytes=9007199254740993,cache_loader_bytes=10)
assert m.parse_io(line('PROBE_IO ',io))==io;trials=1
for field,value in [('enabled',0),('requested_bytes',1),('returned_bytes',1),('cache_hits',-1),('cache_misses',2**64)]:
    d=dict(io);d[field]=value
    try:m.parse_io(line('PROBE_IO ',d))
    except ValueError:trials+=1
    else:raise AssertionError(field)
for text in ['',line('PROBE_IO ',io)*2,line('PROBE_IO ',io)+'ds4: resident-backfill requests=1\n']:
    try:m.parse_io(text)
    except ValueError:trials+=1
    else:raise AssertionError(text)
assert len(m.arms())==3 and m.arms()[0][1]==m.arms()[-1][1]
policy=m.arms()[1][1];prefix='DS4_METAL_V41_RESIDENT_ROUTE_'
assert {k.removeprefix(prefix):v for k,v in policy.items() if k.startswith(prefix)}==dict(GAP='.30',MAX_MASS='.20',MAX_TOTAL_MASS='.40',MAX_RANK='48',MAX_SUBS='3',MODE='apply',POLICY='replace')
assert not any('BACKFILL' in k for _,d in m.arms() for k in d)
route=dict(layers=20480,substitutions=3,native_misses=20,skipped_loads=3,skipped_bytes=300,full_snapshots=5,double_layers=0,planned_substitutions=3,drops=0,triple_layers=1,mean_mass=.1,mean_score_gap=.2,mean_affinity_ratio=.9,mean_rank=20)
assert m.parse_route(line('ds4: resident-route ',route),'broader3')==route;trials+=1
for name,d in [('native-before',route),('broader3',dict(route,layers=0)),('broader3',dict(route,substitutions=0)),('broader3',dict(route,mean_mass=float('nan')))]:
    try:m.parse_route(line('ds4: resident-route ',d),name)
    except ValueError:trials+=1
    else:raise AssertionError(d)
assert m.parse_route('','native-after') is None
# Guard must reject before any model identity getter can be reached.
old=m.subprocess.check_output;m.subprocess.check_output=lambda *a,**k:'python3 unit-test.py'
try:
    try:m.require_main_guard(Path('/nonexistent/confirmation-plan'),'a'*64)
    except RuntimeError as e:assert 'direct parent' in str(e)
    else:raise AssertionError('unguarded launch accepted')
finally:m.subprocess.check_output=old
# Anchor a small synthetic plan to an explicit argv SHA. No model/helper files
# are needed: ownership and authority failures must precede their access.
with tempfile.TemporaryDirectory(dir=root/'speed-bench/ideas/artifacts') as folder:
    output=Path(folder)
    plan=dict(arms=m.arms(),settings=m.SETTINGS,binary_sha256=m.BINARY_SHA,binary_manifest_sha256=m.BUILD_SHA,source_input_freeze_sha256=m.FREEZE_SHA,source_tsv_sha256=m.TSV_SHA,guard_command_template=m.command_template(output),guard_interpreter_identity=dict(path=str(m.GUARD_INTERPRETER),sha256=m.GUARD_INTERPRETER_SHA),guard_sha256=m.GUARD_SHA,files={})
    def pin(value):
        m.write(output/'manifest.json',value);digest=m.sha(output/'manifest.json')
        m.write(output/'prepared-command.json',dict(command=m.guard_command(output,digest),manifest_sha256=digest))
        return digest
    digest=pin(plan);assert m.verified_plan(output,digest)['settings']==m.SETTINGS;trials+=1
    altered=dict(plan,files={'probe':'0'*64});newdigest=pin(altered)
    try:m.verified_plan(output,digest)
    except ValueError as e:assert 'frozen plan changed' in str(e);trials+=1
    else:raise AssertionError('rewritten manifest/command accepted old argv authority')
    assert newdigest!=digest
    digest=pin(plan)
    c=json.loads((output/'prepared-command.json').read_text());c['command'].append('unapproved');m.write(output/'prepared-command.json',c)
    try:m.verified_plan(output,digest)
    except ValueError as e:assert 'command/manifest authority' in str(e);trials+=1
    else:raise AssertionError('changed prepared argv accepted')
    for key,value in [('guard_command_template',[]),('guard_interpreter_identity',{}),('arms',[]),('source_input_freeze_sha256','0'*64)]:
        d=pin(dict(plan,**{key:value}))
        try:m.verified_plan(output,d)
        except ValueError:trials+=1
        else:raise AssertionError(key)
    digest=pin(plan);expected=m.guard_command(output,digest)
    oldps=m.subprocess.check_output;oldlock=m.fcntl.flock
    def held(*args):raise BlockingIOError(errno.EAGAIN,'held by direct guard')
    try:
        m.fcntl.flock=held
        for argv in [['python3','unrelated.py',str(m.GUARD),str(output/'pressure')],expected+['unapproved'],[str(m.GUARD_INTERPRETER),*expected[1:-1],'0'*64],[str(root/'fake-python'),*expected[1:]]]:
            m.subprocess.check_output=lambda *a,_argv=argv,**k:shlex.join(_argv)
            try:m.require_main_guard(output,digest)
            except RuntimeError:trials+=1
            else:raise AssertionError('parent alias/suffix accepted')
        m.subprocess.check_output=lambda *a,**k:shlex.join(expected)
        m.require_main_guard(output,digest);trials+=1
        alias=output/'python-alias';alias.symlink_to(m.GUARD_INTERPRETER)
        m.subprocess.check_output=lambda *a,**k:shlex.join([str(alias),*expected[1:]])
        m.require_main_guard(output,digest);trials+=1
        m.fcntl.flock=lambda *args:None
        try:m.require_main_guard(output,digest)
        except RuntimeError as e:assert 'not held' in str(e);trials+=1
        else:raise AssertionError('unheld MAIN lock accepted')
        # Even with another job holding the lock, path strings in an unrelated
        # parent cannot reach helper/model identity access.
        m.fcntl.flock=held;m.subprocess.check_output=lambda *a,**k:shlex.join(['python3','unrelated.py',str(m.GUARD),str(output/'pressure')])
        oldraw=m.raw_helpers;m.raw_helpers=lambda *a:(_ for _ in ()).throw(AssertionError('helper/model access before guard'))
        try:
            try:m.run(output,digest)
            except RuntimeError as e:assert 'direct parent' in str(e);trials+=1
            else:raise AssertionError('unguarded run accepted')
        finally:m.raw_helpers=oldraw
    finally:m.subprocess.check_output=oldps;m.fcntl.flock=oldlock
# Mandatory file inventory and fixture path alignment are additional checks
# beyond the explicit manifest SHA. Read only frozen metadata, never text.
with tempfile.TemporaryDirectory(dir=root/'speed-bench/ideas/artifacts') as folder:
    output=Path(folder)
    frozen=json.loads((m.FREEZE/'input-freeze.json').read_text());build=json.loads((m.BUILD/'manifest.json').read_text())
    m.write(output/'input-freeze.json',frozen);m.write(output/'binary-manifest.json',build)
    files={'probe':m.BINARY_SHA,'binary-manifest.json':m.BUILD_SHA,'input-freeze.json':m.FREEZE_SHA,'raw_helpers.py':m.RAW_SHA,'run_resident_confirmation.py':m.sha(root/'speed-bench/ideas/run_resident_confirmation.py'),'fixtures.tsv':'1'*64}
    files.update({n:v for n,v in build['files'].items() if n.startswith('metal/')})
    lines=[]
    for i,c in enumerate(frozen['cases']):
        paths=[]
        for kind in ['prompt','continuation']:
            name=f'inputs/{i:03d}-{kind}.txt';files[name]=c[kind+'_sha256'];paths.append(str(output/name))
        lines.append('\t'.join([c['case'],*paths]))
    (output/'fixtures.tsv').write_text('\n'.join(lines)+'\n');assetplan=dict(files=files,inputs=frozen['cases'])
    m.verify_required_assets(output,assetplan);trials+=1
    for field in ['probe','inputs/000-prompt.txt','metal/dsv41.metal']:
        changed=dict(files);changed.pop(field)
        try:m.verify_required_assets(output,dict(assetplan,files=changed))
        except ValueError:trials+=1
        else:raise AssertionError('missing required '+field)
    for field in ['../escape','/absolute']:
        changed=dict(files,**{field:'0'*64})
        try:m.verify_required_assets(output,dict(assetplan,files=changed))
        except ValueError:trials+=1
        else:raise AssertionError(field)
    (output/'fixtures.tsv').write_text('foreign\t/foreign/prompt\t/foreign/continuation\n')
    try:m.verify_required_assets(output,assetplan)
    except ValueError:trials+=1
    else:raise AssertionError('foreign fixture accepted')
raw=m.raw_helpers()
with tempfile.TemporaryDirectory(dir=root/'speed-bench/ideas/artifacts') as folder:
    p=Path(folder)/'synthetic.csv';rows=['case,delta_raw_nll,raw_kl']
    for case in range(16):
        for step in range(32):rows.append(f'p{case},{case-7.5},0')
    p.write_text('\n'.join(rows)+'\n');summary=raw.summarize(p)
    assert len(summary['per_prompt'])==16 and all(v['tokens']==32 for v in summary['per_prompt'].values())
    assert summary['prompt_mean_delta_nll']==0 and summary==raw.summarize(p);trials+=1
print(json.dumps(dict(pass_=True,trials=trials,paired_prompts=16,bootstrap_resamples=2000,seed=20261002,model_opened=False,model_metadata_accessed=False,input_content_inspected=False)))
