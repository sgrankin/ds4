"""Freeze/run19 weight-preservation qualifications; run requires MAIN guard/grant."""
import argparse
import csv
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
import statistics
import struct
import subprocess

ROOT = Path('/Volumes/Models/ds4')
WORK = ROOT / '.workspaces/idea-resident-native-weights-next'
BUILD = WORK / 'speed-bench/ideas/artifacts/resident-native-weights-next-build-plan-v5'
BUILD_SHA = 'e2cfda6426c8ec7c371ec18a12dff60932c1c2f5b166a7ba5b5ccafdc06aa91f'
PLAN_SHA = 'f798a1a977f61b7fc7a0f023da452e91f41a09516614e097e361e6bdd925638f'
INPUT_PROOF = WORK / 'speed-bench/experiments/ds41-ssd/resident-native-weights-next-evidence/build-plan-v3/input-identity-proof.json'
INPUT_SHA = '0839fb8a7d4fd61cccbe6d65a3ded0fb2b4abe0b7e3104466ea5a2ede54aff71'
RAW = ROOT / '.workspaces/idea-fixture-raw-hardened/speed-bench/ideas/artifacts/fixture-raw-hardened-20261002-v1/run_fixture_raw_matrix.py'
RAW_SHA = 'dbeae546e41e6443bff714265b5be64bc2ec4b15b78c0ca8e71afb044a5eb132'
GUARD = ROOT / 'speed-bench/ideas/pressure_run.py'
GUARD_SHA = '461e8c517e82a687b4579fc450c1e7128b1510192d088bab719682b33c946c3c'
PYTHON = Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python')
PYTHON_SHA = 'b80d260a48404da13a32dd44d7283aadc80f6a6e56ed6e03ed5597ec6f46d48d'
P = 'DS4_METAL_V41_RESIDENT_ROUTE_'
SETTINGS = dict(cases=16,tokens=32,rows=512,ctx=4096,cache_gb=64,chunk=512,tile=64,limit=1024,configured_slots=6136)
CSV_FIELDS = 'case step target prompt_tokens base_nll raw_nll policy_nll delta_raw_nll delta_policy_nll raw_kl policy_kl raw_tv policy_tv argmax_flip decode_ms'.split()
IO_FIELDS = set('enabled pread_calls requested_bytes returned_bytes cache_loader_bytes cache_hits cache_misses cache_evictions'.split())
ROUTE_INTS = set('layers substitutions native_misses skipped_loads skipped_bytes full_snapshots double_layers planned_substitutions drops triple_layers'.split())
ROUTE_FLOATS = set('mean_mass mean_score_gap mean_affinity_ratio mean_rank'.split())
QUALITY_INTS = set('cases tokens argmax_flips bit_equal_rows'.split())
QUALITY_FLOATS = set('avg_nll delta_nll policy_delta_nll kl_base_to_raw kl_base_to_policy tv startup_ms prefill_ms decode_ms metrics_ms'.split())
STATE_FIELDS = 'phase kind context tokens ms first_decode_ms'.split()
VOCAB = 129280
SNAPSHOT_HEADER = struct.Struct('<13I')
NUMPY_FILES = {
    '/opt/homebrew/lib/python3.14/site-packages/numpy/__init__.py':'09295a80660f17925ae23765ce8cbd7ff7ceae968d5f2f89349f1cb74c0b9e11',
    '/opt/homebrew/lib/python3.14/site-packages/numpy/_core/_multiarray_umath.cpython-314-darwin.so':'3f300bbe048de52ab5c38882fee8cc58e52b92ded52fa7958916fb129cb97bb8',
}


def strict_json(text):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON field')
            result[key]=value
        return result
    return json.loads(text,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def finite_fp32(data,count):
    if len(data)!=4*count:raise ValueError('truncated FP32 vector')
    import numpy as np
    if np.__version__!='2.5.3' or str(Path(np.__file__)) not in NUMPY_FILES:raise ValueError('NumPy identity differs')
    words=np.frombuffer(data,dtype='<u4')
    if np.any((words & 0x7f800000)==0x7f800000):raise ValueError('nonfinite raw FP32 coefficient')


def validate_raw_finite(path,metadata,raw):
    header,keys,_=metadata
    vocab=raw.RAW_HEADER.unpack(header)[1]
    if vocab!=VOCAB:raise ValueError('unexpected full vocabulary')
    with Path(path).open('rb') as f:
        if f.read(len(header))!=header:raise ValueError('raw header differs')
        for key in keys:
            if f.read(raw.RAW_ROW_META.size)!=key:raise ValueError('raw metadata differs')
            finite_fp32(f.read(vocab*4),vocab)
        if f.read(1):raise ValueError('trailing raw vector bytes')
    return dict(vectors=len(keys),fp32_values=len(keys)*vocab,finite=True)


def state_history(fixture,phases):
    words=Path(fixture).read_text().split();offset=0;history=[]
    for phase in phases:
        if offset+3>len(words) or words[offset]!=phase['kind']:raise ValueError('state fixture phase differs')
        try:cached,count=map(int,words[offset+1:offset+3])
        except ValueError:raise ValueError('state fixture integers invalid')
        offset+=3
        if cached!=phase['cached'] or count!=phase['tokens']:raise ValueError('state fixture boundaries differ')
        block=words[offset:offset+count];offset+=count
        if len(block)!=count or any(not re.fullmatch('[0-9]+',s) for s in block):raise ValueError('state fixture tokens incomplete')
        tokens=list(map(int,block))
        if any(t>=VOCAB for t in tokens):raise ValueError('state fixture token outside vocabulary')
        if phase['kind']=='P':
            if cached>len(history):raise ValueError('state fixture cache prefix invalid')
            history=history[:cached]
        elif cached!=0:raise ValueError('state decode cache prefix invalid')
        history+=tokens
        if len(history)!=phase['live']:raise ValueError('state live length differs')
    if offset!=len(words):raise ValueError('state fixture trailing phases/tokens')
    return history


def snapshot_bytes(pos):
    spans=40*min(pos,128)*512*4
    spans+=sum((pos//(2 if i<3 else 1))*(512+128)*4 for i in range(4))
    if pos&1:spans+=3*2*512*4
    return SNAPSHOT_HEADER.size+(pos+VOCAB)*4+spans


def validate_state_artifacts(logits,snapshot,fixture,phases):
    history=state_history(fixture,phases);pos=len(history)
    if not 0<pos<8192 or Path(logits).stat().st_size!=len(phases)*VOCAB*4:raise ValueError('state logit extent differs')
    last=None
    with Path(logits).open('rb') as f:
        for _ in phases:last=f.read(VOCAB*4);finite_fp32(last,VOCAB)
        if f.read(1):raise ValueError('trailing phase logit bytes')
    if Path(snapshot).stat().st_size!=snapshot_bytes(pos):raise ValueError('canonical snapshot extent differs')
    expected=(0x34565344,2,8192,1,128,128,8193,pos,40,512,128,VOCAB,0x413431)
    with Path(snapshot).open('rb') as f:
        if f.read(SNAPSHOT_HEADER.size)!=SNAPSHOT_HEADER.pack(*expected):raise ValueError('canonical snapshot header differs')
        token_bytes=f.read(pos*4)
        if token_bytes!=struct.pack('<'+str(pos)+'I',*history):raise ValueError('snapshot history differs from frozen replay')
        final=f.read(VOCAB*4);finite_fp32(final,VOCAB)
        if final!=last:raise ValueError('snapshot final logits differ from final phase')
    return dict(header=list(expected),bytes=snapshot_bytes(pos),history_tokens=pos,phase_vectors=len(phases),finite_logit_values=(len(phases)+1)*VOCAB,internal_state_sentinels_preserved=True)


def validate_state_csv(path,phases):
    with Path(path).open() as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=STATE_FIELDS:raise ValueError('state CSV schema differs')
        rows=list(reader)
    if len(rows)!=len(phases):raise ValueError('state phase count differs')
    for i,(row,phase) in enumerate(zip(rows,phases)):
        if set(row)!=set(STATE_FIELDS) or None in row.values():raise ValueError('state CSV columns differ')
        expected=dict(phase=i,kind=phase['kind'],context=phase['live'],tokens=phase['tokens'])
        if any(row[k]!=str(v) for k,v in expected.items()):raise ValueError('state phase metadata differs')
        if any(not math.isfinite(float(row[k])) or float(row[k])<0 for k in ('ms','first_decode_ms')):raise ValueError('state durations invalid')
        if float(row['first_decode_ms'])>float(row['ms']) or (phase['kind']=='P' and float(row['first_decode_ms'])!=0):raise ValueError('state first-decode duration invalid')
    return rows


def validate_quality_json(text,rows):
    result=strict_json(text)
    if not isinstance(result,dict) or result.keys()!=QUALITY_INTS|QUALITY_FLOATS:raise ValueError('quality JSON schema differs')
    if any(type(result[k]) is not int or not 0<=result[k]<=512 for k in QUALITY_INTS):raise ValueError('quality JSON counter invalid')
    if result['cases']!=16 or result['tokens']!=512:raise ValueError('quality JSON coverage invalid')
    if any(type(result[k]) not in (int,float) or not math.isfinite(result[k]) for k in QUALITY_FLOATS):raise ValueError('quality JSON finite fields invalid')
    if any(result[k]<0 for k in ('avg_nll','tv','startup_ms','prefill_ms','decode_ms','metrics_ms')) or any(result[k]<-1e-12 for k in ('kl_base_to_raw','kl_base_to_policy')):raise ValueError('quality JSON negative field')
    checks={'avg_nll':'raw_nll','delta_nll':'delta_raw_nll','policy_delta_nll':'delta_policy_nll','kl_base_to_raw':'raw_kl','kl_base_to_policy':'policy_kl','tv':'raw_tv'}
    for aggregate,column in checks.items():
        if not math.isclose(result[aggregate],statistics.mean(float(r[column]) for r in rows),rel_tol=2e-9,abs_tol=2e-10):raise ValueError('quality JSON/CSV arithmetic differs')
    if result['argmax_flips']!=sum(int(r['argmax_flip']) for r in rows):raise ValueError('quality JSON/CSV flip count differs')
    return result


def original_inputs(info):
    if sha(info['manifest'])!=info['sha256']:raise ValueError('original TSV changed')
    cases=[];seen=set();files={}
    for line in Path(info['manifest']).read_text().splitlines():
        if not line or line.startswith('#'):continue
        parts=line.split('\t')
        if len(parts)!=3 or parts[0] in seen:raise ValueError('original TSV case schema differs')
        name,prompt,continuation=parts;seen.add(name)
        for path in (prompt,continuation):
            if path not in info['files']:raise ValueError('original case path not pinned')
            files[path]=info['files'][path]
        cases.append(dict(case=name,prompt_sha256=files[prompt],continuation_sha256=files[continuation]))
    if len(cases)!=16 or files!=info['files']:raise ValueError('original TSV input inventory differs')
    return cases


def sha(path):
    path=Path(path)
    if path.suffix.lower()=='.gguf':raise ValueError('GGUF hashing forbidden')
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def raw_module(path=RAW):
    if sha(path)!=RAW_SHA:raise ValueError('raw helper changed')
    spec=importlib.util.spec_from_file_location('resident_weights_raw',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def arm_specs():
    specs=[dict(name='state-'+name,kind='state',binary=binary,mode=mode,flag='0')
           for name,binary,mode in [('old-native','old7c6e','native'),('new-native0','new8a47','native'),
                                    ('old-approx0','old7c6e','apply'),('new-approx0','new8a47','apply')]]
    primary=[('old-native','old7c6e','native','0'),('old-approx0','old7c6e','apply','0'),
             ('new-native0','new8a47','native','0'),('new-approx0','new8a47','apply','0')]
    for group in ('calibration16','heldout16'):
        arms=list(primary)
        if group=='calibration16':arms += [('flag1-'+mode,'new8a47',mode,'1') for mode in ('native','lookup','noop')]
        arms += [('native-weights1','new8a47','apply','1'),('old-native-after','old7c6e','native','0')]
        specs += [dict(name=group+'-'+name,kind='quality',group=group,binary=binary,mode=mode,flag=flag)
                  for name,binary,mode,flag in arms]
    return specs


def policy(spec):
    result={'DS4_METAL_V41_SELECTED_TILE':'64','DS4_METAL_V41_SELECTED_LIMIT':'1024',
            'DS4_METAL_STREAMING_EXPERT_IO_METRICS':'1',P+'NATIVE_WEIGHTS':spec['flag'],P+'MODE':spec['mode']}
    if spec['mode']=='native' and spec['flag']=='0':result[P+'GAP']='0'
    else:result.update({P+k:v for k,v in dict(GAP='.30',MAX_MASS='.20',MAX_TOTAL_MASS='.40',MAX_RANK='48',MAX_SUBS='3',POLICY='replace').items()})
    validate_policy(result)
    return result


def validate_policy(env):
    flag=env[P+'NATIVE_WEIGHTS']
    if flag not in ('0','1') or env[P+'MODE'] not in ('native','lookup','noop','apply'):raise ValueError('invalid strict flag/mode')
    gap=float(env[P+'GAP'])
    if not math.isfinite(gap) or not 0<=gap<=1:raise ValueError('invalid finite GAP')
    if flag=='1' and (gap<=0 or env.get(P+'POLICY')!='replace'):raise ValueError('FLAG1 requires positive GAP and replace')


def command_template(output):
    return [str(PYTHON),str(GUARD),str(output/'pressure'),'--',str(PYTHON),str(output/'driver.py'),
            '--run',str(output),'--manifest-sha256','<manifest_sha256>']


def guard_command(output,digest):return command_template(output)[:-1]+[digest]


def require_main_guard(output,digest):
    if Path.cwd()!=ROOT or sha(GUARD)!=GUARD_SHA or sha(PYTHON)!=PYTHON_SHA:raise RuntimeError('MAIN cwd/guard/interpreter identity differs')
    args=shlex.split(subprocess.check_output(['/bin/ps','-p',str(os.getppid()),'-o','command='],text=True,timeout=2))
    if not args or args[1:]!=guard_command(output,digest)[1:]:raise RuntimeError('exact prepared MAIN guard must be direct parent')
    if not os.path.samefile(args[0],PYTHON):raise RuntimeError('parent interpreter is not pinned framework Python')
    with (ROOT/'speed-bench/ideas/artifacts/model-job.lock').open('r+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EAGAIN,errno.EWOULDBLOCK):return
            raise
        fcntl.flock(lock,fcntl.LOCK_UN)
    raise RuntimeError('MAIN lock not held')


def parse_io(text):
    lines=[s.removeprefix('PROBE_IO ') for s in text.splitlines() if s.startswith('PROBE_IO ')]
    if len(lines)!=1:raise ValueError('missing/duplicate exact IO footer')
    fields={}
    for item in lines[0].split():
        key,sep,value=item.partition('=')
        if not sep or key in fields or not re.fullmatch('[0-9]+',value) or int(value)>2**64-1:raise ValueError('malformed IO')
        fields[key]=int(value)
    if fields.keys()!=IO_FIELDS or fields['enabled']!=1 or fields['requested_bytes']!=fields['returned_bytes']:raise ValueError('IO count/enable/read mismatch')
    if 'resident-backfill' in text or 'staged experts probes=' in text:raise ValueError('unexpected background/staged work')
    return fields


def parse_cache(text,state=False):
    budgets=re.findall(r'^ds4:   streaming expert cache budget=(\d+) experts entries=(\d+) ',text,re.M)
    if len(budgets)!=1 or int(budgets[0][0])!=6136 or not 0<=int(budgets[0][1])<=6136:raise ValueError('cache regime differs')
    if state:
        phases=re.findall(r'^REPLAY_CACHE phase=(\d+) configured_slots=(\d+) resident_entries=(\d+)$',text,re.M)
        if len(phases)!=7 or any(int(p)!=i or int(c)!=6136 or not 0<=int(r)<=6136 for i,(p,c,r) in enumerate(phases)):raise ValueError('phase cache regime differs')
    result=dict(configured_slots=int(budgets[0][0]),resident_entries=int(budgets[0][1]))
    line=next(s for s in text.splitlines() if s.startswith('ds4:   streaming expert cache budget='))
    for key in ('hits','misses','wraps','evictions','buffer_allocs','buffer_reuses'):
        values=re.findall(r'\b'+key+r'=([0-9]+)(?:\s|$)',line)
        if len(values)!=1 or int(values[0])>2**64-1:raise ValueError('cache integer footer invalid')
        result[key]=int(values[0])
    return result


def parse_route(text,spec,layers):
    lines=[s.removeprefix('ds4: resident-route ') for s in text.splitlines() if s.startswith('ds4: resident-route ')]
    signed=[s.removeprefix('ds4: resident-route-native-weights ') for s in text.splitlines() if s.startswith('ds4: resident-route-native-weights ')]
    if spec['mode']=='native' and spec['flag']=='0':
        if lines or signed:raise ValueError('default native unexpectedly routed')
        return None
    if len(lines)!=1:raise ValueError('route footer missing/duplicate')
    values={}
    for item in lines[0].split():
        key,sep,value=item.partition('=')
        if not sep or key in values:raise ValueError('invalid route item')
        if key in ROUTE_INTS:
            if not re.fullmatch('[0-9]+',value) or int(value)>2**64-1:raise ValueError('route integer invalid')
            values[key]=int(value)
        elif key in ROUTE_FLOATS:
            values[key]=float(value)
            if not math.isfinite(values[key]) or values[key]<0:raise ValueError('route float invalid')
        else:raise ValueError('unknown route field')
    if values.keys()!=ROUTE_INTS|ROUTE_FLOATS or values['layers']!=(0 if spec['mode']=='native' else layers) or values['drops'] or values['substitutions']!=values['skipped_loads']:raise ValueError('route counters differ')
    if spec['mode']=='apply':
        if not values['substitutions'] or values['substitutions']!=values['planned_substitutions'] or 2*values['double_layers']+3*values['triple_layers']>values['substitutions']:raise ValueError('no/invalid actual committed substitutions')
    elif values['substitutions'] or values['skipped_bytes'] or values['double_layers'] or values['triple_layers']:raise ValueError('no-action mode committed')
    if spec['mode']=='lookup' and (values['full_snapshots'] or values['planned_substitutions']):raise ValueError('lookup did full planning')
    if spec['mode']=='noop' and not values['planned_substitutions']:raise ValueError('noop did not exercise a plan')
    if spec['flag']=='1':
        if signed!=['requested=1 substitutions='+str(values['substitutions'])]:raise ValueError('signed actual weight-substitution proof differs')
    elif signed:raise ValueError('flag0 emitted native-weight footer')
    return values


def validate_csv(path,metadata,cases,raw):
    with Path(path).open() as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=CSV_FIELDS:raise ValueError('CSV schema differs')
        rows=list(reader)
    if len(rows)!=512 or len(metadata[1])!=512:raise ValueError('CSV/raw incomplete')
    seen=set()
    for row,key_bytes in zip(rows,metadata[1]):
        if set(row)!=set(CSV_FIELDS) or None in row.values():raise ValueError('CSV missing/extra columns')
        case,step,target,prompt_tokens,_=raw.RAW_ROW_META.unpack(key_bytes)
        for key,expected in [('step',step),('target',target),('prompt_tokens',prompt_tokens)]:
            if row[key]!=str(expected):raise ValueError('CSV/raw key mismatch')
        if row['case']!=cases[case]['case'] or (case,step) in seen:raise ValueError('CSV case/order/duplicate differs')
        seen.add((case,step))
        if row['argmax_flip'] not in ('0','1'):raise ValueError('invalid flip')
        for key in CSV_FIELDS[4:13]+['decode_ms']:
            if not math.isfinite(float(row[key])):raise ValueError('nonfinite CSV field')
        if any(float(row[k])<0 for k in ('base_nll','raw_nll','policy_nll','decode_ms')):raise ValueError('negative NLL/duration')
        if any(not -1e-12<=float(row[k])<=1+1e-12 for k in ('raw_tv','policy_tv')):raise ValueError('invalid total variation')
        if any(float(row[k]) < -1e-12 for k in ('raw_kl','policy_kl')):raise ValueError('invalid KL divergence')
        if any(not math.isclose(float(row[delta]),float(row[nll])-float(row['base_nll']),rel_tol=2e-9,abs_tol=2e-10) for delta,nll in [('delta_raw_nll','raw_nll'),('delta_policy_nll','policy_nll')]):raise ValueError('NLL/CSV arithmetic differs')
        if float(row['decode_ms'])<0:raise ValueError('negative decode duration')
    if seen!={(case,step) for case in range(16) for step in range(32)}:raise ValueError('case/step coverage differs')
    return rows


def prepare(output):
    if sha(BUILD/'build-manifest.json')!=BUILD_SHA or sha(BUILD/'build-plan.json')!=PLAN_SHA or sha(INPUT_PROOF)!=INPUT_SHA:raise ValueError('source build/input proof changed')
    build=json.loads((BUILD/'build-manifest.json').read_text()); source=json.loads((BUILD/'build-plan.json').read_text()); identities=json.loads(INPUT_PROOF.read_text())
    raw=raw_module();raw.check_frozen_assets({Path(k):(k,v) for k,v in build['files'].items()})
    raw.check_frozen_assets({Path(k):(k,v) for k,v in NUMPY_FILES.items()})
    if sha(GUARD)!=GUARD_SHA or sha(PYTHON)!=PYTHON_SHA:raise ValueError('guard/interpreter changed')
    output.mkdir(parents=True,exist_ok=False)
    groups={}
    for group,info in source['inputs'].items():
        destination=output/group;destination.mkdir()
        fixtures,inputs=raw.freeze_inputs(Path(info['manifest']),destination,16)
        if sha(identities[group]['prior'])!=identities[group]['sha256']:raise ValueError('prior case authority differs')
        prior=strict_json(Path(identities[group]['prior']).read_text())
        expected=prior['inputs'] if 'inputs' in prior else prior['cases']
        if inputs!=expected or inputs!=original_inputs(info) or [case['case'] for case in inputs]!=identities[group]['ordered_case_ids']:raise ValueError('case identity/order differs')
        groups[group]=dict(fixtures=str(fixtures),inputs=inputs)
    for source_path,name in [(RAW,'raw_helpers.py'),(Path(__file__),'driver.py'),(BUILD/'build-manifest.json','build-manifest.json'),(BUILD/'build-plan.json','build-plan.json'),(INPUT_PROOF,'input-identity-proof.json')]:shutil.copyfile(source_path,output/name)
    (output/'metal').mkdir()
    for path in sorted((WORK/'metal').glob('*.metal')):shutil.copyfile(path,output/'metal'/path.name)
    external=dict(build['files']);external.update(NUMPY_FILES);external.update({str(BUILD/'build-manifest.json'):BUILD_SHA,str(BUILD/'build-plan.json'):PLAN_SHA,str(INPUT_PROOF):INPUT_SHA,str(GUARD):GUARD_SHA,str(PYTHON):PYTHON_SHA,str(Path(__file__)):sha(__file__)})
    plan=dict(build_sha256=BUILD_SHA,source_plan_sha256=PLAN_SHA,settings=SETTINGS,arms=arm_specs(),groups=groups,model_from_prior=source['model_from_prior_only'],state_fixture=source['optional_state_control'],guard_command_template=command_template(output),guard_sha256=GUARD_SHA,guard_interpreter=dict(path=str(PYTHON),sha256=PYTHON_SHA),external_files=external,files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()},model_stat_or_open_in_preparation=False,root_model_grant_required=True,first_attempt_only=True,timing_claim=False,estimated_minutes=[20,25],scope='Full old/new default teacher/raw/state/slots +flag1 no-action then fixed weights1 quality. IO directly measures preclose P+D; no-stage teardown submits no new expert reads by separate source audit, no direct postclose counter or physical SSD claim.')
    write(output/'manifest.json',plan);digest=sha(output/'manifest.json')
    write(output/'prepared-command.json',dict(command=guard_command(output,digest),manifest_sha256=digest,launch_requires_root_grant=True))
    print(json.dumps(dict(prepared=str(output),manifest_sha256=digest,arms=len(plan['arms']),model_opened=False)),flush=True)


def verified_plan(output,digest):
    if not re.fullmatch('[0-9a-f]{64}',digest or '') or sha(output/'manifest.json')!=digest:raise ValueError('explicit plan authority differs')
    plan=strict_json((output/'manifest.json').read_text());command=strict_json((output/'prepared-command.json').read_text())
    if command.get('manifest_sha256')!=digest or command.get('command')!=guard_command(output,digest):raise ValueError('prepared command changed')
    if plan['build_sha256']!=BUILD_SHA or plan['source_plan_sha256']!=PLAN_SHA or plan['settings']!=SETTINGS or plan['arms']!=arm_specs() or plan['guard_command_template']!=command_template(output) or plan['guard_sha256']!=GUARD_SHA or plan['guard_interpreter']!=dict(path=str(PYTHON),sha256=PYTHON_SHA):raise ValueError('frozen study differs')
    return plan


def assets_and_inventory(output,plan):
    if sha(output/'build-manifest.json')!=BUILD_SHA or sha(output/'build-plan.json')!=PLAN_SHA or sha(output/'input-identity-proof.json')!=INPUT_SHA:raise ValueError('mandatory build/input authority differs')
    build=json.loads((output/'build-manifest.json').read_text()); source=json.loads((output/'build-plan.json').read_text()); identities=json.loads((output/'input-identity-proof.json').read_text())
    if plan['model_from_prior']!=source['model_from_prior_only'] or plan['state_fixture']!=source['optional_state_control']:raise ValueError('model/state identity differs from frozen source plan')
    required={'raw_helpers.py':RAW_SHA,'driver.py':sha(__file__),'build-manifest.json':BUILD_SHA,'build-plan.json':PLAN_SHA,'input-identity-proof.json':INPUT_SHA}
    for group,info in plan['groups'].items():
        if group not in ('calibration16','heldout16') or info['inputs']!=original_inputs(source['inputs'][group]) or [x['case'] for x in info['inputs']]!=identities[group]['ordered_case_ids']:raise ValueError('case names/order/hash mapping changed')
        lines=[]
        for i,case in enumerate(info['inputs']):
            paths=[]
            for kind in ('prompt','continuation'):
                name=f'{group}/inputs/{i:03d}-{kind}.txt';paths.append(str(output/name));required[name]=case[kind+'_sha256']
            lines.append('\t'.join([case['case'],*paths]))
        fixture=output/group/'fixtures.tsv'
        if info['fixtures']!=str(fixture) or fixture.read_bytes()!=('\n'.join(lines)+'\n').encode():raise ValueError('fixture paths/order differs')
        required[str(fixture.relative_to(output))]=sha(fixture)
    if set(plan['groups'])!={'calibration16','heldout16'}:raise ValueError('input group missing')
    for path,value in build['files'].items():
        p=Path(path)
        if p.parent==WORK/'metal':required['metal/'+p.name]=value
    if sum(name.startswith('metal/') for name in required)!=26 or plan['files']!=required:raise ValueError('mandatory prepared inventory differs')
    expected=dict(build['files']);expected.update(NUMPY_FILES);expected.update({str(BUILD/'build-manifest.json'):BUILD_SHA,str(BUILD/'build-plan.json'):PLAN_SHA,str(INPUT_PROOF):INPUT_SHA,str(GUARD):GUARD_SHA,str(PYTHON):PYTHON_SHA,str(WORK/'speed-bench/ideas/prepare_resident_native_weights_study.py'):required['driver.py']})
    if plan['external_files']!=expected:raise ValueError('mandatory external runtime inventory differs')
    return {**{Path(k):(k,v) for k,v in expected.items()},**{output/k:(k,v) for k,v in required.items()}}


def execute(output,digest):
    plan=verified_plan(output,digest);require_main_guard(output,digest)
    raw=raw_module(output/'raw_helpers.py');assets=assets_and_inventory(output,plan)
    assets[output/'manifest.json']=('manifest',digest);assets[output/'prepared-command.json']=('command',sha(output/'prepared-command.json'))
    raw.check_frozen_assets(assets)
    model=Path(plan['model_from_prior']['path']);identity=raw.identity(model)
    if identity!=plan['model_from_prior']:raise ValueError('guarded model identity differs')
    run=output/'run';run.mkdir(exist_ok=False);write(run/'model-identity.json',dict(identity=identity,guard_verified_before_stat=True))
    env={k:v for k,v in os.environ.items() if not k.startswith('DS4')}
    env.update({'DS4_METAL_'+p.stem.upper()+'_SOURCE':str(p) for p in (output/'metal').glob('*.metal')})
    results={};metadata={};references={};records={}
    for spec in plan['arms']:
        require_main_guard(output,digest);raw.check_frozen_assets(assets)
        if raw.identity(model)!=identity:raise ValueError('model identity changed')
        name=spec['name'];childenv=dict(env,**policy(spec))
        if spec['kind']=='state':
            logits=run/(name+'.logits');snapshot=run/(name+'.snapshot')
            childenv.update(DS4_REPLAY_CONTEXT='8192',DS4_REPLAY_CACHE_GB='64',DS4_REPLAY_PREFILL_CHUNK='512',DS4_REPLAY_CACHE_REGIME='1')
            argv=[str(BUILD/('replay-'+spec['binary'])),str(model),plan['state_fixture']['fixture'],str(logits),str(snapshot)]
        else:
            group=spec['group'];recording=name==group+'-old-native';reference=run/(group+'-reference.raw');record=reference if recording else run/(name+'.raw')
            argv=[str(BUILD/('quality-'+spec['binary'])),str(model),plan['groups'][group]['fixtures'],str(run/(name+'.csv')),str(reference),'record' if recording else 'compare','--cases','16','--tokens','32','--ctx','4096','--cache-gb','64','--chunk','512']
            if not recording:argv+=['--candidate-record',str(record)]
        write(run/(name+'-command.json'),dict(argv=argv,DS4_environment={k:v for k,v in childenv.items() if k.startswith('DS4')},first_attempt=True))
        with (run/(name+'.stdout')).open('x') as out,(run/(name+'.log')).open('x') as err:
            completed=subprocess.run(argv,cwd=ROOT,env=childenv,stdout=out,stderr=err)
        write(run/(name+'-exit.json'),dict(returncode=completed.returncode))
        if completed.returncode:raise RuntimeError(name+' helper failed rc='+str(completed.returncode))
        raw.check_frozen_assets(assets)
        if raw.identity(model)!=identity:raise ValueError('model identity changed during arm')
        text=(run/(name+'.log')).read_text();cache=parse_cache(text,spec['kind']=='state')
        if spec['kind']=='state':
            rows=validate_state_csv(run/(name+'.stdout'),plan['state_fixture']['phases'])
            state=validate_state_artifacts(logits,snapshot,plan['state_fixture']['fixture'],plan['state_fixture']['phases'])
            row=dict(name=name,logits_sha256=sha(logits),snapshot_sha256=sha(snapshot),state_validation=state,cache=cache,route=parse_route(text,spec,343*40))
            counterpart='state-old-native' if spec['mode']=='native' else 'state-old-approx0'
            if spec['binary']=='new8a47' and any(row[k]!=results[counterpart][k] for k in ('logits_sha256','snapshot_sha256','cache','route')):raise ValueError('old/new state/default route control differs')
        else:
            if recording:
                metadata[group]=raw.raw_record_metadata(reference,16,32,512);references[group]=sha(reference);assets[reference]=('frozen fresh '+group+' native reference',references[group])
                write(run/(group+'-reference-proof.json'),dict(sha256=references[group],metadata=metadata[group][2],created_before_candidate=True))
            layout=raw.validate_candidate_record(record,metadata[group],16,32,512)
            finite=validate_raw_finite(record,metadata[group],raw)
            rows=validate_csv(run/(name+'.csv'),metadata[group],plan['groups'][group]['inputs'],raw)
            row=validate_quality_json((run/(name+'.stdout')).read_text(),rows)
            row.update(name=name,raw_sha256=sha(record),layout=layout,finite_validation=finite,cache=cache,io=parse_io(text),route=parse_route(text,spec,512*40),**raw.summarize(run/(name+'.csv')))
            if len(row['per_prompt'])!=16 or any(p['tokens']!=32 for p in row['per_prompt'].values()):raise ValueError('prompt completeness differs')
            native=results.get(group+'-old-native',row)
            noaction=spec['mode'] in ('native','lookup','noop')
            if noaction and (row['raw_sha256']!=references[group] or row['bit_equal_rows']!=512 or any(row[k]!=0 for k in ('delta_nll','kl_base_to_raw','tv','argmax_flips')) or row['io']!=native['io'] or row['cache']!=native['cache']):raise ValueError('native/no-action fullraw/IO/cache control differs')
            if name==group+'-new-approx0':
                old=results[group+'-old-approx0']
                if any(row[k]!=old[k] for k in ('raw_sha256','io','cache','route')):raise ValueError('old/new approximation flag0 raw/IO/routes differ')
            records[name]=str(record)
        results[name]=row
        write(run/(name+'-checked.json'),row)
        print(json.dumps(dict(name=name,checked=True,returncode=0,raw_sha256=row.get('raw_sha256'),state_sha256=row.get('snapshot_sha256'),delta_nll=row.get('delta_nll'),kl=row.get('kl_base_to_raw'),io=row.get('io'),route=row.get('route')),allow_nan=False),flush=True)
    write(run/'summary.json',dict(status='complete',arms=results,model_identity=identity,guarded=True,full_default_state_raw_teacher_io_controls=True,positive_flag1_noaction_controls=True,first_attempt_only=True,timing_claim=False,io_scope=plan['scope']))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',type=Path);parser.add_argument('--run',type=Path);parser.add_argument('--manifest-sha256');args=parser.parse_args()
    if bool(args.prepare)==bool(args.run):parser.error('choose prepare or run')
    if args.prepare and args.manifest_sha256:parser.error('manifest SHA only applies to run')
    if args.run and not args.manifest_sha256:parser.error('run requires pinned manifest SHA')
    if args.prepare:prepare(args.prepare.resolve())
    else:execute(args.run.resolve(),args.manifest_sha256)


if __name__=='__main__':main()
