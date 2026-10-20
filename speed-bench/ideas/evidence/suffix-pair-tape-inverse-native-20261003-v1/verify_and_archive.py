#!/usr/bin/env python3
"""CPU-only retained-file verification/archive for the frozen inverse-parity pending-pair pilot.

Never executes producers, guards, tokenizers or engines and never reads model
metadata. Large native states/vectors/binaries stay at their original paths.
"""
from array import array
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import sys

REPO=Path('/Volumes/Models/ds4')
SOURCE=REPO/'.workspaces/idea-suffix-followup'
PREPARED=SOURCE/'speed-bench/ideas/artifacts/suffix-followup-inverse-parity-pending-prepared-20261003-v1'
OUT=Path(__file__).parent
MANIFEST_SHA='9781896e85d6ae9eee030f034a9fab0a0e0f3e2290c2c3930be20a5a36e32ca2'
COMMAND_SHA='0db865613cf9b96b8e96ba469f9b1e4990d8f0ef8bc2ccbe33b9e3d0f16d9111'
GEOMETRY=dict(prefix=1408,shift=65,span=65,tail=1,decode=16,ctx=4096,chunk=512,cache_gb=64,capacity=1024)
ARMS=[('old-native-before','native'),('new-native','native'),('new-tape-native','tape-native'),('suffix-reuse','reuse'),('old-native-after','native')]

def sha(path):
    path=Path(path)
    assert path.suffix.lower()!='.gguf','Model access forbidden'
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def identity(path):
    path=Path(path)
    assert path.suffix.lower()!='.gguf','Model metadata forbidden'
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))

def load(path):return json.loads(Path(path).read_text())
def put(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def copy_file(path,name):
    destination=OUT/name;destination.parent.mkdir(parents=True,exist_ok=True)
    assert not destination.exists(),str(destination)
    shutil.copy2(path,destination)
    assert sha(path)==sha(destination)

def close(a,b):return math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10)
def floats(data):
    values=array('f');values.frombytes(data)
    if sys.byteorder!='little':values.byteswap()
    assert all(math.isfinite(x) for x in values)
    return values

def z(values):
    peak=max(values)
    return peak+math.log(math.fsum(math.exp(x-peak) for x in values))

def raw_info(path,approximate):
    with path.open('rb') as f:
        header=struct.unpack('<8I',f.read(32));assert header==(0x53505431,129280,16,1408,65,65,1,int(approximate))
        targets=f.read(64);ids=struct.unpack('<16I',targets);assert all(0<=i<header[1] for i in ids)
        reference=hashlib.sha256()
        for row in range(18):
            data=f.read(129280*4);assert len(data)==129280*4
            floats(data);reference.update(data)
        candidate=hashlib.sha256()
        if approximate:
            for row in range(17):
                data=f.read(129280*4);assert len(data)==129280*4
                floats(data);candidate.update(data)
        assert f.read(1)==b''
    return dict(**identity(path),reference_rows=18,candidate_rows=17 if approximate else 0,
        reference_sha256=reference.hexdigest(),candidate_sha256=candidate.hexdigest() if approximate else None,
        teacher_token_hex=targets.hex(),teacher_ids=ids)

def recompute(raw,targets):
    size=129280*4;first=96;rows=[]
    with raw.open('rb') as reference,raw.open('rb') as candidate:
        reference.seek(first+size);candidate.seek(first+18*size)
        for step in range(17):
            pb,qb=reference.read(size),candidate.read(size)
            assert len(pb)==size and len(qb)==size
            p,q=floats(pb),floats(qb);lp,lq=z(p),z(q)
            kl=math.fsum(math.exp(x-lp)*((x-y)+(lq-lp)) for x,y in zip(p,q))
            if kl<0 and kl>-1e-10:kl=0
            tv=.5*math.fsum(abs(math.exp(x-lp)-math.exp(y-lq)) for x,y in zip(p,q))
            target=targets[step] if step<16 else -1
            dnll=(lq-q[target])-(lp-p[target]) if step<16 else 0
            rows.append(dict(step=step,target=target,delta_nll=dnll,kl=kl,tv=tv,
                argmax_flip=int(max(range(len(p)),key=p.__getitem__)!=max(range(len(q)),key=q.__getitem__)),bit_equal=int(pb==qb)))
    return rows

def main():
    assert sha(PREPARED/'manifest.json')==MANIFEST_SHA
    assert sha(PREPARED/'prepared-command.json')==COMMAND_SHA
    plan=load(PREPARED/'manifest.json');assert plan['geometry']==GEOMETRY and plan['case']=='inverse-parity-pending'
    frozen={}
    for name,pin in plan['files'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        actual=identity(PREPARED/name);assert actual['bytes']==pin['bytes'] and actual['sha256']==pin['sha256'],name
        frozen[name]=actual
        if name not in ['baseline-probe','new-probe']:
            copy_file(PREPARED/name,'frozen/'+name)
    command=load(PREPARED/'prepared-command.json');assert command['cwd']==str(REPO) and command['manifest_sha256']==MANIFEST_SHA
    guard=Path(command['command'][1]);guard_pin='461e8c517e82a687b4579fc450c1e7128b1510192d088bab719682b33c946c3c'
    assert sha(guard)==guard_pin;copy_file(guard,'frozen/MAIN-pressure_run.py')
    for name in ['manifest.json','prepared-command.json','root-launch-proof.json','root-preflight-proof.json','root-release-proof.json']:
        copy_file(PREPARED/name,name)
    for name in ['result.json','command.json','pressure.jsonl']:copy_file(PREPARED/'pressure'/name,'pressure/'+name)
    guard_result=load(PREPARED/'pressure/result.json');assert guard_result['status']=='complete' and type(guard_result['child_returncode']) is int and guard_result['child_returncode']==0
    release=load(PREPARED/'root-release-proof.json')
    assert release['guard_result']==guard_result and release['lock_reacquired'] is True
    assert type(release['guard_result']['child_returncode']) is int and release['guard_result']['child_returncode']==0
    assert type(release['live_model_processes']) is list and release['live_model_processes']==[]
    assert type(release['time']) in (int,float) and math.isfinite(release['time'])
    pressure=[json.loads(line) for line in (PREPARED/'pressure/pressure.jsonl').read_text().splitlines()]
    assert all(type(x[k]) in (int,float) and math.isfinite(x[k]) for x in pressure for k in ['time','swap_gb','wired_gb','compressor_gb'])
    assert all(type(release[k]) in (int,float) and math.isfinite(release[k]) for k in ['swap_growth_gb','peak_wired_compressor_gib'])
    assert pressure and all(type(x['level']) is int and x['level']==1 and x['reason'] is None and x['consecutive_bad']==0 for x in pressure)
    swap_growth=max(x['swap_gb'] for x in pressure)-min(x['swap_gb'] for x in pressure)
    peak=max(x['wired_gb']+x['compressor_gb'] for x in pressure)
    assert type(release['normal_pressure_samples']) is int and release['normal_pressure_samples']==len(pressure)==215
    assert type(release['bad_pressure_samples']) is int and release['bad_pressure_samples']==0
    assert close(swap_growth,release['swap_growth_gb']) and swap_growth==0
    assert close(peak,release['peak_wired_compressor_gib']) and release['time']>=max(x['time'] for x in pressure)
    all_raw={};all_states={};controls=None;parsed=[]
    saved=load(PREPARED/'run/summary.json');assert saved['status']=='COMPLETE' and saved['native_controls_exact'] is True and saved['timing_promoted'] is False
    assert [x['arm'] for x in saved['arms']]==[a for a,_ in ARMS]
    for arm,mode in ARMS:
        base=PREPARED/'run'/arm
        exit_value=load(PREPARED/'run'/(arm+'-exit.json'));assert type(exit_value['returncode']) is int and exit_value['returncode']==0
        summary=load(str(base)+'.summary.json');assert summary['mode']==mode
        assert all(type(summary[k]) is int and summary[k]==GEOMETRY[k] for k in ['prefix','shift','span','tail','decode'])
        assert summary['endpoint_partition_aligned'] is True and summary['identity_native_checked'] is False and summary['identity_native_state_equal'] is False and summary['identity_native_logits_equal'] is False
        assert summary['raw_reference_rows']==18 and summary['raw_candidate_rows']==(17 if mode=='reuse' else 0)
        assert type(summary['peak_staging_bytes_including_reserve']) is int and 0<summary['peak_staging_bytes_including_reserve']<summary['staging_budget_bytes']==128<<20
        raw=raw_info(Path(str(base)+'.raw.bin'),mode=='reuse');all_raw[arm]=raw
        state={name:identity(str(base)+name) for name in ['.source.state.bin','.fresh.state.bin']};all_states[arm]=state
        parity=dict(reference_sha256=raw['reference_sha256'],teacher_token_hex=raw['teacher_token_hex'],state={k:v['sha256'] for k,v in state.items()})
        if controls is None:controls=parity
        else:assert parity==controls,arm
        checked=load(PREPARED/'run'/(arm+'-checked.json'))
        assert checked['summary']==summary and checked['native_parity'] is True
        assert checked['raw']['full_sha256']==raw['sha256'] and checked['raw']['reference_sha256']==raw['reference_sha256'] and checked['raw']['teacher_token_hex']==raw['teacher_token_hex']
        assert checked['state']==parity['state']
        for suffix in ['-command.json','-exit.json','-checked.json','.summary.json','.csv','.log']:
            copy_file(PREPARED/'run'/(arm+suffix),'run/'+arm+suffix)
        parsed.append(dict(arm=arm,summary=summary,native_reference_parity=True,raw=raw,state=state))
    candidate=next(x for x in parsed if x['arm']=='suffix-reuse');summary=candidate['summary']
    assert all(summary[k] is True for k in ['coverage_rejections_preserve_state','precancel_preserves_state','partial_cancel_invalidates','approximate_export_blocked'])
    assert summary['pair_inputs_reused']==195 and summary['pair_rows_rebuilt']==96 and summary['pair_boundary_rows']==0 and summary['reused_tokens']==65 and summary['fresh_tail_tokens']==1 and summary['identity_reuse_state_equal'] is False
    recomputed=recompute(Path(candidate['raw']['path']),candidate['raw']['teacher_ids'])
    with (PREPARED/'run/suffix-reuse.csv').open() as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['step','target','delta_nll','kl','tv','argmax_flip','bit_equal'];csv_rows=list(reader)
    assert len(csv_rows)==17
    for a,b in zip(recomputed,csv_rows):
        assert all(a[k]==int(b[k]) for k in ['step','target','argmax_flip','bit_equal'])
        assert all(close(a[k],float(b[k])) for k in ['delta_nll','kl','tv'])
    metrics=dict(mean_kl=math.fsum(x['kl'] for x in recomputed)/17,mean_tv=math.fsum(x['tv'] for x in recomputed)/17,
        sum_delta_nll=math.fsum(x['delta_nll'] for x in recomputed),argmax_flips=sum(x['argmax_flip'] for x in recomputed),bit_equal_rows=sum(x['bit_equal'] for x in recomputed))
    assert all(close(metrics[k],summary[k]) for k in ['mean_kl','mean_tv','sum_delta_nll']) and all(metrics[k]==summary[k] for k in ['argmax_flips','bit_equal_rows'])
    private=Path(str(PREPARED/'run/suffix-reuse')+'.private.state.bin');private_id=identity(private)
    with private.open('rb') as f:private_header=f.read(52)
    with Path(candidate['state']['.fresh.state.bin']['path']).open('rb') as f:fresh_header=f.read(52)
    assert private_header[:4]==b'DSSD' and fresh_header[:4]==b'DSV4' and private_header[4:]==fresh_header[4:]
    assert private_id['bytes']==candidate['state']['.fresh.state.bin']['bytes']
    assert private_id['sha256']==saved['arms'][3]['private_state']['sha256']
    all_states['suffix-reuse']['.private.state.bin']=private_id
    copy_file(PREPARED/'run/summary.json','run/summary.json');copy_file(PREPARED/'run/run-proof.json','run/run-proof.json')
    for name in ['tests/test_suffix_pair_tape_followup.py','tests/test_suffix_pair_tape_followup_state.c']:
        copy_file(SOURCE/name,'frozen/'+name)
    structural_checks=dict(source_odd_destination_even=True,geometry_expected_first_tail_completes_pending_pair=True,pair_boundary_rows=0,production_fresh_tail_recommendation=False)
    cpu_evidence=SOURCE/'speed-bench/ideas/evidence/suffix-pair-tape-followup-cpu-20261003-v1'
    for path in sorted(cpu_evidence.iterdir()):
        if path.is_file():copy_file(path,'cpu/'+path.name)
    mainlog=REPO/'speed-bench/ideas/artifacts/suffix-followup-inverse-root-launch-20261003-v1.log'
    copy_file(mainlog,'ROOT-main-launch.log')
    put('verified-results.json',dict(status='RETAINED_NATIVE_INVERSE_PARITY_VERIFIED',source_revision=plan['source_revision'],prepared_directory=str(PREPARED),prepared_manifest_sha256=MANIFEST_SHA,command_sha256=COMMAND_SHA,
        native_control_identity=controls,arms=parsed,private_state=private_id,independent_math=dict(method='Streamed FP32 rows; FP64 math.fsum/logsumexp, independent log-ratio arithmetic',rows=recomputed,summary=metrics),
        pressure=dict(samples=len(pressure),normal_samples=len(pressure),bad_samples=0,swap_growth_gb=swap_growth,peak_wired_plus_compressor_gib=peak),historical_release_proof=release,
        structural_controls=structural_checks,main_launch_log=identity(mainlog),timing_diagnostic=True,accuracy_claim=False,forgetting_claim=False,default_promotion=False,archive_model_assets_accessed=False))
    put('retained-large-assets.json',dict(prepared_binaries={n:frozen[n] for n in ['baseline-probe','new-probe']},raw_vectors=all_raw,native_and_diagnostic_states=all_states,frozen_inventory=frozen,
        all_large_files_retained_original=True,no_large_state_vector_binary_copies=True))
    assert sha(PREPARED/'manifest.json')==MANIFEST_SHA and sha(PREPARED/'prepared-command.json')==COMMAND_SHA
    print(json.dumps(dict(status='CPU_RETAINED_EVIDENCE_VERIFIED',arms=5,native_reference_rows=18,candidate_rows=17,independent_quality=metrics,structural_controls=structural_checks,pressure_samples=len(pressure),model_assets_accessed=False),sort_keys=True))

if __name__=='__main__':main()
