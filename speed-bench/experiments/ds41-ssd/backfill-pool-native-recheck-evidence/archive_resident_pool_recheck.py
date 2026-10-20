#!/usr/bin/env python3
"""CPU-only archive of the completed frozen native-resident admission screen."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tarfile

ROOT=Path('/Volumes/Models/ds4')
PREP=ROOT/'.workspaces/idea-resident-pool-recheck-study-fixed/speed-bench/ideas/artifacts/resident-pool-recheck-prepared-20261002-v2'
BUILD=ROOT/'.workspaces/idea-resident-backfill-pool-recheck/speed-bench/ideas/artifacts/resident-pool-recheck-build'
SOURCE=ROOT/'.workspaces/idea-resident-pool-recheck-study-fixed'
MANIFEST_SHA='04ea5a8c9b599d958fc1635019a610143d39e2bc1fad64b6ef8dcc4af78b8bc7'
COMMAND_SHA='a33f2c4f148666dc0dcb4b60b897327a012e98df33e871bfa508970ae5b2062b'
WRAPPER_SHA='1ff7d11572750c8bd8999cd4327fb576df52c50ad0d329fc0a1de76ad05a5246'
CPU_SHA='cf97af6a76da9068cb9daac7cba762888207232310324d92f6a0735a08edf71f'

def require(test,message):
    if not test:raise ValueError(message)

def sha(path):
    require(Path(path).suffix.lower()!='.gguf','No model access')
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def release_gate(release,pressure):
    require(type(pressure.get('child_returncode')) is int and pressure==dict(status='complete',child_returncode=0),'completed original guard0')
    require(type(release.get('guard_result')) is dict and release['guard_result']==pressure and
            type(release['guard_result'].get('child_returncode')) is int,'typed original release/guard agreement')
    require(release.get('lock_reacquired') is True and release.get('live_model_processes')==[],'ROOT release lock/process proof')
    for key in ['time','swap_growth_gb','peak_wired_compressor_gib']:
        require(type(release.get(key)) in (int,float) and math.isfinite(release[key]),'typed finite release '+key)
    require(type(release.get('normal_pressure_samples')) is int and release['normal_pressure_samples']>0 and
            type(release.get('bad_pressure_samples')) is int and release['bad_pressure_samples']==0 and release['swap_growth_gb']==0,
            'healthy released pressure proof')

def archive(output):
    # Typed ROOT release precedes any completed quality/counter/raw outcome read.
    release=json.loads((PREP/'root-release-proof.json').read_text());pressure=json.loads((PREP/'pressure/result.json').read_text())
    release_gate(release,pressure)
    for name,want in [('manifest.json',MANIFEST_SHA),('prepared-command.json',COMMAND_SHA),('run_resident_pool_recheck_study.py',WRAPPER_SHA),('cpu-proof.json',CPU_SHA)]:
        require(sha(PREP/name)==want,'frozen study '+name)
    study=module('fixed_recheck_archive',PREP/'run_resident_pool_recheck_study.py')
    prepared=study.verified_plan(PREP,MANIFEST_SHA);study.verify_required_assets(PREP,prepared)
    pins={PREP/name:want for name,want in prepared['files'].items()}
    for name in ['manifest.json','prepared-command.json','cpu-proof.json','root-release-proof.json']:
        pins[PREP/name]=sha(PREP/name)
    for path,want in pins.items():require(sha(path)==want,'prepared64/source pin differs '+str(path))
    cpu=json.loads((PREP/'cpu-proof.json').read_text())
    for name,want in cpu['source_files_sha256'].items():require(sha(SOURCE/name)==want,'literal CPU source differs '+name)
    require(cpu['cpu_test_source_sha256']==cpu['source_files_sha256']['tests/test_resident_pool_recheck_study.py'],'CPU test proof source identity')
    build=json.loads((BUILD/'manifest.json').read_text())
    require(sha(BUILD/'manifest.json')==study.BINARY_MANIFEST_SHA and build['source_commit']==study.INFERENCE_COMMIT,'frozen runtime authority')
    for name,want in build['files'].items():require(sha(BUILD/name)==want,'frozen runtime asset differs '+name)
    require(sha(study.GUARD)==study.GUARD_SHA,'original MAIN pressure source')
    raw=study.load_raw(PREP/'raw_helpers.py');run=PREP/'run'
    reference=run/'reference.bin';require(sha(reference)==study.REF_SHA,'retained full native reference')
    verified_large={reference:study.REF_SHA}
    metadata=raw.raw_record_metadata(reference,16,32,512)
    rows=json.loads((run/'summary.json').read_text())
    require(type(rows) is list and [r['name'] for r in rows]==[name for name,env in study.arms()],'six completed ordered arms')
    prior=json.loads((PREP/'reference-manifest.json').read_text());checks=[]
    for row,(name,extra) in zip(rows,study.arms()):
        require(json.loads((run/(name+'-exit.json')).read_text())==dict(returncode=0),'original arm0')
        candidate=run/(name+'-raw.bin');digest=sha(candidate)
        verified_large[candidate]=digest
        require(digest==row['candidate_record_sha256'],'all retained unmodified raw float bytes '+name)
        require(raw.validate_candidate_record(candidate,metadata,16,32,512)==row['candidate_record_layout'],'full512 raw row metadata '+name)
        csv_layout=study.validate_csv(run/(name+'.csv'),metadata,prior['inputs'])
        require(csv_layout==row['csv_layout'],'complete finite CSV/raw identity '+name)
        for key,value in raw.summarize(run/(name+'.csv')).items():require(row[key]==value,'paired16 quality/bootstrap '+key+' '+name)
        log=(run/(name+'.log')).read_text();pool=name.startswith('pool')
        require(study.parse_io(log,pool,pool,'recheck' in name)==row['io'],'exact logical I/O and ownership balances '+name)
        require(study.parse_late(log,pool)==row['late'] and study.parse_route(log,name)==row['route'],'exact route/late transactions '+name)
        study.check_control(row)
        if pool:
            require(row['io']['pool']['ready_age']==1600 and row['io']['pool']['consumed_late']==row['late']['installs'] and
                    row['io']['pool']['requests']==row['late']['tentative'],'pool actual late scope '+name)
        command=json.loads((run/(name+'-command.json')).read_text())
        expected=[str(PREP/'probe'),prepared['model_from_prior']['path'],str(run/'fixtures.tsv'),str(run/(name+'.csv')),str(reference),'compare',
            '--cases','16','--tokens','32','--ctx','4096','--cache-gb','64','--chunk','512','--candidate-record',str(candidate)]
        shaders={'DS4_METAL_'+p.stem.upper()+'_SOURCE':str(p) for p in (PREP/'metal').glob('*.metal')}
        require(command==dict(command=expected,env=dict(shaders,**extra)),'literal frozen per-arm argv/environment '+name)
        checks.append(dict(name=name,full512_metadata_and_csv=True,fullraw_sha256=digest,all_balances=True))
    original_checks=json.loads((run/'checks.json').read_text())
    for key in ['native_reference_exact','default_off_native_reference_exact','bf0_previous_fullraw_exact','native_demand_io_equal']:
        require(original_checks[key] is True,'required native/BF0 control '+key)
    require(rows[0]['io']['demand']==rows[-1]['io']['demand'],'native demand counter repetition')
    output.mkdir(parents=True,exist_ok=False)
    compact=dict(status='CPU_ARCHIVE_INTEGRITY_PASS',original_guard_returncode=0,source_study_commit='7040eef1359deeb802ae1f1686707fd43ac80a7e',
        inference_commit=study.INFERENCE_COMMIT,original_results=rows,controls=original_checks,independent_checks=checks,release_proof=release,
        scope=prepared['scope'],interpretation='Recheck drops31/34 admissions, but background total remains above BF0; no adoption, latency or physical-SSD claim. Async approximate repeats may differ.',
        model_opened=False,model_metadata_read=False,all_originals_retained=True)
    write(output/'compact.json',compact)
    # Preserve small original sources/proofs/logs/CSVs; pin all large arrays/binaries.
    saved=[]
    original_files=[*PREP.rglob('*'),*BUILD.rglob('*')]
    for path in sorted(set(original_files)):
        if not path.is_file():continue
        digest=verified_large[path] if path in verified_large else sha(path)
        saved.append(dict(path=str(path),sha256=digest,bytes=path.stat().st_size))
        if path.suffix in ('.bin','.o') or path.name in ('probe','fixture-pool-recheck-io-quality','replay-pool-recheck-metrics'):continue
        relative=Path('prepared')/path.relative_to(PREP) if PREP in path.parents else Path('runtime-build')/path.relative_to(BUILD)
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        require(sha(target)==digest,'small original archive copy')
    for name,want in cpu['source_files_sha256'].items():
        path=SOURCE/name;target=output/'study-source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        require(sha(target)==want,'immutable7040 CPU source copy')
    write(output/'retained-original-pins.json',saved)
    shutil.copy2(Path(__file__),output/'archive_resident_pool_recheck.py')
    # The runtime source/log bundle is compact; compiled slabs and raw arrays stay pinned.
    with tarfile.open(output/'runtime-source-and-proofs.tar.gz','w:gz') as tar:tar.add(output/'runtime-build',arcname='runtime-build')
    write(output/'index.json',dict(files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        model_metadata_read=False,all_originals_retained=True))
    print(json.dumps(dict(status=compact['status'],archive=str(output),index_sha256=sha(output/'index.json')),allow_nan=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();archive(a.output)
