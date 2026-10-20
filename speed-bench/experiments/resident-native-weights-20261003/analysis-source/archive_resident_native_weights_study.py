"""CPU-only archive of the ROOT-released nineteen-arm first-attempt study."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import stat

WORK=Path('/Volumes/Models/ds4/.workspaces/idea-resident-native-weights-next')
PREP=WORK/'speed-bench/ideas/artifacts/resident-native-weights-study-prepared-v2'
SOURCE=WORK/'speed-bench/ideas/prepare_resident_native_weights_study.py'
SOURCE_SHA='5e0b20bd28b39ac94eed5e40a73aac20a7552628d6c3572d71ec287453e8505d'
MANIFEST_SHA='6199c1d5b5f2aa71a13c915a16c8b4b16e198207966b79a50fb83b6753c406e1'
COMMAND_SHA='dc930fe3d9c601cbfcf0e119eb9f9a50b320f5df166da54fb142db059128cae1'
SOURCE_REVISION='6d970cb0374733f109ef4e210861ab5248f1f7b9'


def require(condition,message):
    if not condition:raise ValueError(message)


def released(release):
    require(type(release) is dict,'release schema')
    require(release.get('guard_result')==dict(status='complete',child_returncode=0) and type(release['guard_result'].get('child_returncode')) is int,'original guard incomplete/failed')
    require(release.get('lock_reacquired') is True and type(release.get('live_model_processes')) is list and release['live_model_processes']==[],'ROOT release not proven')
    for key in ('normal_pressure_samples','bad_pressure_samples'):
        require(type(release.get(key)) is int and release[key]>=0,'pressure sample type')
    require(release['normal_pressure_samples']>0 and release['bad_pressure_samples']==0,'pressure not healthy')
    for key in ('time','swap_growth_gb','peak_wired_compressor_gib'):
        require(type(release.get(key)) in (int,float) and math.isfinite(release[key]) and release[key]>=0,'pressure/time type')
    require(release['swap_growth_gb']==0,'swap growth')


def physical(root):
    files={};dirs=[]
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),'symlink in archive/source inventory')
        name=str(p.relative_to(root))
        if p.is_dir():dirs.append(name)
        elif p.is_file():files[name]=p
        else:raise ValueError('special file in evidence inventory')
    return files,dirs


def paired(a,b):
    require(list(a['per_prompt'])==list(b['per_prompt']) and len(a['per_prompt'])==16,'paired case coverage')
    values=[b['per_prompt'][case]['delta_nll']-a['per_prompt'][case]['delta_nll'] for case in a['per_prompt']]
    rng=random.Random(20261002)
    boots=sorted(statistics.mean(rng.choices(values,k=16)) for _ in range(2000))
    return dict(mean_delta_nll_difference=statistics.mean(values),prompt_bootstrap_95=[boots[49],boots[1949]],seed=20261002,draws=2000,paired_prompts=16)


def archive(output):
    spec=importlib.util.spec_from_file_location('study',SOURCE);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    require(s.sha(SOURCE)==SOURCE_SHA,'frozen source differs')
    release=s.strict_json((PREP/'root-release-proof.json').read_text());released(release)
    launch=s.strict_json((PREP/'root-launch-proof.json').read_text())
    require(launch['source_revision']==SOURCE_REVISION and launch['manifest_sha256']==MANIFEST_SHA and launch['command_sha256']==COMMAND_SHA and launch['driver_sha256']==SOURCE_SHA and launch['status']=='ROOT_EXPLICIT_MODEL_GRANT' and launch['arms']==19,'ROOT launch authority differs')
    plan=s.verified_plan(PREP,MANIFEST_SHA);require(s.sha(PREP/'prepared-command.json')==COMMAND_SHA,'command pin differs')
    raw=s.raw_module(PREP/'raw_helpers.py');assets=s.assets_and_inventory(PREP,plan);raw.check_frozen_assets(assets)
    run=PREP/'run';summary=s.strict_json((run/'summary.json').read_text());guard=s.strict_json((PREP/'pressure/result.json').read_text())
    require(guard==dict(status='complete',child_returncode=0) and type(guard['child_returncode']) is int,'original pressure result failed')
    require(summary.keys()==set('status arms model_identity guarded full_default_state_raw_teacher_io_controls positive_flag1_noaction_controls first_attempt_only timing_claim io_scope'.split()),'typed final summary schema')
    require(summary['status']=='complete' and summary['model_identity']==plan['model_from_prior'] and all(summary[k] is True for k in ('guarded','full_default_state_raw_teacher_io_controls','positive_flag1_noaction_controls','first_attempt_only')) and summary['timing_claim'] is False and summary['io_scope']==plan['scope'],'typed final summary status')
    require(list(summary['arms'])==[a['name'] for a in s.arm_specs()],'final ordered arm coverage')
    results={};meta={};refs={};large={}
    for arm in s.arm_specs():
        name=arm['name'];exit=s.strict_json((run/(name+'-exit.json')).read_text())
        require(exit==dict(returncode=0) and type(exit['returncode']) is int,'original first-attempt helper exit failed')
        text=(run/(name+'.log')).read_text();cache=s.parse_cache(text,arm['kind']=='state')
        saved=s.strict_json((run/(name+'-checked.json')).read_text())
        if arm['kind']=='state':
            logits=run/(name+'.logits');snapshot=run/(name+'.snapshot')
            s.validate_state_csv(run/(name+'.stdout'),plan['state_fixture']['phases'])
            state=s.validate_state_artifacts(logits,snapshot,plan['state_fixture']['fixture'],plan['state_fixture']['phases'])
            row=dict(name=name,logits_sha256=s.sha(logits),snapshot_sha256=s.sha(snapshot),state_validation=state,cache=cache,route=s.parse_route(text,arm,343*40))
            large[str(logits)]=row['logits_sha256'];large[str(snapshot)]=row['snapshot_sha256']
            if arm['binary']=='new8a47':
                old=results['state-old-native' if arm['mode']=='native' else 'state-old-approx0']
                require(all(row[k]==old[k] for k in ('logits_sha256','snapshot_sha256','cache','route')),'state old/new control')
        else:
            group=arm['group'];ref=run/(group+'-reference.raw');recording=name==group+'-old-native';record=ref if recording else run/(name+'.raw')
            if recording:meta[group]=raw.raw_record_metadata(ref,16,32,512);refs[group]=s.sha(ref)
            layout=raw.validate_candidate_record(record,meta[group],16,32,512);finite=s.validate_raw_finite(record,meta[group],raw)
            rows=s.validate_csv(run/(name+'.csv'),meta[group],plan['groups'][group]['inputs'],raw)
            row=s.validate_quality_json((run/(name+'.stdout')).read_text(),rows)
            row.update(name=name,raw_sha256=s.sha(record),layout=layout,finite_validation=finite,cache=cache,io=s.parse_io(text),route=s.parse_route(text,arm,512*40),**raw.summarize(run/(name+'.csv')))
            large[str(record)]=row['raw_sha256']
            native=results.get(group+'-old-native',row)
            if arm['mode'] in ('native','lookup','noop'):
                require(row['raw_sha256']==refs[group] and row['bit_equal_rows']==512 and all(row[k]==0 for k in ('delta_nll','kl_base_to_raw','tv','argmax_flips')) and row['io']==native['io'] and row['cache']==native['cache'],'native/no-action raw/IO/cache control')
            if name==group+'-new-approx0':
                old=results[group+'-old-approx0'];require(all(row[k]==old[k] for k in ('raw_sha256','io','cache','route')),'flag0 approximation control')
        require(row==saved and row==summary['arms'][name],'independently reconstructed checked/final row differs')
        results[name]=row
    # Every historical guard/source/CPU failure remains first-attempt evidence.
    prior=WORK/'speed-bench/experiments/ds41-ssd/resident-native-weights-next-evidence'
    output.mkdir(parents=True,exist_ok=False)
    retained={}
    def copy_tree(source,destination,skip_large=False):
        files,_=physical(source)
        for name,p in files.items():
            target=destination/name
            if skip_large and p.suffix in ('.raw','.logits','.snapshot'):
                digest=s.sha(p);require(large.get(str(p))==digest,'unexpected/unverified large model output')
                retained[str(p)]=dict(bytes=p.stat().st_size,sha256=digest,mode_octal=oct(stat.S_IMODE(p.stat().st_mode)),streaming_sha_verified=True,reason='full original array retained in released run; independent streaming SHA and FP32/layout validation passed')
                continue
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    copy_tree(PREP,output/'prepared-and-run',True)
    for name in ('f2-first-failures','cpu-pass-v3','build-plan-v3','build-plan-v5','build-pass-v5','study-prepare-v1','study-prepared-v2'):
        copy_tree(prior/name,output/'prior-evidence'/name)
    # Copy immutable recursive C/header snapshots; keep built objects/binaries
    # at their hash-pinned durable locations rather than copying them again.
    build_plan=s.strict_json((s.BUILD/'build-plan.json').read_text())
    for kind in ('old7c6e','new8a47'):
        snapshot=WORK/'speed-bench/ideas/artifacts/resident-native-weights-next-build-plan-v3/source'/kind
        copy_tree(snapshot,output/'runtime-source'/kind)
    for p in (Path(__file__),SOURCE,WORK/'tests/test_resident_native_weights_study.py'):
        destination=output/'analysis-source'/p.name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,destination)
    for path,value in assets.items():
        if path.suffix in ('.o',) or path.name.startswith(('quality-','replay-')):
            require(s.sha(path)==value[1],'retained runtime asset differs');retained[str(path)]=dict(bytes=path.stat().st_size,sha256=value[1],mode_octal=oct(stat.S_IMODE(path.stat().st_mode)),streaming_sha_verified=True,reason='frozen linked runtime/helper retained with full build proof')
    comparisons={}
    keys='delta_nll prompt_bootstrap_95 worst_prompt_delta_nll kl_base_to_raw kl_p99 kl_max argmax_flips bit_equal_rows raw_sha256 io route'.split()
    for group in ('calibration16','heldout16'):
        a=results[group+'-old-approx0'];b=results[group+'-native-weights1'];n=results[group+'-old-native']
        comparisons[group]=dict(dataset_role='calibration' if group=='calibration16' else 'previously used confirmation16; not an untouched new holdout',affinity={k:a[k] for k in keys},original_gpu_weights={k:b[k] for k in keys},paired_weight_minus_affinity=paired(a,b),native_requested_bytes=n['io']['requested_bytes'],candidate_minus_affinity_requested_bytes=b['io']['requested_bytes']-a['io']['requested_bytes'],candidate_relative_native_requested_pct=(b['io']['requested_bytes']/n['io']['requested_bytes']-1)*100)
    compact=dict(status='complete',first_attempts=19,all_helper_returncodes=0,source_revision=SOURCE_REVISION,manifest_sha256=MANIFEST_SHA,command_sha256=COMMAND_SHA,all_old_new_default_state_and_teacher_controls_exact=True,all_positive_flag1_noaction_fullraw_io_cache_controls_exact=True,all_native_brackets_exact=True,all_raw_vectors_independently_finite=True,comparisons=comparisons,release=release,logical_io_scope='Direct cumulative explicit cache-loader requested/returned bytes through preclose P+D, including the final unscored teacher eval. No background stage enabled. Teardown source audit shows no new submissions; postclose counters were not measured. No physical SSD or speed claim.',coefficient_scope='Flag1 successful ID-only commit counts are measured. Actual GPU six-slot coefficient tensor bits are not exported by this frontend; preservation follows source isolation plus focused actual-function CPU bit tests.',conclusion='Mixed quality; mean delta NLL improves uncertainly, KL worsens on both datasets, tail/argmax changes mixed; demand logical bytes essentially unchanged from affinity. Remains opt-in; no adoption or latency claim.',prior_mixed_native_weights_evidence='/Volumes/Models/ds4/speed-bench/ideas/evidence/resident-native-weights-calibration16-20261002/compact-summary.json')
    s.write(output/'compact-summary.json',compact);s.write(output/'retained-large-files.json',retained)
    # Preserve the complete physical inventory, including interpreter caches
    # and hidden producer files, even if repository-wide ignores match them.
    (output/'.gitignore').write_text('!**\n')
    files,dirs=physical(output)
    inventory=dict(physical_files_including_hidden={name:dict(bytes=p.stat().st_size,sha256=s.sha(p),mode_octal=oct(stat.S_IMODE(p.stat().st_mode))) for name,p in files.items()},physical_directories_including_hidden=dirs,self_excluded='archive-manifest.json',expected_tracked_files=sorted(files.keys()|{'archive-manifest.json'}))
    s.write(output/'archive-manifest.json',inventory)
    actual,actual_dirs=physical(output)
    require(set(actual)==set(inventory['expected_tracked_files']) and actual_dirs==dirs,'complete physical inventory differs')
    print(json.dumps(dict(archive=str(output),files=len(actual),directories=len(dirs),retained_large=len(retained),archive_manifest_sha256=s.sha(output/'archive-manifest.json'),compact_summary_sha256=s.sha(output/'compact-summary.json'),models_opened=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args();archive(a.output.resolve())
