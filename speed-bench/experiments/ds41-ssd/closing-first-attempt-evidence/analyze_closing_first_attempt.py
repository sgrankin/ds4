#!/usr/bin/env python3
"""Strict CPU-only analysis/archive of the frozen first closing-token pilot."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

PREPARED=Path('/Volumes/Models/ds4/.workspaces/idea-quant-closing/speed-bench/ideas/artifacts/closing-prepared-20261002-v1')
PLAN_SHA='cca9ffbc4e8f5d77c77f7b2c68f622dc144cbd54d426adc80b854418ba40ed65'
COMMAND_SHA='c4634fe3b5005492bf6d2b14193ee559ba20258332d334f717fb980246831998'
LAUNCHER_SHA='37a57726f2030653d1018ef2a78a52b18e7c67e84016b0762be6282fcc5c4f78'
ANALYSIS_SHA='fd559a7c66d0d46e0e1196b197f0489e7d78bc45963d506051c676667c57d486'
BINARY_SHA='847ca79f2d850901d4d553a99535fdd5ccf1783ae10322585c0551e639e5be39'
ARMS=[('native-before',0),('closing-add2',2),('native-after',0)]
DISTRIBUTION=('Same-prefix full-vocabulary pre-filter softmax at actual Tfloat '
              '0.6000000238418579 and actual rounded FP32 ADD delta to ID128822. '
              'Not post-top-p probabilities, T1, corpus NLL or cross-trajectory KL.')

def require(test,message):
    if not test:raise ValueError(message)

def sha(path):
    path=Path(path);require(path.suffix.lower()!='.gguf','No model hashing/access')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as target:json.dump(value,target,indent=2,allow_nan=False);target.write('\n')

def load_frozen():
    for name,expected in [('plan.json',PLAN_SHA),('prepared-command.json',COMMAND_SHA),
                          ('launch_closing.py',LAUNCHER_SHA),('summarize_closing.py',ANALYSIS_SHA),('closing-pilot',BINARY_SHA)]:
        require(sha(PREPARED/name)==expected,'Frozen '+name+' differs')
    plan=json.loads((PREPARED/'plan.json').read_text())
    require([(a['name'],a['boost']) for a in plan['arms']]==ARMS,'Frozen arm order/policy differs')
    require(plan['settings']==dict(profile='closing',direction='ADD',boost=2,after=1024,token_id=128822,
            exact_piece_hex='3c2f7468696e6b3e',think_level=25,temperature_float=0.6000000238418579,
            seed=20261002,cap=2048,cases=4),'Frozen study settings differ')
    require(len(plan['asset_sha256'])==194,'Frozen mandatory inventory differs')
    for path,expected in plan['asset_sha256'].items():require(sha(path)==expected,'Frozen asset differs: '+path)
    spec=importlib.util.spec_from_file_location('closing_validated_analysis',PREPARED/'summarize_closing.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return plan,module

def validate_release(release,pressure_result):
    require(pressure_result.get('status')=='complete' and type(pressure_result.get('child_returncode')) is int
            and pressure_result['child_returncode'] in (0,3),'First guarded producer incomplete/infrastructure failure')
    guard=release.get('guard_result')
    require(type(guard) is dict and guard.get('status')=='complete' and
            type(guard.get('child_returncode')) is int and guard['child_returncode']==pressure_result['child_returncode'],
            'Original guard returncode differs')
    require(release.get('lock_reacquired') is True and release.get('live_model_processes')==[],
            'ROOT release proof incomplete')
    require(type(release.get('time')) in (int,float) and math.isfinite(release['time']) and release['time']>0 and
            type(release.get('normal_pressure_samples')) is int and release['normal_pressure_samples']>0 and
            type(release.get('bad_pressure_samples')) is int and release['bad_pressure_samples']==0 and
            type(release.get('swap_growth_gb')) in (int,float) and math.isfinite(release['swap_growth_gb']) and release['swap_growth_gb']==0 and
            type(release.get('peak_wired_compressor_gib')) in (int,float) and math.isfinite(release['peak_wired_compressor_gib']) and release['peak_wired_compressor_gib']>0,
            'ROOT release pressure/time proof incomplete')

def inspect_arm(analysis,arm,fixture,reference,rc):
    report,rows,infos=analysis.validate_arm(Path(arm['results_dir']),arm['boost'],fixture,rc)
    parity=analysis.check_reference(Path(arm['results_dir']),rows,infos,reference,arm['boost']>0)
    old=analysis.jsonl(Path(reference)/'results.jsonl')
    cases=[]
    for row,info,baseline in zip(rows,infos,old):
        oldtrace=analysis.jsonl(analysis.local(reference,baseline['diagnostics']['token_trace']))
        same_tokens=len(oldtrace)==len(info['trace']) and all(all(a[k]==b[k] for k in analysis.TOKEN_FIELDS)
                          for a,b in zip(info['trace'],oldtrace))
        same_result=all(row[k]==baseline[k] for k in analysis.PARITY_FIELDS)
        samples=[t for t in info['trace'] if 'raw_marker_mass_pre_filter' in t]
        active_samples=[t for t in samples if t['policy_active']]
        cases.append(dict(case=row['case'],status=row['status'],cap_hit=row['cap_hit'],stopped=row['stopped'],
            think_closed=row['think_closed'],correct_exact=row['correct_exact'],
            completed_correct=row['status']=='CORRECT' and not row['cap_hit'] and row['stopped'] and row['think_closed'],
            generated_tokens=row['generated_tokens'],generated_token_difference_vs_native25=row['generated_tokens']-baseline['generated_tokens'],
            reasoning_tokens=row['reasoning_tokens'],active_steps=info['active_steps'],
            token_rng_piece_status_parity_vs_native25=same_tokens and same_result,
            trajectory_changed=not (same_tokens and same_result),prefilter_samples=len(samples),active_prefilter_samples=len(active_samples),
            mean_kl_raw_to_policy_prefilter=sum(t['kl_raw_to_policy_pre_filter'] for t in samples)/len(samples) if samples else 0.,
            mean_active_kl_raw_to_policy_prefilter=sum(t['kl_raw_to_policy_pre_filter'] for t in active_samples)/len(active_samples) if active_samples else 0.,
            selected_delta_nll_sum_prefilter=sum(t['selected_delta_nll_pre_filter'] for t in samples),
            selected_active_closing_tokens=sum(t['selected_marker'] and t['policy_active'] for t in info['trace']),
            full_token_trace_sha256=info['token_trace_sha256'],raw_output_sha256=info['raw_trace_sha256']))
    require(cases[0]['case']=='circular10' and cases[0]['active_steps']==0 and
            cases[0]['token_rng_piece_status_parity_vs_native25'],'Circular inactive native control differs')
    result=dict(report,arm=arm['name'],boost=arm['boost'],original_producer_returncode=rc,
                old25_reference_control_pass=parity,changed_cases=sum(c['trajectory_changed'] for c in cases),
                per_case=cases,distribution=DISTRIBUTION)
    return result,rows,infos

def analyze_completed(plan,analysis):
    output=Path(plan['output']);pressure=Path(plan['pressure'])
    release=json.loads((output/'root-release-proof.json').read_text())
    pressure_result=json.loads((pressure/'result.json').read_text());validate_release(release,pressure_result)
    producer=json.loads((output/'summary.json').read_text())
    require(producer.get('status')=='COMPLETE' and producer.get('all_first_attempts_retained') is True,
            'Original first-attempt aggregate incomplete')
    require([r['arm'] for r in producer['arms']]==[n for n,b in ARMS],'Original reports missing/reordered')
    fixture=json.loads(Path(plan['fixture']).read_text());reports=[]
    for arm,recorded in zip(plan['arms'],producer['arms']):
        process=json.loads((output/(arm['name']+'-process.json')).read_text())
        command=json.loads((output/(arm['name']+'-command.json')).read_text())
        require(command['argv']==arm['argv'] and command['cwd']==plan['run_cwd'] and
                command['ds4_environment']==plan['environment'],'Recorded first-attempt command differs')
        report,rows,infos=inspect_arm(analysis,arm,fixture,Path(plan['reference']),process['returncode'])
        for key in ['cases','correct','completed_correct','capped_correct','incorrect','invalid','cap_hits','tokens','reasoning_tokens','active_steps','producer_returncode']:
            require(type(recorded[key]) is type(report[key]) and recorded[key]==report[key],'Original typed report differs: '+key)
        reports.append(report)
    want_rc=3 if any(r['invalid'] for r in reports) else 0
    require(pressure_result['child_returncode']==want_rc,'Original aggregate INVALID-only rc differs')
    return dict(status='FIRST_ATTEMPT_CLOSING_INTEGRITY_PASS',arms=reports,original_guard_returncode=want_rc,
                all_original_attempts_retained=True,original_statuses_rewritten=False,root_release_proof=release,
                distribution=DISTRIBUTION,model_opened=False,model_metadata_read=False,
                interpretation='Four related counting fixtures; caps/wrong/invalid attempts retained. No quantization causality, speed, forced closure or default promotion claim.')

def prepare(output):
    plan,module=load_frozen()
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),output/'analyze_closing_first_attempt.py')
    shutil.copy2(PREPARED/'summarize_closing.py',output/'validated_summarize_closing.py')
    write(output/'cpu-frozen-plan.json',dict(original_plan_sha256=PLAN_SHA,prepared_command_sha256=COMMAND_SHA,
        launcher_sha256=LAUNCHER_SHA,validated_analysis_sha256=ANALYSIS_SHA,binary_sha256=BINARY_SHA,
        source_sha256=sha(Path(__file__)),original_prepared=str(PREPARED),mandatory_assets=194,
        exact_arms=ARMS,distribution=DISTRIBUTION,live_output_read=False,model_opened=False,model_metadata_read=False))

def archive(output):
    plan,module=load_frozen();report=analyze_completed(plan,module)
    output.mkdir(parents=True,exist_ok=False)
    sources={Path(path):expected for path,expected in plan['asset_sha256'].items()}
    for name in ['plan.json','prepared-command.json','launch_closing.py','summarize_closing.py']:
        sources[PREPARED/name]=sha(PREPARED/name)
    live=Path(plan['output']);pressure=Path(plan['pressure'])
    for path in [*live.iterdir(),*pressure.iterdir()]:
        if path.is_file():sources[path]=sha(path)
    retained=[]
    for arm in plan['arms']:
        for path in sorted(Path(arm['results_dir']).iterdir()):
            if path.is_file():sources[path]=sha(path)
    for path,expected in sources.items():
        require(sha(path)==expected,'Original/source changed before archive')
        retained.append(dict(path=str(path),sha256=expected,bytes=path.stat().st_size))
        if path.suffix=='.o' or path.name in ('closing-pilot','overthink_pilot'):continue
        if live in path.parents:relative=Path('original-run')/path.relative_to(live)
        elif pressure in path.parents:relative=Path('pressure')/path.relative_to(pressure)
        elif PREPARED in path.parents:relative=Path('original-preparation')/path.relative_to(PREPARED)
        elif any(Path(a['results_dir']) in path.parents for a in plan['arms']):
            a=next(a for a in plan['arms'] if Path(a['results_dir']) in path.parents);relative=Path(a['name'])/path.name
        else:continue
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);require(sha(target)==expected,'Archive copy differs')
    shutil.copy2(Path(__file__),output/'analyze_closing_first_attempt.py')
    write(output/'compact.json',report);write(output/'retained-source-pins.json',retained)
    for path,expected in sources.items():require(sha(path)==expected,'Original/source changed through archive')
    write(output/'index.json',dict(files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
                                  original_sources_unchanged=True,all_large_originals_retained=True,model_metadata_read=False))
    print(json.dumps(dict(status=report['status'],guard_rc=report['original_guard_returncode'],archive=str(output),
                         index_sha256=sha(output/'index.json')),allow_nan=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__);choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare',type=Path);choice.add_argument('--archive',type=Path)
    args=parser.parse_args()
    if args.prepare:prepare(args.prepare)
    else:archive(args.archive)

if __name__=='__main__':main()
