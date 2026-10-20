"""Fixed unused mixed24 confirmation; preparation requires released ROOT selection.

No policy is selected from confirmation outcomes. No model identity is accessed
until the exact MAIN guard, held lock, and ROOT grant have been checked.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess

WORK=Path('/Volumes/Models/ds4/.workspaces/idea-resident-unused-mixed24')
SOURCE=WORK/'speed-bench/ideas/prepare_resident_mixed24_confirmation.py'
TEST=WORK/'tests/test_resident_mixed24_confirmation.py'
SPEC=WORK/'speed-bench/ideas/resident_mixed24_confirmation_spec.json'
SPEC_SHA='dfc1c57ed41dc975add4ea9c1430fb88f30e8c75accda482a166e432eef39704'
CAL_SOURCE=Path('/Volumes/Models/ds4/.workspaces/idea-resident-env-calibration-next/speed-bench/ideas/prepare_resident_env_calibration.py')
CAL_SHA='7184efeb16794d88f59f7f6fef5507d09c750189be708b8187d55895a5a0e8ff'
CAL=CAL_SOURCE.parent/'artifacts/resident-env-calibration-prepared-20261003-v2'
CAL_MANIFEST='1b537cb53a217d25fd13ee5957b405955a9a0ae63da9e372dde23092ccfe7fcd'
FREEZE=Path('/Volumes/Models/ds4/.workspaces/idea-resident-native-weights-next/speed-bench/ideas/artifacts/resident-unused-mixed24-input-freeze-20261003-v1/input-freeze.json')
FREEZE_SHA='3295e9bd9b5361c2cb870233f8038b6377e4dca91d778b467092cead2cdec7a4'
SETTINGS=dict(cases=24,tokens=32,rows=768,ctx=4096,cache_gb=64,chunk=512,tile=64,limit=1024,configured_slots=6136)
CSV_FIELDS='case step target prompt_tokens base_nll raw_nll policy_nll delta_raw_nll delta_policy_nll raw_kl policy_kl raw_tv policy_tv argmax_flip decode_ms'.split()
QUALITY_INTS=set('cases tokens argmax_flips bit_equal_rows'.split())
QUALITY_FLOATS=set('avg_nll delta_nll policy_delta_nll kl_base_to_raw kl_base_to_policy tv startup_ms prefill_ms decode_ms metrics_ms'.split())
SELECTION_FIELDS={'status','root_explicit_selection','calibration_manifest_sha256','calibration_driver_sha256','calibration_summary_sha256','calibration_decision_sha256','calibration_release_sha256','selected_policy','time'}


def require(value,message):
    if not value:raise ValueError(message)


def strict_json(text):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate JSON field');result[key]=value
        return result
    return json.loads(text,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def sha(path):
    path=Path(path);require(path.suffix.lower()!='.gguf','model hashing forbidden')
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def calibration_module():
    require(sha(CAL_SOURCE)==CAL_SHA,'frozen calibration source differs')
    spec=importlib.util.spec_from_file_location('mixed24_fixed_calibration',CAL_SOURCE)
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);return c


def frozen_spec():
    require(sha(SPEC)==SPEC_SHA,'pre-outcome confirmation specification changed')
    spec=strict_json(SPEC.read_text())
    require(spec['cases']==24 and spec['teacher_tokens_per_case']==32 and spec['full_vocabulary_rows']==768,'fixed geometry differs')
    return spec


def inputs():
    require(sha(FREEZE)==FREEZE_SHA,'unused mixed24 input authority changed')
    frozen=strict_json(FREEZE.read_text());cases=frozen['cases']
    require(len(cases)==24 and [c['case'] for c in cases]==frozen['ordered_cases'] and len(set(frozen['ordered_cases']))==24,'input order/coverage differs')
    counts={}
    for case in cases:counts[case['category']]=counts.get(case['category'],0)+1
    require(counts==frozen['category_counts']==dict(agent=4,tool=4,code=4,math=4,string=4,rare=4),'six fixed categories differ')
    # The 77 immutable input pins include this top input-freeze.json itself.
    require(len(frozen['files'])==76 and frozen['teacher_tokens_requested_later']==32 and frozen['expected_rows_later']==768 and frozen['model_or_tokenizer_opened'] is False,'input freeze contract differs')
    expected_lines=[]
    for case in cases:
        for kind in ('prompt','continuation','gold'):
            path=Path(case[kind]);require(path.parent==FREEZE.parent and frozen['files'][path.name]==case[kind+'_sha256'],'input path/hash binding differs')
        expected_lines.append('\t'.join([case['case'],case['prompt'],case['continuation']]))
    require((FREEZE.parent/'fixtures.tsv').read_text()==''.join(line+'\n' for line in expected_lines),'frozen TSV paths/order differ')
    return frozen


def selection_authority(path,c):
    """Check explicit ROOT authority BEFORE any current calibration outcome read."""
    proof=strict_json(Path(path).read_text())
    require(type(proof) is dict and set(proof)==SELECTION_FIELDS,'ROOT selection schema differs')
    require(proof['status']=='ROOT_FIXED_CALIBRATION_SELECTION' and proof['root_explicit_selection'] is True,'ROOT explicit frozen selection required')
    require(proof['calibration_manifest_sha256']==CAL_MANIFEST and proof['calibration_driver_sha256']==CAL_SHA,'selection calibration authority differs')
    require(type(proof['time']) in (int,float) and math.isfinite(proof['time']) and proof['time']>0,'selection time invalid')
    for field in ('calibration_summary_sha256','calibration_decision_sha256','calibration_release_sha256'):
        require(type(proof[field]) is str and re.fullmatch('[0-9a-f]{64}',proof[field]),'selection digest malformed')
    require(proof['selected_policy'] in [a['name'] for a in c.arms()[1:-1]],'selection policy outside fixed calibration list')
    require(sha(CAL/'manifest.json')==CAL_MANIFEST,'calibration manifest changed')
    require(sha(CAL/'root-release-proof.json')==proof['calibration_release_sha256'],'calibration release evidence changed')
    release=strict_json((CAL/'root-release-proof.json').read_text())
    require(release['guard_result']==dict(status='complete',child_returncode=0) and type(release['guard_result']['child_returncode']) is int and release['lock_reacquired'] is True and release['live_model_processes']==[] and type(release['live_model_processes']) is list,'calibration not successfully released')
    for field,name in [('calibration_summary_sha256','run/summary.json'),('calibration_decision_sha256','run/decision.json')]:require(sha(CAL/name)==proof[field],'released calibration evidence changed')
    summary=strict_json((CAL/'run/summary.json').read_text());decision=strict_json((CAL/'run/decision.json').read_text())
    require(summary['status']=='complete' and summary['first_attempt_only'] is True and summary['full_native_brackets_and_anchor_controls_exact'] is True,'calibration controls incomplete')
    require(decision==summary['decision']==c.decision(summary['arms']),'fixed calibration decision arithmetic differs')
    expected=decision['selected_calibration_policy'] if decision['advance'] else 'anchor-affinity3'
    require(proof['selected_policy']==expected,'ROOT selection must follow frozen advance decision')
    return proof


def arms(selected,c):
    fixed=c.arms();by_name={a['name']:a for a in fixed}
    require(selected in by_name and selected not in ('native-before','native-after'),'invalid selected policy')
    names=['native-before','anchor-affinity3']+([selected] if selected!='anchor-affinity3' else [])+['native-after']
    return [dict(by_name[name]) for name in names]


def validate_csv(path,metadata,cases,raw,case_count=24,token_limit=32):
    """Private explicit geometry; never monkeypatch qualified 16/512 validators."""
    count=case_count*token_limit
    require(len(cases)==case_count and len(metadata[1])==count,'CSV/raw/cases geometry differs')
    with Path(path).open() as f:
        reader=csv.DictReader(f);require(reader.fieldnames==CSV_FIELDS,'CSV schema differs');rows=list(reader)
    require(len(rows)==count,'CSV incomplete')
    for index,(row,key) in enumerate(zip(rows,metadata[1])):
        require(set(row)==set(CSV_FIELDS) and None not in row.values(),'CSV missing/extra columns')
        case,step,target,prompt_tokens,_=raw.RAW_ROW_META.unpack(key)
        require((case,step)==divmod(index,token_limit),'CSV/raw ordered case/step coverage differs')
        require(row['case']==cases[case]['case'] and all(row[k]==str(v) for k,v in [('step',step),('target',target),('prompt_tokens',prompt_tokens)]),'CSV/raw teacher keys differ')
        require(row['argmax_flip'] in ('0','1'),'invalid flip')
        require(all(math.isfinite(float(row[k])) for k in CSV_FIELDS[4:13]+['decode_ms']),'nonfinite CSV field')
        require(all(float(row[k])>=0 for k in ('base_nll','raw_nll','policy_nll','decode_ms')),'negative NLL/duration')
        require(all(-1e-12<=float(row[k])<=1+1e-12 for k in ('raw_tv','policy_tv')) and all(float(row[k])>=-1e-12 for k in ('raw_kl','policy_kl')),'invalid KL/TV')
        require(all(math.isclose(float(row[d]),float(row[n])-float(row['base_nll']),rel_tol=2e-9,abs_tol=2e-10) for d,n in [('delta_raw_nll','raw_nll'),('delta_policy_nll','policy_nll')]),'CSV NLL arithmetic differs')
    return rows


def validate_quality_json(text,rows,case_count=24,token_limit=32):
    result=strict_json(text);count=case_count*token_limit
    require(type(result) is dict and set(result)==QUALITY_INTS|QUALITY_FLOATS,'quality JSON schema differs')
    require(len(rows)==count and all(type(result[k]) is int and 0<=result[k]<=count for k in QUALITY_INTS) and result['cases']==case_count and result['tokens']==count,'quality JSON geometry/count differs')
    require(all(type(result[k]) in (int,float) and math.isfinite(result[k]) for k in QUALITY_FLOATS),'quality JSON finite fields invalid')
    require(all(result[k]>=0 for k in ('avg_nll','tv','startup_ms','prefill_ms','decode_ms','metrics_ms')) and all(result[k]>=-1e-12 for k in ('kl_base_to_raw','kl_base_to_policy')),'negative quality field')
    checks={'avg_nll':'raw_nll','delta_nll':'delta_raw_nll','policy_delta_nll':'delta_policy_nll','kl_base_to_raw':'raw_kl','kl_base_to_policy':'policy_kl','tv':'raw_tv'}
    require(all(math.isclose(result[k],statistics.mean(float(r[col]) for r in rows),rel_tol=2e-9,abs_tol=2e-10) for k,col in checks.items()),'quality JSON/CSV arithmetic differs')
    require(result['argmax_flips']==sum(int(r['argmax_flip']) for r in rows),'quality JSON flip total differs')
    return result


def paired(candidate,anchor,cases):
    keys=[case['case'] for case in cases]
    require(len(keys)==24 and list(candidate['per_prompt'])==list(anchor['per_prompt'])==keys,'paired ordered24 coverage differs')
    delta=[candidate['per_prompt'][k]['delta_nll']-anchor['per_prompt'][k]['delta_nll'] for k in keys]
    require(all(type(x) in (int,float) and math.isfinite(x) for x in delta),'paired values invalid')
    rng=random.Random(20261002);boot=sorted(statistics.mean(rng.choices(delta,k=24)) for _ in range(2000))
    return dict(mean=statistics.mean(delta),ci95=[boot[49],boot[1949]],paired_prompts=24,draws=2000,seed=20261002)


def confirmation(results,selected,cases,spec,c):
    require(list(results)==[a['name'] for a in arms(selected,c)],'all fixed first-attempt arms required')
    native=results['native-before'];anchor=results['anchor-affinity3'];screen=spec['screen'];reports={}
    require(type(native['io']['requested_bytes']) is int and native['io']['requested_bytes']>0,'native requested bytes invalid')
    for arm in arms(selected,c)[1:-1]:
        name=arm['name'];row=results[name];pair=paired(row,anchor,cases)
        limits=[('mean_delta_nll',row['delta_nll'],screen['mean_delta_nll_max']),('mean_kl',row['kl_base_to_raw'],screen['mean_kl_max']),('p99_kl',row['kl_p99'],screen['p99_kl_max']),('max_kl',row['kl_max'],screen['max_kl_max']),('argmax_flips',row['argmax_flips'],screen['argmax_flips_max']),('worst_prompt_delta_nll',row['worst_prompt_delta_nll'],screen['worst_prompt_delta_nll_max']),('paired_nll_ci_upper',pair['ci95'][1],screen['paired_candidate_minus_anchor_nll_ci_upper_max'])]
        require(all(type(value) in (int,float) and math.isfinite(value) for _,value,_ in limits) and type(row['argmax_flips']) is int and 0<=row['argmax_flips']<=768 and type(row['io']['requested_bytes']) is int and row['io']['requested_bytes']>=0,'screen typed/finite fields invalid')
        failures=[key for key,value,limit in limits if value>limit]
        if row['io']['requested_bytes']*5>native['io']['requested_bytes']*4:failures.append('native_requested_byte_reduction')
        categories={}
        for category in ('agent','tool','code','math','string','rare'):
            ids=[case['case'] for case in cases if case['category']==category];require(len(ids)==4,'category coverage differs')
            categories[category]=dict(prompts=4,tokens=128,mean_delta_nll=statistics.mean(row['per_prompt'][k]['delta_nll'] for k in ids),mean_kl=statistics.mean(row['per_prompt'][k]['kl'] for k in ids),paired_candidate_minus_anchor_mean=statistics.mean(row['per_prompt'][k]['delta_nll']-anchor['per_prompt'][k]['delta_nll'] for k in ids))
        reports[name]=dict(screen_pass=not failures,failures=failures,paired_candidate_minus_anchor_nll=pair,native_requested_byte_reduction=1-row['io']['requested_bytes']/native['io']['requested_bytes'],categories=categories)
    return dict(fixed_selected_policy=selected,selected_confirmation_pass=reports[selected]['screen_pass'],reports=reports,no_policy_selection_from_confirmation=True,no_automatic_models=True,timing_claim=False,scope=spec['scope'])


def cpu_proof(path):
    proof=strict_json(Path(path).read_text())
    require(proof['source_sha256']==sha(SOURCE) and proof['test_sha256']==sha(TEST) and proof['model_opened'] is False and type(proof['returncode']) is int and proof['returncode']==0 and type(proof['tests']) is int and proof['tests']>=18,'passing frozen CPU proof required')
    return Path(path)


def source_assets(c,s,qualplan,frozen,selection_path,proof):
    assets=c.reference_assets(s,qualplan)
    for name,digest in frozen['files'].items():assets[FREEZE.parent/name]=(name,digest)
    assets.update({FREEZE:('mixed24 freeze',FREEZE_SHA),SPEC:('prespecified confirmation',SPEC_SHA),CAL_SOURCE:('fixed calibration decision source',CAL_SHA),CAL/'manifest.json':('calibration manifest',CAL_MANIFEST),SOURCE:('confirmation source',sha(SOURCE)),TEST:('confirmation tests',sha(TEST)),Path(selection_path):('ROOT frozen selection',sha(selection_path)),Path(proof):('passing CPU proof',sha(proof))})
    selection=strict_json(Path(selection_path).read_text())
    for field,name in [('calibration_release_sha256','root-release-proof.json'),('calibration_summary_sha256','run/summary.json'),('calibration_decision_sha256','run/decision.json')]:assets[CAL/name]=(name,selection[field])
    return assets


def prepare(output,proof_path,selection_path):
    c=calibration_module();spec=frozen_spec();frozen=inputs();selection=selection_authority(selection_path,c);proof=cpu_proof(proof_path)
    s=c.module();qualplan,_=c.qualification(s);assets=source_assets(c,s,qualplan,frozen,selection_path,proof);s.raw_module().check_frozen_assets(assets)
    require(os.access(s.BUILD/'quality-new8a47',os.X_OK),'qualified helper lost execute permission')
    output.mkdir(parents=True,exist_ok=False)
    copies=[(SOURCE,'driver.py'),(TEST,'tests.py'),(SPEC,'spec.json'),(FREEZE,'input-freeze.json'),(Path(selection_path),'selection.json'),(proof,'cpu-proof.json')]
    for source,name in copies:shutil.copy2(source,output/name)
    plan=dict(source_sha256=sha(SOURCE),test_sha256=sha(TEST),spec_sha256=SPEC_SHA,input_freeze_sha256=FREEZE_SHA,selection=selection,selection_path=str(selection_path),selection_sha256=sha(selection_path),cpu_proof=dict(path=str(proof),sha256=sha(proof)),arms=arms(selection['selected_policy'],c),settings=SETTINGS,screen=spec['screen'],statistics=spec['statistics'],model_from_prior=qualplan['model_from_prior'],fixed_fixture=dict(fixtures=str(FREEZE.parent/'fixtures.tsv'),inputs=frozen['cases']),shader_env=c.shader_environment(qualplan),command_template=s.command_template(output),files={name:sha(output/name) for _,name in copies},external_files={str(path):value[1] for path,value in assets.items()},first_attempt_only=True,root_model_grant_required=True,model_stat_or_open_in_preparation=False,no_automatic_models=True,timing_claim=False,scope=spec['scope'])
    write(output/'manifest.json',plan);digest=sha(output/'manifest.json');write(output/'prepared-command.json',dict(command=s.guard_command(output,digest),manifest_sha256=digest,launch_requires_root_grant=True,automatic_holdout_run=False))
    print(json.dumps(dict(prepared=str(output),manifest_sha256=digest,command_sha256=sha(output/'prepared-command.json'),arms=len(plan['arms']),model_opened=False)),flush=True)


def verified(output,digest,c,s,qualplan):
    require(type(digest) is str and re.fullmatch('[0-9a-f]{64}',digest) and sha(output/'manifest.json')==digest,'explicit manifest authority differs')
    plan=strict_json((output/'manifest.json').read_text());cmd=strict_json((output/'prepared-command.json').read_text());spec=frozen_spec();frozen=inputs()
    require(cmd==dict(command=s.guard_command(output,digest),manifest_sha256=digest,launch_requires_root_grant=True,automatic_holdout_run=False),'exact prepared command differs')
    selection=selection_authority(plan['selection_path'],c);proof=cpu_proof(plan['cpu_proof']['path'])
    require(plan['selection']==selection and plan['selection_sha256']==sha(plan['selection_path']) and plan['cpu_proof']['sha256']==sha(proof),'selection/CPU authority changed')
    require(plan['arms']==arms(selection['selected_policy'],c) and plan['settings']==SETTINGS and plan['screen']==spec['screen'] and plan['statistics']==spec['statistics'],'frozen arms/geometry/screens differ')
    require(plan['source_sha256']==sha(SOURCE)==sha(__file__) and plan['test_sha256']==sha(TEST) and plan['spec_sha256']==SPEC_SHA and plan['input_freeze_sha256']==FREEZE_SHA,'source/spec/input authority changed')
    require(plan['command_template']==s.command_template(output) and plan['model_from_prior']==qualplan['model_from_prior'] and plan['fixed_fixture']==dict(fixtures=str(FREEZE.parent/'fixtures.tsv'),inputs=frozen['cases']) and plan['shader_env']==c.shader_environment(qualplan),'model/input/guard/shader authority differs')
    require(all(plan[k] is True for k in ('first_attempt_only','root_model_grant_required','no_automatic_models')) and plan['model_stat_or_open_in_preparation'] is False and plan['timing_claim'] is False and plan['scope']==spec['scope'],'fixed execution scope differs')
    external=source_assets(c,s,qualplan,frozen,plan['selection_path'],proof)
    required={'driver.py':sha(SOURCE),'tests.py':sha(TEST),'spec.json':SPEC_SHA,'input-freeze.json':FREEZE_SHA,'selection.json':sha(plan['selection_path']),'cpu-proof.json':sha(proof)}
    require(plan['files']==required and plan['external_files']=={str(path):value[1] for path,value in external.items()},'mandatory inventory differs')
    external.update({output/name:(name,digest) for name,digest in required.items()});external[output/'manifest.json']=('explicit manifest',digest);external[output/'prepared-command.json']=('exact command',sha(output/'prepared-command.json'))
    return plan,external


def execute(output,digest):
    c=calibration_module();s=c.module();qualplan,_=c.qualification(s);plan,assets=verified(output,digest,c,s,qualplan)
    s.require_main_guard(output,digest);c.require_root_grant(output,digest,s)
    raw=s.raw_module();raw.check_frozen_assets(assets)
    model=Path(plan['model_from_prior']['path']);identity=raw.identity(model);require(identity==plan['model_from_prior'],'guarded model identity differs')
    run=output/'run';run.mkdir(exist_ok=False);write(run/'model-identity.json',dict(identity=identity,guard_verified_before_stat=True))
    env={k:v for k,v in os.environ.items() if not k.startswith('DS4')};env.update(plan['shader_env'])
    results={};metadata=None;ref=run/'native-before.raw';ref_sha=None
    for arm in plan['arms']:
        s.require_main_guard(output,digest);c.require_root_grant(output,digest,s);raw.check_frozen_assets(assets);require(raw.identity(model)==identity,'model identity changed')
        name=arm['name'];record=run/(name+'.raw');scores=run/(name+'.csv');recording=name=='native-before';childenv=dict(env,**c.policy(arm))
        argv=[str(s.BUILD/'quality-new8a47'),str(model),plan['fixed_fixture']['fixtures'],str(scores),str(ref),'record' if recording else 'compare','--cases','24','--tokens','32','--ctx','4096','--cache-gb','64','--chunk','512']
        if not recording:argv+=['--candidate-record',str(record)]
        write(run/(name+'-command.json'),dict(argv=argv,DS4_environment={k:v for k,v in childenv.items() if k.startswith('DS4')},first_attempt=True))
        with (run/(name+'.stdout')).open('x') as out,(run/(name+'.log')).open('x') as err:completed=subprocess.run(argv,cwd=s.ROOT,env=childenv,stdout=out,stderr=err)
        write(run/(name+'-exit.json'),dict(returncode=completed.returncode));require(completed.returncode==0,name+' first-attempt helper failure')
        raw.check_frozen_assets(assets);require(raw.identity(model)==identity,'model changed during arm')
        if recording:
            metadata=raw.raw_record_metadata(ref,24,32,768);ref_sha=sha(ref);assets[ref]=('mixed24 first native reference',ref_sha)
            write(run/'native-reference-proof.json',dict(sha256=ref_sha,metadata=metadata[2],created_before_candidate=True))
        text=(run/(name+'.log')).read_text();layout=raw.validate_candidate_record(record,metadata,24,32,768);finite=s.validate_raw_finite(record,metadata,raw);csv_sha=sha(scores);rows=validate_csv(scores,metadata,plan['fixed_fixture']['inputs'],raw)
        row=validate_quality_json((run/(name+'.stdout')).read_text(),rows);row.update(name=name,raw_sha256=sha(record),csv_sha256=csv_sha,layout=layout,finite_validation=finite,cache=s.parse_cache(text),io=s.parse_io(text),route=s.parse_route(text,arm,768*40),**raw.summarize(scores));require(sha(scores)==csv_sha,'CSV changed during checked arithmetic')
        require(list(row['per_prompt'])==[case['case'] for case in plan['fixed_fixture']['inputs']] and all(p['tokens']==32 for p in row['per_prompt'].values()),'ordered24 prompt completeness differs')
        native=results.get('native-before',row)
        if arm['mode']=='native':require(row['raw_sha256']==ref_sha and row['bit_equal_rows']==768 and all(row[k]==0 for k in ('delta_nll','kl_base_to_raw','tv','argmax_flips')) and row['io']==native['io'] and row['cache']==native['cache'],'native fullraw/teacher/IO/cache bracket control differs')
        if arm['mode']=='apply':
            require(int(arm['subs'])>=2 or (row['route']['double_layers']==0 and row['route']['triple_layers']==0),'sub1 multiple slots committed')
            require(int(arm['subs'])>=3 or row['route']['triple_layers']==0,'sub2 triple slot committed')
        results[name]=row;write(run/(name+'-checked.json'),row)
        print(json.dumps(dict(name=name,checked=True,raw_sha256=row['raw_sha256'],delta_nll=row['delta_nll'],kl=row['kl_base_to_raw'],requested_bytes=row['io']['requested_bytes']),allow_nan=False),flush=True)
    report=confirmation(results,plan['selection']['selected_policy'],plan['fixed_fixture']['inputs'],frozen_spec(),c)
    write(run/'confirmation.json',report);write(run/'summary.json',dict(status='complete',arms=results,confirmation=report,model_identity=identity,first_attempt_only=True,full_native_brackets_exact=True,no_confirmation_policy_selection=True,timing_claim=False,scope=plan['scope']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',type=Path);p.add_argument('--cpu-proof',type=Path);p.add_argument('--selection',type=Path);p.add_argument('--run',type=Path);p.add_argument('--manifest-sha256');a=p.parse_args()
    if bool(a.prepare)==bool(a.run):p.error('choose prepare or run')
    if a.prepare and (not a.cpu_proof or not a.selection):p.error('passing CPU proof and explicit released ROOT selection required')
    if a.run and not a.manifest_sha256:p.error('explicit manifest SHA required')
    if a.prepare:prepare(a.prepare.absolute(),a.cpu_proof.absolute(),a.selection.absolute())
    else:execute(a.run.absolute(),a.manifest_sha256)
