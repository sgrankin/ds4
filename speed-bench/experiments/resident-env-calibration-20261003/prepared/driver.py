"""Freeze eight guarded calibration arms against the qualified sixteen cases."""
import argparse
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

WORK=Path('/Volumes/Models/ds4/.workspaces/idea-resident-env-calibration-next')
QUAL_WORK=Path('/Volumes/Models/ds4/.workspaces/idea-resident-native-weights-next')
QUAL=QUAL_WORK/'speed-bench/ideas/artifacts/resident-native-weights-study-prepared-v2'
QUAL_SOURCE=QUAL_WORK/'speed-bench/ideas/prepare_resident_native_weights_study.py'
QUAL_SOURCE_SHA='5e0b20bd28b39ac94eed5e40a73aac20a7552628d6c3572d71ec287453e8505d'
QUAL_MANIFEST_SHA='6199c1d5b5f2aa71a13c915a16c8b4b16e198207966b79a50fb83b6753c406e1'
QUAL_COMMAND_SHA='dc930fe3d9c601cbfcf0e119eb9f9a50b320f5df166da54fb142db059128cae1'
TEST_SOURCE=WORK/'tests/test_resident_env_calibration.py'
SHADER_FILES=('flash_attn.metal','dense.metal','glm53_bf16.metal','glm53_vision.metal',
 'deepseek4_vision.metal','glm53_kda.metal','moe.metal','dsv4_hc.metal','unary.metal',
 'dsv4_kv.metal','dsv41.metal','dsv4_rope.metal','dsv4_misc.metal','argsort.metal',
 'cpy.metal','concat.metal','get_rows.metal','sum_rows.metal','softmax.metal','repeat.metal',
 'glu.metal','norm.metal','bin.metal','set_rows.metal','qwen4.metal','qwen4_vision.metal')
QUAL_PINS={
 'root-release-proof.json':'a2fc29cebc7f593b6436c44471b6b749412111f54d9c6204f19ba2134fa9314c',
 'run/summary.json':'c143efdd78fc352a3a6a6e0b76fac82f571fa56c58355103bd08be6d4b98ea64',
 'run/calibration16-old-approx0-checked.json':'18fab65962dc79ae9cc7bcd9eb46c98cb397061622a5a2c9c80a813ee58ed301',
 'run/calibration16-native-weights1-checked.json':'cfee32fb98f5bca78502bf60947430b7650b3390622d77e789e801553089168c',
 'run/calibration16-old-native-checked.json':'214d18029c71b1be0cb11b9708d5453fcb22f65aa39acfedf5ad354fded5b993',
}
REF_SHA='1993aa5ff886b3f51401c1d1015e3882550120777809c31ea263936e9d63dda8'
ANCHOR_RAWS={'anchor-affinity3':'89b455eb38161b5937d81d26742ce6971c36129ae1891754b1924d0d29701e61','anchor-weights3':'48f13f8291728241cd42da081db0923bc1cb6e86b79cb89286be79a1e29025b4'}
P='DS4_METAL_V41_RESIDENT_ROUTE_'
SCREEN={
 'mean_delta_nll_max':0.0096013288621,
 'mean_kl_max':0.0182143927637,
 'p99_kl_max':0.179189521279,
 'max_kl_max':1.64426740744,
 'argmax_flips_max':16,
 'worst_prompt_delta_nll_max':0.10611760921684572,
 'paired_candidate_minus_anchor_nll_ci_upper_max':0.005,
 'minimum_native_requested_byte_reduction':0.20,
 'bytes_tie_fraction':0.01,
 'paired_bootstrap_seed':20261002,
 'paired_bootstrap_draws':2000,
 'choice':'Among eligible, take the minimum requested bytes; within1% of that minimum choose lowest mean KL, then p99 KL, then requested bytes, then fixed arm order.',
 'advance':'Selected nonanchor policy must strictly reduce anchor requested bytes OR weakly improve mean NLL, mean KL, p99 KL, max KL, flips and worst-prompt NLL with at least one strict improvement. Otherwise record no advance, preserving every failure and Pareto tradeoff.',
 'scope':'Calibration screen only; no default/task guarantee, no holdout automatic run, no physical SSD/latency claim.',
}


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    path=Path(path);require(path.suffix.lower()!='.gguf','model hashing forbidden')
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def module():
    require(sha(QUAL_SOURCE)==QUAL_SOURCE_SHA,'qualified source differs')
    spec=importlib.util.spec_from_file_location('qualified_resident',QUAL_SOURCE)
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);return s


def qualification(s):
    plan=s.verified_plan(QUAL,QUAL_MANIFEST_SHA)
    require(sha(QUAL/'prepared-command.json')==QUAL_COMMAND_SHA,'qualified command differs')
    for name,digest in QUAL_PINS.items():require(sha(QUAL/name)==digest,'qualified completed proof differs')
    release=s.strict_json((QUAL/'root-release-proof.json').read_text())
    require(release['guard_result']==dict(status='complete',child_returncode=0) and type(release['guard_result']['child_returncode']) is int and release['lock_reacquired'] is True and type(release['live_model_processes']) is list and release['live_model_processes']==[],'qualified model not released')
    summary=s.strict_json((QUAL/'run/summary.json').read_text())
    require(summary['status']=='complete' and len(summary['arms'])==19 and summary['full_default_state_raw_teacher_io_controls'] is True and summary['positive_flag1_noaction_controls'] is True,'nineteen-arm qualification incomplete')
    return plan,summary


def arms():
    return [
      dict(name='native-before',flag='0',subs='3',gap='0',mass='.20',total='.40',rank='48',mode='native'),
      dict(name='anchor-affinity3',flag='0',subs='3',gap='.30',mass='.20',total='.40',rank='48',mode='apply'),
      dict(name='anchor-weights3',flag='1',subs='3',gap='.30',mass='.20',total='.40',rank='48',mode='apply'),
      dict(name='wider-weights3',flag='1',subs='3',gap='.30',mass='.20',total='.50',rank='48',mode='apply'),
      dict(name='closer-weights3',flag='1',subs='3',gap='.15',mass='.20',total='.40',rank='24',mode='apply'),
      dict(name='protected-weights2',flag='1',subs='2',gap='.30',mass='.18',total='.35',rank='48',mode='apply'),
      dict(name='protected-weights1',flag='1',subs='1',gap='.30',mass='.12',total='.40',rank='24',mode='apply',total_mass_ignored=True),
      dict(name='native-after',flag='0',subs='3',gap='0',mass='.20',total='.40',rank='48',mode='native'),
    ]


def policy(arm):
    return {'DS4_METAL_V41_SELECTED_TILE':'64','DS4_METAL_V41_SELECTED_LIMIT':'1024','DS4_METAL_STREAMING_EXPERT_IO_METRICS':'1',**{P+k:v for k,v in dict(NATIVE_WEIGHTS=arm['flag'],MAX_SUBS=arm['subs'],GAP=arm['gap'],MAX_MASS=arm['mass'],MAX_TOTAL_MASS=arm['total'],MAX_RANK=arm['rank'],MODE=arm['mode'],POLICY='replace').items()}}


def shader_environment(qualplan):
    # Qualified manifest keys are authority. Never enumerate the directory:
    # an unpinned case alias such as Dense.metal must not override DENSE_SOURCE.
    inventory={name:value for name,value in qualplan['files'].items() if name.startswith('metal/')}
    require(set(inventory)=={'metal/'+name for name in SHADER_FILES} and len(inventory)==26,
            'exact qualified recognized shader inventory differs')
    env={'DS4_METAL_'+Path(name).stem.upper()+'_SOURCE':str(QUAL/name) for name in inventory}
    recognized={'DS4_METAL_'+Path(name).stem.upper()+'_SOURCE' for name in SHADER_FILES}
    require(len(recognized)==26 and len(env)==26 and set(env)==recognized,
            'unique recognized shader variables differ')
    return env


def require_root_grant(output,digest,s):
    proof=s.strict_json((output/'root-launch-proof.json').read_text())
    require(type(proof) is dict and proof.get('status')=='GRANTED' and
            proof.get('sole_model_grant') is True and proof.get('manifest_sha256')==digest and
            proof.get('command')==s.guard_command(output,digest),
            'explicit ROOT sole-model grant must bind exact frozen manifest and guard command')
    stamp=proof.get('time')
    require(type(stamp) in (int,float) and math.isfinite(stamp) and stamp>0,
            'finite positive ROOT grant time required')
    return proof


def paired(candidate,anchor):
    require(list(candidate['per_prompt'])==list(anchor['per_prompt']) and len(anchor['per_prompt'])==16,'paired sixteen-case coverage differs')
    values=[candidate['per_prompt'][name]['delta_nll']-anchor['per_prompt'][name]['delta_nll'] for name in anchor['per_prompt']]
    require(all(type(v) in (int,float) and math.isfinite(v) for v in values),'paired NLL nonfinite')
    rng=random.Random(SCREEN['paired_bootstrap_seed'])
    boots=sorted(statistics.mean(rng.choices(values,k=16)) for _ in range(SCREEN['paired_bootstrap_draws']))
    return dict(mean=statistics.mean(values),ci95=[boots[49],boots[1949]],paired_prompts=16)


def decision(results):
    require(list(results)==[a['name'] for a in arms()],'all eight first attempts must be retained in order')
    native=results['native-before'];anchor=results['anchor-affinity3'];rows={}
    require(type(native['io']['requested_bytes']) is int and native['io']['requested_bytes']>0,'native logical byte baseline invalid')
    for arm in arms()[1:-1]:
        row=results[arm['name']];pair=paired(row,anchor)
        limits=[('mean_delta_nll',row['delta_nll'],SCREEN['mean_delta_nll_max']),('mean_kl',row['kl_base_to_raw'],SCREEN['mean_kl_max']),('p99_kl',row['kl_p99'],SCREEN['p99_kl_max']),('max_kl',row['kl_max'],SCREEN['max_kl_max']),('argmax_flips',row['argmax_flips'],SCREEN['argmax_flips_max']),('worst_prompt_delta_nll',row['worst_prompt_delta_nll'],SCREEN['worst_prompt_delta_nll_max']),('paired_nll_ci_upper',pair['ci95'][1],SCREEN['paired_candidate_minus_anchor_nll_ci_upper_max'])]
        reduction=1-row['io']['requested_bytes']/native['io']['requested_bytes']
        require(type(row['io']['requested_bytes']) is int and row['io']['requested_bytes']>=0 and all(type(value) in (int,float) and math.isfinite(value) for _,value,_ in limits),'screen metrics nonfinite/invalid')
        failures=[name for name,value,limit in limits if value>limit]
        # Exact integer comparisons keep the fixed 20% and 1% boundaries inclusive.
        if row['io']['requested_bytes']*5>native['io']['requested_bytes']*4:failures.append('native_requested_byte_reduction')
        rows[arm['name']]=dict(eligible=not failures,failures=failures,native_requested_byte_reduction=reduction,paired_candidate_minus_anchor_nll=pair)
    dimensions=('delta_nll','kl_base_to_raw','kl_p99','kl_max','argmax_flips','worst_prompt_delta_nll')
    names=list(rows)
    def vector(name):return [results[name]['io']['requested_bytes']]+[results[name][k] for k in dimensions]
    for name in names:
        a=vector(name)
        rows[name]['pareto_nondominated']=not any(all(y<=x for x,y in zip(a,vector(other))) and any(y<x for x,y in zip(a,vector(other))) for other in names if other!=name)
    eligible=[name for name in names if rows[name]['eligible']]
    choice=None;advance=False
    if eligible:
        minimum=min(results[name]['io']['requested_bytes'] for name in eligible)
        ties=[name for name in eligible if results[name]['io']['requested_bytes']*100<=minimum*101]
        choice=min(ties,key=lambda name:(results[name]['kl_base_to_raw'],results[name]['kl_p99'],results[name]['io']['requested_bytes'],names.index(name)))
        chosen=results[choice]
        quality_dominates=all(chosen[k]<=anchor[k] for k in dimensions) and any(chosen[k]<anchor[k] for k in dimensions)
        advance=choice!='anchor-affinity3' and (chosen['io']['requested_bytes']<anchor['io']['requested_bytes'] or quality_dominates)
    return dict(candidates=rows,selected_calibration_policy=choice,advance=advance,selection_is_only_calibration_screen=True,root_final_policy_selection_required=True,holdout_model_grant_required=True,automatic_holdout_run=False)


def reference_assets(s,qualplan):
    assets=s.assets_and_inventory(QUAL,qualplan)
    for name,digest in QUAL_PINS.items():assets[QUAL/name]=(name,digest)
    assets[QUAL/'manifest.json']=('qualification manifest',QUAL_MANIFEST_SHA)
    assets[QUAL/'prepared-command.json']=('qualification command',QUAL_COMMAND_SHA)
    assets[QUAL_SOURCE]=('qualified source',QUAL_SOURCE_SHA)
    ref=QUAL/'run/calibration16-reference.raw';assets[ref]=('full qualified native reference',REF_SHA)
    for name,digest in ANCHOR_RAWS.items():
        stem='calibration16-old-approx0' if name=='anchor-affinity3' else 'calibration16-native-weights1'
        assets[QUAL/'run'/(stem+'.raw')]=(name+' retained full reference',digest)
    return assets


def cpu_proof(path):
    path=Path(path);proof=module().strict_json(path.read_text())
    require(proof['source_sha256']==sha(__file__) and proof['test_sha256']==sha(TEST_SOURCE) and proof['model_opened'] is False and proof['returncode']==0 and type(proof['returncode']) is int and type(proof['tests']) is int and proof['tests']>=12,'CPU proof/source contract differs')
    return path


def prepare(output,proof_path):
    proof_path=cpu_proof(proof_path)
    s=module();qualplan,summary=qualification(s);assets=reference_assets(s,qualplan);s.raw_module().check_frozen_assets(assets)
    for arm in arms():s.validate_policy(policy(arm))
    output.mkdir(parents=True,exist_ok=False)
    for source,name in [(Path(__file__),'driver.py'),(TEST_SOURCE,'test_resident_env_calibration.py'),(proof_path,'cpu-proof.json'),(QUAL/'manifest.json','qualification-manifest.json'),(QUAL/'run/summary.json','qualification-summary.json'),(QUAL/'root-release-proof.json','qualification-release.json')]:shutil.copyfile(source,output/name)
    external={str(path):value[1] for path,value in assets.items()};external[str(Path(__file__))]=sha(__file__)
    external.update({str(TEST_SOURCE):sha(TEST_SOURCE),str(proof_path):sha(proof_path)})
    plan=dict(arms=arms(),settings=s.SETTINGS,screen=SCREEN,qualification_manifest_sha256=QUAL_MANIFEST_SHA,source_sha256=sha(__file__),test_sha256=sha(TEST_SOURCE),cpu_proof=dict(path=str(proof_path),sha256=sha(proof_path)),files={str(p.relative_to(output)):sha(p) for p in sorted(output.iterdir()) if p.is_file()},external_files=external,command_template=s.command_template(output),shader_env=shader_environment(qualplan),model_from_prior=qualplan['model_from_prior'],fixed_fixture=qualplan['groups']['calibration16'],sampler='teacher forced; NONE thinking; no RNG or generation',all_inherited_DS4_environment_cleared=True,anchor_source='CURRENT qualified calibration16 raw/IO/routes, not older parameters',first_attempt_only=True,model_stat_or_open_in_preparation=False,root_model_grant_required=True,holdout_not_included=True,timing_claim=False,estimated_minutes=[6,9])
    s.write(output/'manifest.json',plan);digest=sha(output/'manifest.json');s.write(output/'prepared-command.json',dict(command=s.guard_command(output,digest),manifest_sha256=digest,launch_requires_root_grant=True,automatic_holdout_run=False))
    print(json.dumps(dict(prepared=str(output),manifest_sha256=digest,command_sha256=sha(output/'prepared-command.json'),arms=8,model_opened=False)),flush=True)


def verified(output,digest,s,qualplan):
    require(re.fullmatch('[0-9a-f]{64}',digest or '') is not None and sha(output/'manifest.json')==digest,'explicit manifest authority differs')
    plan=s.strict_json((output/'manifest.json').read_text());cmd=s.strict_json((output/'prepared-command.json').read_text())
    require(cmd==dict(command=s.guard_command(output,digest),manifest_sha256=digest,launch_requires_root_grant=True,automatic_holdout_run=False),'prepared command differs')
    require(plan['arms']==arms() and plan['settings']==s.SETTINGS and plan['screen']==SCREEN and plan['qualification_manifest_sha256']==QUAL_MANIFEST_SHA and plan['source_sha256']==sha(__file__) and plan['command_template']==s.command_template(output),'fixed policy/screen/source differs')
    require(plan['model_from_prior']==qualplan['model_from_prior'] and plan['fixed_fixture']==qualplan['groups']['calibration16'] and plan['holdout_not_included'] is True,'immutable model/input role differs')
    require(plan['shader_env']==shader_environment(qualplan) and plan['root_model_grant_required'] is True,
            'fixed shader bindings or enforced ROOT grant contract differs')
    proof_path=cpu_proof(plan['cpu_proof']['path'])
    require(plan['test_sha256']==sha(TEST_SOURCE) and plan['cpu_proof']['sha256']==sha(proof_path),'CPU source/proof pin differs')
    required={'driver.py':sha(__file__),'test_resident_env_calibration.py':sha(TEST_SOURCE),'cpu-proof.json':sha(proof_path),'qualification-manifest.json':QUAL_MANIFEST_SHA,'qualification-summary.json':QUAL_PINS['run/summary.json'],'qualification-release.json':QUAL_PINS['root-release-proof.json']}
    expected={str(path):value[1] for path,value in reference_assets(s,qualplan).items()};expected[str(WORK/'speed-bench/ideas/prepare_resident_env_calibration.py')]=sha(__file__)
    expected.update({str(TEST_SOURCE):sha(TEST_SOURCE),str(proof_path):sha(proof_path)})
    require(plan['files']==required and plan['external_files']==expected,'mandatory frozen inventory differs')
    assets={Path(p):(p,v) for p,v in expected.items()};assets.update({output/p:(p,v) for p,v in required.items()});assets[output/'manifest.json']=('explicit manifest',digest);assets[output/'prepared-command.json']=('command',sha(output/'prepared-command.json'))
    return plan,assets


def execute(output,digest):
    s=module();qualplan,qualified=qualification(s);plan,assets=verified(output,digest,s,qualplan)
    s.require_main_guard(output,digest);require_root_grant(output,digest,s);raw=s.raw_module();raw.check_frozen_assets(assets)
    model=Path(plan['model_from_prior']['path']);identity=raw.identity(model);require(identity==plan['model_from_prior'],'guarded model identity differs')
    run=output/'run';run.mkdir(exist_ok=False);s.write(run/'model-identity.json',dict(identity=identity,guard_verified_before_stat=True))
    ref=QUAL/'run/calibration16-reference.raw';metadata=raw.raw_record_metadata(ref,16,32,512);s.validate_raw_finite(ref,metadata,raw)
    env={k:v for k,v in os.environ.items() if not k.startswith('DS4')};env.update(shader_environment(qualplan))
    results={}
    for arm in arms():
        s.require_main_guard(output,digest);require_root_grant(output,digest,s);raw.check_frozen_assets(assets);require(raw.identity(model)==identity,'model identity changed')
        name=arm['name'];record=run/(name+'.raw');scores=run/(name+'.csv');childenv=dict(env,**policy(arm))
        argv=[str(s.BUILD/'quality-new8a47'),str(model),plan['fixed_fixture']['fixtures'],str(scores),str(ref),'compare','--cases','16','--tokens','32','--ctx','4096','--cache-gb','64','--chunk','512','--candidate-record',str(record)]
        s.write(run/(name+'-command.json'),dict(argv=argv,DS4_environment={k:v for k,v in childenv.items() if k.startswith('DS4')},first_attempt=True))
        with (run/(name+'.stdout')).open('x') as out,(run/(name+'.log')).open('x') as err:completed=subprocess.run(argv,cwd=s.ROOT,env=childenv,stdout=out,stderr=err)
        s.write(run/(name+'-exit.json'),dict(returncode=completed.returncode))
        require(completed.returncode==0,name+' first-attempt helper failure')
        raw.check_frozen_assets(assets);require(raw.identity(model)==identity,'model changed during arm')
        text=(run/(name+'.log')).read_text();layout=raw.validate_candidate_record(record,metadata,16,32,512);finite=s.validate_raw_finite(record,metadata,raw);rows=s.validate_csv(scores,metadata,plan['fixed_fixture']['inputs'],raw)
        row=s.validate_quality_json((run/(name+'.stdout')).read_text(),rows);row.update(name=name,raw_sha256=sha(record),layout=layout,finite_validation=finite,cache=s.parse_cache(text),io=s.parse_io(text),route=s.parse_route(text,arm,512*40),**raw.summarize(scores))
        native=qualified['arms']['calibration16-old-native']
        if arm['mode']=='native':require(row['raw_sha256']==REF_SHA and row['bit_equal_rows']==512 and all(row[k]==0 for k in ('delta_nll','kl_base_to_raw','tv','argmax_flips')) and row['io']==native['io'] and row['cache']==native['cache'],'native fullraw/IO bracket control')
        if name in ANCHOR_RAWS:
            prior=qualified['arms']['calibration16-old-approx0' if name=='anchor-affinity3' else 'calibration16-native-weights1']
            require(row['raw_sha256']==ANCHOR_RAWS[name] and all(row[k]==prior[k] for k in ('io','cache','route')),'current anchor fullraw/IO/routes differs')
        if arm['mode']=='apply':
            require(int(arm['subs'])>=2 or (row['route']['double_layers']==0 and row['route']['triple_layers']==0),'sub1 committed multiple slots')
            require(int(arm['subs'])>=3 or row['route']['triple_layers']==0,'sub2 committed triple slot')
        results[name]=row;s.write(run/(name+'-checked.json'),row)
        print(json.dumps(dict(name=name,checked=True,raw_sha256=row['raw_sha256'],delta_nll=row['delta_nll'],kl=row['kl_base_to_raw'],requested_bytes=row['io']['requested_bytes'],route=row['route']),allow_nan=False),flush=True)
    selection=decision(results);s.write(run/'decision.json',selection)
    s.write(run/'summary.json',dict(status='complete',arms=results,decision=selection,first_attempt_only=True,full_native_brackets_and_anchor_controls_exact=True,model_identity=identity,timing_claim=False,automatic_holdout_run=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',type=Path);p.add_argument('--cpu-proof',type=Path);p.add_argument('--run',type=Path);p.add_argument('--manifest-sha256');a=p.parse_args()
    if bool(a.prepare)==bool(a.run):p.error('choose prepare or run')
    if a.run and not a.manifest_sha256:p.error('explicit manifest SHA required')
    if a.prepare and not a.cpu_proof:p.error('passing frozen CPU proof required')
    if a.prepare:prepare(a.prepare.resolve(),a.cpu_proof.resolve())
    else:execute(a.run.resolve(),a.manifest_sha256)
