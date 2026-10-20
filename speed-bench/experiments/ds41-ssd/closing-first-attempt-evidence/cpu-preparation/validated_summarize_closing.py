#!/usr/bin/env python3
"""Typed analysis for the frozen additive closing-only pilot; no model access."""
import argparse,hashlib,json,math,re
from pathlib import Path
TOKEN=128822
TEMP=0.6000000238418579
GOLDS={'circular10':123,'restricted_derangements8':10860,'lattice8x6':2002,'subset_mod7':132}

def require(test,message):
    if not test:raise ValueError(message)
def integer(x,label):
    require(type(x) is int and 0<=x<=2147483647,label+': signed integer');return x
def number(x,label):
    require(type(x) in (int,float) and math.isfinite(x),label+': finite number');return x
def boolean(x,label):
    require(type(x) is bool,label+': bool');return x
def digest(path):
    require(Path(path).suffix.lower()!='.gguf','No model hashing');return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jsonl(path):
    raw=Path(path).read_bytes();require(raw.endswith(b'\n'),'truncated JSONL')
    rows=[json.loads(line) for line in raw.splitlines()]
    require(all(type(r) is dict for r in rows),'JSONL objects');return rows

def local(directory,name):
    require(type(name) is str and name and Path(name).name==name,'plain local trace filename')
    return Path(directory)/name

def validate_trace(directory,row):
    d=row['diagnostics'];require(type(d) is dict,'diagnostics object')
    require(integer(d['stride'],'stride')==32,'stride32')
    require(number(d['temperature_float_value'],'temperature')==TEMP,'Tfloat0.6')
    trace_path=local(directory,d['token_trace']);raw_path=local(directory,row['raw_trace'])
    trace=jsonl(trace_path);require(len(trace)==row['generated_tokens']+int(row['stopped']),'trace length')
    raw=bytearray();final=bytearray();closed=False;reasoning=0;previous=f"{row['seed']:016x}"
    counts=dict(active_steps=0,thinking_marker_selections=0,active_marker_selections=0,pre_filter_samples=0,sampled_argmax_changes=0)
    masses=[];kls=[];dnlls=[]
    for i,t in enumerate(trace):
        require(integer(t['step'],'step')==i,'contiguous step')
        token=integer(t['token_id'],'token_id');require(token<129280,'native vocabulary range')
        for key in ['stop','inside_thinking','policy_active','selected_marker']:boolean(t[key],key)
        prior=integer(t['generated_thinking_tokens'],'prior thinking');require(prior<=i,'prior thinking count')
        active=row['closing_boost']>0 and t['inside_thinking'] and prior>=1024
        require(t['policy_active']==active,'ADD closing gate')
        require(t['selected_marker']==(token==TOKEN),'single closing marker')
        delta=number(t['applied_logit_delta'],'actual rounded delta')
        require(0<=delta<=4 if active else delta==0,'actual delta gate/range')
        for key in ['rng_before','rng_after']:require(type(t[key]) is str and re.fullmatch('[0-9a-f]{16}',t[key]),'RNG hex')
        require(t['rng_before']==previous,'RNG chain');previous=t['rng_after']
        require(type(t['piece_hex']) is str and re.fullmatch('(?:[0-9a-f]{2})*',t['piece_hex']),'piece hex')
        piece=bytes.fromhex(t['piece_hex'])
        if token==TOKEN:require(piece==b'</think>','exact closing piece')
        require(t['stop']==(row['stopped'] and i==row['generated_tokens']),'stop position')
        counts['active_steps']+=active
        counts['thinking_marker_selections']+=token==TOKEN and t['inside_thinking']
        counts['active_marker_selections']+=token==TOKEN and active
        sampled=i%32==0 and t['inside_thinking']
        require(('raw_marker_mass_pre_filter' in t)==sampled,'diagnostic placement')
        if sampled:
            m=number(t['raw_marker_mass_pre_filter'],'closing raw mass');require(0<=m<=1,'mass range')
            alpha=delta/TEMP;logz=math.log1p(m*math.expm1(alpha))
            kl=max(0,logz-alpha*m);dnll=logz-alpha*(token==TOKEN)
            require(abs(number(t['kl_raw_to_policy_pre_filter'],'KL')-kl)<1e-9,'ADD KL/T pre-filter formula')
            require(abs(number(t['selected_delta_nll_pre_filter'],'deltaNLL')-dnll)<1e-9,'ADD deltaNLL/T pre-filter formula')
            number(t['marker_best_minus_other_logit'],'margin')
            for key in ['raw_argmax','policy_argmax']:require(integer(t[key],key)<129280,'argmax range')
            if not active:require(t['raw_argmax']==t['policy_argmax'],'inactive argmax parity')
            counts['pre_filter_samples']+=1;counts['sampled_argmax_changes']+=t['raw_argmax']!=t['policy_argmax']
            masses.append(m);kls.append(t['kl_raw_to_policy_pre_filter']);dnlls.append(t['selected_delta_nll_pre_filter'])
        if t['stop']:continue
        raw.extend(piece)
        if not closed:reasoning+=1
        if piece==b'</think>':closed=True;final.clear()
        elif closed:final.extend(piece)
    require(bytes(raw)==raw_path.read_bytes(),'raw bytes from pieces')
    require(row['reasoning_tokens']==reasoning and row['think_closed']==closed,'reasoning closure count')
    require(type(row['final']) is str and row['final'].encode()==bytes(final),'every-close final reset')
    require(integer(d['sampling_steps'],'sampling_steps')==len(trace),'sample count')
    for key,want in counts.items():require(integer(d[key],key)==want,'aggregate '+key)
    for key,want in [('raw_marker_mass_mean',sum(masses)/len(masses) if masses else 0),('raw_marker_mass_max',max(masses,default=0)),('policy_kl_mean_pre_filter',sum(kls)/len(kls) if kls else 0),('selected_delta_nll_sum_pre_filter',sum(dnlls))]:
        require(abs(number(d[key],key)-want)<1e-8,'aggregate '+key)
    require(number(d['cpu_seconds'],'cpu time')>=0,'nonnegative diagnostic time')
    return dict(token_trace_sha256=digest(trace_path),raw_trace_sha256=digest(raw_path),active_steps=counts['active_steps'],trace=trace)

def validate_arm(directory,boost,fixture,rc):
    cases=fixture['cases'];require(type(cases) is list and len(cases)==4,'four cases')
    require({c['id']:c['answer'] for c in cases}==GOLDS,'four gold identities')
    require(all(type(c['id']) is str and type(c['answer']) is int and type(c['prompt']) is str for c in cases),'typed fixture')
    rows=jsonl(Path(directory)/'results.jsonl');require(len(rows)==4,'four native/single candidate records')
    require([r['case'] for r in rows]==[c['id'] for c in cases],'ordered unique cases')
    infos=[]
    options=dict(closing_boost=boost,policy_direction='ADD',closing_token_id=TOKEN,scope='think',marker_profile='closing',mapped_markers=1,
        engine_penalty=0,engine_marker_profile='hesitation',engine_overthink_after=0,think_level=25,seed=20261002,temperature=.6,top_p=.95,min_p=0,
        max_tokens=2048,cache_gib_requested=64.,closing_after=1024,case_offset=0,case_limit=4,suite='counting',scorer='strict-final-v2')
    for r,c in zip(rows,cases):
        for key,want in options.items():
            if type(want) is int:integer(r[key],key)
            elif type(want) is float:number(r[key],key)
            else:require(type(r[key]) is str,'typed '+key)
            require(r[key]==want,'configuration '+key)
        require(type(r['expected']) is str and r['expected']==str(c['answer']),'gold')
        require(r['prompt']==c['prompt'] and type(r['prompt']) is str,'prompt unchanged')
        for key in ['parse_ok','think_closed','stopped','cap_hit','correct_exact']:boolean(r[key],key)
        n=integer(r['generated_tokens'],'generated');reason=integer(r['reasoning_tokens'],'reasoning')
        require(reason<=n<=2048 and r['cap_hit']==(n==2048) and r['stopped']!=r['cap_hit'],'token/cap/stop flags')
        require(r['status'] in ['CORRECT','INCORRECT','INVALID'],'strict producer status')
        require(type(r['parsed_answer']) is str and type(r['parse_reason']) is str,'typed scorer result')
        if r['status']=='INVALID':require(not r['parse_ok'] and not r['correct_exact'] and r['parse_reason'] in ['unclosed_reasoning','empty_final','unsupported_final_format'],'INVALID status')
        else:
            require(r['parse_ok'] and r['think_closed'] and r['parse_reason']=='none','parsed producer result')
            require(r['correct_exact']==(r['status']=='CORRECT') and r['correct_exact']==(r['parsed_answer']==r['expected']),'strict recorded result')
        for key in ['prefill_seconds','decode_seconds']:require(number(r[key],key)>=0,'timing range')
        infos.append(validate_trace(directory,r))
    invalid=sum(r['status']=='INVALID' for r in rows)
    require(type(rc) is int and rc==(3 if invalid else 0),'original INVALID-only producer rc rule')
    report=dict(cases=4,correct=sum(r['status']=='CORRECT' for r in rows),completed_correct=sum(r['status']=='CORRECT' and not r['cap_hit'] and r['stopped'] and r['think_closed'] for r in rows),
        capped_correct=sum(r['status']=='CORRECT' and r['cap_hit'] for r in rows),incorrect=sum(r['status']=='INCORRECT' for r in rows),invalid=invalid,cap_hits=sum(r['cap_hit'] for r in rows),
        tokens=sum(r['generated_tokens'] for r in rows),reasoning_tokens=sum(r['reasoning_tokens'] for r in rows),active_steps=sum(i['active_steps'] for i in infos),producer_returncode=rc,
        cases_status=[{k:r[k] for k in ['case','status','cap_hit','generated_tokens','reasoning_tokens','final','parsed_answer']} for r in rows],record_integrity='PASS')
    return report,rows,infos

PARITY_FIELDS=('status','parse_ok','parse_reason','parsed_answer','final','expected','prompt','generated_tokens','reasoning_tokens','think_closed','stopped','cap_hit','correct_exact')
TOKEN_FIELDS=('step','token_id','stop','inside_thinking','generated_thinking_tokens','rng_before','rng_after','piece_hex')
def check_reference(directory,rows,infos,reference,only_circular=False):
    old=jsonl(Path(reference)/'results.jsonl');require(len(old)==4,'old25 four records')
    for r,i,o in zip(rows,infos,old):
        require(r['case']==o['case'],'old25 order')
        if only_circular and r['case']!='circular10':continue
        require(all(r[k]==o[k] for k in PARITY_FIELDS),'old25 strict/raw result parity '+r['case'])
        oldtrace=jsonl(local(reference,o['diagnostics']['token_trace']))
        require(len(oldtrace)==len(i['trace']) and all(all(a[k]==b[k] for k in TOKEN_FIELDS) for a,b in zip(oldtrace,i['trace'])),'old25 full-token/RNG/piece parity '+r['case'])
        require(local(directory,r['raw_trace']).read_bytes()==local(reference,o['raw_trace']).read_bytes(),'old25 raw bytes parity')
        if only_circular:require(i['active_steps']==0,'circular must inactive')
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());fixture=json.loads(Path(plan['fixture']).read_text());reports=[]
    for arm in plan['arms']:
        rc=json.loads((Path(plan['output'])/(arm['name']+'-process.json')).read_text())['returncode']
        report,rows,infos=validate_arm(arm['results_dir'],arm['boost'],fixture,rc)
        report.update(arm=arm['name'],old25_reference_parity=check_reference(arm['results_dir'],rows,infos,plan['reference'],arm['boost']>0));reports.append(report)
    result=dict(status='ANALYSIS_INTEGRITY_PASS',arms=reports,all_first_attempts_retained=True,
        distribution='Same-prefix full-vocabulary pre-filter softmax at explicit Tfloat0.6000000238418579; not nucleus probabilities or T1/cross-trajectory KL.',
        interpretation='Strict producer outcomes unchanged. Completed-correct requires !cap, stopped and closed. No quantization-causality or speed/default promotion claim.')
    with a.output.open('x') as out:json.dump(result,out,indent=2);out.write('\n')
if __name__=='__main__':main()
