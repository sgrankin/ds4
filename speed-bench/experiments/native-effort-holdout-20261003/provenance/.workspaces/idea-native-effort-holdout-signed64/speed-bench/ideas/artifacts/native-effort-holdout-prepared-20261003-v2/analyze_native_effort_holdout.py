#!/usr/bin/env python3
"""Strict first-attempt native-effort counting analysis; no model access."""
import hashlib
import json
import math
from pathlib import Path
import re

TEMP=0.6000000238418579
TOP_P=0.949999988079071
CAP=2048
TOKEN_FIELDS=('step','token_id','stop','inside_thinking','generated_thinking_tokens','policy_active','penalty','rng_before','rng_after','piece_hex')
PARITY_FIELDS=('case','suite','think_level','native_penalty','think_enabled','prompt_supplied_think_close','generated_think_close','status','parse_ok','parse_reason','scorer','seed','temperature_float_value','top_p_float_value','min_p','max_tokens','cache_gib_requested','context_size','prefill_chunk','prompt_tokens','generated_tokens','reasoning_tokens','think_closed','stopped','cap_hit','correct_exact','parsed_answer','final','expected','prompt')

def require(test,message):
    if not test:raise ValueError(message)

def digest(path):
    require(Path(path).suffix.lower()!='.gguf','No model access')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def integer(value,label):
    require(type(value) is int and 0<=value<=2147483647,'typed integer '+label);return value

def boolean(value,label):
    require(type(value) is bool,'typed bool '+label);return value

def number(value,label):
    require(type(value) in (int,float) and math.isfinite(value),'finite number '+label);return value

def local(root,name):
    require(type(name) is str and name and Path(name).name==name,'plain local filename')
    return Path(root)/name

def jsonl(path):
    raw=Path(path).read_bytes();require(raw.endswith(b'\n'),'truncated JSONL')
    values=[json.loads(line) for line in raw.splitlines()]
    require(all(type(v) is dict for v in values),'JSONL objects');return values

def parsed_integer(final):
    # Match the literal C strict-final-v2 wrapper order and ASCII whitespace.
    s=final.strip(' \t\n\r\v\f')
    if len(s)>=6 and s.startswith('```') and s.endswith('```'):
        s=s[3:-3]
        if s.startswith('python\n'):s=s[7:]
        elif s.startswith('text\n'):s=s[5:]
        s=s.strip(' \t\n\r\v\f')
    if len(s)>=4 and s.startswith('**') and s.endswith('**'):s=s[2:-2].strip(' \t\n\r\v\f')
    if len(s)>=2 and s.startswith('`') and s.endswith('`'):s=s[1:-1].strip(' \t\n\r\v\f')
    if len(s)>=9 and s.startswith('\\boxed{') and s.endswith('}'):s=s[7:-1].strip(' \t\n\r\v\f')
    if not re.fullmatch('[+-]?[0-9]+',s):return None
    negative=s.startswith('-')
    digits=(s[1:] if s[0] in '+-' else s).lstrip('0') or '0'
    limit='9223372036854775808' if negative else '9223372036854775807'
    if len(digits)>len(limit) or (len(digits)==len(limit) and digits>limit):return None
    # Conversion is now bounded to signed64 magnitude, independent of Python's
    # global digit limit. Long leading-zero inputs retain C strtoll semantics.
    return str(int(('-' if negative else '')+digits))

def validate_trace(root,row,level):
    trace_path=local(root,row['token_trace']);raw_path=local(root,row['raw_trace']);prompt_path=local(root,row['prompt_trace'])
    trace=jsonl(trace_path)
    require(len(trace)==row['generated_tokens']+int(row['stopped']),'complete sampled rows')
    prompts=prompt_path.read_bytes();require(prompts.endswith(b'\n'),'prompt newline')
    ids=prompts.splitlines();require(len(ids)==row['prompt_tokens'] and ids,'complete prompt IDs')
    require(all(re.fullmatch(b'[0-9]+',i) and int(i)<129280 for i in ids),'prompt ID range')
    raw=bytearray();final=bytearray();closed=level==0;generated_close=False;reasoning=0;previous=f'{row["seed"]:016x}'
    for step,t in enumerate(trace):
        require(set(t)==set(TOKEN_FIELDS),'exact token schema')
        require(integer(t['step'],'step')==step,'contiguous token steps')
        require(integer(t['token_id'],'token_id')<129280,'vocabulary range')
        for key in ['stop','inside_thinking','policy_active']:boolean(t[key],key)
        require(t['policy_active'] is False and number(t['penalty'],'penalty')==0,'native policy only')
        require(integer(t['generated_thinking_tokens'],'prior thinking')<=step,'bounded prior thinking')
        if level==0:require(not t['inside_thinking'] and t['generated_thinking_tokens']==0,'native no-thinking state')
        for key in ['rng_before','rng_after']:require(type(t[key]) is str and re.fullmatch('[0-9a-f]{16}',t[key]),'RNG hex')
        require(t['rng_before']==previous,'RNG chain');previous=t['rng_after']
        require(type(t['piece_hex']) is str and re.fullmatch('(?:[0-9a-f]{2})*',t['piece_hex']),'piece hex')
        piece=bytes.fromhex(t['piece_hex'])
        require(t['stop']==(row['stopped'] and step==row['generated_tokens']),'stop position')
        if t['stop']:continue
        raw.extend(piece)
        if not closed:reasoning+=1
        if piece==b'</think>':closed=True;generated_close=True;final.clear()
        elif closed:final.extend(piece)
    require(bytes(raw)==raw_path.read_bytes(),'raw bytes reconstructed from all non-stop pieces')
    require(row['reasoning_tokens']==reasoning and row['think_closed']==closed and row['generated_think_close']==generated_close,'closure counts')
    require(type(row['final']) is str and row['final'].encode()==bytes(final),'strict offset0/last-close final region')
    return dict(trace=trace,raw_sha256=digest(raw_path),token_sha256=digest(trace_path),prompt_sha256=digest(prompt_path))

def validate_arm(root,level,fixture,rc):
    require(type(level) is int and level in (0,25,75),'study native level')
    cases=fixture['cases'];require(type(cases) is list and len(cases)==6 and len({c['case'] for c in cases})==6,'six unique frozen cases')
    rows=jsonl(Path(root)/'results.jsonl');require(len(rows)==6 and [r['case'] for r in rows]==[c['case'] for c in cases],'complete ordered six first attempts')
    options=dict(suite='independent-counting-v1',think_level=level,native_penalty=0,think_enabled=level>0,
        prompt_supplied_think_close=level==0,scorer='strict-final-v2',seed=20261002,temperature_float_value=TEMP,
        top_p_float_value=TOP_P,min_p=0,max_tokens=CAP,cache_gib_requested=64,context_size=8192,prefill_chunk=512)
    infos=[]
    for row,case in zip(rows,cases):
        for key,want in options.items():
            if type(want) is bool:boolean(row[key],key)
            elif type(want) is int:integer(row[key],key)
            elif type(want) is float:number(row[key],key)
            else:require(type(row[key]) is str,'typed '+key)
            require(row[key]==want,'fixed configuration '+key)
        require(row['expected']==case['answer'] and row['prompt']==case['prompt'],'frozen gold/prompt')
        for key in ['generated_think_close','parse_ok','think_closed','stopped','cap_hit','correct_exact']:boolean(row[key],key)
        n=integer(row['generated_tokens'],'generated');reasoning=integer(row['reasoning_tokens'],'reasoning')
        require(reasoning<=n<=CAP and row['cap_hit']==(n==CAP) and row['stopped']!=row['cap_hit'],'native cap/stop counts')
        for key in ['prefill_seconds','decode_seconds']:require(number(row[key],key)>=0,'nonnegative timing')
        info=validate_trace(root,row,level);infos.append(info)
        parsed=parsed_integer(row['final']) if row['think_closed'] else None
        expected_status='INVALID' if parsed is None else 'CORRECT' if parsed==case['answer'] else 'INCORRECT'
        expected_reason='none' if parsed is not None else 'unclosed_reasoning' if not row['think_closed'] else 'empty_final' if not row['final'].strip(' \t\n\r\v\f') else 'unsupported_final_format'
        require(row['parsed_answer']==(parsed or '') and row['parse_ok']==(parsed is not None) and row['status']==expected_status and
                row['parse_reason']==expected_reason and row['correct_exact']==(expected_status=='CORRECT'),'independent exact final scorer')
    invalid=sum(r['status']=='INVALID' for r in rows)
    require(type(rc) is int and rc==(3 if invalid else 0),'preserved INVALID-only producer rc')
    report=dict(level=level,cases=6,completed_correct=sum(r['status']=='CORRECT' and r['stopped'] and not r['cap_hit'] for r in rows),
        correct=sum(r['status']=='CORRECT' for r in rows),capped_correct=sum(r['status']=='CORRECT' and r['cap_hit'] for r in rows),
        incorrect=sum(r['status']=='INCORRECT' for r in rows),invalid=invalid,cap_hits=sum(r['cap_hit'] for r in rows),
        generated_tokens=sum(r['generated_tokens'] for r in rows),reasoning_tokens=sum(r['reasoning_tokens'] for r in rows),producer_returncode=rc,
        per_case=[dict(case=r['case'],status=r['status'],cap_hit=r['cap_hit'],stopped=r['stopped'],generated_tokens=r['generated_tokens'],
                      reasoning_tokens=r['reasoning_tokens'],parsed_answer=r['parsed_answer'],**{k:i[k] for k in ['raw_sha256','token_sha256','prompt_sha256']}) for r,i in zip(rows,infos)])
    return report,rows,infos

def require_repeat(first,second):
    a,ar,ai=first;b,br,bi=second
    require(a['level']==b['level'],'same repeated native level')
    require(len(ar)==len(br)==6,'complete repeat')
    for x,y,p,q in zip(ar,br,ai,bi):
        require(all(x[k]==y[k] for k in PARITY_FIELDS),'repeat first-attempt fields '+x['case'])
        require(p['trace']==q['trace'] and p['raw_sha256']==q['raw_sha256'] and p['prompt_sha256']==q['prompt_sha256'],
                'repeat exact full token/RNG/piece/raw/prompt '+x['case'])
    return True
