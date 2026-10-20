"""Author fixed inputs/golds once, then independently check their semantics."""
import copy
import json
from pathlib import Path
import random
from gold_oracles import dedupe_gold, bucket_gold

HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'server-confirmation-hour17-20261004/practical/spec.json'


def events(pairs, field):
    return [{'t':t,field:value} for t,value in pairs]


def dedupe_check(payload, config):
    clocks, accepted, rejected = {}, [], []
    for i,e in enumerate(payload):
        if e['key'] not in clocks or e['t'] >= clocks[e['key']]+config['ttl']:
            clocks[e['key']]=e['t']; accepted.append(i)
        else:
            rejected.append(i)
    return {'accepted':accepted,'rejected':rejected}


def bucket_check(payload, config):
    # Refill one time unit at a time, independently of the gold's delta formula.
    tokens, clock = config['capacity'], 0
    accepted, rejected, remaining = [], [], []
    for i,e in enumerate(payload):
        while clock < e['t']:
            tokens=min(config['capacity'],tokens+config['rate']); clock+=1
        if e['cost'] > tokens:
            rejected.append(i)
        else:
            accepted.append(i); tokens-=e['cost']
        remaining.append(tokens)
    return {'accepted':accepted,'rejected':rejected,'remaining':remaining,'tokens':tokens}


def main():
    dedupe_sample=events([(0,'a'),(4,'a'),(5,'a'),(5,'b'),(9,'a'),(10,'a')],'key')
    dedupe_edges=[([],5), ([(0,'a')],0), ([(0,'a'),(0,'a')],5),
        ([(0,'a'),(4,'a'),(5,'a')],5), ([(0,'a'),(4,'a'),(8,'a'),(10,'a')],5),
        ([(0,'a'),(4,'a'),(5,'a'),(9,'a'),(10,'a')],5),
        ([(0,'a'),(0,'b'),(0,'a')],1), ([(0,'a'),(0,'a'),(0,'a')],0),
        ([(0,''),(1,''),(2,'')],2), ([(0,'é'),(0,'e\u0301'),(1,'é')],2),
        ([(0,'a'),(0,'A'),(1,'a')],2), ([(0,'😀'),(1,'😀'),(2,'😀')],2),
        ([(2,'a'),(3,'b'),(4,'a'),(5,'b')],2),
        ([(0,'a'),(50,'a'),(50,'a')],50),
        ([(3,'a'),(4,'a'),(7,'a')],4),
        ([(0,'a'),(1,'b'),(2,'a'),(3,'b'),(4,'a')],3)]
    dedupe_inputs=[(dedupe_sample,{'ttl':5})]+[(events(p,'key'),{'ttl':ttl}) for p,ttl in dedupe_edges]
    rng=random.Random(20261004321)
    for _ in range(144):
        t=0; data=[]
        for _ in range(rng.randrange(35)):
            t+=rng.choice([0,0,1,2,4,9])
            data.append({'t':t,'key':rng.choice(['','a','b','A','é','e\u0301','😀'])})
        dedupe_inputs.append((data,{'ttl':rng.choice([0,1,2,5,9,20])}))
    bucket_sample=events([(0,4),(1,4),(2,1),(10,8),(10,5),(10,0)],'cost')
    bucket_edges=[([],5,1),([],0,0),([(0,0),(1,1),(2,0)],0,3),
        ([(0,2),(2,1)],3,0), ([(0,3),(0,0),(0,1)],3,1),
        ([(0,3),(1,1),(2,2)],3,1), ([(0,5),(1,6),(5,5)],5,1),
        ([(0,5),(2,6),(3,3)],5,1), ([(0,6),(0,5),(0,0)],5,0),
        ([(10,5),(11,1),(12,1)],5,1), ([(0,1),(1,1),(2,1)],1,0),
        ([(0,2),(1,4),(1,3)],3,2), ([(0,5),(2,5),(2,1)],5,3),
        ([(0,0),(0,0),(3,0)],4,0), ([(0,10**16),(2,7),(3,9)],10**16,3),
        ([(0,1),(0,4),(2,3),(2,2)],5,1)]
    bucket_inputs=[(bucket_sample,{'capacity':5,'rate':1})]+[
        (events(p,'cost'),{'capacity':cap,'rate':rate}) for p,cap,rate in bucket_edges]
    rng=random.Random(20261004322)
    for _ in range(144):
        t=0; data=[]
        for _ in range(rng.randrange(35)):
            t+=rng.choice([0,0,1,2,4,9])
            data.append({'t':t,'cost':rng.choice([0,1,2,3,5,8,13,21])})
        bucket_inputs.append((data,{'capacity':rng.choice([0,1,2,5,8,11]),
                                   'rate':rng.choice([0,1,2,4])}))
    inputs={'dedupe':dedupe_inputs,'bucket':bucket_inputs}
    golds={'dedupe':dedupe_gold,'bucket':bucket_gold}
    rows={key:[{'payload':data,'config':config,'gold':golds[key](data,config)}
               for data,config in cases] for key,cases in inputs.items()}
    assert all(len(cases)==161 for cases in rows.values())
    (HERE/'cases.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    old=json.loads(OLD.read_text())
    contract=old['tasks'][0]['prompt'].split('The project contains config.json',1)[1]
    prompts={
        'dedupe':('Task: Time-window deduplication.\nCreate solution.py containing def solve(payload, config). '
            'payload is a list of events with nonnegative integer t timestamps in nondecreasing order and string key values. '
            'config ttl is a nonnegative integer. Start with no accepted events. Accept the first occurrence of each exact key. '
            'For later occurrences, reject only if t minus that key\'s LAST ACCEPTED timestamp is less than ttl. '
            'Rejected duplicates do not refresh or change that timestamp. Equality expires: t minus the stored accepted timestamp >= ttl is accepted and replaces it. '
            'ttl0 accepts every event, including equal timestamps. Return {accepted, rejected}, both lists of zero-based payload indices in their original order. '
            'Compare string keys exactly without Unicode normalization, preserve empty keys, and do not mutate payload or config.\n'),
        'bucket':('Task: Integer token bucket event reducer.\nCreate solution.py containing def solve(payload, config). '
            'payload is a list of events with nonnegative integer t timestamps in nondecreasing order and nonnegative integer cost values. '
            'config capacity and rate are nonnegative integers. Begin at time0 with tokens equal to capacity. '
            'At EVERY event, first refill tokens by (t minus previous event timestamp) times rate, capped at capacity, then set previous timestamp to t. '
            'If cost <= tokens, accept and subtract cost; otherwise reject without spending any tokens. '
            'Oversized costs are rejected but still perform the refill and timestamp advance. Zero cost is accepted even at capacity0. '
            'Return {accepted, rejected, remaining, tokens}. accepted/rejected are zero-based payload indices in order; remaining contains the balance AFTER each event; '
            'tokens is the final balance (capacity for an empty payload). Use exact integer arithmetic and do not mutate payload or config.\n')}
    spec={'task_order':['dedupe','bucket'],'tests_per_task':161,
          'scope':'Unused event-processing content, fixed before any model attempt.', 'tasks':[]}
    for key in ['dedupe','bucket']:
        sample,config=inputs[key][0]
        spec['tasks'].append({'key':key,'prompt':prompts[key]+'The project contains config.json'+contract,
            'system':old['tasks'][0]['system'],'config':dict(config,sample=sample)})
    (HERE/'spec.json').write_text(json.dumps(spec,indent=2,ensure_ascii=False)+'\n')
    # Hand answers precede the independent full-fixture comparison.
    assert dedupe_gold(dedupe_sample,{'ttl':5})=={'accepted':[0,2,3,5],'rejected':[1,4]}
    assert dedupe_gold(events([(0,'a'),(4,'a'),(5,'a')],'key'),{'ttl':5})=={'accepted':[0,2],'rejected':[1]}
    assert bucket_gold(bucket_sample,{'capacity':5,'rate':1})=={
        'accepted':[0,2,4,5],'rejected':[1,3],'remaining':[1,2,2,5,0,0],'tokens':0}
    assert bucket_gold(events([(0,5),(2,6),(3,3)],'cost'),{'capacity':5,'rate':1})=={
        'accepted':[0,2],'rejected':[1],'remaining':[0,2,0],'tokens':0}
    for key,check in [('dedupe',dedupe_check),('bucket',bucket_check)]:
        for row in rows[key]:
            assert check(copy.deepcopy(row['payload']),copy.deepcopy(row['config']))==row['gold']
    result={'status':'GOLD_SANITY_PASS','cases':{key:len(cases) for key,cases in rows.items()},
            'hand_edges':4,'independent_checks':'dedupe accepted-history scan vs per-key clock; bucket delta arithmetic vs unit-time refill',
            'model_or_tokenizer_or_compiler_executed':False}
    (HERE/'check-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
