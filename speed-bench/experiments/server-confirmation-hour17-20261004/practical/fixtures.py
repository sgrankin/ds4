"""Private deterministic checks for two new first-attempt tasks."""
import copy
import random


def ledger_gold(events, config):
    live, deleted = [], []
    for key in sorted({e['key'] for e in events}):
        candidates = [(e['revision'], i, e) for i, e in enumerate(events) if e['key'] == key]
        event = max(candidates, key=lambda x: (x[0], x[1]))[2]
        if event['op'] == 'put':
            live.append({'key':key, 'revision':event['revision'], 'value':copy.deepcopy(event['value'])})
        elif config['include_deleted']:
            deleted.append({'key':key, 'revision':event['revision']})
    return {'live':live, 'deleted':deleted}


def lru_gold(accesses, config):
    capacity = config['capacity']
    if capacity <= 0:
        return {'cache':[], 'hits':0, 'misses':len(accesses), 'evictions':[]}
    last, hits, misses, evictions = {}, 0, 0, []
    for position, key in enumerate(accesses):
        if key in last:
            hits += 1
        else:
            misses += 1
        last[key] = position
        if len(last) > capacity:
            victim = min(last, key=last.get)
            evictions.append(victim)
            del last[victim]
    return {'cache':sorted(last, key=last.get), 'hits':hits, 'misses':misses, 'evictions':evictions}


def put(key, revision, value):
    return {'key':key, 'revision':revision, 'op':'put', 'value':value}


def delete(key, revision):
    return {'key':key, 'revision':revision, 'op':'delete'}


def fixtures(key):
    rng = random.Random(20261004 + (17 if key == 'ledger' else 18))
    if key == 'ledger':
        sample = [put('x',2,'new'), delete('x',1), put('y',3,False), delete('y',3), put('',0,None)]
        edges = [[], [delete('a',0)], [put('a',0,None)],
            [put('a',2,1), put('a',1,2)], [put('a',2,1), delete('a',2)],
            [delete('a',2), put('a',2,1)], [delete('a',3), put('a',2,1)],
            [delete('a',2), put('a',3,1)], [put('',0,False), delete('',0)],
            [put('é',1,1), put('e\u0301',1,2)], [put('b',1,1), put('a',1,2)],
            [put('a',0,{}), put('a',0,[])], [put('😀',9,{'n':[None,False]})],
            [delete('b',9), delete('a',8)], [put('a',9,0), delete('a',0)],
            [put('a',1,True), put('a',1,False), delete('a',1), put('a',1,None)]]
        inputs = [(sample, {'include_deleted':True})]
        inputs += [(events, {'include_deleted':bool(i % 2)}) for i,events in enumerate(edges)]
        keys = ['', 'a', 'b', 'é', 'e\u0301', '😀', 'A']
        values = [None, False, True, 0, -4, '', [0,None], {'x':[False,2]}]
        for i in range(144):
            events=[]
            for j in range(rng.randrange(0,25)):
                k, revision = rng.choice(keys), rng.choice([0,1,2,3,9])
                events.append(delete(k,revision) if rng.randrange(4)==0 else put(k,revision,copy.deepcopy(rng.choice(values))))
            inputs.append((events, {'include_deleted':bool(rng.randrange(2))}))
        gold = ledger_gold
    elif key == 'lru':
        inputs = [(['a','b','a','c','b','d'], {'capacity':2})]
        edges = [([],0),([],3),(['a'],-1),(['a','a'],0),(['a','a'],1),
            (['a','b','a'],1),(['a','b','a'],2),(['a','b','c','a'],2),
            (['a','b','a','c'],2),(['a','b','c'],8),(['','a',''],1),
            (['é','e\u0301','é'],2),(['a','A','a'],1),(['😀','a','😀'],1),
            (['a','b','a','b','a','b'],2),(['a','b','c','d','a','b'],3)]
        inputs += [(accesses, {'capacity':capacity}) for accesses,capacity in edges]
        for i in range(144):
            inputs.append(([rng.choice(['','a','b','c','é','e\u0301','😀']) for j in range(rng.randrange(0,45))],
                           {'capacity':rng.choice([-2,0,1,2,3,5,9])}))
        gold = lru_gold
    else:
        raise ValueError('unknown task '+key)
    assert len(inputs) == 161
    return [(payload,config,gold(payload,config)) for payload,config in inputs]
