#!/usr/bin/env python3
"""Offline equal-size expert-cache screening. No inference or runtime timing.

The trace omits prefill. reset models independent cold decode stretches;
carry models an artificial decode-only stream, NOT the actual session.
Current six-expert groups are protected; older GPU work is assumed complete.
"""
import argparse
from collections import OrderedDict, Counter
import hashlib
import heapq
import json
from pathlib import Path
from route_stats import read_routes, route_bytes


def oldest(q, protected):
    return next((x for x in q if x not in protected), None)


class Cache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.resident = set()
        self.clock = 0

    def prepare(self, group, token, future):
        self.protected = set(group)

    def access(self, key):
        raise NotImplementedError


class Ranked(Cache):
    """LRU, all-request LFU with history, decaying LFU, or future-aware eviction."""
    def __init__(self, capacity, mode, decay=0):
        super().__init__(capacity)
        self.mode, self.decay = mode, decay
        self.frequency = Counter()
        self.stamps, self.ranks, self.heap = {}, {}, []
        self.epoch = 0
        self.next_use = {}

    def rank(self, key):
        if self.mode == 'lru': return (self.stamps[key], key)
        if self.mode == 'future': return (-self.next_use[key], key)
        return (self.frequency[key], self.stamps[key], key)

    def rebuild(self):
        self.ranks = {k:self.rank(k) for k in self.resident}
        self.heap = [(rank,k) for k,rank in self.ranks.items()]
        heapq.heapify(self.heap)

    def prepare(self, group, token, future):
        super().prepare(group, token, future)
        epoch = token // self.decay if self.decay else 0
        if epoch != self.epoch:
            shift = epoch - self.epoch
            self.frequency = Counter({k:v >> shift for k,v in self.frequency.items()})
            self.epoch = epoch
            self.rebuild()
        # The native loader counts the entire selected group before eviction.
        self.frequency.update(group)
        self.next_use.update(zip(group, future))
        for k in group:
            if k in self.resident:
                self.ranks[k] = self.rank(k)
                heapq.heappush(self.heap, (self.ranks[k], k))

    def access(self, key):
        hit = key in self.resident
        if not hit and len(self.resident) == self.capacity:
            held = []
            while True:
                rank, victim = heapq.heappop(self.heap)
                if self.ranks.get(victim) != rank: continue
                if victim in self.protected:
                    held.append((rank, victim)); continue
                self.resident.remove(victim)
                del self.ranks[victim]
                break
            for item in held: heapq.heappush(self.heap, item)
        self.resident.add(key)
        self.clock += 1
        self.stamps[key] = self.clock
        self.ranks[key] = self.rank(key)
        heapq.heappush(self.heap, (self.ranks[key], key))
        if len(self.heap) > 4 * self.capacity: self.rebuild()
        return hit


class Queues(Cache):
    """SLRU (80% protected), full 2Q (25% FIFO, 50% ghost), or ARC.

    ARC follows the FAST03 four-list algorithm with integer target adaptation.
    Batch adaptation: skip protected keys; if the desired queue is entirely
    pinned, evict from the other resident queue. This is reported as a variant.
    """
    def __init__(self, capacity, mode):
        super().__init__(capacity)
        self.mode = mode
        self.t1, self.t2, self.b1, self.b2 = [OrderedDict() for _ in range(4)]
        self.p = 0

    def evict(self, prefer_t1, ghost=True):
        queues = (self.t1, self.t2) if prefer_t1 else (self.t2, self.t1)
        for q in queues:
            victim = oldest(q, self.protected)
            if victim is not None:
                del q[victim]; self.resident.remove(victim)
                if ghost:
                    b = self.b1 if q is self.t1 else self.b2
                    b[victim] = None
                return
        raise AssertionError('all resident slots protected')

    def replace_arc(self, key):
        self.evict(bool(self.t1) and
                   (len(self.t1) > self.p or (key in self.b2 and len(self.t1) == self.p)))

    def access(self, key):
        hit = key in self.resident
        c = self.capacity
        if hit:
            if key in self.t2: self.t2.move_to_end(key)
            elif self.mode != 'twoq':
                del self.t1[key]; self.t2[key] = None
            if self.mode == 'slru' and len(self.t2) > max(1, int(.8*c)):
                k,_ = self.t2.popitem(last=False); self.t1[k] = None
            return True
        if self.mode == 'arc':
            if key in self.b1:
                self.p = min(c, self.p + max(1, len(self.b2)//len(self.b1)))
                self.replace_arc(key); del self.b1[key]; self.t2[key] = None
            elif key in self.b2:
                self.p = max(0, self.p - max(1, len(self.b1)//len(self.b2)))
                self.replace_arc(key); del self.b2[key]; self.t2[key] = None
            else:
                if len(self.t1) + len(self.b1) == c:
                    if len(self.t1) < c:
                        self.b1.popitem(last=False); self.replace_arc(key)
                    else: self.evict(True, ghost=False)
                elif len(self.t1) + len(self.b1) < c:
                    total = len(self.t1)+len(self.t2)+len(self.b1)+len(self.b2)
                    if total >= c:
                        if total == 2*c: self.b2.popitem(last=False)
                        self.replace_arc(key)
                self.t1[key] = None
        elif self.mode == 'twoq':
            repeat = key in self.b1
            if len(self.resident) == c:
                # Full 2Q uses FIFO for A1in and LRU for Am.
                self.evict(len(self.t1) > max(1, c//4) or not self.t2)
            if repeat:
                del self.b1[key]; self.t2[key] = None
            else: self.t1[key] = None
            while len(self.b1) > max(1,c//2): self.b1.popitem(last=False)
            self.b2.clear()  # Am evictions do not enter 2Q history.
        else:
            if len(self.resident) == c: self.evict(True, ghost=False)
            self.t1[key] = None
        self.resident.add(key)
        return False


POLICIES = ('lru','lfu','aging16','aging64','aging256','slru','twoq','arc','future')


def factory(name, capacity):
    if name in ('slru','twoq','arc'): return Queues(capacity,name)
    return Ranked(capacity, 'lfu' if name.startswith('aging') else name,
                  int(name[5:]) if name.startswith('aging') else 0)


def segments(tokens, experts):
    result = []
    for i,(pos,_,routes) in enumerate(tokens):
        if i == 0 or pos != tokens[i-1][0]+1: result.append([])
        result[-1].append([[layer*experts+k for k in group]
                          for layer,group in enumerate(routes)])
    return result


def future_indices(groups):
    last = {}; result = [None]*len(groups)
    for i in range(len(groups)-1,-1,-1):
        result[i] = [last.get(k,len(groups)+1) for k in groups[i]]
        last.update(dict.fromkeys(groups[i],i))
    return result


def simulate(stretches, name, capacity, scenario, warmup=64):
    if capacity < max(len(g) for s in stretches for t in s for g in t):
        raise ValueError('capacity smaller than one protected group')
    cache = None; results=[]; global_token=0
    # In carry mode the oracle can see across the artificial removed gaps too.
    all_groups=[g for s in stretches for t in s for g in t]
    future_all=future_indices(all_groups); offset=0
    for si,stretch in enumerate(stretches):
        if scenario == 'reset' or cache is None: cache=factory(name,capacity)
        groups=[g for t in stretch for g in t]
        future=future_indices(groups) if scenario=='reset' else future_all[offset:offset+len(groups)]
        counts=Counter(); steady=Counter(); gi=0
        for ti,token in enumerate(stretch):
            for group in token:
                hits_before=sum(k in cache.resident for k in group)
                cache.prepare(group,ti if scenario=='reset' else global_token,future[gi]);gi+=1
                misses=sum(not cache.access(k) for k in group)
                assert misses == len(group)-hits_before
                row={'requests':len(group),'misses':misses,'groups':1,'all_hit_groups':int(misses==0)}
                counts.update(row)
                if ti>=warmup:steady.update(row)
            global_token+=1
        offset+=len(groups)
        results.append(dict(segment=si,tokens=len(stretch),all=dict(counts),after_warmup=dict(steady)))
    def aggregate(field):
        c=Counter()
        for r in results:c.update(r[field])
        return dict(c,miss_rate=c['misses']/c['requests'] if c['requests'] else None,
                    all_hit_fraction=c['all_hit_groups']/c['groups'] if c['groups'] else None)
    return dict(policy=name,capacity=capacity,scenario=scenario,all=aggregate('all'),
                after_warmup=aggregate('after_warmup'),segments=results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--routes',type=Path,default=Path(__file__).with_name('session-v1.routes.bin.gz'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capacities',default='4096,7822,8253')
    p.add_argument('--policies',default=','.join(POLICIES))
    p.add_argument('--warmup',type=int,default=64)
    a=p.parse_args()
    layers,experts,used,tokens=read_routes(a.routes)
    ss=segments(tokens,experts)
    result=dict(routes_sha256=hashlib.sha256(route_bytes(a.routes)).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        layers=layers,experts_per_layer=experts,selected=used,tokens=len(tokens),
        segment_lengths=[len(s) for s in ss],warmup_tokens_per_segment=a.warmup,
        limitations=['Prefill and its cache seeding/frequency changes are absent.',
            'reset starts every stretch empty with zero history; carry ignores all missing prefill.',
            'Equal-size global cache; no layer quotas, slab constraints, prefetch or older in-flight work.',
            'Current selected group protected. ARC/2Q skip pinned entries and may use the other queue.',
            'aging16 is a route-only approximation of native policy, not calibrated native cache replay.',
            'future uses next-use eviction with group protection; not a proven group-optimal bound.',
            '64-token warmup is accounting only; no future information used by online policies.',
            'No latency or inference speed is measured.'],results=[])
    for scenario in ('reset','carry'):
        for capacity in map(int,a.capacities.split(',')):
            for name in a.policies.split(','):
                if name not in POLICIES:raise ValueError(name)
                r=simulate(ss,name,capacity,scenario,a.warmup);result['results'].append(r)
                print(scenario,capacity,name,r['all']['misses'],r['after_warmup']['misses'],flush=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
