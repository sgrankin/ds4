import itertools
import random
import unittest
from functools import lru_cache
from cache_policy_study import factory, future_indices, POLICIES, simulate


def run(name,capacity,groups,decay_tokens=None):
    c=factory(name,capacity); future=future_indices(groups); misses=0
    for i,g in enumerate(groups):
        c.prepare(g,i if decay_tokens is None else decay_tokens[i],future[i])
        misses+=sum(not c.access(k) for k in g)
        assert set(g)<=c.resident
        assert len(c.resident)<=capacity
        if hasattr(c,'t1'):
            queues=[set(c.t1),set(c.t2),set(c.b1),set(c.b2)]
            assert c.resident==queues[0]|queues[1]
            assert sum(map(len,queues))==len(set.union(*queues))
            if name=='arc':
                assert len(c.t1)+len(c.b1)<=capacity
                assert sum(map(len,queues))<=2*capacity
                assert 0<=c.p<=capacity
    return misses,c


class CacheTests(unittest.TestCase):
    def test_lru_reuse_distance(self):
        self.assertEqual(run('lru',2,[[x] for x in [1,2,1,3,1,2]])[0],4)

    def test_group_hits_cannot_be_evicted_before_access(self):
        for p in POLICIES:
            # 1 is old, but is protected while 3 is inserted before its hit.
            misses,c=run(p,2,[[1,2],[3,1]])
            self.assertEqual(misses,3,p)
            self.assertEqual(c.resident,{1,3})

    def test_aging_matches_independent_scan(self):
        rng=random.Random(8)
        groups=[rng.sample(range(12),3) for _ in range(150)]
        for policy,period in [('lfu',0),('aging16',16),('aging64',64)]:
            freq={};stamp={};resident=set();clock=0;misses=0
            for token,g in enumerate(groups):
                if period and token and token%period==0:freq={k:v>>1 for k,v in freq.items()}
                for k in g:freq[k]=freq.get(k,0)+1
                for k in g:
                    if k not in resident:
                        misses+=1
                        if len(resident)==5:
                            victim=min(resident-set(g),key=lambda x:(freq[x],stamp[x],x))
                            resident.remove(victim)
                        resident.add(k)
                    clock+=1;stamp[k]=clock
            actual,c=run(policy,5,groups)
            self.assertEqual(actual,misses)
            self.assertEqual(c.resident,resident)

    def test_arc_adapts_to_both_ghost_lists(self):
        _,c=run('arc',2,[[x] for x in [1,2,1,3,2]])
        self.assertEqual(c.p,1);self.assertEqual(set(c.b2),{1})
        c.prepare([1],5,[99]);self.assertFalse(c.access(1))
        self.assertEqual(c.p,0);self.assertEqual(c.resident,{1,2})

    def test_twoq_cold_hit_does_not_promote_ghost_hit_does(self):
        _,c=run('twoq',2,[[1],[1],[2],[3]])
        self.assertIn(1,c.b1);self.assertFalse(c.t2)
        c.prepare([1],4,[99]);c.access(1)
        self.assertIn(1,c.t2)

    def test_random_group_invariants(self):
        rng=random.Random(912)
        groups=[rng.sample(range(30),4) for _ in range(1000)]
        for policy in POLICIES:
            for capacity in (4,5,12,30):run(policy,capacity,groups)

    def test_future_matches_exhaustive_singleton_optimum(self):
        for seq in itertools.product(range(3),repeat=6):
            @lru_cache(None)
            def best(i,state):
                if i==len(seq):return 0
                x=seq[i];cache=set(state)
                if x in cache:return best(i+1,state)
                if len(cache)<2:return 1+best(i+1,tuple(sorted(cache|{x})))
                return 1+min(best(i+1,tuple(sorted(cache-{v}|{x}))) for v in cache)
            self.assertEqual(run('future',2,[[x] for x in seq])[0],best(0,()))

    def test_reset_and_carry_distinguish_recording_gaps(self):
        segments=[[[[1,2]]],[[[1,2]]]]
        a=simulate(segments,'lru',2,'reset',0)
        b=simulate(segments,'lru',2,'carry',0)
        self.assertEqual(a['all']['misses'],4)
        self.assertEqual(b['all']['misses'],2)
        self.assertEqual(a['all'],a['after_warmup'])

    def test_online_policies_do_not_use_future_information(self):
        groups=[[1,2],[3,1],[2,4],[1,3],[4,2]]
        for name in POLICIES:
            if name=='future':continue
            a=factory(name,3);b=factory(name,3)
            for i,g in enumerate(groups):
                a.prepare(g,i,[0]*len(g));b.prepare(g,i,[999]*len(g))
                self.assertEqual([a.access(k) for k in g],[b.access(k) for k in g])
                self.assertEqual(a.resident,b.resident)

    def test_full_capacity_only_compulsory_misses(self):
        for p in POLICIES:
            self.assertEqual(run(p,4,[[1,2],[3,4],[1,4],[2,3]])[0],4)

if __name__=='__main__':unittest.main()
