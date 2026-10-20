#!/usr/bin/env python3
"""Independent exact golds for six frozen native-effort counting cases.

CPU only: no model, tokenizer, outcome file or floating-point approximation.
Every gold is checked by two independently expressed counting methods.
"""
import argparse
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, permutations, product
import json
from math import comb, gcd
from pathlib import Path

CASES = [
    ('fixed_points9', 'How many permutations p of {1,2,3,4,5,6,7,8,9} have exactly three fixed points? A fixed point is a position i with p(i)=i. There are no other restrictions. Put only the integer count in your final answer.'),
    ('necklaces12_weight4', 'How many necklaces consist of twelve beads, exactly four black and eight white, when rotations are identified but reflections are counted as different unless they are also rotations? Put only the integer count in your final answer.'),
    ('onto7to4', 'How many functions from the labeled set {1,2,3,4,5,6,7} to the labeled set {A,B,C,D} are onto? Onto means that every one of A,B,C,D occurs at least once as an image. Put only the integer count in your final answer.'),
    ('domino3by8', 'How many tilings of a 3-by-8 rectangular grid use twelve identical 1-by-2 dominoes? Dominoes may be horizontal or vertical. Grid positions are fixed, and rotations or reflections of a tiling are counted separately if they occupy different grid cells. Put only the integer count in your final answer.'),
    ('partitions20_5_max7', 'How many nondecreasing five-tuples of positive integers (a,b,c,d,e) satisfy a+b+c+d+e=20 and e<=7? Count each nondecreasing tuple once. Put only the integer count in your final answer.'),
    ('connected_graph6_edges7', 'How many simple undirected connected graphs have the labeled vertex set {1,2,3,4,5,6} and exactly seven edges? Loops and multiple edges are forbidden. Put only the integer count in your final answer.'),
]

def fixed_points_enumeration():
    return sum(sum(i==v for i,v in enumerate(p))==3 for p in permutations(range(9)))

def fixed_points_formula():
    a,b=1,0
    for n in range(2,7):a,b=b,(n-1)*(a+b)
    return comb(9,3)*b

def necklace_orbits():
    words={sum(1<<i for i in c) for c in combinations(range(12),4)}
    mask=(1<<12)-1
    return len({min(((w<<r)|(w>>(12-r)))&mask for r in range(12)) for w in words})

def necklace_burnside():
    # A shift has gcd(12,r) cycles, each of length12/gcd(12,r).
    fixed=0
    for r in range(12):
        cycles=gcd(12,r);length=12//cycles
        if 4%length==0:fixed+=comb(cycles,4//length)
    assert fixed%12==0
    return fixed//12

def onto_enumeration():
    return sum(len(set(f))==4 for f in product(range(4),repeat=7))

def onto_inclusion_exclusion():
    return sum((-1)**k*comb(4,k)*(4-k)**7 for k in range(5))

@lru_cache(None)
def domino_board(mask=0):
    if mask==(1<<24)-1:return 1
    i=next(i for i in range(24) if not mask&(1<<i))
    total=0
    # Row-major grid width8; place domino at first empty cell.
    for j in ([i+1] if i%8<7 else [])+([i+8] if i<16 else []):
        if not mask&(1<<j):total+=domino_board(mask|(1<<i)|(1<<j))
    return total

def domino_closed_recurrence():
    # For even widths, T(n)=4*T(n-2)-T(n-4), T(0)=1,T(2)=3.
    a,b=1,3
    for _ in range(4,9,2):a,b=b,4*b-a
    return b

def partition_enumeration():
    return sum(sum(c)==20 for c in combinations_with_replacement(range(1,8),5))

@lru_cache(None)
def partition_recurrence(total=20,count=5,maximum=7):
    if count==0:return int(total==0)
    if total<1 or maximum<1:return 0
    return partition_recurrence(total,count,maximum-1)+partition_recurrence(total-maximum,count-1,maximum)

def graph_enumeration():
    edges=list(combinations(range(6),2));total=0
    for picked in combinations(edges,7):
        neighbors=[set() for _ in range(6)]
        for a,b in picked:neighbors[a].add(b);neighbors[b].add(a)
        reached={0};pending=[0]
        while pending:
            for j in neighbors[pending.pop()]-reached:reached.add(j);pending.append(j)
        total+=len(reached)==6
    return total

def choose(n,k):return comb(n,k) if 0<=k<=n else 0

@lru_cache(None)
def graph_component_recurrence(n=6,m=7):
    if m<0 or m>n*(n-1)//2:return 0
    if n==1:return int(m==0)
    all_graphs=choose(n*(n-1)//2,m)
    # Subtract graphs whose component containing vertex1 has size k<n.
    disconnected=sum(choose(n-1,k-1)*graph_component_recurrence(k,j)*
        choose((n-k)*(n-k-1)//2,m-j) for k in range(1,n) for j in range(m+1))
    return all_graphs-disconnected

METHODS=[(fixed_points_enumeration,fixed_points_formula),(necklace_orbits,necklace_burnside),
         (onto_enumeration,onto_inclusion_exclusion),(domino_board,domino_closed_recurrence),
         (partition_enumeration,partition_recurrence),(graph_enumeration,graph_component_recurrence)]

def verified_cases():
    result=[]
    for (name,prompt),(a,b) in zip(CASES,METHODS):
        first,second=a(),b()
        if type(first) is not int or type(second) is not int or first!=second:
            raise ValueError('Independent gold mismatch: '+name)
        result.append(dict(case=name,prompt=prompt,answer=str(first),kind='integer',
            gold_methods=[a.__name__,b.__name__],gold_results=[first,second]))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path)
    args=parser.parse_args();cases=verified_cases()
    document=dict(version=1,cases=cases,model_opened=False,model_metadata_read=False,
                  inputs_selected_without_model_outcomes=True)
    if args.freeze:
        args.freeze.parent.mkdir(parents=True,exist_ok=True)
        with args.freeze.open('x') as f:json.dump(document,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({c['case']:c['answer'] for c in cases},sort_keys=True))

if __name__=='__main__':main()
