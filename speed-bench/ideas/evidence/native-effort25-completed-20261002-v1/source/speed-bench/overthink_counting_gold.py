#!/usr/bin/env python3
"""Independent enumeration + DP checks for four frozen fresh counting cases."""
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "experiments/ds41-ssd/overthink-provenance/counting-study/cases-v1.json"

circle_enumeration = sum(not any((mask >> i & 1) and (mask >> ((i+1) % 10) & 1)
    for i in range(10)) for mask in range(1 << 10))
circle_dp = 0
for first in (0,1):
    state = {first: 1}
    for position in range(1,10):
        following = {0: 0, 1: 0}
        for last, count in state.items():
            for bit in (0,1):
                if not (last and bit): following[bit] += count
        state = following
    circle_dp += sum(count for last,count in state.items() if not (first and last))
assert circle_enumeration == circle_dp == 123

derangement_enumeration = sum(all(value != i+1 for i,value in enumerate(p))
    and p[0] != 2 and p[1] != 1 for p in itertools.permutations(range(1,9)))
dp = {0: 1}
for row in range(8):
    next_dp = {}
    for mask,count in dp.items():
        for column in range(8):
            if mask >> column & 1 or row == column or (row,column) in {(0,1),(1,0)}: continue
            following = mask | (1 << column)
            next_dp[following] = next_dp.get(following,0) + count
    dp = next_dp
assert derangement_enumeration == dp[(1 << 8)-1] == 10860

lattice_enumeration = 0
for ups in itertools.combinations(range(14),6):
    up_positions = set(ups)
    x = y = 0
    for step in range(14):
        if step in up_positions: y += 1
        else: x += 1
        if y > x+1: break
    else: lattice_enumeration += 1
grid = [[0]*7 for _ in range(9)]
for x in range(9):
    for y in range(7):
        if y > x+1: continue
        if x == y == 0: grid[x][y] = 1
        else: grid[x][y] = (grid[x-1][y] if x else 0) + (grid[x][y-1] if y else 0)
assert lattice_enumeration == grid[8][6] == math.comb(14,6)-math.comb(14,4) == 2002

subset_enumeration = sum(sum(values)%7 == 2 for values in itertools.combinations(range(1,13),6))
residues = [[0]*7 for _ in range(7)]
residues[0][0] = 1
for value in range(1,13):
    for size in range(6,0,-1):
        for residue,count in enumerate(residues[size-1]):
            residues[size][(residue+value)%7] += count
assert subset_enumeration == residues[6][2] == 132

answers = dict(circular10=circle_enumeration, restricted_derangements8=derangement_enumeration,
               lattice8x6=lattice_enumeration, subset_mod7=subset_enumeration)
fixture = json.loads(FIXTURE.read_text())
assert {case['id']:case['answer'] for case in fixture['cases']} == answers
expected = '/* Frozen counting-v1 entries; independently checked by overthink_counting_gold.py. */\n'
expected += '\n'.join('        {'+json.dumps(c['id'])+', '+json.dumps(c['prompt'])+', '
                      +json.dumps(str(c['answer']))+', ANSWER_INTEGER},' for c in fixture['cases'])+'\n'
assert (HERE/'overthink_counting_cases.h').read_text() == expected
print(json.dumps(dict(status='PASS_ENUMERATION_AND_INDEPENDENT_DP',answers=answers,
                     fixture_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest()),indent=2))
