#!/usr/bin/env python3
"""Independently enumerate exact answers for the harder branching math pilot."""
import itertools
import json
import math

circle = sum(
    bits.bit_count() == 6
    and not any((bits >> i & 1) and (bits >> ((i + 1) % 16) & 1) for i in range(16))
    for bits in range(1 << 16)
)
assert circle == 16 * math.comb(10, 6) // 10 == 336

lattice = 0
for ups in itertools.combinations(range(14), 7):
    up_positions = set(ups)
    x = y = 0
    for i in range(14):
        if i in up_positions:
            y += 1
        else:
            x += 1
        if y > x or (x, y) in {(2, 2), (4, 4)}:
            break
    else:
        lattice += 1
assert lattice == 429 - (2 * 42 + 14 * 5 - 2 * 2 * 5) == 295

triples = sum(
    b <= c <= 30 and math.gcd(math.gcd(a, b), c) == 1
    and sum(x % 2 == 0 for x in (a, b, c)) == 1
    for a in range(1, 61)
    for b in range(a, 61)
    for c in [60 - a - b]
)
assert triples == 52

digits = sum(
    d[0] != 0 and all(d[i] % 2 != d[i + 1] % 2 for i in range(5))
    and sum(v * 10 ** (5 - i) for i, v in enumerate(d)) % 7 == 0
    for d in itertools.permutations(range(8), 6)
)
assert digits == 156

triangles = sum(
    a + b > 17 and (a + b + 17) % 2 == 1
    and math.gcd(math.gcd(a, b), 17) == 1
    for a in range(1, 18)
    for b in range(a, 18)
)
assert triangles == 44

even_permutations = 0
for p in itertools.permutations(range(1, 9)):
    fixed = [i for i, v in enumerate(p, 1) if v == i]
    if len(fixed) != 2 or fixed[1] - fixed[0] <= 1:
        continue
    inversions = sum(p[i] > p[j] for i in range(8) for j in range(i + 1, 8))
    even_permutations += inversions % 2 == 0
assert even_permutations == math.comb(7, 2) * 130 == 2730

print(json.dumps({
    "circular_binary": circle,
    "restricted_lattice": lattice,
    "primitive_triples": triples,
    "alternating_digits": digits,
    "primitive_triangles": triangles,
    "even_permutations": even_permutations,
}, indent=2))
