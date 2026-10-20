"""Direct independent arithmetic/record checks for the20 authored fixture golds."""
import csv
import io
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / 'fixtures'
WORDS = {'three': 3, 'seven': 7, 'seventeen': 17}


def compute(case, prompt):
    name = case.split('_', 2)[2]
    if name == 'python_slices':
        xs = json.loads(re.search(r'xs=(\[[^]]+\])', prompt)[1])
        return dict(forward=xs[1:6:2], backward=xs[::-3])
    if name == 'dictionary_update':
        values = json.loads(re.search(r'starts as (\{[^}]+\})', prompt)[1])
        for key, value in re.findall(r'assign (\w+)=(-?\d+)', prompt, re.I):
            values[key] = int(value)
        return dict(items=[list(pair) for pair in values.items()], sum=sum(values.values()))
    if name == 'floor_division':
        a, b = map(int, re.search(r'dividend=(-?\d+) and divisor=(\d+)', prompt).groups())
        q, r = divmod(a, b)
        return dict(quotient=q, remainder=r, reconstructed=q*b+r)
    if name == 'quoted_csv':
        data = prompt.split('fields: ', 1)[1].split('. Report ', 1)[0]
        rows = list(csv.DictReader(io.StringIO(data)))
        return dict(names=[r['name'] for r in rows], total_quantity=sum(int(r['quantity']) for r in rows))
    if name == 'boolean_filter':
        rows = json.loads(prompt.split('Rows are ', 1)[1].split('. Retain', 1)[0])
        selected = [r for r in rows if r['active'] and r['score'] >= 8]
        return dict(ids=[r['id'] for r in selected], score_sum=sum(r['score'] for r in selected))
    if name == 'modular_power':
        base, exponent, modulus = map(int, re.search(r'(\d+)\^(\d+) modulo (\d+)', prompt).groups())
        return dict(remainder=pow(base, exponent, modulus))
    if name == 'fraction_sum':
        fractions = [Fraction(x) for x in re.findall(r'\d+/\d+', prompt)]
        result = fractions[0] + fractions[1] - fractions[2]
        return dict(numerator=result.numerator, denominator=result.denominator)
    if name == 'combined_work':
        a, b = map(int, re.search(r'in (\d+) hours and (\d+) hours', prompt).groups())
        hours = 1 / (Fraction(1, a) + Fraction(1, b))
        whole = int(hours)
        minutes = (hours-whole)*60
        assert minutes.denominator == 1
        return dict(time_numerator=hours.numerator, time_denominator=hours.denominator,
                    whole_hours=whole, minutes=int(minutes))
    if name == 'two_colors':
        balls = list('RRRRBBB')
        ordered = list(itertools.permutations(range(len(balls)), 2))
        favorable = sum(balls[a] != balls[b] for a,b in ordered)
        probability = Fraction(favorable, len(ordered))
        return dict(favorable=favorable, total=len(ordered),
                    numerator=probability.numerator, denominator=probability.denominator)
    if name == 'gcd_lcm':
        a, b = map(int, re.search(r'integers (\d+) and (\d+)', prompt).groups())
        return dict(gcd=math.gcd(a,b), lcm=math.lcm(a,b))
    if name == 'interval_schedule':
        intervals = [(name,int(start),int(end)) for name,start,end in
                     re.findall(r'(I\d+)=\[(\d+),(\d+)\)', prompt)]
        valid = []
        for count in range(len(intervals)+1):
            for subset in itertools.combinations(intervals, count):
                ordered = sorted(subset, key=lambda x: x[1])
                if all(a[2] <= b[1] for a,b in zip(ordered,ordered[1:])):
                    valid.append(ordered)
        best = max(valid, key=len)
        assert sum(len(x) == len(best) for x in valid) == 1
        return dict(selected=[x[0] for x in best], count=len(best))
    if name == 'shipping_revision':
        count, color, dock = re.search(r'Approved revision: (\w+) (\w+) crates go to Dock (\w+)', prompt).groups()
        return dict(crates=WORDS[count], color=color, dock=dock)
    if name == 'catalog_cross_reference':
        catalog = dict(re.findall(r'(Kestrel|Mica|Alder) as a (\w+)', prompt))
        first, second, owner = re.search(r'Order O17 contains (\w+) and (\w+); its owner is (\w+)', prompt).groups()
        return dict(owner=owner, materials=[catalog[first],catalog[second]])
    if name == 'custom_units':
        rods = int(re.search(r'league equals (\d+) rods', prompt)[1])
        meters = int(re.search(r'rod equals (\d+) meters', prompt)[1])
        leagues = WORDS[re.search(r'route is (\w+) leagues', prompt)[1]]
        distance = rods*meters*leagues
        return dict(meters=distance, kilometers=distance/1000)
    if name == 'ready_rule':
        record = dict(re.findall(r'(permit|lock|alarm)=(\w+)', prompt))
        requirements = dict(permit='approved',lock='open',alarm='acknowledged')
        failed = [key for key,value in requirements.items() if record[key] != value]
        assert len(failed) == 1
        return dict(ready=not failed, blocking_field=failed[0])
    if name == 'fictional_day_cycle':
        names = prompt.split('in order: ',1)[1].split('.',1)[0].split(', ')
        start = re.search(r'Today is (\w+)', prompt)[1]
        days = WORDS[re.search(r'exactly (\w+) days', prompt)[1]]
        cycles, remainder = divmod(days,len(names))
        return dict(day=names[(names.index(start)+days)%len(names)],complete_cycles=cycles,remainder=remainder)
    if name == 'bitmask':
        a,b,c = [int(x,16) for x in re.findall(r'0x([0-9A-F]+)',prompt)[:3]]
        value = (a&b)^c
        return dict(decimal=value, hex=f'{value:02X}')
    if name == 'current_access':
        obsolete = re.search(r'obsolete access code (\d+)',prompt)[1]
        current = re.search(r'current update replaces it with (\d+)',prompt)[1]
        return dict(current_code=current,obsolete_code=obsolete)
    if name == 'recurrence':
        a,b = map(int,re.search(r'a0=(\d+), a1=(\d+)',prompt).groups())
        values = [a,b]
        while len(values) <= 6:
            values.append(2*values[-1]-values[-2])
        differences = {y-x for x,y in zip(values,values[1:])}
        assert len(differences) == 1
        return dict(a6=values[6],first_difference=differences.pop())
    if name == 'stack':
        stack,popped = [],[]
        commands = prompt.split('Execute ',1)[1].split('. DUP ',1)[0].split(', ')
        for command in commands:
            if command.startswith('PUSH '): stack.append(int(command[5:]))
            elif command == 'POP': popped.append(stack.pop())
            elif command == 'DUP': stack.append(stack[-1])
            elif command == 'ADD': stack.append(stack.pop()+stack.pop())
            else: raise ValueError(command)
        return dict(stack=stack,popped=popped)
    raise ValueError(case)


def main():
    results = []
    for manifest in ('teacher.tsv','generation.tsv'):
        with (HERE/manifest).open(newline='') as file:
            rows = list(csv.reader(file,delimiter='\t'))
        for case,prompt_path,teacher_path in rows:
            prompt = Path(prompt_path).read_text()
            expected = json.loads((FIXTURES/(case+'.gold.json')).read_text())['gold']
            actual = compute(case,prompt)
            teacher = json.loads(Path(teacher_path).read_text().splitlines()[0])
            assert actual == expected, (case,actual,expected)
            assert teacher == expected, (case,'teacher JSON differs from gold')
            results.append(dict(case=case,independently_computed=actual,gold_equal=True,teacher_json_equal=True))
    result = dict(status='PASS',teacher_cases=16,generation_cases=4,results=results,
                  scope='Direct stdlib arithmetic/enumeration and supplied-record extraction; no model/tokenizer/native helper.',
                  command='python3 speed-bench/experiments/confirmation-hour5-20261003/check_gold.py')
    (HERE/'check-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS:16 teacher and4 generation golds; all teacher JSON prefixes match.')


if __name__ == '__main__':
    main()
