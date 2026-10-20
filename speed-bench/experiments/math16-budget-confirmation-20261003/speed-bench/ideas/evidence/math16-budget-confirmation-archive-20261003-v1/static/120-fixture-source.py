"""Source-only unused16 math-budget fixture and serial private-caller build plan."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil

ROOT=Path('/Volumes/Models/ds4')
WORK=ROOT/'.workspaces/idea-reasoning-budget-confirmation'
BUILD=ROOT/'.workspaces/idea-reasoning-budget/speed-bench/ideas/artifacts/reasoning-budget-native-build-v2'
PILOT=ROOT/'.workspaces/idea-reasoning-budget-runner/speed-bench/ideas/artifacts/reasoning-budget-pilot-prepared-20261003-v2'
MIXED=ROOT/'.workspaces/idea-resident-native-weights-next/speed-bench/ideas/artifacts/resident-unused-mixed24-input-freeze-20261003-v1/input-freeze.json'
MIXED_SHA='3295e9bd9b5361c2cb870233f8038b6377e4dca91d778b467092cead2cdec7a4'
PARENT_SOURCE_SHA='48bf48e6ae5b16adffa53129571c9ff3279ed1e5372e3ca8d56738f0aeed519c'
SUITE='math-budget-confirmation16-v1'
SPECS=[
 ('circular14_five','Count binary strings of length 14 with exactly five 1s and no adjacent 1s, including adjacency between positions 14 and 1. Positions are labeled, so rotations are different.'),
 ('linear12_four_gap2','Count binary strings of length 12 with exactly four 1s and at least two 0s between consecutive 1s. There is no circular adjacency constraint.'),
 ('permutations9_two_forbidden','Count permutations p of {1,...,9} such that p(i) differs from i and from i+1 for every i, with the cyclic successor of9 defined as1.'),
 ('deranged8_first_two_odd','Count permutations p of {1,...,8} with p(i)!=i for every i and both p(1) and p(2) odd. Permutations are ordered bijections.'),
 ('dyck7_three_interior_returns','Count right/up unit-step paths from(0,0) to(7,7) with y<=x at every visited point and exactly three visited points on y=x strictly between start and end.'),
 ('bounded10x8_avoid5x4','Count right/up unit-step paths from(0,0) to(10,8) with y<=x+2 at every visited point that never visit(5,4).'),
 ('subset15_seven_mod5','Count seven-element subsets of {1,...,15} whose sum is congruent to3 modulo5. Subsets are unordered and elements distinct.'),
 ('subset16_six_nonconsecutive_threeodd','Count six-element subsets of {1,...,16} with no two consecutive integers and exactly three odd elements.'),
 ('tiles2x8_domino_square','Count tilings of a 2 by 8 rectangular grid using 1 by 2 dominoes (either orientation) and 2 by 2 squares. Tile placements distinguish tilings; same-shaped tiles are indistinguishable.'),
 ('composition18_five_max6','Count ordered lists of five positive integers with sum 18 and each entry at most 6.'),
 ('balls9_four_bins_max4','Distribute nine indistinguishable balls among four labeled bins, at most four balls per bin. Count the possible ordered occupancy lists, allowing empty bins.'),
 ('cycle9_three_colors','Count proper colorings of the labeled 9-vertex cycle with three labeled colors. Adjacent vertices, including vertices 9 and 1, must differ; no quotient by rotations or color permutations.'),
 ('ladder2x5_three_colors','Count colorings of a 2 by 5 grid with three labeled colors such that cells sharing a side differ. Cells are labeled and there are no diagonal adjacency constraints.'),
 ('words_A3B4C3_noAB','Count length 10 words containing exactly three A, four B and three C characters, with no contiguous substring AB. Identical characters are indistinguishable.'),
 ('BANANA_no_adjacent_A','Count distinct anagrams of BANANA with no two adjacent A characters. Identical copies of the same letter are indistinguishable.'),
 ('digits8_sum12_no_equal_neighbors','Count ordered length 8 sequences from digits {0,1,2,3} with sum 12 and no equal neighboring digits. Leading 0s are allowed; there is no circular adjacency constraint.'),
]


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    path=Path(path);require(path.suffix.lower()!='.gguf','no model reads')
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def put(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def assignment_dp(size,allowed):
    states={0:1}
    for row in range(size):
        nxt={}
        for mask,count in states.items():
            for col in range(size):
                if not mask>>col&1 and allowed(row,col):key=mask|1<<col;nxt[key]=nxt.get(key,0)+count
        states=nxt
    return states[(1<<size)-1]


def lattice_dp(width,height,ceiling,forbidden=None,returns=None):
    states={(0,0,0):1}
    for x in range(width+1):
        for y in range(height+1):
            if (x,y)==(0,0) or y>x+ceiling or (x,y)==forbidden:continue
            for k in range((returns or 0)+1):
                extra=int(returns is not None and x==y and 0<x<width)
                if k>=extra:states[x,y,k]=states.get((x-1,y,k-extra),0)+states.get((x,y-1,k-extra),0)
    return states.get((width,height,returns or 0),0)


def lattice_enumerate(width,height,ceiling,forbidden=None,returns=None):
    count=0
    for up in itertools.combinations(range(width+height),height):
        up=set(up);x=y=seen=0;valid=True
        for i in range(width+height):
            if i in up:y+=1
            else:x+=1
            if y>x+ceiling or (x,y)==forbidden:valid=False;break
            seen+=int(x==y and 0<x<width)
        count+=valid and (returns is None or seen==returns)
    return count


def gold_pairs():
    pairs=[]
    pairs.append((sum(sum(b)==5 and not any(b[i] and b[(i+1)%14] for i in range(14)) for b in itertools.product((0,1),repeat=14)),14*math.comb(9,5)//9))
    pairs.append((sum(sum(b)==4 and all(not b[i] or all(not b[j] for j in range(i+1,min(12,i+3))) for i in range(12)) for b in itertools.product((0,1),repeat=12)),math.comb(12-2*(4-1),4)))
    pairs.append((sum(all(p[i]!=i and p[i]!=(i+1)%9 for i in range(9)) for p in itertools.permutations(range(9))),assignment_dp(9,lambda i,j:j!=i and j!=(i+1)%9)))
    pairs.append((sum(all(p[i]!=i+1 for i in range(8)) and p[0]%2==p[1]%2==1 for p in itertools.permutations(range(1,9))),assignment_dp(8,lambda i,j:j!=i and (i>=2 or (j+1)%2==1))))
    pairs.append((lattice_enumerate(7,7,0,returns=3),lattice_dp(7,7,0,returns=3)))
    pairs.append((lattice_enumerate(10,8,2,forbidden=(5,4)),lattice_dp(10,8,2,forbidden=(5,4))))
    subset=sum(sum(c)%5==3 for c in itertools.combinations(range(1,16),7));states={(0,0):1}
    for value in range(1,16):
        nxt=dict(states)
        for (n,r),count in states.items():
            if n<7:key=(n+1,(r+value)%5);nxt[key]=nxt.get(key,0)+count
        states=nxt
    pairs.append((subset,states[7,3]))
    subset=sum(sum(v%2 for v in c)==3 and all(b-a>1 for a,b in zip(c,c[1:])) for c in itertools.combinations(range(1,17),6));states={(0,0,False):1}
    for value in range(1,17):
        nxt={}
        for (n,odd,prior),count in states.items():
            key=(n,odd,False);nxt[key]=nxt.get(key,0)+count
            if not prior and n<6:key=(n+1,odd+value%2,True);nxt[key]=nxt.get(key,0)+count
        states=nxt
    pairs.append((subset,sum(count for (n,o,p),count in states.items() if n==6 and o==3)))
    def tile_words(remaining):
        if not remaining:return {()}
        result={('V',)+word for word in tile_words(remaining-1)}
        if remaining>=2:result|={(kind,)+word for kind in ('Hpair','square') for word in tile_words(remaining-2)}
        return result
    tiles=[1,1]
    for i in range(2,9):tiles.append(tiles[-1]+2*tiles[-2])
    pairs.append((len(tile_words(8)),tiles[8]))
    for lower,upper,parts,total in [(1,6,5,18),(0,4,4,9)]:
        direct=sum(sum(v)==total for v in itertools.product(range(lower,upper+1),repeat=parts));states={0:1}
        for _ in range(parts):
            nxt={}
            for partial,count in states.items():
                for value in range(lower,upper+1):key=partial+value;nxt[key]=nxt.get(key,0)+count
            states=nxt
        pairs.append((direct,states[total]))
    pairs.append((sum(all(c[i]!=c[(i+1)%9] for i in range(9)) for c in itertools.product(range(3),repeat=9)),2**9-2))
    direct=sum(all(c[i]!=c[i+5] for i in range(5)) and all(c[i]!=c[i+1] for i in (0,1,2,3,5,6,7,8)) for c in itertools.product(range(3),repeat=10));columns=[p for p in itertools.product(range(3),repeat=2) if p[0]!=p[1]];states={p:1 for p in columns}
    for _ in range(4):states={p:sum(n for old,n in states.items() if old[0]!=p[0] and old[1]!=p[1]) for p in columns}
    pairs.append((direct,sum(states.values())))
    direct=sum(c.count('A')==3 and c.count('B')==4 and c.count('C')==3 and not any(a=='A' and b=='B' for a,b in zip(c,c[1:])) for c in itertools.product('ABC',repeat=10));states={(0,0,0,''):1};target=(3,4,3)
    for _ in range(10):
        nxt={}
        for (a,b,c,last),count in states.items():
            for i,letter in enumerate('ABC'):
                counts=[a,b,c]
                if counts[i]>=target[i] or last=='A' and letter=='B':continue
                counts[i]+=1;key=tuple(counts)+(letter,);nxt[key]=nxt.get(key,0)+count
        states=nxt
    pairs.append((direct,sum(states.values())))
    pairs.append((sum('AA' not in ''.join(p) for p in set(itertools.permutations('BANANA'))),math.factorial(3)//math.factorial(2)*math.comb(4,3)))
    direct=sum(sum(c)==12 and all(a!=b for a,b in zip(c,c[1:])) for c in itertools.product(range(4),repeat=8));states={(0,-1):1}
    for _ in range(8):
        nxt={}
        for (partial,last),count in states.items():
            for value in range(4):
                if value!=last and partial+value<=12:key=(partial+value,value);nxt[key]=nxt.get(key,0)+count
        states=nxt
    pairs.append((direct,sum(n for (s,last),n in states.items() if s==12)))
    require(len(pairs)==16 and all(type(a) is int and a==b and 0<a<2**63 for a,b in pairs),'bounded two-method gold mismatch')
    return pairs


def fixture(pairs):
    return dict(suite=SUITE,cases=[dict(case=ident,prompt=prompt+' Put only the integer count in your final answer.',answer=str(pair[0])) for (ident,prompt),pair in zip(SPECS,pairs)])


def frontend():
    source=WORK/'speed-bench/reasoning_budget_probe.c';require(sha(source)==PARENT_SOURCE_SHA,'literal parent caller differs')
    text=source.read_text();require(text.count('#include "reasoning_budget_cases.h"')==1 and text.count('counting-v1')==1,'only intended source substitution sites')
    changed=text.replace('#include "reasoning_budget_cases.h"','#include "reasoning_budget_confirmation_cases.h"').replace('counting-v1',SUITE)
    require(changed.replace('#include "reasoning_budget_confirmation_cases.h"','#include "reasoning_budget_cases.h"').replace(SUITE,'counting-v1')==text,'policy/sampler/native-body change')
    return changed


def prepare(output,proof):
    proof=Path(proof);info=json.loads(proof.read_text());require(info['source_sha256']==sha(__file__) and info['returncode']==0 and type(info['returncode']) is int and info['model_or_metadata_opened'] is False,'CPU proof authority')
    pairs=gold_pairs();items=fixture(pairs);require(len(items['cases'])==16 and len({x['case'] for x in items['cases']})==16,'ordered fixed16 coverage')
    require(sha(MIXED)==MIXED_SHA,'mixed input freeze changed');mixed=json.loads(MIXED.read_text());old=json.loads((PILOT/'fixture.json').read_text());used={r['prompt_sha256'] for r in mixed['cases']}|{hashlib.sha256(r['prompt'].encode()).hexdigest() for r in old['cases']}
    require(all(hashlib.sha256(r['prompt'].encode()).hexdigest() not in used for r in items['cases']),'exact previous prompt reused')
    output.mkdir(parents=True,exist_ok=False);source=WORK/'speed-bench/reasoning_budget_confirmation_probe.c';header=WORK/'speed-bench/reasoning_budget_confirmation_cases.h'
    require(not source.exists() and not header.exists(),'private source outputs must be fresh')
    source.write_text(frontend());header.write_text('/* Fixed unused16 math-budget cases; gold proofs are in the input freeze. */\n'+''.join('        {'+','.join(json.dumps(row[k]) for k in ('case','prompt','answer'))+',ANSWER_INTEGER},\n' for row in items['cases']))
    put(output/'fixture.json',items);put(output/'gold-proof.json',dict(methods='independent exhaustive enumeration versus DP/closed formula, each explicitly implemented in fixture-source.py',ordered_cases=[r['case'] for r in items['cases']],pairs=[list(p) for p in pairs],all_equal=True,model_or_metadata_opened=False))
    shutil.copy2(__file__,output/'fixture-source.py');shutil.copy2(proof,output/'cpu-proof.json');shutil.copy2(source,output/source.name);shutil.copy2(header,output/header.name)
    build=json.loads((BUILD/'build-manifest.json').read_text());compile_argv=build['compile_argv'][:];oldsource=compile_argv.index(str(ROOT/'.workspaces/idea-reasoning-budget/speed-bench/reasoning_budget_probe.c'));compile_argv[oldsource]=str(source)
    include=compile_argv.index('-I');compile_argv[include+1]=str(WORK);binary=output/'reasoning-budget-confirmation-probe';compile_argv[compile_argv.index('-o')+1]=str(binary)
    borrowed={str(BUILD/'build-manifest.json'):sha(BUILD/'build-manifest.json'),str(WORK/'speed-bench/reasoning_budget_probe.c'):PARENT_SOURCE_SHA,**build['objects'],**build['sdk_sha256'],build['compiler']['path']:build['compiler']['sha256']}
    for name,pin in build['source_sha256'].items():
        borrowed[str(ROOT/'.workspaces/idea-reasoning-budget'/name)]=pin
        borrowed[str(WORK/name)]=pin
    for name in ['ds4.h','ds4_gpu.h']:
        old=ROOT/'.workspaces/idea-reasoning-budget'/name;require(sha(WORK/name)==sha(old),'public native API header differs')
        borrowed[str(WORK/name)]=sha(old);borrowed[str(old)]=sha(old)
    borrowed[str(PILOT/'reasoning_budget_signed64.py')]='73a2263294f2d2257fb3e3c20848c8aa5da99ebcccfc4ec4a6a80aff8bb92d5e';borrowed[str(MIXED)]=MIXED_SHA
    for path,pin in borrowed.items():require(sha(path)==pin,'borrowed source/build pin differs')
    put(output/'build-plan.json',dict(status='SOURCE_ONLY_UNRUN',source_parent='7b63dca033e747583e9d613db4ad8bced66b6e8a',caller_parent_source='a3a726027492fd7e74f15e7a3b7e21fdc71612f9',source_sha256=sha(source),header_sha256=sha(header),compile_argv=compile_argv,selftest_argv=[str(binary),'--self-test'],all_outputs_must_be_absent=[str(binary)],max_concurrent_compilers=1,only_private_frontend_link=True,core_or_Metal_rebuild=False,compile_or_link_granted=False,borrowed_pins=borrowed,compiler_version=build['compiler']['version'],sdk_path=build['sdk_path']))
    plan=dict(role='UNUSED16 fixed math-budget confirmation; source-only',suite=SUITE,ordered_cases=[r['case'] for r in items['cases']],arms=[dict(name=n,budget=b) for n,b in [('native-before',0),('budget1024',1024),('native-after',0)]],settings=dict(effort=25,cap=2048,seed=20261002,temperature_float_value=.6000000238418579,top_p_float_value=.949999988079071,top_k=0,min_p=0,cache_gib=64,context_size=8192,prefill_chunk=512,threads=2,mtp_draft=1,native_penalty=0),scorer='literal strict signed64 final-v2 wrapper order; independent libc cross-check before study',all_first_attempts=48,maximum_generated_tokens=98304,estimated_minutes=[75,110],native_before_after_full_token_RNG_piece_prompt_raw_exact_required=True,pre_force_same_prompt_state_and_native_trace_required=True,only_forced_close_token128822_no_RNG_advance_then_native_outside_thinking=True,all_wrong_cap_INVALID_attempts_retained=True,primary='completed strict exact answers; capped correct answers reported separately',discordance='report both native-correct/budget-failed and native-failed/budget-correct across all16 fixed cases',no_cross_prefix_KL_NLL=True,no_quantization_causality_claim=True,no_default_promotion=True,existing_four_calibration_cases_remain_distinct=True,mixed24_remains_distinct=True,automatic_model_or_tokenization_run=False,model_or_tokenizer_metadata_opened=False,model_grant_required_after_source_and_build_qualification=True,files={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    put(output/'input-freeze.json',plan)
    print(json.dumps(dict(output=str(output),input_freeze_sha256=sha(output/'input-freeze.json'),cases=16,golds=[a for a,b in pairs],core_or_private_frontend_compiled=False,model_or_metadata_opened=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',type=Path,required=True);parser.add_argument('--cpu-proof',type=Path,required=True);args=parser.parse_args();prepare(args.prepare.resolve(),args.cpu_proof.resolve())
