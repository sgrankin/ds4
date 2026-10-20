"""Independent CPU-only archive of the first released protocol-budget pilot."""
import argparse
import ctypes
import errno
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import stat
import sys

ROOT=Path('/Volumes/Models/ds4')
WORK=ROOT/'.workspaces/idea-reasoning-budget-runner'
PREP=WORK/'speed-bench/ideas/artifacts/reasoning-budget-pilot-prepared-20261003-v2'
PLAN_SHA='d78f72e601fd811280d62adc9d2971bd7612ab10c58b29fdde47abfcb026a286'
RELEASE_SHA='a95b2d2b37d0a03077fab9e9c5646d45f6f4c5c62c2e9168f66fa57b561720cc'
ANALYZER_SHA='fd57c19c551a09049303f35b38299e6b403d850debdde80f3ecdb26107766377'
DRIVER_SHA='d3afdee0cb0b03083df0b29e087d58ac38fc5f69db2710198d7825e1a9645f2c'


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    path=Path(path);require(path.suffix.lower()!='.gguf','no model hashing/open')
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def put(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def golds():
    circular=sum(not any(bits[i] and bits[(i+1)%10] for i in range(10)) for bits in itertools.product((0,1),repeat=10))
    circular2=0
    for first in (0,1):
        states={first:1}
        for _ in range(9):states={n:sum(count for old,count in states.items() if not old*n) for n in (0,1)}
        circular2+=sum(n for last,n in states.items() if not last*first)
    perms=sum(all(p[i]!=i+1 for i in range(8)) and p[0]!=2 and p[1]!=1 for p in itertools.permutations(range(1,9)))
    dp={0:1}
    for i in range(8):
        nxt={}
        for mask,count in dp.items():
            for j in range(8):
                if mask>>j&1 or j==i or (i,j) in ((0,1),(1,0)):continue
                target=mask|1<<j;nxt[target]=nxt.get(target,0)+count
        dp=nxt
    lattice={(0,0):1}
    for x in range(9):
        for y in range(7):
            if (x,y)==(0,0) or y>x+1:continue
            lattice[x,y]=lattice.get((x-1,y),0)+lattice.get((x,y-1),0)
    lattice2=0
    for ups in itertools.combinations(range(14),6):
        x=y=0;valid=True
        for step in range(14):
            if step in ups:y+=1
            else:x+=1
            if y>x+1:valid=False;break
        lattice2+=valid
    subsets=sum(sum(c)%7==2 for c in itertools.combinations(range(1,13),6));table={(0,0):1}
    for value in range(1,13):
        nxt=dict(table)
        for (count,rem),n in table.items():
            if count<6:key=(count+1,(rem+value)%7);nxt[key]=nxt.get(key,0)+n
        table=nxt
    pairs=dict(circular10=(circular,circular2),restricted_derangements8=(perms,dp[255]),lattice8x6=(lattice[8,6],lattice2),subset_mod7=(subsets,table[6,2]))
    require(all(a==b for a,b in pairs.values()),'independent gold methods disagree');return pairs


def independent_score(final):
    # Separate native libc strtoll conversion, retaining literal wrapper order.
    s=final.strip(' \t\n\r\v\f')
    if len(s)>=6 and s.startswith('```') and s.endswith('```'):
        s=s[3:-3]
        if s.startswith('python\n'):s=s[7:]
        elif s.startswith('text\n'):s=s[5:]
        s=s.strip(' \t\n\r\v\f')
    for prefix,suffix,minimum in [('**','**',4),('`','`',2),('\\boxed{','}',9)]:
        if len(s)>=minimum and s.startswith(prefix) and s.endswith(suffix):s=s[len(prefix):-len(suffix)].strip(' \t\n\r\v\f')
    if not re.fullmatch('[+-]?[0-9]+',s):return None
    libc=ctypes.CDLL(None,use_errno=True);libc.strtoll.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_char_p),ctypes.c_int];libc.strtoll.restype=ctypes.c_longlong
    end=ctypes.c_char_p();ctypes.set_errno(0);number=libc.strtoll(s.encode('ascii'),ctypes.byref(end),10)
    if ctypes.get_errno()==errno.ERANGE or end.value!=b'':return None
    return str(number)


def identity(path):
    path=Path(path);require(path.suffix.lower()!='.gguf','no model identity/stat')
    info=path.stat();record=dict(path=str(path),sha256=sha(path),bytes=info.st_size,mode=stat.S_IMODE(info.st_mode))
    if path.is_symlink():record.update(symlink=True,resolved_path=str(path.resolve()))
    return record


def archive(output):
    require(sha(PREP/'plan.json')==PLAN_SHA and sha(PREP/'root-release-proof.json')==RELEASE_SHA,'ROOT frozen plan/release authority')
    require(sha(PREP/'analyze_reasoning_budget.py')==ANALYZER_SHA and sha(PREP/'run_reasoning_budget.py')==DRIVER_SHA,'reviewed analyzer/driver authority')
    sys.path.insert(0,str(PREP));import analyze_reasoning_budget as analysis;import run_reasoning_budget as driver
    require(Path(analysis.__file__).resolve()==PREP/'analyze_reasoning_budget.py' and Path(driver.__file__).resolve()==PREP/'run_reasoning_budget.py','literal prepared imports')
    plan=analysis.load(PREP/'plan.json');release=analysis.load(PREP/'root-release-proof.json');guard=analysis.load(PREP/'pressure/result.json')
    analysis.validate_release(release,guard);require(guard==dict(status='complete',child_returncode=3),'retain expected first-attempt rc3')
    require(type(release['peak_wired_compressor_gib']) in (int,float) and math.isfinite(release['peak_wired_compressor_gib']) and release['peak_wired_compressor_gib']>=0,'typed pressure peak')
    driver.validate_plan(plan,PREP/'plan.json');driver.require_grant(plan,PREP/'plan.json',PLAN_SHA);analysis.validate_pressure(plan,PLAN_SHA,release);driver.check_assets(plan)
    index=analysis.load(PREP/'freeze-index.json')
    for name,pin in index['files'].items():require(identity(PREP/name)==dict(path=pin['original_path'],sha256=pin['sha256'],bytes=pin['bytes'],mode=pin['mode']),'original prepared freeze pin differs')
    output.mkdir(parents=True,exist_ok=False)
    checked=analysis.analyze(PREP/'plan.json',PLAN_SHA,PREP/'root-release-proof.json',RELEASE_SHA,output/'independent-offline-analysis.json')
    require(checked==analysis.load(PREP/'offline-analysis-v1.json'),'independent rerun differs from retained original analysis')
    calculated=golds();require({name:str(a) for name,(a,b) in calculated.items()}==analysis.GOLDS,'independent problem gold differs')
    independent=[]
    for arm in plan['arms']:
        directory=Path(arm['results_dir']);rows=analysis.jsonl(directory/'results.jsonl')
        for row in rows:
            parsed=independent_score(row['final']) if row['think_closed'] else None
            status='INVALID' if parsed is None else 'CORRECT' if parsed==str(calculated[row['case']][0]) else 'INCORRECT'
            require(status==row['status'] and (parsed or '')==row['parsed_answer'],'independent libc score differs')
            trace=analysis.jsonl(directory/row['token_trace']);forced=[t for t in trace if t['policy_active']]
            require(len(forced)==int(row['forced_think_close']),'independent force count')
            if forced:
                t=forced[0];require(t['step']==1024 and t['generated_thinking_tokens']==1024 and t['token_id']==128822 and t['rng_before']==t['rng_after'] and trace[1025]['inside_thinking'] is False,'independent forced-close identity/phase/RNG')
            independent.append(dict(arm=arm['name'],case=row['case'],status=status,parsed_answer=parsed,expected=calculated[row['case']][0],cap_hit=row['cap_hit'],stopped=row['stopped'],generated_tokens=row['generated_tokens'],reasoning_tokens=row['reasoning_tokens'],forced_steps=[t['step'] for t in forced]))
    copied={};external={}
    def copy_file(source,target):
        source=Path(source);require(not source.is_symlink(),'literal archived files cannot be symlinks');pin=identity(source)
        dest=output/target;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        now=identity(dest);require(all(now[k]==pin[k] for k in ('sha256','bytes','mode')),'literal copy identity differs');copied[str(target)]=pin
    def copy_tree(source,target):
        source=Path(source)
        for path in sorted(source.rglob('*')):
            if path.is_file():copy_file(path,Path(target)/path.relative_to(source))
    copy_tree(PREP,'prepared')
    for arm in plan['arms']:copy_tree(arm['results_dir'],Path('first-attempts')/arm['name'])
    copy_tree(plan['reference_dir'],'historical-native')
    cpu=Path(plan['cpu_proof']).parent;copy_tree(cpu,'cpu-proof-and-original-refusal')
    proof=analysis.load(plan['cpu_proof'])
    for name,pin in proof['sources'].items():require(sha(name)==pin,'CPU source proof differs');copy_file(name,Path('cpu-tested-sources')/Path(name).name)
    for name in ('stdout_log','first_stdout_log','fixture_result'):copy_file(proof[name],Path('cpu-logs')/Path(proof[name]).name)
    for path in Path(proof['result']['fixtures']).rglob('*'):
        if path.is_file():external[str(path)]=identity(path)
    buildpath=Path(plan['build_manifest']);build=analysis.load(buildpath);failed=Path(build['first_failed_manifest'])
    require(sha(failed)==build['first_failed_manifest_sha256'] and sha(failed.parent/'compile.log')==build['first_failed_compile_log_sha256'],'first compiler refusal pins differ')
    for source,label in [(buildpath.parent,'build-v2'),(failed.parent,'build-first-refusal')]:
        for path in sorted(source.rglob('*')):
            if path.is_file():
                if path.name=='reasoning-budget-probe' or path.suffix in ('.o','.dylib'):external[str(path)]=identity(path)
                else:copy_file(path,Path(label)/path.relative_to(source))
    for path,pin in plan['assets'].items():
        require(identity(path)['sha256']==pin['sha256'],'prepared source/runtime pin changed')
        if Path(path).suffix=='.o' or '/python@' in path or path==plan['binary']:external[path]=identity(path)
        else:copy_file(path,Path('pinned-assets')/Path(path).relative_to('/'))
    for path,pin in build['sdk_sha256'].items():require(sha(path)==pin,'build SDK pin changed');external[path]=identity(path)
    compiler=build['compiler'];require(sha(compiler['path'])==compiler['sha256'],'selected build compiler pin changed');external[compiler['path']]=identity(compiler['path'])
    copy_file(Path(__file__),'archive-source.py')
    native,candidate=checked['arms'][:2];summary=dict(status='COMPLETE_FIRST_ATTEMPTS',root_release_sha256=RELEASE_SHA,original_guard_rc=3,independent_analyzer_matches_original=True,independent_two_method_golds={k:list(v) for k,v in calculated.items()},independent_libc_scores=independent,native_generated_tokens=native['generated_tokens'],budget_generated_tokens=candidate['generated_tokens'],token_reduction_fraction=1-candidate['generated_tokens']/native['generated_tokens'],native_completed_correct=native['completed_correct'],budget_completed_correct=candidate['completed_correct'],native_caps=native['cap_hits'],budget_caps=candidate['cap_hits'],wrong_completed_lattice_retained=True,subset_capped_outside_thinking_retained=True,raw_trace_meaning='UTF8 generated text reconstructed from token pieces, not full-vocabulary logit arrays',limitations=checked['limitations'],model_or_metadata_opened=False,native_or_helper_build=False)
    put(output/'compact-summary.json',summary);put(output/'external-references.json',dict(files=external,all_streaming_sha_verified=True,literal_archive_files_are_separate=True))
    put(output/'README.json',dict(prepared=str(PREP),immutable_parent='e1b80a601c8795dc0f0e10f3c46935959dd2d10e',original_first_attempts_retained=True,original_build_and_prepare_failures_retained=True,archive_administrative_refusal='Initial workspace add --ignore-working-copy created metadata then refused checkout; materialized in private workspace and selected exact immutable parent. No study artifact changed.',interpretation='Token reduction with identical two correct completions on four calibration cases; one INVALID cap became completed INCORRECT, and another still capped outside thinking. No equal-quality, generalization, quantization causality, KL/NLL or latency claim.'))
    files={str(p.relative_to(output)):identity(p) for p in sorted(output.rglob('*')) if p.is_file()};dirs=sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_dir())
    put(output/'archive-index.json',dict(files=files,directories=dirs,literal_file_count=len(files),copied_source_map=copied,external_reference_count=len(external),index_not_self_hashed=True))
    for name,pin in files.items():require(identity(output/name)==pin,'final archive file changed')
    require(sha(PREP/'plan.json')==PLAN_SHA and sha(PREP/'root-release-proof.json')==RELEASE_SHA,'study authority changed during archive')
    print(json.dumps(dict(output=str(output),index_sha256=sha(output/'archive-index.json'),compact_sha256=sha(output/'compact-summary.json'),literal_files=len(files)+1,external_references=len(external),model_or_metadata_opened=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();archive(a.output.resolve())
