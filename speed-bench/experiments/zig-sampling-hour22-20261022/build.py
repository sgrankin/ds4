import hashlib,json,pathlib,subprocess,time
root=pathlib.Path('/Volumes/Models/ds4');out=root/'.workspaces/zig-sampling-20261022';rows=[]
files=['build.zig','Makefile','ds4.c','ds4_sampling.h','zig/sampling.zig','tests/test_sampling.c']
def hashes():return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}
source_before=hashes();(out/'source-before.json').write_text(json.dumps(source_before,indent=2)+'\n')
def run(label,command,required=True):
    t=time.perf_counter()
    with (out/'build'/f'{label}.stdout.log').open('wb') as stdout,(out/'build'/f'{label}.stderr.log').open('wb') as stderr:
        p=subprocess.run(command,cwd=root,stdout=stdout,stderr=stderr)
    row=dict(label=label,command=command,rc=p.returncode,seconds=time.perf_counter()-t);rows.append(row)
    (out/'build/results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
    if required and p.returncode:raise SystemExit(p.returncode)
for native in [False,True]:
    arm='native' if native else 'c'
    for mode in ['ReleaseFast','Debug']:
        label=arm+('-debug' if mode=='Debug' else '');prefix=out/label
        command=['./tools/zig-build','-Dnative='+str(native).lower(),'-Doptimize='+mode,'--cache-dir',str(out/(arm+'-local-cache')),'--prefix',str(prefix)]
        command+=['install','sampling-test'] if mode=='ReleaseFast' else ['sampling-test']
        run(label+'-build',command)
        run(label+'-sampling',[str(prefix/'bin/test-sampling')],required=False)
s={'policy':'Primary: actual Make RF engine fast math; test source finite override. Debug engine/test Debug; native sampler matches selected host object.','runs':rows,'source_sha256':source_before,'source_after_sha256':hashes(),'binaries':{}}
assert s['source_sha256']==s['source_after_sha256']
for p in sorted(out.glob('*/bin/*')):
    if p.is_file():s['binaries'][str(p.relative_to(out))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
s['all_fixtures_pass']=all(r['rc']==0 for r in rows if r['label'].endswith('-sampling'))
(out/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
print('Primary build/fixture sequence finished; see individual fixture return codes. NO GPU/MODELS',flush=True)
raise SystemExit(0 if s['all_fixtures_pass'] else 1)
