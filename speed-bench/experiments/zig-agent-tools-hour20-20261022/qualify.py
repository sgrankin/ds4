import hashlib,json,pathlib,subprocess,time
root=pathlib.Path('/Volumes/Models/ds4');out=root/'.workspaces/zig-agent-tools-20261022';rows=[]
files=['build.zig','Makefile','ds4_agent.c','ds4_agent_tools.h','ds4_tool_text.h','zig/agent_tools.zig','tests/ds4_agent_test.c']
def hashes():return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}
source_before=hashes();(out/'source-before.json').write_text(json.dumps(source_before,indent=2)+'\n')
def run(label,command):
    t=time.perf_counter()
    with (out/'build'/f'{label}.stdout.log').open('wb') as stdout,(out/'build'/f'{label}.stderr.log').open('wb') as stderr:
        p=subprocess.run(command,cwd=root,stdout=stdout,stderr=stderr)
    r=dict(label=label,command=command,rc=p.returncode,seconds=time.perf_counter()-t)
    rows.append(r);(out/'build/results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(r),flush=True)
    if p.returncode:raise SystemExit(p.returncode)
for native in (False,True):
    arm='native' if native else 'c'
    for mode in ('ReleaseFast','Debug'):
        label=arm+('-debug' if mode=='Debug' else '');prefix=out/label
        cmd=['./tools/zig-build','-Dnative='+str(native).lower(),'-Doptimize='+mode,'--cache-dir',str(out/(arm+'-local-cache')),'--prefix',str(prefix)]
        cmd+=['install','agent-test'] if mode=='ReleaseFast' else ['agent-test']
        run(label+'-build',cmd)
        run(label+'-agent',[str(prefix/'bin/ds4_agent_test')])
s={'runs':rows,'source_sha256':source_before,'source_after_sha256':hashes(),'binaries':{}}
assert s['source_sha256']==s['source_after_sha256']
for p in sorted(out.glob('*/bin/*')):
    if p.is_file():s['binaries'][str(p.relative_to(out))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(out/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
print('ALL BUILDS AND FOUR AGENT FIXTURES PASS; NO GPU/MODELS',flush=True)
