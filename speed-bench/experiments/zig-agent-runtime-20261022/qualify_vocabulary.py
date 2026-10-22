import hashlib, json, pathlib, subprocess, time
root=pathlib.Path(__file__).resolve().parents[3]; out=root/'.workspaces/zig-agent-context-vocabulary-rerun'
out.mkdir()
model=root/'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
results=[]
for label in ['context-control','context','context-debug']:
    here=out/label;here.mkdir()
    binary=root/'.workspaces/zig-agent-qualification'/label/'bin/test-agent-live-context'
    cmd=[str(binary),str(model),str(here)]
    t=time.monotonic()
    with (here/'stdout').open('wb') as so,(here/'stderr').open('wb') as se:
        r=subprocess.run(cmd,stdout=so,stderr=se,timeout=90)
    row=dict(label=label,command=cmd,returncode=r.returncode,seconds=time.monotonic()-t,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    results.append(row);print(json.dumps(row),flush=True)
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    if r.returncode: raise SystemExit(1)
