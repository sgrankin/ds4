from pathlib import Path
import hashlib,json,os,subprocess,time
base=Path.cwd()/'.workspaces/zig-cli-frontend-20261022';out=base/'debug-real';out.mkdir()
command=next(r['command'] for r in json.loads((base/'real-2/results.json').read_text()) if r['label']=='sampled-zig');command[0]=str(base/'native-debug/bin/ds4')
start=time.monotonic();p=subprocess.run(command,capture_output=True,env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))},timeout=180)
(out/'stdout').write_bytes(p.stdout);(out/'stderr').write_bytes(p.stderr)
row=dict(command=command,returncode=p.returncode,seconds=time.monotonic()-start,identical_output=p.stdout==(base/'real-2/sampled-zig.stdout').read_bytes(),binary_sha256=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest());(out/'result.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
assert p.returncode==0 and row['identical_output']
