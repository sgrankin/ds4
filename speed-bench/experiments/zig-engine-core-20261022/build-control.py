from pathlib import Path
import hashlib,json,subprocess,sys
w=Path('.workspaces/zig-engine-core-20261022');paths=[Path('build.zig'),Path('zig/engine/graph.zig')];saved={p:p.read_bytes() for p in paths}
try:
 p=paths[0];s=p.read_text();assert s.count(', "-DDS4_NATIVE_ENGINE_CORE=1"')==1;p.write_text(s.replace(', "-DDS4_NATIVE_ENGINE_CORE=1"','',1))
 p=paths[1];s=p.read_text();entry=Path('speed-bench/experiments/zig-engine-core-20261022/core_entry.zig.inc').read_text();assert s.endswith(entry);p.write_text(s[:-len(entry)])
 (w/'integration/control-selection.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
 subprocess.run([sys.executable,'-B','.workspaces/zig-server-input-20261022/record.py',str(w/'integration/control-fixture-build'),'tools/zig-build','graph-test','-Dnative=true','-Doptimize=ReleaseFast','--prefix',str(w/'control')],check=True)
finally:
 for p,b in saved.items():p.write_bytes(b)
