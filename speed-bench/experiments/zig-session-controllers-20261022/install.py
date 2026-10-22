import pathlib as p,shutil,subprocess
r=p.Path.cwd(); w=r/'.workspaces/zig-session-integration-20261022'
def apply(owner,name):
 f=r/'.workspaces'/owner/name
 subprocess.run(['patch','--no-backup-if-mismatch','-p1','-i',str(f)],check=True)
def copy(owner,src,dst):
 shutil.copy2(r/'.workspaces'/owner/src,r/dst)
co='zig-session-coordinator-20261022'; su='zig-session-suffix-20261022'; ba='zig-session-batch-20261022'; sp='zig-session-speculation-20261022'; ro='zig-engine-route-services-20261022'; vi='zig-engine-vision-runtime-20261022'
copy(co,'ds4_session_control.h','ds4_session_control.h')
copy(ba,'ds4_session_batch.h','ds4_session_batch.h')
for owner,files in [(co,['coordinator.zig']),(su,['suffix_transaction.zig']),(ba,['batch_transaction.zig','coordinator.zig']),(sp,['session_speculation.zig']),(ro,['route_services.zig','route_oracle.zig'])]:
 for f in files:copy(owner,f,'zig/engine/deepseek41/'+f)
copy(ro,'resident_route_entry.zig','zig/resident_route_entry.zig')
copy(vi,'vision_runtime.zig','zig/engine/vision_runtime.zig');copy(vi,'ds4_engine_vision.h','ds4_engine_vision.h')
for owner,names in [(co,['control-record.patch','activation.patch']),(su,['activation.patch']),(ba,['activation.patch']),(sp,['view-statistics.patch','activation.patch','graph-adapter.patch','batch-visibility.patch','selection-visibility.patch','cycle-composition.patch']),(ro,['resident-math.patch','executor.zig.patch','moe.zig.patch','prefill.zig.patch','graph.patch','activation.patch']),(vi,['activation.patch'])]:
 for name in names:apply(owner,name)
f=r/'zig/engine/deepseek41/graph_api.h';s=f.read_text();at=s.rfind('#endif'); assert at>=0
parts=[(r/'.workspaces'/o/'api-prototypes.h').read_text() for o in [co,su,ba,sp]]
parts+=['#include "../../../ds4_engine_vision.h"\n']
f.write_text(s[:at]+'\n'.join(parts)+'\n'+s[at:])
f=r/'zig/engine/graph.zig';s=f.read_text();f.write_text(s+'\n'+'\n'.join((r/'.workspaces'/o/'entry-snippet.zig').read_text() for o in [co,su,ba,sp]))
f=r/'zig/engine/input.zig';f.write_text(f.read_text()+'\n'+(r/'.workspaces'/vi/'entry-snippet.zig').read_text())
f=r/'build.zig';s=f.read_text();anchor='"-DDS4_NATIVE_SESSION_SAMPLING=1"';assert s.count(anchor)==1
s=s.replace(anchor,anchor+', '+', '.join('"-D'+v+'=1"' for v in ['DS4_NATIVE_SESSION_COORDINATOR','DS4_NATIVE_SESSION_SUFFIX','DS4_NATIVE_SESSION_BATCH','DS4_NATIVE_SESSION_SPECULATION','DS4_NATIVE_ENGINE_ROUTE_SERVICES','DS4_NATIVE_ENGINE_VISION_RUNTIME']))
s=s.replace('"zig/resident_route.zig", "resident_route_c", api','"zig/resident_route_entry.zig", "gguf_c", api');f.write_text(s)
