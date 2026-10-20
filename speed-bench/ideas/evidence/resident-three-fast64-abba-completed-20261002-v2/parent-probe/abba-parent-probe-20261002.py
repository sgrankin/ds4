import os,json,subprocess,sys
from pathlib import Path
p=Path(__file__).with_suffix(".json")
actual=subprocess.check_output(["ps","-p",str(os.getppid()),"-o","command="],text=True,timeout=2).strip()
d={"parent_ps_command":actual,"child_sys_executable":sys.executable,"model_metadata_used":False}
p.write_text(json.dumps(d,indent=2)+"\n")
print(json.dumps(d))
