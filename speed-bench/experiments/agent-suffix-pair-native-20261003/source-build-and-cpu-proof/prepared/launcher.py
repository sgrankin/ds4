OUTPUT='/Volumes/Models/ds4/.workspaces/idea-agent-suffix-pair/speed-bench/ideas/artifacts/agent-suffix-pair-native-prepared-20261003-v1'
MANIFEST_SHA='7f9b80a35e67b9caefe959193555cbce084e9983b9ef695d4e42ebb910653cb6'
COMMAND_SHA='bfebb2fbe5178eaffb4a750d81759ddb669693bcfb94cbe9bb1d07a3a86e87f5'
import importlib.util,json,os,subprocess,sys
from pathlib import Path
p=Path(OUTPUT)
spec=importlib.util.spec_from_file_location("frozen_driver",p/"run_agent_suffix_pair_native.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.sha(p/"prepared-command.json")==COMMAND_SHA
m.verify(p,MANIFEST_SHA)
if sys.argv[1:]!=["--execute"]:
 assert sys.argv[1:]==[]
 print("PREFLIGHT_PASS; no model metadata or process")
else:
 env={k:v for k,v in os.environ.items() if not k.startswith("DS4")}
 raise SystemExit(subprocess.run(json.loads((p/"prepared-command.json").read_text())["command"],cwd=m.ROOT,env=env).returncode)
