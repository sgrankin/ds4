"""Reuse the qualification oracle for archived topology tasks only."""
import importlib.util
import json
from pathlib import Path
import sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(root):
    oracle_path = Path(__file__).resolve().parent.parent / "2026-11-04-resident-agent-qualification/check-agent.py"
    oracle = load("qualification_oracle", oracle_path)
    results = {}
    for arm in ["published", "current_gpu_off", "gpu_refresh4", "gpu_refresh8", "gpu_refresh16"]:
        dest = root / arm
        try:
            result = oracle.check_task(load(arm, dest / "workdir/layers.py"), "layers")
            result["agent_exit_code"] = json.loads((dest / "exit.json").read_text())["exit_code"]
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        results[arm] = result
    (root / "oracle-result.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return int(any(r.get("error") or r.get("failed_evaluations") or r.get("agent_exit_code") for r in results.values()))


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
