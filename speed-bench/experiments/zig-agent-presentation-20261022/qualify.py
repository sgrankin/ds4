#!/usr/bin/env python3
"""Run the existing terminal/agent fixture with an explicit artifact directory."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path.cwd()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("binary", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
binary = str(args.binary.resolve())
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
spec = importlib.util.spec_from_file_location("terminal_fixture", ROOT / "tests/ds4_agent_terminal_test.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
fixture.check_fixtures(binary, output)
fixture.check_hints(output)
fixture.check_pty(binary, output)
screen = fixture.pyte.Screen(80, 24)
fixture.pyte.Stream(screen).feed((output / "presentation-code.ansi").read_text().replace("\n", "\r\n"))
visible = "\n".join(screen.display)
assert "still old */ if (ok) x = 7;" in visible
assert "caf\u00e9" in visible and "\u4e2d" in visible
(output / "presentation-screen.txt").write_text(visible + "\n")
result = {
    "binary": binary,
    "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
    "all_pass": True,
    "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(output.iterdir()) if path.is_file()},
}
(output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print("Production agent, terminal, hints, PTY and code screen: PASS")
