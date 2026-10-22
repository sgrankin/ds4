#!/usr/bin/env python3
"""Run one existing fixture with explicit shader paths and clean DS4 settings."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
root=Path.cwd()
env={key:value for key,value in os.environ.items() if not key.startswith(('DS4','PT_'))}
for key,source in re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',(root/'ds4_metal.m').read_text()):
    env[key]=str(root/source)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--settings",type=json.loads,default={})
parser.add_argument("command",nargs=argparse.REMAINDER)
args=parser.parse_args()
env.update(args.settings)
command=args.command
command[0]=str(Path(command[0]).resolve())
os.execve(command[0],command,env)
