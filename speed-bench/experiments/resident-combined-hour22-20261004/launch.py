#!/usr/bin/env python3
"""Resident routing plus fixed READ/WRITE/FINAL reasoning policies; ROOT launches."""
import argparse
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DRIVER = ROOT / 'speed-bench/experiments/final-thinking-hour20-20261004/run.py'
ARMS = [('native-before',0),('subs3',3),('subs1',1),('native-after',0)]


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--server',type=Path,default=ROOT/'ds4-server')
    p.add_argument('--visible',type=Path,default=ROOT/'.workspaces/server-agent-day-20261004/confirmation-visible')
    p.add_argument('--global-cap',type=int,default=2048)
    p.add_argument('--max-wall-seconds',type=int,default=1500)
    p.add_argument('--case',action='append',choices=['ledger','lru'])
    p.add_argument('--arm',action='append',choices=[a for a,b in ARMS])
    args = p.parse_args()
    parent = subprocess.check_output(['ps','-p',str(os.getppid()),'-o','command='],text=True)
    if 'speed-bench/ideas/pressure_run.py' not in parent:
        p.error('launch as the direct child of MAIN speed-bench/ideas/pressure_run.py')
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    shaders=dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                           (ROOT/'ds4_metal.m').read_text()))
    total_started=time.monotonic()
    records=[]
    for name,substitutions in ARMS:
        if args.arm and name not in args.arm:
            continue
        remaining=args.max_wall_seconds-(time.monotonic()-total_started)
        if remaining<=0:
            records.append({'arm':name,'not_run':'whole experiment wall bound exhausted'})
            break
        out=args.output/name
        out.mkdir()
        env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))}
        env.update({k:str(ROOT/v) for k,v in shaders.items()})
        env.update(DS4_STATE_CACHE_DIR=str(out/'cache'),DS4_STATE_CACHE_HOT_MB='0',
            DS4_STATE_CACHE_SSD_MB='1024',DS4_SERVER_RESPONSE_STORE_DIR=str(out/'store'),
            DS4_METAL_V41_RESIDENT_ROUTE_GAP='0',DS4_METAL_V41_SELECTED_TILE='64',
            DS4_METAL_V41_SELECTED_LIMIT='1024',DS4_SERVER_CONTEXT_SUFFIX_REUSE='0')
        route_settings = {'enabled':bool(substitutions),'gap':.90 if substitutions else 0,
            'max_mass':.25,'max_rank':64,'max_substitutions':substitutions,
            'max_total_mass':.50,'policy':'replace','native_weights':bool(substitutions)}
        if substitutions:
            env.update(DS4_METAL_V41_RESIDENT_ROUTE_GAP='.90',
                DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS='.25',DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK='64',
                DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS=str(substitutions),
                DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS='.50',
                DS4_METAL_V41_RESIDENT_ROUTE_POLICY='replace',DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS='1')
        else:
            env['DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS']='0'
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
        command=[str(args.server.resolve()),'--model',str(ROOT/'gguf/DeepSeek-V4.1-Flash-Q2.gguf'),
            '--host','127.0.0.1','--port',str(port),'--ssd-streaming',
            '--ssd-streaming-cache-experts','64GB','--ctx','8192','--prefill-chunk','512',
            '--batched-session','1','--tokens',str(args.global_cap)]
        save(out/'execution.json',{'command':command,'cwd':str(ROOT),
            'environment':{k:v for k,v in env.items() if k.startswith('DS4')},
            'arm':name,'protocol':'natural','write_budget':128,'final_policy':'budget1','global_output_cap_per_task':args.global_cap,
            'read_write_effort':'high','final_effort':'high','route_settings':route_settings,
            'temperature':0,'seed_ignored':True,'new_cache_per_arm':True,
            'scope':'Resident profile varies; READ high/0, WRITE high/128, FINAL high/1, suffix off, stable task instructions/tools.'})
        started=time.monotonic()
        with (out/'server.stdout.log').open('wb') as stdout,(out/'server.stderr.log').open('wb') as stderr:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            record={'arm':name,'write_budget':128,'final_policy':'budget1','returncode':None}
            try:
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError('server startup exit '+str(proc.returncode))
                    if time.monotonic()-started>=120:
                        raise TimeoutError('server startup timeout')
                    conn=http.client.HTTPConnection('127.0.0.1',port,timeout=1)
                    try:
                        conn.request('GET','/v1/models');response=conn.getresponse();response.read()
                        if response.status==200:break
                    except OSError:pass
                    finally:conn.close()
                    time.sleep(.2)
                save(out/'startup.json',{'seconds':time.monotonic()-started,'pid':proc.pid,'port':port})
                runner=[sys.executable,'-B',str(DRIVER),'--base-url',f'http://127.0.0.1:{port}',
                    '--output',str(out/'checks'),'--visible',str(args.visible.resolve()),
                    '--arm',name,'--final-policy','budget1','--global-cap',str(args.global_cap)]
                for case in args.case or []:runner+=['--case',case]
                remaining=args.max_wall_seconds-(time.monotonic()-total_started)
                if remaining<=0:raise TimeoutError('whole experiment wall bound exhausted')
                completed=subprocess.run(runner,cwd=ROOT,timeout=remaining)
                record.update(returncode=completed.returncode,argv=runner,seconds=time.monotonic()-started)
            except Exception as exc:
                record.update(error=type(exc).__name__+': '+str(exc),seconds=time.monotonic()-started)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=2)
                record['server_returncode_after_shutdown']=proc.returncode
                save(out/'result.json',record)
                records.append(record)
                save(args.output/'runs.json',records)
                print(json.dumps(record),flush=True)
        if record.get('error'):break
    return 0 if records and all(r.get('returncode') in [0,3] for r in records) else 2


if __name__=='__main__':
    raise SystemExit(main())
