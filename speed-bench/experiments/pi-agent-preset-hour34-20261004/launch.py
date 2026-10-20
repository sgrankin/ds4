#!/usr/bin/env python3
"""Actual ds4-local persistent-fast plus isolated installed Pi CLI; ROOT launches."""
import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PRACTICAL=ROOT/'speed-bench/experiments/fresh-clm-coding-hour32-20261004/practical'


def save(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    parent=subprocess.check_output(['/bin/ps','-p',str(os.getppid()),'-o','command='],text=True)
    if 'speed-bench/ideas/pressure_run.py' not in parent:
        p.error('launch as the direct child of MAIN pressure_run.py')
    project,agent=out/'project',out/'pi-agent'; project.mkdir(); agent.mkdir()
    task=next(t for t in json.loads((PRACTICAL/'spec.json').read_text())['tasks'] if t['key']=='dedupe')
    save(project/'settings.json',task['config'])
    (out/'settings.before.json').write_bytes((project/'settings.json').read_bytes())
    prompt=task['prompt'].replace('config.json',str(project/'settings.json')).replace('solution.py',str(project/'solution.py'))
    prompt=prompt.replace('with whole=true and raw=true','with path only and no offset or limit')
    (out/'prompt.txt').write_text(prompt); (out/'system.txt').write_text(task['system'])
    save(agent/'settings.json',{'retry':{'enabled':False,'provider':{'maxRetries':0}},
                               'compaction':{'enabled':False}})
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1',0)); port=reservation.getsockname()[1]
    save(agent/'models.json',{'providers':{'local-ds4':{
        'baseUrl':f'http://127.0.0.1:{port}/v1','api':'openai-responses','apiKey':'local',
        'models':[{'id':'ds4','name':'DS4 local','reasoning':True,'input':['text'],
            'contextWindow':8192,'maxTokens':2048,
            'thinkingLevelMap':{'off':'none'},
            'cost':{'input':0,'output':0,'cacheRead':0,'cacheWrite':0}}]}}})
    env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_','PI_'))}
    env.update(DS4_STATE_CACHE_DIR=str(out/'cache'),DS4_STATE_CACHE_HOT_MB='0',
               DS4_STATE_CACHE_SSD_MB='4096',DS4_SERVER_RESPONSE_STORE_DIR=str(out/'store'),
               PI_CODING_AGENT_DIR=str(agent))
    command=[str(ROOT/'ds4-local'),'--server','persistent-fast','--model',
        str(ROOT/'gguf/DeepSeek-V4.1-Flash-Q2.gguf'),'--port',str(port),'--tokens','2048']
    cli=['/opt/homebrew/bin/pi','--provider','local-ds4','--model','ds4','--api-key','local',
        '--thinking','off','--tools','read,write','--no-session','--mode','json',
        '--no-extensions','--no-skills','--no-prompt-templates','--no-themes','--no-context-files',
        '--system-prompt',task['system'],'-p',prompt]
    save(out/'execution.json',{'server_argv':command,'server_cwd':str(ROOT),'pi_argv':cli,
        'pi_cwd':str(project),'environment':{k:v for k,v in env.items() if k.startswith(('DS4','PI_'))},
        'scope':'Actual installed Pi CLI compatibility/first-write coding; no latency comparison.'})
    started=time.monotonic(); deadline=started+600; result={'returncode':None}
    with (out/'server.stdout.log').open('wb') as stdout,(out/'server.stderr.log').open('wb') as stderr:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        try:
            while True:
                if proc.poll() is not None: raise RuntimeError('server startup exit '+str(proc.returncode))
                if time.monotonic()>=min(started+120,deadline): raise TimeoutError('server startup timeout')
                conn=http.client.HTTPConnection('127.0.0.1',port,timeout=1)
                try:
                    conn.request('GET','/v1/models'); response=conn.getresponse(); response.read()
                    if response.status==200: break
                except OSError: pass
                finally: conn.close()
                time.sleep(.2)
            save(out/'startup.json',{'seconds':time.monotonic()-started,'pid':proc.pid,'port':port})
            with (out/'pi.events.jsonl').open('wb') as events,(out/'pi.stderr.log').open('wb') as pi_stderr:
                completed=subprocess.run(cli,cwd=project,env=env,stdout=events,stderr=pi_stderr,
                                         timeout=max(.1,deadline-time.monotonic()))
            save(out/'pi-process.json',{'argv':cli,'cwd':str(project),'returncode':completed.returncode})
            with (out/'check.stdout').open('wb') as check_stdout,(out/'check.stderr').open('wb') as check_stderr:
                result['returncode']=subprocess.run([sys.executable,'-B',str(HERE/'check.py'),'--output',str(out)],
                    cwd=ROOT,stdout=check_stdout,stderr=check_stderr,timeout=10).returncode
        except Exception as exc: result['error']=type(exc).__name__+': '+str(exc)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=2)
            result.update(seconds=time.monotonic()-started,server_returncode_after_shutdown=proc.returncode)
            save(out/'result.json',result)
    return result['returncode'] if result['returncode'] is not None else 2


if __name__=='__main__': sys.exit(main())
