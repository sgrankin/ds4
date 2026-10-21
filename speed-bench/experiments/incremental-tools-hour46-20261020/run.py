#!/usr/bin/env python3
"""First-output tool discovery/retrieval ABBA. Run once under pressure_run.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]

def save(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')

def text(body):
    return ''.join(p.get('text', '') for item in body.get('output', []) if item.get('type') == 'message'
                   for p in item.get('content', []) if p.get('type') == 'output_text').strip()

def stop(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=3)
    return proc.returncode if proc else None

def http(base, name, payload, out, timeout):
    save(out/(name+'.request.json'), payload)
    started = time.monotonic()
    req = urllib.request.Request(base+'/responses', data=json.dumps(payload).encode(),
        headers={'Content-Type':'application/json', 'Connection':'close'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status, wire = response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        status, wire = exc.code, exc.read().decode(errors='replace')
    row = {'status':status, 'seconds':time.monotonic()-started, 'wire':wire}
    save(out/(name+'.wire.json'), row)
    if status != 200: raise RuntimeError(f'{name}: HTTP {status}; first wire retained')
    body = json.loads(wire)
    save(out/(name+'.response.json'), body)
    return body, row['seconds']

def one_call(body, kind):
    calls = [x for x in body.get('output', []) if x.get('type') in ('tool_search_call','function_call')]
    if len(calls) != 1 or calls[0]['type'] != kind:
        raise ValueError(f'expected one {kind}; actual first output retained')
    if body.get('status') != 'completed': raise ValueError('response did not complete')
    return calls[0]

def scenario(args, base, common, fixture, out, parent_check=False):
    out.mkdir()
    history = [{'role':'system','content':fixture['system']}, {'role':'user','content':fixture['prompt']}]
    result = {'case':fixture['case'], 'history_mode':fixture['history_mode'], 'pass':False, 'error':None, 'requests':[]}
    pending_input = history
    previous_id = None
    origin = time.monotonic()
    try:
        for step in range(1, 4):
            payload = common | {'input':history, 'prompt_cache_key':fixture['case']}
            if previous_id and fixture['history_mode'] == 'parent':
                payload.update(input=pending_input, previous_response_id=previous_id)
            body, seconds = http(base, f'{step:02d}', payload, out, args.timeout)
            previous_id = body['id']
            result['requests'].append({'step':step,'seconds':seconds,'usage':body.get('usage'),
                'status':body.get('status'), 'output_types':[x.get('type') for x in body.get('output', [])]})
            history += body.get('output', [])
            if step == 1:
                call = one_call(body, 'tool_search_call')
                result['first_search_arguments'] = call.get('arguments')
                pending_input = [{'type':'tool_search_output', 'call_id':call['call_id'],
                    'execution':'client','status':'completed','tools':fixture['discovered']}]
                history += pending_input
            elif step == 2:
                call = one_call(body, 'function_call')
                arguments = call['arguments']
                if isinstance(arguments, str): arguments = json.loads(arguments)
                if call['name'] != 'read_note' or call.get('namespace') != fixture['namespace']:
                    raise ValueError('wrong discovered tool name/namespace')
                if arguments != {'path':fixture['note_path']}:
                    raise ValueError('wrong read arguments; no unknown path executed')
                before = time.monotonic()
                note = Path(arguments['path']).read_text()
                result['tool_seconds'] = time.monotonic()-before
                result['actual_file_read'] = True
                result['read_sha256'] = hashlib.sha256(note.encode()).hexdigest()
                pending_input = [{'type':'function_call_output', 'call_id':call['call_id'], 'output':note}]
                history += pending_input
                result['parent_id_after_read_call'] = body['id']
                saved_output = history[-1]
            else:
                answer = text(body)
                result['answer'] = answer
                result['pass'] = body.get('status') == 'completed' and json.loads(answer) == fixture['expected']
        result['workflow_seconds'] = time.monotonic()-origin
        if parent_check:
            parent, seconds = http(base, '04-parent', common | {
                'previous_response_id':result['parent_id_after_read_call'], 'input':[saved_output],
                'prompt_cache_key':fixture['case']}, out, args.timeout)
            result['parent_check'] = {'seconds':seconds, 'usage':parent.get('usage'),
                'pass':parent.get('status') == 'completed' and json.loads(text(parent)) == fixture['expected']}
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['total_seconds'] = time.monotonic()-origin
    save(out/'result.json', result)
    return result

def arm(args, index, enabled, fixtures):
    label = 'default' if enabled is None else 'late' if enabled else 'native'
    out = args.output / f'{index:02d}-{label}'
    out.mkdir()
    env = {k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))}
    env.update(DS4_SERVER_CONTEXT_SUFFIX_REUSE='0',
        DS4_SERVER_RESPONSE_STORE_DIR=str(out/'responses'), DS4_METAL_V41_RESIDENT_ROUTE_GAP='0',
        DS4_METAL_V41_RESIDENT_PREFILL_GAP='0', DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS='0')
    if enabled is not None:
        env['DS4_SERVER_INCREMENTAL_TOOL_SCHEMAS'] = str(int(enabled))
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1',0)); port = reserve.getsockname()[1]
    command = [str(args.server), '--model',str(args.model),'--host','127.0.0.1','--port',str(port),
        '--ssd-streaming','--ssd-streaming-cache-experts','64GB','--ctx','16384','--prefill-chunk','2048',
        '--batched-session','1','--mtp-draft','1']
    save(out/'execution.json', {'command':command,'cwd':str(ROOT),
        'env':{k:v for k,v in env.items() if k.startswith('DS4')},
        'server_sha256':hashlib.sha256(args.server.read_bytes()).hexdigest(),
        'application_ssd_kv_cache':'disabled', 'timing':'workflow includes requests and actual file read; startup separate'})
    proc=None; result={'arm':index,'enabled':enabled,'cases':[],'error':None}
    started=time.monotonic()
    try:
        with (out/'stdout.log').open('wb') as stdout, (out/'stderr.log').open('wb') as stderr:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            base=f'http://127.0.0.1:{port}/v1'
            deadline=started+120
            while True:
                if proc.poll() is not None: raise RuntimeError(f'server startup exit {proc.returncode}')
                if time.monotonic()>deadline: raise TimeoutError('startup deadline')
                try:
                    with urllib.request.urlopen(base+'/models',timeout=1) as response: models=json.load(response)['data']
                    if models: break
                except (OSError, ValueError, KeyError): pass
                time.sleep(.2)
            result['startup_seconds']=time.monotonic()-started
            common={'model':models[0]['id'],'temperature':0,'reasoning':{'effort':'none'},
                'max_output_tokens':128,'store':True,'tools':[{'type':'tool_search','execution':'client',
                'description':'Discover a tool by capability or name. Return the matching tool definitions.',
                'parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}}]}
            for fixture in fixtures:
                result['cases'].append(scenario(args,base,common,fixture,out/fixture['case'],
                    parent_check=args.parent_check and fixture['case']=='ordinary'))
    except Exception as exc: result['error']=f'{type(exc).__name__}: {exc}'
    finally: result['server_returncode']=stop(proc)
    result['arm_seconds']=time.monotonic()-started
    result['pass']=not result['error'] and len(result['cases'])==len(fixtures) and all(r['pass'] and not r['error'] for r in result['cases'])
    save(out/'result.json',result)
    print(json.dumps({'arm':index,'enabled':enabled,'pass':result['pass'],'error':result['error']}),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--server',type=Path,default=ROOT/'ds4-server')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--order',default='0110',choices=('0','1','0110','u','0uu0'),
        help='0=legacy, 1=incremental, u=unset flag to qualify the server default')
    p.add_argument('--cases',default='ordinary,favorable')
    p.add_argument('--timeout',type=float,default=600)
    p.add_argument('--parent-check',action='store_true',help='compatibility after timed workflow, excluded from workflow time')
    args=p.parse_args();args.model=args.model.resolve();args.server=args.server.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    fixtures=[]
    for case in args.cases.split(','):
        lines=12 if case=='ordinary' else 180
        work=args.output/'fixtures'/case;work.mkdir(parents=True)
        path=work/'note.json'; path.write_text(json.dumps({'archived_code':'MAPLE-4826','receipt':'READ-46-'+case})+'\n')
        namespace=None if case=='ordinary' else 'archive__'
        read={'type':'function','name':'read_note','description':'Read the exact local note file at path; returns its JSON contents.',
            'parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}}
        tools=[read]
        if case=='favorable':
            tools += [{'type':'function','name':f'catalog_metric_{i}',
                'description':('Return a historical catalog metric. This is unrelated to reading note files. '*3),
                'parameters':{'type':'object','properties':{'ticket':{'type':'string'}},'required':['ticket']}} for i in range(12)]
            discovered=[{'type':'namespace','name':namespace,'description':'Archived note functions and historical metrics.','tools':tools}]
        else: discovered=tools
        logs=''.join(f'historical audit {i:04d}: module={i%8} count={(i*37)%997:03d} state=reviewed immutable=true; no current decisions.\n' for i in range(lines))
        prompt=('Historical context only:\n<log>\n'+logs+'</log>\nCurrent key: SABLE-1937.\n'
            f'Task: use tool_search to discover read_note, then use that tool to read {path}. '
            'After actually reading it, reply with exactly one JSON object with archived_code and receipt from the file, '
            'and current_key from this request. Do not answer before the read. No prose or Markdown.')
        fixture={'case':case,'history_mode':'full' if case=='ordinary' else 'parent','system':'Follow the requested tool workflow. Use only defined tools, and do not invent tool results.',
            'prompt':prompt,'note_path':str(path),'namespace':namespace,'discovered':discovered,
            'expected':{'archived_code':'MAPLE-4826','receipt':'READ-46-'+case,'current_key':'SABLE-1937'}}
        save(work/'fixture.json',fixture);fixtures.append(fixture)
    def interrupt(signum,_frame): raise InterruptedError(f'signal {signum}')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,interrupt)
    rows=[]
    for i,flag in enumerate(args.order,1):
        rows.append(arm(args,i,None if flag=='u' else flag=='1',fixtures));save(args.output/'result.json',rows)
        if rows[-1]['error']: break
    return 0 if len(rows)==len(args.order) and all(r['pass'] for r in rows) else 3
if __name__=='__main__': raise SystemExit(main())
