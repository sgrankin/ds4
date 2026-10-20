#!/usr/bin/env python3
"""First-attempt READ -> WRITE -> FINAL against an already guarded server."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PRACTICAL = ROOT / 'speed-bench/experiments/confirmation-hour5-20261003/practical'
TOOLS = [
    {'type':'function', 'name':'read', 'description':'Read the advertised config file in full.',
     'parameters':{'type':'object','properties':{'path':{'type':'string'},
        'whole':{'type':'boolean'},'raw':{'type':'boolean'}},
        'required':['path','whole','raw'],'additionalProperties':False}},
    {'type':'function', 'name':'write', 'description':'Write the complete first implementation to the advertised solution path.',
     'parameters':{'type':'object','properties':{'path':{'type':'string'},'content':{'type':'string'}},
        'required':['path','content'],'additionalProperties':False}}
]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def final_text(body):
    return ''.join(part.get('text','') for item in body.get('output',[])
        if item.get('type') == 'message' for part in item.get('content',[])
        if part.get('type') == 'output_text')


def grade(case, code, out):
    command = [sys.executable, '-B', str(PRACTICAL / 'grade.py'), case, str(code)]
    with (out/'grader.stdout').open('wb') as stdout, (out/'grader.stderr').open('wb') as stderr:
        try:
            result = subprocess.run(command, cwd=PRACTICAL, stdout=stdout, stderr=stderr,
                                    timeout=4, stdin=subprocess.DEVNULL)
            record = json.loads((out/'grader.stdout').read_text())
            record['returncode'] = result.returncode
            return record
        except Exception as exc:
            return {'correct':False,'reason':type(exc).__name__,'message':str(exc)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--visible', type=Path, default=ROOT/'.workspaces/server-agent-day-20261004/thinking-visible')
    p.add_argument('--arm', required=True)
    p.add_argument('--budget', type=int, choices=[0,256,512], required=True)
    p.add_argument('--global-cap', type=int, default=2048)
    p.add_argument('--case', action='append', choices=['join','window_argmax'])
    args = p.parse_args()
    if args.global_cap < 1:
        p.error('--global-cap must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    spec = json.loads((PRACTICAL/'spec.json').read_text())
    rows = []
    for task in spec['tasks']:
        case = task['key']
        if args.case and case not in args.case:
            continue
        out = args.output/case
        out.mkdir()
        project = args.visible.resolve()/case
        project.mkdir(parents=True, exist_ok=True)
        config, solution = project/'config.json', project/'solution.py'
        config_bytes = (json.dumps(task['config'], ensure_ascii=False, indent=2)+'\n').encode()
        if config.exists() and config.read_bytes() != config_bytes:
            raise ValueError('advertised config changed between arms')
        config.write_bytes(config_bytes)
        solution.unlink(missing_ok=True)
        prompt = task['prompt'].replace('config.json',str(config)).replace('solution.py',str(solution))
        save(out/'task.json', {'key':case,'prompt':prompt,'system':task['system'],
                              'config':task['config'],'tests_per_task':spec['tests_per_task']})
        pending_input = [{'role':'user','content':prompt}]
        previous_id = None
        calls, replies, errors, history = [], [], [], list(pending_input)
        read_count = write_count = output_tokens = reasoning_tokens = 0
        complete = False
        text = ''
        first_code = out/'first-solution.py'
        started = time.monotonic()
        for index, role in enumerate(['READ','WRITE','FINAL']):
            if output_tokens >= args.global_cap:
                errors.append('global output-token cap exhausted before '+role)
                break
            if role == 'WRITE' and read_count != 1:
                errors.append('WRITE requires exactly one successful actual read')
                break
            if role == 'FINAL' and write_count != 1:
                errors.append('FINAL requires exactly one successful actual write')
                break
            stage_instruction = {
                'READ':'This is the READ stage. Make exactly the advertised read call now; do not write or implement until its successful result is supplied.',
                'WRITE':'The advertised read succeeded. This is the WRITE stage. Make exactly one write call containing the complete implementation now. Do not read again or call other tools.',
                'FINAL':'The first implementation was successfully written. This is the FINAL stage. Return a brief plain completion message without tools or code changes.'
            }[role]
            budget = args.budget if role == 'WRITE' else 0
            payload = {'model':'deepseek-v4.1-flash','input':pending_input,
                'instructions':task['system']+'\n'+stage_instruction,
                'tools':TOOLS,'tool_choice':'auto','temperature':0,
                'reasoning':{'effort':'high'},'ds4_reasoning_budget':budget,
                'store':True,'stream':False,'max_output_tokens':args.global_cap-output_tokens,
                'prompt_cache_key':'ds4-hour16-'+case}
            if previous_id:
                payload['previous_response_id'] = previous_id
            save(out/f'request-{index}.json', payload)
            request = urllib.request.Request(args.base_url.rstrip('/')+'/v1/responses',
                data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
            body, status = {}, None
            t = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=360) as response:
                    status, raw = response.status, response.read()
                (out/f'response-{index}.json').write_bytes(raw)
                body = json.loads(raw)
            except urllib.error.HTTPError as exc:
                status = exc.code
                (out/f'response-{index}.json').write_bytes(exc.read())
                errors.append(str(exc))
            except Exception as exc:
                errors.append(type(exc).__name__+': '+str(exc))
            usage = body.get('usage') or {}
            generated = usage.get('output_tokens',0)
            thinking = (usage.get('output_tokens_details') or {}).get('reasoning_tokens',0)
            output_tokens += generated
            reasoning_tokens += thinking
            replies.append({'role':role,'budget':budget,'seconds':time.monotonic()-t,
                'http_status':status,'response_status':body.get('status'),'usage':usage,
                'requested_max_output_tokens':payload['max_output_tokens'],
                'incomplete_details':body.get('incomplete_details'), 'stop_token_observed':None})
            output = body.get('output') or []
            history.extend(output)
            text = final_text(body)
            if status != 200 or body.get('status') != 'completed':
                errors.append(role+' request did not complete')
                break
            previous_id = body.get('id')
            if not previous_id:
                errors.append('completed response has no continuation ID')
                break
            functions = [item for item in output if item.get('type') == 'function_call']
            if role == 'FINAL':
                complete = bool(text.strip()) and not functions
                if not complete:
                    errors.append('FINAL must contain text and no tool calls')
                break
            required_tool = 'read' if role == 'READ' else 'write'
            if len(functions) != 1 or functions[0].get('name') != required_tool:
                calls.extend({'role':role,'name':f.get('name'),'call_id':f.get('call_id'),
                    'arguments':f.get('arguments'),'executed':False} for f in functions)
                errors.append(role+' expected exactly one '+required_tool+' call; no tools executed')
                break
            call = functions[0]
            record = {'role':role,'name':call.get('name'),'call_id':call.get('call_id'),
                      'arguments':call.get('arguments'),'executed':False}
            tool_started = time.monotonic()
            try:
                supplied = json.loads(call.get('arguments','{}'))
                expected = config if role == 'READ' else solution
                if Path(supplied['path']).resolve() != expected or not call.get('call_id'):
                    raise ValueError('tool must use advertised path and a nonempty call ID')
                if role == 'READ':
                    if supplied.get('whole') is not True or supplied.get('raw') is not True:
                        raise ValueError('read requires whole=true/raw=true')
                    tool_result = config.read_text()
                    read_count += 1
                else:
                    source = supplied['content']
                    if not isinstance(source,str):
                        raise ValueError('write content must be a string')
                    first_code.write_text(source)
                    solution.write_text(source)
                    write_count += 1
                    tool_result = 'Successfully wrote '+str(solution)
                record.update(executed=True,path=str(expected),output=tool_result,
                              tool_ms=(time.monotonic()-tool_started)*1000)
            except Exception as exc:
                record['error'] = type(exc).__name__+': '+str(exc)
                errors.append(record['error'])
                calls.append(record)
                break
            calls.append(record)
            pending_input = [{'type':'function_call_output','call_id':call['call_id'],'output':tool_result}]
            history.extend(pending_input)
            save(out/'history.json', history)
        wall = time.monotonic()-started
        grading = grade(case,first_code,out) if first_code.exists() else {'correct':False,'reason':'no first write'}
        config_unchanged = config.read_bytes() == config_bytes
        row = {'case':case,'arm':args.arm,'write_budget':args.budget,'global_cap':args.global_cap,
            'wall_seconds':wall,'http_output_tokens':output_tokens,'http_reasoning_tokens':reasoning_tokens,
            'actual_read_count':read_count,'actual_write_count':write_count,'calls':calls,'replies':replies,
            'first_code_path':str(first_code) if first_code.exists() else None,'grade':grading,
            'complete':complete,'cap':any(r['response_status']=='incomplete' for r in replies) or output_tokens>=args.global_cap,
            'final_text':text,'errors':errors,'config_unchanged':config_unchanged,
            'seed_note':'No seed field: this server ignores seed; temperature0 requests native greedy.',
            'reasoning_history':'Stored previous_response_id preserves private reasoning; current stage instructions replace prior instructions.',
            'timing_scope':'Whole READ/WRITE/FINAL task including actual read/write; excludes startup and external code grading.'}
        row['pass'] = bool(complete and grading.get('correct') and read_count==write_count==1
                           and config_unchanged and not errors)
        save(out/'history.json',history)
        save(out/'result.json',row)
        rows.append(row)
        save(args.output/'results.json',rows)
        print(json.dumps({k:row[k] for k in ['case','arm','wall_seconds','http_output_tokens',
                          'http_reasoning_tokens','complete','cap','pass','grade']}), flush=True)
    return 0 if rows and all(r['pass'] for r in rows) else 3


if __name__ == '__main__':
    raise SystemExit(main())
