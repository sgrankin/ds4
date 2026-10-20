#!/usr/bin/env python3
"""First actual config read, authored CLM note removal, first write and final."""
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
PRACTICAL = HERE / 'practical'
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
    command = [sys.executable, '-B', str(PRACTICAL / 'grade.py'), case, str(code.resolve())]
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
    p.add_argument('--visible', type=Path, default=ROOT/'.workspaces/server-agent-day-20261004/fresh32-visible')
    p.add_argument('--arm', required=True)
    p.add_argument('--global-cap', type=int, default=2048)
    p.add_argument('--case', action='append', choices=['dedupe','bucket'])
    args = p.parse_args()
    args.final_policy = 'budget1'
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
        block = '[OBSOLETE_CODING_NOTE_BEGIN]\n' + 'old shipment note checked.\n' * 64 + '[OBSOLETE_CODING_NOTE_END]'
        initial = ('The marked dispatch notes are obsolete and are not part of the coding requirement.\n'
                   + block + '\n' + block + '\nThe unchanged coding requirement follows:\n' + prompt)
        pending_input = [{'role':'user','content':initial}]
        calls, replies, errors, history = [], [], [], list(pending_input)
        read_count = write_count = output_tokens = reasoning_tokens = 0
        complete = False
        text = ''
        edit = None
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
            if role == 'WRITE':
                edit_started = time.monotonic()
                before = history[0]['content']
                after = before
                for _ in range(2):
                    lo = after.index('[OBSOLETE_CODING_NOTE_BEGIN]')
                    hi = after.index('[OBSOLETE_CODING_NOTE_END]', lo) + len('[OBSOLETE_CODING_NOTE_END]')
                    after = after[:lo] + '[dispatch note archived]' + after[hi:]
                (out/'context.before.txt').write_text(before)
                (out/'context.after.txt').write_text(after)
                history[0]['content'] = after
                history.append({'role':'user','content':
                    'The configuration read succeeded. Implement the unchanged coding requirement using that result. '
                    'Make the single advertised write now; do not repeat the read.'})
                save(out/'edited-history-before-write.json', history)
                edit = {'removed_blocks':2,'before_bytes':len(before.encode()),
                        'after_bytes':len(after.encode()),'seconds':time.monotonic()-edit_started}
                save(out/'edit.json', edit)
            budget = 128 if role == 'WRITE' else (1 if role == 'FINAL' else 0)
            effort = 'high'
            payload = {'model':'deepseek-v4.1-flash','input':history,
                'instructions':task['system'],
                'tools':TOOLS,'tool_choice':'auto','temperature':0,
                'reasoning':{'effort':effort},'ds4_reasoning_budget':budget,
                'store':False,'stream':False,'max_output_tokens':args.global_cap-output_tokens,
                'prompt_cache_key':'ds4-hour32-'+case}
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
            replies.append({'role':role,'budget':budget,'effort':effort,'seconds':time.monotonic()-t,
                'http_status':status,'response_status':body.get('status'),'usage':usage,
                'requested_max_output_tokens':payload['max_output_tokens'],
                'incomplete_details':body.get('incomplete_details'), 'stop_token_observed':None})
            output = body.get('output') or []
            history.extend(output)
            text = final_text(body)
            if status != 200 or body.get('status') != 'completed':
                errors.append(role+' request did not complete')
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
        row = {'case':case,'arm':args.arm,'protocol':'full-input-clm','write_budget':128,'final_policy':args.final_policy,'global_cap':args.global_cap,
            'wall_seconds':wall,'http_output_tokens':output_tokens,'http_reasoning_tokens':reasoning_tokens,
            'actual_read_count':read_count,'actual_write_count':write_count,'calls':calls,'replies':replies,
            'first_code_path':str(first_code) if first_code.exists() else None,'grade':grading,
            'complete':complete,'cap':any(r['response_status']=='incomplete' for r in replies) or output_tokens>=args.global_cap,
            'final_text':text,'final_text_length':len(text),'final_text_bytes':len(text.encode()),
            'first_code_bytes':first_code.stat().st_size if first_code.exists() else None,
            'errors':errors,'config_unchanged':config_unchanged,'context_edit':edit,
            'seed_note':'No seed field: this server ignores seed; temperature0 requests native greedy.',
            'reasoning_history':'Actual response items and tool outputs are replayed in full input; stable task.system/tools, no previous_response_id. Public reasoning is preserved as returned.',
            'timing_scope':'Whole READ/CLM-edit/WRITE/FINAL task including actual read/write; excludes startup and external code grading.'}
        row['pass'] = bool(complete and grading.get('correct') and read_count==write_count==1
                           and config_unchanged and not errors and not row['cap'])
        save(out/'history.json',history)
        save(out/'result.json',row)
        rows.append(row)
        save(args.output/'results.json',rows)
        print(json.dumps({k:row[k] for k in ['case','arm','wall_seconds','http_output_tokens',
                          'http_reasoning_tokens','complete','cap','pass','grade']}), flush=True)
    return 0 if rows and all(r['pass'] for r in rows) else 3


if __name__ == '__main__':
    raise SystemExit(main())
