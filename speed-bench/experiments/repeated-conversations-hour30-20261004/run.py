#!/usr/bin/env python3
"""Three real item deletions per durable Conversation; ROOT owns the server."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('hour17_suffix_workflow',
    ROOT/'speed-bench/experiments/server-confirmation-hour17-20261004/suffix_workflow.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
FACTS = {'A': {'batches':23,'items_per_batch':14,'reserved_items':9,
               'codes':['BEACON-46']*3},
         'B': {'batches':31,'items_per_batch':8,'reserved_items':17,
               'codes':['ORBIT-57','ORBIT-58','ORBIT-59']}}


def http(a, out, name, method, route, payload=None):
    workflow.dump(out/(name+'-request.json'), {'method':method,'route':route,'body':payload})
    request = urllib.request.Request(a.base_url.rstrip('/')+'/v1'+route,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method, headers={'Content-Type':'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    (out/(name+'-response.json')).write_bytes(raw)
    body = json.loads(raw)
    if status != 200:
        raise RuntimeError(f'{method} {route}: HTTP{status}: {body}')
    return body, time.monotonic()-started


def snapshot(a, out, name, cid):
    public, _ = http(a,out,name+'-items','GET',f'/conversations/{cid}/items?order=asc&limit=100')
    # Version is local-store metadata, not a public Conversations API field.
    record = json.loads((a.store/'conversations'/('resp_'+cid+'.json')).read_text())
    result = {'conversation':cid,'version':record['version'],'items':public['data'],
              'has_more':public['has_more'],'private_replay_present':record.get('replay') is not None}
    workflow.dump(out/(name+'-snapshot.json'),result)
    return result


def turn(a, out, name, session, cid, query, wanted_read, gold, offset):
    started = time.monotonic()
    replies, calls, errors = [], [], []
    text, complete, read_count, cap = '', False, 0, False
    pending = [{'role':'user','content':query}]
    for index in range(3):
        payload = {'model':'deepseek-v4.1-flash','conversation':cid,'input':pending,
            'instructions':workflow.INSTRUCTIONS,'tools':workflow.TOOLS,'tool_choice':'auto',
            'store':False,'stream':False,'temperature':0,'reasoning':{'effort':'none'},
            'max_output_tokens':128}
        try:
            body, seconds = http(a,out,f'{name}-{index}','POST','/responses',payload)
        except Exception as exc:
            errors.append(type(exc).__name__+': '+str(exc))
            break
        usage = body.get('usage') or {}
        complete = body.get('status') == 'completed'
        cap = cap or body.get('status') == 'incomplete' or usage.get('output_tokens',0) >= 128
        replies.append({'seconds':seconds,'response_status':body.get('status'),'usage':usage,
            'id':body.get('id'),'conversation':body.get('conversation'),
            'incomplete_details':body.get('incomplete_details'),'requested_max_output_tokens':128})
        snapshot(a,out,f'{name}-{index}',cid)
        output = body.get('output') or []
        text = workflow.final_text(body)
        functions = [item for item in output if item.get('type')=='function_call']
        if not complete or not functions:
            break
        if len(functions) != 1 or read_count:
            errors.append('expected exactly one fresh read, followed by a plain answer')
            break
        call = functions[0]
        record = {'name':call.get('name'),'call_id':call.get('call_id'),
                  'arguments':call.get('arguments'),'read':False}
        try:
            args = json.loads(call.get('arguments','{}'))
            path = Path(args['path']).resolve()
            if (call.get('name')!='read' or not call.get('call_id') or path!=wanted_read
                    or args.get('whole') is not True or args.get('raw') is not True):
                raise ValueError('read must use the current advertised file, whole=true/raw=true')
            result = path.read_text()
            record.update(read=True,path=str(path),text=result)
            read_count += 1
            pending = [{'type':'function_call_output','call_id':call['call_id'],'output':result}]
        except Exception as exc:
            record['error'] = type(exc).__name__+': '+str(exc)
            errors.append(record['error'])
        calls.append(record)
        if errors:
            break
    else:
        errors.append('three HTTP responses exhausted without a final answer')
        complete = False
    correct, answer = workflow.answer_grade(text,gold)
    log, offset = workflow.read_log(a.server_log,offset)
    (out/(name+'-server.log')).write_text(log)
    row = {'step':name,'session':session,'conversation':cid,'seconds':time.monotonic()-started,
        'text':text,'answer':answer,'gold':gold,'correct':correct,'complete':complete,'cap':cap,
        'fresh_read':read_count==1,'calls':calls,'replies':replies,'errors':errors,
        'suffix_log_lines':[line for line in log.splitlines() if 'suffix' in line.lower()],
        'exact_hit_log_lines':[line for line in log.splitlines() if 'state cache exact hit' in line]}
    row['pass'] = bool(correct and complete and read_count==1 and not errors and not cap)
    workflow.dump(out/(name+'-result.json'),row)
    print(json.dumps({k:row[k] for k in ['step','seconds','text','pass','cap','fresh_read']},
                     ensure_ascii=False),flush=True)
    return row, offset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True)
    p.add_argument('--phase',choices=['before','after'],required=True)
    p.add_argument('--mode',choices=['native','reuse'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shared',type=Path,required=True)
    p.add_argument('--store',type=Path,required=True)
    p.add_argument('--server-log',type=Path)
    a = p.parse_args()
    a.output, a.shared, a.store = a.output.resolve(), a.shared.resolve(), a.store.resolve()
    a.output.mkdir(parents=True,exist_ok=True)
    out = a.output/a.phase
    out.mkdir(exist_ok=False)
    state_file = a.output/'conversations.json'
    rows, edits, started = [], [], time.monotonic()
    offset = a.server_log.stat().st_size if a.server_log and a.server_log.exists() else 0
    if a.phase == 'before':
        if state_file.exists():
            p.error('before requires a fresh conversation namespace')
        state = {'mode':a.mode,'shared':str(a.shared),'store':str(a.store),
                 'sessions':{},'results':[]}
        for session, fact in FACTS.items():
            directory = a.shared/session
            directory.mkdir(parents=True,exist_ok=True)
            active, archive = directory/'active.json', directory/'archive.json'
            data = {key:fact[key] for key in ['batches','items_per_batch','reserved_items']}
            data['notes'] = [f'Dispatch note {i:02d}: current batch counts are authoritative; obsolete shipment logs may be archived.' for i in range(24)]
            workflow.dump(active,data)
            workflow.dump(archive,{'archived_code':None})
            initial = [{'role':'user','content':f'Obsolete dispatch note {n+1}; safe to archive.\n'
                        +'old shipment note checked.\n'*64} for n in range(3)]
            initial.append({'role':'user','content':
                'Retain the active inventory tool result for later calculations. '
                'The historical dispatch notes are obsolete and not authoritative. '
                'For each newest archive question, read its current advertised file again; '
                'calculate batches * items_per_batch - reserved_items from the retained inventory result.'})
            created, _ = http(a,out,session+'-create','POST','/conversations',
                {'metadata':{'label':'repeated-suffix-'+session},'items':initial})
            cid = created['id']
            initial_state = snapshot(a,out,session+'-initial',cid)
            state['sessions'][session] = {'conversation':cid,
                'obsolete_ids':[item['id'] for item in initial_state['items'][:3]]}
            query = f'Read {active} once using read with whole=true and raw=true. Retain the full result, then reply READY only.'
            row, offset = turn(a,out,session+'-warm',session,cid,query,active,'READY',offset)
            rows.append(row); state['results'].append(row)
            workflow.dump(state_file,state)
    else:
        state = json.loads(state_file.read_text())
        if (state['mode']!=a.mode or state['shared']!=str(a.shared) or state['store']!=str(a.store)):
            p.error('after must continue the same mode/files/store')
    for round_number in ([1] if a.phase=='before' else [2,3]):
        for session, fact in FACTS.items():
            cid = state['sessions'][session]['conversation']
            name = f'{session}-edit{round_number}'
            edit_started = time.monotonic()
            before = snapshot(a,out,name+'-before',cid)
            item_id = state['sessions'][session]['obsolete_ids'][round_number-1]
            http(a,out,name+'-delete','DELETE',f'/conversations/{cid}/items/{item_id}')
            after = snapshot(a,out,name+'-after',cid)
            expected_ids = [item['id'] for item in before['items'] if item['id']!=item_id]
            if [item['id'] for item in after['items']] != expected_ids:
                raise RuntimeError('deletion changed items other than the selected obsolete note')
            code = fact['codes'][round_number-1]
            archive = a.shared/session/'archive.json'
            workflow.dump(archive,{'archived_code':code})
            edit = {'session':session,'round':round_number,'item_id':item_id,
                'before_version':before['version'],'after_version':after['version'],
                'before_item_ids':[i['id'] for i in before['items']], 'after_item_ids':expected_ids,
                'seconds':time.monotonic()-edit_started,'archive_code':code}
            edits.append(edit)
            workflow.dump(out/(name+'-edit.json'),edit)
            query = (f'Read {archive} again now using read with whole=true and raw=true. '
                'A fresh read is required even if an earlier answer contained a code. '
                'Use the retained active inventory tool result to calculate batches * items_per_batch - reserved_items. '
                'Reply only JSON with active_total (integer) and archived_code (string) from the current file.')
            gold = {'active_total':fact['batches']*fact['items_per_batch']-fact['reserved_items'],
                    'archived_code':code}
            row, offset = turn(a,out,name,session,cid,query,archive,gold,offset)
            row.update(seconds_including_edit=time.monotonic()-edit_started,
                       edit_seconds=edit['seconds'],round=round_number)
            workflow.dump(out/(name+'-result.json'),row)
            rows.append(row); state['results'].append(row)
            workflow.dump(state_file,state)
    lines = [line for row in rows for line in row['suffix_log_lines']]
    result = {'phase':a.phase,'mode':a.mode,'steps':rows,'edits':edits,
        'seconds':time.monotonic()-started,'pass':all(row['pass'] for row in rows),
        'fresh_reads':sum(row['fresh_read'] for row in rows),
        'suffix_grafts_observed':sum('suffix APPROXIMATE' in line for line in lines),
        'approximate_ssd_sources_observed':sum('suffix APPROXIMATE' in line and
            'source_ssd=1' in line and 'source_approximate=1' in line for line in lines),
        'restart_owned_by_launcher':True,'client_concurrency':1}
    workflow.dump(out/'result.json',result)
    fields = ['step','session','round','seconds','seconds_including_edit','correct',
              'fresh_read','complete','cap','pass','text']
    with (out/'results.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    return 0 if result['pass'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
