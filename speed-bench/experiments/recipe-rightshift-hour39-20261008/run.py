#!/usr/bin/env python3
"""Actual config read, rightward planning-context expansion, first write and final."""
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
PRACTICAL = ROOT / 'speed-bench/experiments/fresh-clm-coding-hour32-20261004/practical'
TOOLS = [
    {'type':'function', 'name':'read', 'description':'Read the advertised config file in full.',
     'parameters':{'type':'object','properties':{'path':{'type':'string'},
        'whole':{'type':'boolean'},'raw':{'type':'boolean'}},
        'required':['path','whole','raw'],'additionalProperties':False}},
    {'type':'function', 'name':'write', 'description':'Write the complete first implementation to the advertised solution path.',
     'parameters':{'type':'object','properties':{'path':{'type':'string'},'content':{'type':'string'}},
        'required':['path','content'],'additionalProperties':False}}
]

# Authored benchmark discussion, expanded after the actual config read.
# Contracts and config/tool results remain outside this replaced block.
DISCUSSIONS = {
    'dedupe': """Planning review: this helper is for a batch of timestamped events, not a long-running service. The caller already gives timestamps in nondecreasing order. We agreed that accepted and rejected are lists of original zero-based positions, so a duplicate-looking event must still have its own index in the result. The function must leave the events and configuration unchanged.

The first sketch used a dictionary of last-seen times. During review we noticed that this would make every rejected duplicate extend the waiting period. That is the wrong rule for this task. The settled dictionary records only the last ACCEPTED event for each exact key. A rejected event leaves its stored timestamp alone, however close it is to another rejection.

We discussed whether the timeout means strictly after the boundary or at the boundary. The accepted contract is at the boundary: t minus the last accepted time greater than or equal to ttl accepts. A key's first event always accepts. The ttl comes from config; it is not a constant chosen from a design example. Reading the real config before implementation is part of the task.

An early pseudocode page sorted events by key before checking them. We crossed that out because it obscures original positions and changes event order. The input order is already useful. A single pass can keep a small dictionary and append each current index to exactly one of two lists. There is no need to sort the input or assemble a second array of event records.

Someone asked whether keys should be stripped or made lowercase. That would merge distinct keys and is outside the contract. Empty strings and Unicode are ordinary exact keys. We also rejected storing a short hash in place of the key, because collision handling adds complexity and could merge unrelated streams. The original string itself is the dictionary key.

For an empty payload, both index lists are empty. With ttl zero every event accepts, including several events for one key at the same timestamp. These follow directly from the first-occurrence and greater-than-or-equal rules; they are not special business exceptions. Values are guaranteed to have the specified types and ordering, so extra input repair is unnecessary.

The implementation discussion wandered into keeping rejected-event reasons. We decided not to add them. The requested output has just accepted and rejected. The caller might present reasons later, but this function should not print a report, attach fields to input dictionaries, or record anything in a file. A diagnostic log would also distract from the plain return value.

We compared scanning the entire accepted history for each event with a dictionary lookup. The scan can get expensive and makes the intended last-accepted rule harder to see. Keep the simple dictionary approach. A generator was also suggested to stream results, but the public result must contain ordinary complete lists. Return those lists in the requested dictionary.

There was a separate release checklist about naming the module, where a future command-line wrapper might live, and how an operator would distinguish an empty run from a skipped run. Those presentation choices are deferred. They do not change this function's payload format, index numbering, key identity, ttl rule, or nonmutation requirement.

Handoff: implement one plain solve(payload, config) function using the allowed builtins. Do not build a service, class hierarchy, validator, or command-line interface. The final coding requirement and the actual config read are authoritative. Finish the required first write and a brief completion reply; there is no revision or retry step in this handoff.""",
    'bucket': """Planning review: this helper evaluates a supplied batch of timestamped costs. It does not consult the wall clock or run a background refill timer. Event timestamps are nonnegative integers in nondecreasing order. Capacity, rate, and costs are also nonnegative integers. We settled on exact integer arithmetic throughout.

The bucket starts at time zero with tokens equal to capacity, including when the first event arrives later than zero. For each event, refill from the previous event timestamp using elapsed time times rate, cap the balance at capacity, then consider the cost. Capacity and rate come from the actual config file, not from a number in an old planning example.

Our first draft advanced the clock only when a request was accepted. Review found that this charges elapsed time more than once after a rejection. We rejected that draft. Advance the remembered timestamp for EVERY event after its refill, including rejected and oversized requests. The next event measures elapsed time from this event, regardless of whether any tokens were spent.

An event accepts when cost is less than or equal to the refilled balance. Acceptance subtracts that cost. Rejection spends nothing. An oversized cost still refills and advances the clock; it simply cannot pass the capacity-limited balance. A zero-cost event accepts even with capacity zero. Neither case needs a special external action or a retry mechanism.

The result is a dictionary containing accepted and rejected original zero-based indices, remaining balance after each event, and the final tokens balance. The remaining list has one entry for every input event, after any spending decision. For an empty payload it is empty and final tokens equals capacity. Output order follows supplied event order.

We considered keeping fractional credits to support future rates such as half a token per second. That is not part of this contract and was dropped. We also discarded rounding the refill or using floating point for timestamp arithmetic. The given values permit direct integer operations, even for large time gaps or costs. There is no simulated sleep, real clock read, or periodic callback.

A preliminary diagram split accepted and rejected events into separate processing queues. That would make it easy to forget that both queues affect the same timestamp and balance. Keep a single ordered loop instead. Sorting events is unnecessary because their timestamp order is guaranteed, and reordering them could change both original indices and same-time decisions.

The review noted that a zero refill rate still has a full starting balance, and that several events can have equal timestamps. Those follow from the general initial-state and per-event rules. Avoid changing input records to attach balances. Build separate index and remaining lists and leave both payload and configuration untouched.

We briefly debated logging every denied event and adding a reason string to the return value. Both ideas were deferred. The current caller wants only the four specified result fields. A UI or monitoring layer can make its own presentation later. No file output, print statements, network interaction, or additional public fields belong in solve.

Handoff: the final coding requirement and the actual configuration read govern implementation. Use one plain solve(payload, config) function with the permitted builtins rather than an asynchronous service or a reusable scheduler. Complete the one required first write and then a short final reply. The current task has no retry, correction, or extra tools stage."""
}

SUMMARIES = {
    'dedupe': """Settled design: process the supplied nondecreasing event order once, preserving original zero-based indices and leaving payload/config unchanged. Use exact keys, including empty/Unicode strings, with no stripping, case folding, hash substitution, or sorting. Track only each key's LAST ACCEPTED timestamp. First occurrence accepts; later occurrence accepts when t-last_accepted >= config ttl and updates that timestamp. Rejection does not refresh it. ttl0 accepts every event; empty payload returns two empty lists. Return only {accepted, rejected}; a simple dictionary and ordinary lists suffice. History scans, generators, extra validation, diagnostic reasons, logging, UI and service wrappers were rejected or deferred. Read the actual config, make the required first write, then finish briefly under the unchanged coding contract.""",
    'bucket': """Settled design: use exact integers and one pass in supplied nondecreasing event order, without sorting or mutating payload/config. Begin at time0 with tokens=capacity. At EVERY event refill min(capacity, tokens+(t-previous_t)*rate), then advance previous_t even on rejection or oversized costs. Accept cost<=tokens and subtract; reject otherwise without spending. cost0 accepts even at capacity0. Return only {accepted, rejected, remaining, tokens}: original zero-based index lists, balance AFTER each event, final balance; empty payload leaves tokens=capacity. Capacity/rate come from the actual config. Fractional credits, rounding, separate queues, real clocks/timers, logging, reason fields and service wrappers were rejected or deferred. Make the required actual read and first write, then finish briefly under the unchanged coding contract."""
}

REVIEW_APPENDIX = """Deferred review appendix: the team discussed how a future repository could present small helpers to readers. One proposal was an examples directory with short prose introductions. Another was a notebook containing screenshots of a demonstration. Nobody selected a publishing format, and neither proposal describes the current function's inputs, outputs, configuration, or implementation.

A separate conversation concerned the appearance of a possible project page. The designer sketched a narrow sidebar, a large title, and a list of links to background notes. The group also considered a plain text index that would be easier to maintain. These were presentation alternatives for an unspecified future page, with no bearing on this coding handoff.

The naming brainstorm produced several categories rather than a final name: descriptive utility names, short abstract names, and names tied to the team's internal vocabulary. We left the brainstorm open. The advertised module and function in the coding requirement remain the current interface, so this note does not supply a replacement name or a new public API.

During a documentation review, people compared placing examples inline with collecting them in an appendix. Inline examples might interrupt readers who already know the terminology; an appendix might be hard for new readers to find. The reviewers agreed to revisit that editorial question after an actual documentation draft exists. No documentation task was added to the present implementation.

Someone proposed maintaining a visual map of discussions: sketches, meeting notes, and discarded ideas could have separate colors. Others preferred ordinary dated documents with links. That organizational debate was parked because there is no current archive interface to design. It adds no event data, configuration values, acceptance rule, or required return field to the helper.

The group briefly discussed how to present later performance investigations. A chart could compare two implementations, while a compact table might make individual measurements easier to inspect. A narrative could explain the workload that produced each result. No measurements or performance requirements were established in this discussion; it is only a list of presentation options for future work.

Finally, we considered whether old planning notes should be kept as separate files or condensed into a short handoff. Keeping them might explain why some ideas were deferred; condensing them might make the active work easier to read. This appendix preserves that ordinary review conversation. The settled design above, unchanged coding requirement below, and actual configuration read remain authoritative."""


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
    p.add_argument('--visible', type=Path, default=ROOT/'.workspaces/recipe-20261008/right-visible')
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
        summary = SUMMARIES[case]
        expanded = summary + '\n\n' + REVIEW_APPENDIX if case == 'dedupe' else DISCUSSIONS[case]
        initial = ('The following working discussion records earlier planning. '
                   'Its settled decisions carry forward; the coding requirement below is authoritative.\n'
                   '[WORKING_DISCUSSION_BEGIN]\n' + summary + '\n[WORKING_DISCUSSION_END]\n'
                   'The unchanged coding requirement follows:\n' + prompt)
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
                lo = before.index('[WORKING_DISCUSSION_BEGIN]')
                hi = before.index('[WORKING_DISCUSSION_END]', lo) + len('[WORKING_DISCUSSION_END]')
                after = before[:lo] + '[WORKING_DISCUSSION_BEGIN]\n' + expanded + '\n[WORKING_DISCUSSION_END]' + before[hi:]
                save(out/'history.before-edit.json', history)
                (out/'context.before.txt').write_text(before)
                (out/'context.after.txt').write_text(after)
                history[0]['content'] = after
                history.append({'role':'user','content':
                    'The configuration read succeeded. Implement the unchanged coding requirement using that result. '
                    'Make the single advertised write now; do not repeat the read.'})
                save(out/'edited-history-before-write.json', history)
                edit = {'removed_blocks':0,'expanded_blocks':1,
                        'source_kind':'irrelevant review insertion' if case == 'dedupe' else 'faithful settled-summary expansion',
                        'original_summary':summary,'expanded_discussion':expanded,'before_bytes':len(before.encode()),
                        'after_bytes':len(after.encode()),'seconds':time.monotonic()-edit_started}
                save(out/'edit.json', edit)
            budget = 128 if role == 'WRITE' else (1 if role == 'FINAL' else 0)
            effort = 'high'
            payload = {'model':'deepseek-v4.1-flash','input':history,
                'instructions':task['system'],
                'tools':TOOLS,'tool_choice':'auto','temperature':0,
                'reasoning':{'effort':effort},'ds4_reasoning_budget':budget,
                'store':False,'stream':False,'max_output_tokens':args.global_cap-output_tokens,
                'prompt_cache_key':'ds4-hour39-'+case}
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
            'timing_scope':'Whole READ/right-expansion/WRITE/FINAL task including actual read/write; excludes startup and external code grading.'}
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
