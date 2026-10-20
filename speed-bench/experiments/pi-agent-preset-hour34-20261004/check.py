#!/usr/bin/env python3
"""Inspect actual Pi JSON events and grade its first write, never a repair."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
PRACTICAL=ROOT/'speed-bench/experiments/fresh-clm-coding-hour32-20261004/practical'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); out=a.output.resolve(); project=out/'project'
    events=[]; errors=[]
    for i,line in enumerate((out/'pi.events.jsonl').read_text(errors='replace').splitlines()):
        try: events.append(json.loads(line))
        except ValueError: errors.append('non-JSON event line '+str(i+1))
    starts=[e for e in events if e.get('type')=='tool_execution_start']
    ends={e.get('toolCallId'):e for e in events if e.get('type')=='tool_execution_end'}
    reads=[e for e in starts if e.get('toolName')=='read']
    writes=[e for e in starts if e.get('toolName')=='write']
    messages=[e.get('message',{}) for e in events if e.get('type')=='message_end'
              and e.get('message',{}).get('role')=='assistant']
    first=out/'first-write.py'; grade={'correct':False,'passed':0,'reason':'no first write'}
    if writes and isinstance(writes[0].get('args',{}).get('content'),str):
        first.write_text(writes[0]['args']['content'])
        command=[sys.executable,'-B',str(PRACTICAL/'grade.py'),'dedupe',str(first)]
        with (out/'grader.stdout').open('wb') as stdout,(out/'grader.stderr').open('wb') as stderr:
            try:
                rc=subprocess.run(command,cwd=PRACTICAL,stdout=stdout,stderr=stderr,timeout=4).returncode
                grade=json.loads((out/'grader.stdout').read_text()); grade['returncode']=rc
            except Exception as exc: grade={'correct':False,'reason':type(exc).__name__,'message':str(exc)}
    def allowed(event, name):
        args=event.get('args') or {}; path=Path(args.get('path',''))
        path=(project/path).resolve() if not path.is_absolute() else path.resolve()
        expected=project/('settings.json' if name=='read' else 'solution.py')
        ended=ends.get(event.get('toolCallId'),{})
        return (path==expected and ended.get('isError') is False and
                (name!='read' or ('offset' not in args and 'limit' not in args)))
    cap=any(m.get('stopReason')=='length' or (m.get('usage') or {}).get('output',0)>=2048 for m in messages)
    final=messages[-1] if messages else {}
    final_text=''.join(part.get('text','') for part in final.get('content',[]) if part.get('type')=='text')
    complete=bool(any(e.get('type')=='agent_end' for e in events) and final.get('stopReason')=='stop' and final_text.strip())
    if any(m.get('stopReason') in ['error','aborted'] for m in messages): errors.append('assistant error or abort')
    if any(e.get('type')=='auto_retry_start' for e in events): errors.append('unexpected retry')
    config_unchanged=(project/'settings.json').read_bytes()==(out/'settings.before.json').read_bytes()
    process=json.loads((out/'pi-process.json').read_text())
    tool_ok=(len(starts)==2 and len(reads)==len(writes)==1 and
             starts[0].get('toolName')=='read' and starts[1].get('toolName')=='write' and
             allowed(reads[0],'read') and allowed(writes[0],'write'))
    result={'pi_returncode':process.get('returncode'),'complete':complete,'cap':cap,
        'actual_reads':len(reads),'actual_writes':len(writes),'tool_contract':tool_ok,
        'tool_events':starts,'first_code_bytes':first.stat().st_size if first.exists() else None,
        'grade':grade,'config_unchanged':config_unchanged,'final_text':final_text,
        'assistant_messages':messages,'errors':errors,
        'pass':bool(process.get('returncode')==0 and complete and not cap and tool_ok
                    and config_unchanged and grade.get('correct') and not errors)}
    (out/'check-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return 0 if result['pass'] else 3


if __name__=='__main__': main_rc=main(); sys.exit(main_rc)
