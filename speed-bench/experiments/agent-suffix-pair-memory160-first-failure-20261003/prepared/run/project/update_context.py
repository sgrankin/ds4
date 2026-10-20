#!/usr/bin/env python3
"""Apply the already selected frozen operation; never echoes record/gold text."""
import hashlib,json,os,tempfile
from pathlib import Path


def replace_unique(text,old,new):
    if not old or text.count(old)!=1:raise ValueError('Frozen operation is not unique')
    return text.replace(old,new,1)


def apply(text,task,index):
    if type(index) is not int or not 0<=index<16:raise ValueError('Candidate index differs')
    for pair in [task['edit']['replace_exact_unique_fact'],task['edit']['replace_exact_unique_requirement']]:
        text=replace_unique(text,*pair)
    deletion=task['edit']['candidate_order'][index]
    if deletion['candidate']!=index:raise ValueError('Frozen candidate order differs')
    return replace_unique(text,deletion['delete_padding_prefix'],'')


def execute(output,live,task_path):
    output=Path(output);live=Path(live);task_path=Path(task_path)
    # Exclusive first-attempt record: a repeated model call cannot repair it.
    result_path=output/'update-result.json'
    with result_path.open('x') as record:
        try:
            task=json.loads(task_path.read_text());selection=json.loads((output/'selection.json').read_text())
            if selection['status']!='SELECTED_BEFORE_THIRD_EVAL' or type(selection['candidate']) is not int:raise ValueError('No frozen native selection')
            before=live.read_bytes()
            if len(before)>1<<20:raise ValueError('Live context bound exceeded')
            text=before.decode('utf-8');after=apply(text,task,selection['candidate']).encode('utf-8')
            # The source selection text precedes model generation. Live file
            # now includes appended tool/history, so editing current bytes is
            # essential. Do not replace the live file with selection-after.
            fd,tmp=tempfile.mkstemp(prefix=live.name+'.edit-',dir=live.parent)
            try:
                with os.fdopen(fd,'wb') as stream:stream.write(after);stream.flush();os.fsync(stream.fileno())
                os.replace(tmp,live)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
            (output/'update-before.txt').write_bytes(before);(output/'update-after.txt').write_bytes(after)
            result=dict(status='UPDATED',candidate=selection['candidate'],before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest(),before_bytes=len(before),after_bytes=len(after),appended_history_preserved=True)
            json.dump(result,record,indent=2);record.write('\n')
        except BaseException as error:
            json.dump(dict(status='FIRST_UPDATE_FAILURE_RETAINED',error=type(error).__name__+': '+str(error)),record,indent=2);record.write('\n');raise
    return result


if __name__=='__main__':
    try:
        execute(os.environ['DS4_AGENT_PAIR_ACTUAL_OUTPUT'],os.environ['DS4_AGENT_PAIR_ACTUAL_LIVE_FILE'],os.environ['DS4_AGENT_PAIR_ACTUAL_TASK'])
    except BaseException:
        print('CONTEXT_UPDATE_FAILED');raise SystemExit(3)
    print('CONTEXT_EDITED')
