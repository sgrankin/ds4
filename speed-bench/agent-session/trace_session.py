"""Extract measured rounds and a strict teacher-forced replay from an agent trace."""
from datetime import datetime
import hashlib
import re


def extract(path):
    events, rounds = [], []
    collecting = None
    current = None
    decoded = []
    calls = 0
    for line in path.read_text().splitlines():
        stamp = datetime.strptime(line[:23], '%Y-%m-%d %H:%M:%S.%f')
        if 'compacted_transcript' in line or 'generation context boundary:' in line:
            raise ValueError('Compaction requires a separate replay format')
        m = re.search(r'tokens label=(\w+) start=(\d+) len=(\d+)', line)
        if m:
            label, start, end = m[1], int(m[2]), int(m[3])
            if label not in ('initial_system_prompt', 'prefill_suffix'):
                raise ValueError(f'Unexpected token block: {label}')
            collecting = dict(kind='P', cached=start, count=end-start, tokens=[])
            events.append(collecting)
            if collecting['count'] == 0: collecting = None
        elif (m := re.search(r' token index=(\d+) id=(\d+)', line)):
            token = int(m[2])
            if collecting is not None:
                assert int(m[1]) == collecting['cached'] + len(collecting['tokens'])
                collecting['tokens'].append(token)
                if len(collecting['tokens']) == collecting['count']:
                    collecting = None
            elif current is not None and 'prefill_ms' in current:
                decoded.append(token)
        if (m := re.search(r'prefill tool_round=(\d+) transcript=\d+ prompt=(\d+) cached=(\d+) suffix=(\d+)', line)):
            assert collecting is None
            current = dict(tool_round=int(m[1]), context=int(m[2]), cached=int(m[3]), suffix=int(m[4]), started=stamp)
            decoded = []
        if (m := re.search(r'prefill sync done tool_round=\d+ .* rc=(\d+) ([\d.]+) ms', line)):
            assert current is not None and int(m[1]) == 0
            current['prefill_ms'] = float(m[2]); current['prefilled'] = stamp
        if (m := re.search(r'dsml done calls=(\d+)', line)):
            calls += int(m[1])
        if (m := re.search(r'generation finished tool_round=\d+ generated=(\d+)', line)):
            assert current and len(decoded) == int(m[1])
            current['generated'] = len(decoded)
            current['decode_ms'] = (stamp-current.pop('prefilled')).total_seconds()*1000
            current['response_ms'] = (stamp-current.pop('started')).total_seconds()*1000
            rounds.append(current); current = None
            events.append(dict(kind='D', cached=0, count=len(decoded), tokens=decoded))
    assert collecting is None and current is None and rounds and calls
    history = []
    for ev in events:
        assert len(ev['tokens']) == ev['count']
        if ev['kind'] == 'P':
            assert ev['cached'] <= len(history)
            history = history[:ev['cached']] + ev['tokens']
        else:
            history += ev['tokens']
    replay = ''.join(f"{e['kind']} {e['cached']} {e['count']} " + ' '.join(map(str,e['tokens']))+'\n' for e in events)
    return dict(rounds=rounds, tool_calls=calls, final_context=len(history),
                prefill_ms=sum(r['prefill_ms'] for r in rounds),
                decode_ms=sum(r['decode_ms'] for r in rounds),
                replay_sha256=hashlib.sha256(replay.encode()).hexdigest()), replay
