#!/usr/bin/env python3
"""Pair common teacher histories, summarize the bracket, plot the measured curve."""
import argparse
from array import array
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

def read_arm(path):
    meta = json.loads((path/'tokens.json').read_text())
    timing = {r['case']:r for r in csv.DictReader((path/'timings.csv').open())}
    answers = [json.loads(line) for line in (path/'answers.jsonl').read_text().splitlines()]
    rows = list(csv.DictReader((path/'rows.csv').open()))
    logits = array('f')
    with (path/'logits.bin').open('rb') as f:
        logits.fromfile(f, (path/'logits.bin').stat().st_size // 4)
    if sys.byteorder != 'little': logits.byteswap()
    assert len(logits) == len(rows) * meta['vocab']
    return dict(path=path, meta=meta, timing=timing, answers=answers, rows=rows, logits=logits)

def norm(values):
    assert all(math.isfinite(x) for x in values)
    peak = max(values)
    return peak + math.log(math.fsum(math.exp(x-peak) for x in values))

def paired(reference, other):
    for key in ['vocab','context','chunk','cases']:
        assert reference['meta'][key] == other['meta'][key], key
    assert [(r['case'],r['step'],r['target']) for r in reference['rows']] == [
        (r['case'],r['step'],r['target']) for r in other['rows']]
    v = reference['meta']['vocab']
    rows = []
    for i, original in enumerate(reference['rows']):
        p = reference['logits'][i*v:(i+1)*v]
        q = other['logits'][i*v:(i+1)*v]
        pz, qz = norm(p), norm(q)
        target = int(original['target'])
        pnll, qnll = pz-p[target], qz-q[target]
        kl = math.fsum(math.exp(x-pz) * ((x-pz)-(y-qz)) for x,y in zip(p,q))
        rows.append(dict(case=original['case'],step=int(original['step']),target=target,
            native_nll=pnll,candidate_nll=qnll,delta_nll=qnll-pnll,kl=kl,
            argmax_flip=max(range(v),key=p.__getitem__) != max(range(v),key=q.__getitem__),
            byte_equal=p.tobytes()==q.tobytes()))
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant-key', choices=['rows','compressed_kv_f16'], default='rows')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    arms = [read_arm(a.input/r['directory']) for r in json.loads((a.input/'result.json').read_text())]
    controls = [x for x in arms if x['meta'][a.variant_key]==0]
    assert len(controls)==2
    result = {'reference':controls[0]['path'].name,'timing_formula':'greedy_sync_ms + greedy_ms',
        'quality_formula':'KL(native opening || candidate), shared fixed teachers including EOS', 'arms':[]}
    for arm in arms:
        rows = paired(controls[0], arm)
        per_case = []
        for name, time_row in arm['timing'].items():
            sub = [r for r in rows if r['case']==name]
            natives = [float(c['timing'][name]['greedy_sync_ms']) +
                       float(c['timing'][name]['greedy_ms']) for c in controls]
            native_sync = statistics.mean(float(c['timing'][name]['greedy_sync_ms']) for c in controls)
            practical = float(time_row['greedy_sync_ms'])+float(time_row['greedy_ms'])
            per_case.append(dict(case=name,practical_ms=practical,native_practical_ms=statistics.mean(natives),
                saving=1-practical/statistics.mean(natives),
                native_drift=natives[-1]/natives[0]-1,
                sync_ms=float(time_row['greedy_sync_ms']),native_sync_ms=native_sync,
                mean_delta_nll=statistics.mean(r['delta_nll'] for r in sub),
                mean_kl=statistics.mean(r['kl'] for r in sub),
                answer=next(x for x in arm['answers'] if x['case']==name)))
        total = sum(x['practical_ms'] for x in per_case)
        native_total = sum(x['native_practical_ms'] for x in per_case)
        mean_dnll = statistics.mean(r['delta_nll'] for r in rows)
        entry = dict(directory=arm['path'].name, rows=arm['meta']['rows'],
            variant_key=a.variant_key,variant_value=arm['meta'][a.variant_key],
            practical_ms=total,native_practical_ms=native_total,saving=1-total/native_total,
            mean_delta_nll=mean_dnll,perplexity_ratio=math.exp(mean_dnll),
            mean_kl=statistics.mean(r['kl'] for r in rows),max_kl=max(r['kl'] for r in rows),
            argmax_flips=sum(r['argmax_flip'] for r in rows),teacher_rows=len(rows),
            byte_equal_rows=sum(r['byte_equal'] for r in rows),
            correct=sum(x['exact'] and x['stopped'] for x in arm['answers']),cases=per_case,
            logits_sha256=hashlib.sha256((arm['path']/'logits.bin').read_bytes()).hexdigest())
        if a.variant_key=='compressed_kv_f16':
            actual = re.search(r'KV allocated main=\w+ main_bytes=(\d+) index_bytes=(\d+) raw_bytes=(\d+)',
                               (arm['path']/'stderr.log').read_text())
            assert actual, 'Missing actual KV allocation counters'
            entry['allocated_kv_bytes'] = dict(zip(['main','index','raw'], map(int,actual.groups())))
        result['arms'].append(entry)
        with (a.output/(arm['path'].name+'-paired.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['reference','timing_formula']}))
    for entry in result['arms']:
        print(json.dumps({k:entry[k] for k in ['directory','saving','mean_delta_nll','perplexity_ratio','mean_kl','correct']}))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; summary and paired CSVs retained', file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(7,4.5),layout='constrained')
    plotted = [x for x in result['arms'] if x['variant_value'] != 0]
    if a.variant_key=='compressed_kv_f16':
        # One point per storage format; repeated arms remain in the raw table.
        selected=plotted
        plotted=[dict(selected[0],directory='F16 repeat mean',
            practical_ms=statistics.mean(x['practical_ms'] for x in selected),
            timing_min_ms=min(x['practical_ms'] for x in selected),
            timing_max_ms=max(x['practical_ms'] for x in selected))]
    native_times=[x['practical_ms'] for x in result['arms'] if x['variant_value']==0]
    plotted.insert(0,dict(directory='Native bracket mean',rows=0,
        variant_value=0,practical_ms=statistics.mean(native_times),
        timing_min_ms=min(native_times),timing_max_ms=max(native_times),
        allocated_kv_bytes=result['arms'][0].get('allocated_kv_bytes'),
        perplexity_ratio=1.0,correct=result['arms'][0]['correct'],cases=result['arms'][0]['cases']))
    for entry in plotted:
        if a.variant_key=='compressed_kv_f16':
            x,y=sum(entry['allocated_kv_bytes'].values())/2**20,entry['practical_ms']/1000
            ax.errorbar(x,y,yerr=[[y-entry['timing_min_ms']/1000],
                                 [entry['timing_max_ms']/1000-y]],fmt='o',capsize=4)
        else:
            x,y=entry['practical_ms']/1000,100*(entry['perplexity_ratio']-1)
            ax.scatter(x,y)
        ax.annotate(f"{entry['directory']} ({entry['correct']}/{len(entry['cases'])} answers)",(x,y),
                    xytext=(-5,6) if entry['variant_value']==0 else (5,6),textcoords='offset points',
                    ha='right' if entry['variant_value']==0 else 'left',fontsize=8)
    if a.variant_key=='compressed_kv_f16':
        ax.set(xlabel='Actual allocated main + index + raw KV (MiB)',
            ylabel='Sum of direct prefill + first greedy answer (seconds)',title='Measured runtime KV packing tradeoff')
    else:
        ax.axhline(0,color='gray',linestyle=':',linewidth=.8)
        ax.set(xlabel='Sum of direct prefill + first greedy answer (seconds)',
               ylabel='Fixed-teacher perplexity change (%)',title='Measured decoder replay tradeoff')
    fig.savefig(a.output/'tradeoff.svg')
    fig.savefig(a.output/'tradeoff.png',dpi=150)

if __name__=='__main__':main()
