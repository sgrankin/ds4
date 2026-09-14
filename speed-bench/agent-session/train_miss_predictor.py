#!/usr/bin/env python3
"""Predict cold expert demand with abstention; fixed native cache, no speed claim.

Epoch and global score threshold are selected only on validation, maximizing
covered demand misses with wrong reads <=10% of native validation demand.
The separate task is a reused regression cohort, not a fresh blind holdout.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from train_route_predictor import dataset, inputs, digest
from score_route_prefetch import cache_data, score


def threshold_for_budget(pred, confidence, actual, resident, fraction=.1):
    rows = np.arange(len(actual))[:,None]
    missing = ~resident[rows, actual].astype(bool)
    active = (~resident[rows,pred].astype(bool)) & (confidence>0)
    good = (pred[:,:,None] == actual[:,None,:]).any(axis=2)[active]
    conf = confidence[active]
    order = np.argsort(-conf,kind='stable'); conf=conf[order]; good=good[order]
    if not len(conf): return 7.0
    wrong = np.cumsum(~good)
    ends = np.r_[conf[1:] != conf[:-1], True]
    allowed = np.flatnonzero(ends & (wrong <= int(missing.sum()*fraction)))
    if not len(allowed): return 7.0
    benefit = np.cumsum(good)
    best = allowed[np.argmax(benefit[allowed])]
    return float(conf[best]) if benefit[best] else 7.0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('features','routes','cache','output'): p.add_argument(name,type=Path)
    p.add_argument('--holdout-features',type=Path,required=True)
    p.add_argument('--holdout-routes',type=Path,required=True)
    p.add_argument('--holdout-cache',type=Path,required=True)
    p.add_argument('--baseline-weights',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30)
    a=p.parse_args()
    if a.epochs<1:p.error('positive epochs required')
    d,labels=dataset(a.features,a.routes); cache=cache_data(a.cache,labels)
    hd,hl=dataset(a.holdout_features,a.holdout_routes); hc=cache_data(a.holdout_cache,hl)
    a.output.mkdir(parents=True,exist_ok=False)
    n=len(d)//40; cut1=int(n*.6)*40;cut2=int(n*.8)*40
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    mx.random.seed(1234);rng=np.random.default_rng(1234)
    class Predictor(nn.Module):
        def __init__(self):
            super().__init__();self.stem=nn.Linear(5120,128)
            self.heads=mx.random.normal((40,128,384))*(128**-.5)
            self.bias=mx.full((40,384),-4.0)
        def __call__(self,x,layer):
            return mx.einsum('bi,bij->bj',mx.tanh(self.stem(x)),self.heads[layer])+self.bias[layer]
    model=Predictor();optimizer=optim.Adam(learning_rate=.001)
    def loss_fn(model,x,layer,target,cold):
        logits=model(x,layer)
        loss=mx.logaddexp(0,logits)-target*logits
        return mx.sum(loss*(1+15*target)*cold)/mx.maximum(mx.sum(cold),1)
    vg=nn.value_and_grad(model,loss_fn)

    def predictions(weights,data,c,lo,hi,mode):
        ps=[];cs=[]
        for i in range(lo,hi,512):
            ix=np.arange(i,min(i+512,hi));layer=mx.array(data['meta'][ix,2])
            z=mx.tanh(mx.array(inputs(data,ix))@weights['stem.weight'].T+weights['stem.bias'])
            logits=mx.einsum('bi,bij->bj',z,weights['heads'][layer])+weights['bias'][layer]
            values=np.array(mx.sigmoid(logits) if mode=='miss' else 6*mx.softmax(logits,axis=-1))
            values=np.where(c['resident'][ix],-1,values)
            top=np.argsort(-values,axis=1,kind='stable')[:,:6]
            ps.append(top);cs.append(np.take_along_axis(values,top,axis=1))
        return np.concatenate(ps),np.concatenate(cs)

    def weights_now():
        return {'stem.weight':model.stem.weight,'stem.bias':model.stem.bias,'heads':model.heads,'bias':model.bias}
    def evaluate(weights,data,lab,c,lo,hi,mode,threshold=None):
        pred,conf=predictions(weights,data,c,lo,hi,mode)
        if threshold is None: threshold=threshold_for_budget(pred,conf,lab[lo:hi,3:],c['resident'][lo:hi])
        result=score(pred,conf>=threshold,lab[lo:hi,3:],c['resident'][lo:hi])
        return dict(threshold=threshold,metrics=result)

    history=[];best=-1;best_epoch=None;best_threshold=None
    for epoch in range(1,a.epochs+1):
        total=0
        for ix in np.array_split(rng.permutation(cut1), int(np.ceil(cut1/512))):
            target=np.zeros((len(ix),384),np.float32);target[np.arange(len(ix))[:,None],labels[ix,3:]]=1
            loss,grads=vg(model,mx.array(inputs(d,ix)),mx.array(d['meta'][ix,2]),mx.array(target),mx.array(1-cache['resident'][ix]))
            optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state,loss)
            total+=float(loss)*len(ix)
        val=evaluate(weights_now(),d,labels,cache,cut1,cut2,'miss')
        history.append(dict(epoch=epoch,train_loss=total/cut1,validation=val))
        print(json.dumps(dict(epoch=epoch,train_loss=total/cut1,validation=val)),flush=True)
        if val['metrics']['useful_reads']>best:
            best=val['metrics']['useful_reads'];best_epoch=epoch;best_threshold=val['threshold']
            model.save_weights(str(a.output/'best.safetensors'))
    model.load_weights(str(a.output/'best.safetensors'))
    baseline=mx.load(str(a.baseline_weights))
    bv=evaluate(baseline,d,labels,cache,cut1,cut2,'route')
    result=dict(architecture='5120->128 tanh shared stem,40x128->384 heads',parameters=2636928,
        objective='BCE on nonresident experts only, positive weight16, Adam lr.001, bias init -4',
        selection='Max validation covered misses at wrong reads <=10% of native demand; top6 cold candidates; one global threshold',
        epochs=a.epochs,seed=1234,best_epoch=best_epoch,threshold=best_threshold,
        split_tokens=[cut1//40,(cut2-cut1)//40,(len(d)-cut2)//40],history=history,
        test=evaluate(weights_now(),d,labels,cache,cut2,len(d),'miss',best_threshold),
        separate_task=evaluate(weights_now(),hd,hl,hc,0,len(hd),'miss',best_threshold),
        route_baseline=dict(validation=bv,test=evaluate(baseline,d,labels,cache,cut2,len(d),'route',bv['threshold']),
            separate_task=evaluate(baseline,hd,hl,hc,0,len(hd),'route',bv['threshold'])),
        hashes={name:digest(getattr(a,name)) for name in ('features','routes','cache','holdout_features','holdout_routes','holdout_cache','baseline_weights')},
        weights_sha256=digest(a.output/'best.safetensors'),script_sha256=digest(__file__),mlx_version=importlib.metadata.version('mlx'),
        limits='Reused evaluation cohorts. Native residency fixed; no eviction, I/O contention, overhead or readiness deadlines. Scores are not calibrated probabilities.')
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('history','hashes')}),flush=True)

if __name__=='__main__':main()
