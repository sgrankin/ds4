"""Pure contract validation; no engine/model/guard access or process launch."""
import csv,hashlib,json,math,re,struct
from pathlib import Path
CTX=32768
VOCAB=129280
SLOTS=6136
ORIGINAL_COUNTS=[1859,90,72,8192,8,16384,8,38,8]
COUNTS=[1859,90,72,8320,8,16895,8,38,8]
KINDS=list('PPDPDPDPD')
POSITIONS=[1859,1949,2021,10341,10349,27244,27252,27290,27298]
PREFIXES=[0,1859,0,2021,0,10349,0,27252,0]
HEADER=struct.Struct('<13I')
class Refusal(ValueError):pass
def require(ok,message):
    if not ok:raise Refusal(message)
def strict_pairs(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'duplicate JSON key');result[key]=value
    return result
def load_json(data):return json.loads(data,object_pairs_hook=strict_pairs,parse_constant=lambda x:(_ for _ in ()).throw(Refusal('nonfinite JSON')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def original(text):
    lines=text.splitlines();require(len(lines)==9,'original nine phases')
    blocks=[];history=[]
    for i,line in enumerate(lines):
        w=line.split();require(len(w)>=4 and w[0]==KINDS[i] and all(re.fullmatch('[0-9]+',x) for x in w[1:]),'literal replay record')
        cached,n=map(int,w[1:3]);ids=list(map(int,w[3:]));require(n==len(ids)==ORIGINAL_COUNTS[i] and all(t<VOCAB for t in ids),'original token bounds/count')
        require((w[0]=='P' and cached==len(history)) or (w[0]=='D' and cached==0),'original uninterrupted prefix')
        blocks.append(dict(kind=w[0],cached=cached,ids=ids));history+=ids
    require(len(history)==26659,'original complete history')
    return blocks
def derive(text):
    old=original(text);rows=[];history=[];additions={3:128,5:511}
    for i,b in enumerate(old):
        ids=b['ids']+b['ids'][:additions.get(i,0)]
        cached=len(history) if b['kind']=='P' else 0
        rows.append(' '.join(map(str,[b['kind'],cached,len(ids),*ids])))
        history+=ids;require(len(history)==POSITIONS[i],'derived independent geometry')
    targets={p:old[p+1]['ids']+old[p+2]['ids'][:24] for p in (3,5)}
    require(all(len(t)==32 for t in targets.values()),'32 untouched following teacher targets')
    return '\n'.join(rows)+'\n',''.join(' '.join(map(str,['B',p,32,*targets[p]]))+'\n' for p in (3,5)),targets
def ragged(text):
    blocks=[];history=[]
    lines=text.splitlines();require(len(lines)==9,'complete derived replay')
    for i,line in enumerate(lines):
        w=line.split();require(len(w)>=4 and w[0]==KINDS[i] and all(re.fullmatch('[0-9]+',x) for x in w[1:]),'literal derived replay')
        cached,n=map(int,w[1:3]);ids=list(map(int,w[3:]));require(cached==PREFIXES[i] and n==COUNTS[i] and len(ids)==n and all(t<VOCAB for t in ids),'frozen derived record')
        if w[0]=='P':require(cached==len(history),'full retained derived prefix')
        else:require(cached==0,'decode prefix')
        history+=ids;require(len(history)==POSITIONS[i] and len(history)<CTX,'derived context capacity')
        blocks.append(dict(kind=w[0],cached=cached,ids=ids,history=list(history)))
    return blocks
def dispatch(data):
    require(data and data.endswith(b'\n'),'complete dispatch JSONL')
    rows=[load_json(s) for s in data.splitlines()];groups={p:[] for p in (0,1,3,5,7)};last_phase=-1
    for row in rows:
        require(type(row) is dict and set(row)=={'phase','start','end','total','count'} and all(type(v) is int for v in row.values()),'typed complete dispatch row')
        p=row['phase'];require(p in groups and p>=last_phase,'ordered prefill-only dispatch')
        start=groups[p][-1]['end'] if groups[p] else PREFIXES[p]
        require(row['start']==start and row['total']==POSITIONS[p] and start<row['end']<=row['total'] and row['count']==row['end']-start,'contiguous completed production dispatch')
        groups[p].append(row);last_phase=p
    require(all(g and g[-1]['end']==POSITIONS[p] and sum(r['count'] for r in g)==COUNTS[p] for p,g in groups.items()),'all prefill chunks complete')
    return {p:[r['count'] for r in g] for p,g in groups.items()}
def dispatch_gate(fast0,fast1):
    # Independent expected tails are fixed before outcomes. Bulk observations
    # may report one large dispatch or repeated bulk pieces; compare residual
    # boundaries at the unchanged original long-P count.
    for p,bulk,tail in [(3,8192,128),(5,16384,511)]:
        def residual(seq):
            cursor=0;result=[]
            for n in seq:
                require(n>0,'positive dispatch');cursor+=n
                if cursor>bulk:require(cursor-n>=bulk,'dispatch crosses frozen bulk/tail boundary');result.append(n)
            require(cursor==bulk+tail,'full diagnostic phase');return result
        a=residual(fast0[p]);b=residual(fast1[p])
        require(a==([64,64] if p==3 else [64]*7+[63]),'flag0 fast expected selected residual')
        require(b==([8]*16 if p==3 else [511]),'flag1 native expected residual')
        require(a!=b,'actual dispatch difference mandatory before timing')
    return True
def snapshot_size(pos,vocab=VOCAB):
    require(type(pos) is int and 0<=pos<CTX,'snapshot pos bound')
    return 52+(pos+vocab)*4+40*min(pos,128)*512*4+(3*(pos//2)+pos)*640*4+(3*2*512*4 if pos&1 else 0)
def finite(data,vocab=VOCAB):
    require(len(data)==vocab*4,'full vector extent')
    require(all(u[0]&0x7f800000!=0x7f800000 for u in struct.iter_unpack('<I',data)),'finite FP32 vector')
def canonical(data,history,last,vocab=VOCAB):
    pos=len(history);require(len(data)==snapshot_size(pos,vocab),'full canonical extent')
    require(HEADER.unpack_from(data)==(0x34565344,2,CTX,1,128,128,CTX+1,pos,40,512,128,vocab,0x413431),'canonical full header')
    require(data[52:52+4*pos]==struct.pack('<'+str(pos)+'i',*history),'exact canonical tokens')
    finite(last,vocab);require(data[52+4*pos:52+4*(pos+vocab)]==last,'canonical raw logits match')
def branch_rows(path,targets,positions=POSITIONS):
    with Path(path).open() as f:
        reader=csv.DictReader(f);require(reader.fieldnames==['phase','step','pos','target'],'score-before-target metadata schema');rows=list(reader)
    require(len(rows)==64,'all64 branch predictions')
    for row,(p,j) in zip(rows,((p,j) for p in (3,5) for j in range(32))):
        require(set(row)=={'phase','step','pos','target'} and [row[k] for k in ('phase','step','pos','target')]==list(map(str,(p,j,positions[p]+j,targets[p][j]))),'branch target/step/position identity')
    return rows
def restore_gate(anchor_state,restored_state,anchor_raw,restored_raw,first_branch_raw,history,vocab=VOCAB):
    canonical(anchor_state,history,anchor_raw,vocab)
    canonical(restored_state,history,restored_raw,vocab)
    require(anchor_state==restored_state and anchor_raw==restored_raw==first_branch_raw,'native canonical restore and first branch score full byte identity')
def logit_metrics(old,new,target,vocab=VOCAB):
    finite(old,vocab);finite(new,vocab);require(type(target) is int and 0<=target<vocab,'teacher token bound')
    p=list(struct.unpack('<'+str(vocab)+'f',old));q=list(struct.unpack('<'+str(vocab)+'f',new))
    a=max(p);b=max(q);zp=a+math.log(math.fsum(math.exp(x-a) for x in p));zq=b+math.log(math.fsum(math.exp(x-b) for x in q))
    kl=math.fsum(math.exp(x-zp)*((x-zp)-(y-zq)) for x,y in zip(p,q));tv=.5*math.fsum(abs(math.exp(x-zp)-math.exp(y-zq)) for x,y in zip(p,q))
    require(math.isfinite(kl) and kl>=-1e-12 and 0<=tv<=1+1e-12,'valid full-vector diagnostics')
    return dict(target=target,delta_nll=(zq-q[target])-(zp-p[target]),kl=max(0,kl),tv=tv,bit_equal=old==new,argmax_flip=p.index(a)!=q.index(b))
