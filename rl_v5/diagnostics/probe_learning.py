"""Training-state fit and input sensitivity, explicitly not a held-out evaluation."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'rl_v5'))
from model import load,tensors,setup_torch
from features import SHAPES
import torch,numpy as np
setup_torch()
run=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'rl_v5/runs/structured-split'
results={}
source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
for p in sorted(run.glob('*-s*-ppo-*.pt')):
    meta=p.with_suffix('.json')
    if not meta.exists():continue
    training=json.loads(meta.read_text());name,seedword,_,updateword=p.stem.split('-');seed=int(seedword[1:]);update=int(updateword)
    output=p.with_suffix('.probe.json');fingerprint=hashlib.sha256(p.read_bytes()).hexdigest()
    if output.exists():
        old=json.loads(output.read_text())
        if old['checkpoint_sha256']==fingerprint and old.get('analysis_sha256')==source_hash:results[p.stem]=old;continue
    parts=[]
    for datafile in sorted((run/'episodes').glob(f'{name}-s{seed}-rollout-{update}-*.npz')):
        with np.load(datafile) as d:
            ids=np.linspace(0,len(d['action'])-1,min(256,len(d['action']))).astype(int)
            part={k:d[k][ids] for k in list(SHAPES)+['action','aux','aux_mask']};parts.append(part)
    data={k:np.concatenate([d[k] for d in parts]) for k in parts[0]};obs=tensors({k:data[k] for k in SHAPES})
    previous=run/(f'{name}-initial-{seed}.pt' if update==1 else f'{name}-s{seed}-ppo-{update-1}.pt')
    old=load(previous).eval();m=load(p).eval();actions=torch.as_tensor(data['action']);ids=torch.arange(len(actions))
    def aux_metrics(aux,split_only=False):
        pred=aux[ids,actions].sigmoid().numpy();truth=data['aux'];mask=data['aux_mask'];out={}
        for i,key in enumerate(('child_survival','growth','parent_survival')):
            valid=mask[:,i] & ((data['action']>0) if split_only else True);y=truth[valid,i];v=pred[valid,i]
            out[key]=dict(count=len(y),target_mean=float(y.mean()) if len(y) else None,
                mse=float(((v-y)**2).mean()) if len(y) else None,
                constant_mse=float(((y-y.mean())**2).mean()) if len(y) else None,
                mae=float(abs(v-y).mean()) if len(y) else None)
        return out
    with torch.no_grad():
        pl,_,pa=old(obs);lp,_,aux=m(obs);prob=lp.exp();mode=prob[:,1:].sum(-1)
        greedy=lambda x:torch.where(x[:,1:].exp().sum(-1)>x[:,0].exp(),1+x[:,1:].argmax(-1),0)
        shuffled=dict(obs);shuffled['tail']=obs['tail'].roll(1,0);sh=m(shuffled)[0]
        result=dict(checkpoint_sha256=fingerprint,analysis_sha256=source_hash,states=len(actions),before=aux_metrics(pa),after=aux_metrics(aux),
            before_split_only=aux_metrics(pa,True),after_split_only=aux_metrics(aux,True),
            mean_split_probability=float(mode.mean()),mean_probability_update=float((prob-pl.exp()).abs().mean()),
            greedy_changes_from_previous=int((greedy(lp)!=greedy(pl)).sum()),tail_shuffle_action_changes=int((greedy(sh)!=greedy(lp)).sum()),
            tail_shuffle_mean_probability_change=float((sh.exp()-prob).abs().mean()),
            mode_head_weight_norm=float(m.mode[-1].weight.norm()),size_head_weight_norm=float(m.size.weight.norm()),
            interpretation='Fit on sampled training states only. Tail shuffling is an out-of-distribution sensitivity probe, not a causal intervention or generalization result.')
    output.write_text(json.dumps(result,indent=2));results[p.stem]=result
(run/'learning-probes.json').write_text(json.dumps(results,indent=2));print(json.dumps({k:{q:v[q] for q in ('states','greedy_changes_from_previous','tail_shuffle_action_changes','mean_split_probability')} for k,v in results.items()}))
