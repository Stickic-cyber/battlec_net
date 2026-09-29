"""Controlled PPO reweighting trial on an existing, exactly on-policy batch.

Original runs remain immutable. Each game and each populated 100-round phase
within that game receives equal loss weight. The final update is backtracked
against the exact full-distribution, phase-weighted KL, including a phase cap.
"""
from pathlib import Path
import sys,json,hashlib,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'rl_v5'))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')

def fit(initial,paths,output,cfg,seed):
    import numpy as np
    import torch
    import training
    from model import load,tensors,setup_torch
    from features import SHAPES
    setup_torch();device='cuda' if torch.cuda.is_available() else 'cpu'
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    original_data=training.data_from;sampling={}
    def balanced(files):
        d=original_data(files);n=len(d['action']);weights=[];phases=[];distribution=[]
        for p in files:
            with np.load(p) as x:phase=x['round']//100
            bins,counts=np.unique(phase,return_counts=True)
            w=np.empty(len(phase),np.float32)
            for b,count in zip(bins,counts):w[phase==b]=n/(len(files)*len(bins)*count)
            weights.append(w);phases.append(phase);distribution.append(dict(file=str(p),counts={int(b):int(c) for b,c in zip(bins,counts)}))
        d['weights']=np.concatenate(weights);adv=d['returns']-d['value']
        mean=np.average(adv,weights=d['weights']);std=np.sqrt(np.average((adv-mean)**2,weights=d['weights']))
        d['advantage']=(adv-mean)/max(std,1e-6)
        sampling.update(distribution=distribution,phases=np.concatenate(phases),max_weight=float(d['weights'].max()))
        return d
    cfg=dict(cfg,refinement='Equal game/100-round-phase loss weighting; exact weighted-KL backtracking',
        refinement_source_sha256=sha(__file__),base_training_source_sha256=sha(ROOT/'rl_v5/training.py'))
    training.data_from=balanced
    raw=output.with_name(output.stem+'-raw.pt')
    try:_,metrics=training.train(initial,paths,raw,cfg,seed,1,device)
    finally:training.data_from=original_data
    d=balanced(paths);old=load(initial,device).eval();m=load(raw,device).eval()
    old_state={k:v.detach().clone() for k,v in old.state_dict().items()};raw_state={k:v.detach().clone() for k,v in m.state_dict().items()}
    def compare():
        kl=[];changed=[];rule=[];p_before=[];p_after=[]
        with torch.no_grad():
            for start in range(0,len(d['action']),512):
                ids=np.arange(start,min(start+512,len(d['action'])));obs=tensors({k:d[k][ids] for k in SHAPES},device)
                pl,_,_=old(obs);lp,_,_=m(obs)
                kl.extend((pl.exp()*(pl-lp)).sum(-1).cpu().numpy())
                greedy=lambda x:torch.where(x[:,1:].exp().sum(-1)>x[:,0].exp(),1+x[:,1:].argmax(-1),0)
                g=greedy(lp);changed.extend((g!=greedy(pl)).cpu().numpy());rule.extend((g!=torch.as_tensor(d['teacher'][ids],device=device)).cpu().numpy())
                p_before.extend(pl[:,1:].exp().sum(-1).cpu().numpy());p_after.extend(lp[:,1:].exp().sum(-1).cpu().numpy())
        kl=np.asarray(kl);ph=sampling['phases'];phases={}
        for b in np.unique(ph):
            ix=ph==b;w=d['weights'][ix]
            phases[int(b)]=dict(states=int(ix.sum()),kl=float(np.average(kl[ix],weights=w)),
                split_probability_before=float(np.average(np.asarray(p_before)[ix],weights=w)),
                split_probability_after=float(np.average(np.asarray(p_after)[ix],weights=w)))
        return dict(weighted_kl=float(np.average(kl,weights=d['weights'])),max_phase_kl=max(v['kl'] for v in phases.values()),
            action_changes=int(sum(changed)),rule_disagreements=int(sum(rule)),states=len(kl),phase=phases)
    alpha=1.;diagnostics=compare()
    while diagnostics['weighted_kl']>.02 or diagnostics['max_phase_kl']>.05:
        alpha/=2
        if alpha<1/1024:raise RuntimeError('No finite trust-region projection found')
        m.load_state_dict({k:old_state[k]+alpha*(raw_state[k]-old_state[k]) for k in old_state})
        diagnostics=compare()
    assert training.base_hash(old)==training.base_hash(m)
    m.save(output,stage='phase-balanced-ppo',config=cfg,seed=seed,initial=str(initial),projection_alpha=alpha,
        frozen_sha256=training.base_hash(m))
    result=dict(initial=str(initial),initial_sha256=sha(initial),data_sha256=[sha(p) for p in paths],checkpoint_sha256=sha(output),
        config=cfg,seed=seed,projection_alpha=alpha,diagnostics=diagnostics,sampling={k:v for k,v in sampling.items() if k!='phases'},
        frozen_sha256=training.base_hash(m),raw_training=metrics)
    save(output.with_suffix('.json'),result);print('PHASE_BALANCED',output.stem,json.dumps(diagnostics),flush=True);return output,result

def main():
    p=argparse.ArgumentParser();p.add_argument('--architecture',choices=['spatial','gru'],required=True)
    p.add_argument('--seed',type=int,default=9201);p.add_argument('--batch',type=int,default=2);p.add_argument('--lr',type=float,default=.0012)
    p.add_argument('--epochs',type=int,default=12);args=p.parse_args()
    run=ROOT/'rl_v5/runs/structured-split';n=args.architecture;s=args.seed;u=args.batch
    initial=run/(f'{n}-initial-{s}.pt' if u==1 else f'{n}-s{s}-ppo-{u-1}.pt')
    paths=sorted((run/'episodes').glob(f'{n}-s{s}-rollout-{u}-*.npz'));assert len(paths)==4
    for f in paths:
        j=json.loads(f.with_suffix('.json').read_text())['job']
        assert j['stochastic'] and j['collect'] and j['controller']=='network' and j['weight_sha256']==sha(initial)
    cfg=json.loads((ROOT/'rl_v5/config.json').read_text());cfg.update(lr=args.lr,max_lr=args.lr,epochs=args.epochs,kl_stop=.01)
    out=ROOT/f'rl_v5/runs/phase-balanced/{n}-s{s}-batch{u}.pt'
    fit(initial,paths,out,cfg,14000+s)
if __name__=='__main__':main()
