"""PPO over strategic decisions; exact cached features because movement/GRU is frozen."""
from model import load,tensors
from features import SHAPES
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import json,hashlib

def save_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def base_hash(model):
    h=hashlib.sha256()
    for k,v in model.base.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def data_from(paths):
    parts=[]
    for p in paths:
        with np.load(p) as d:parts.append({k:d[k] for k in d.files})
    keys=list(SHAPES)+['action','logp','value','aux','aux_mask','teacher']
    total=sum(len(p['action']) for p in parts)
    data={k:np.concatenate([p[k] for p in parts]) for k in keys}
    data['returns']=np.concatenate([np.full(len(p['action']),float(p['team_reward']),np.float32) for p in parts])
    data['weights']=np.concatenate([np.full(len(p['action']),total/len(parts)/len(p['action']),np.float32) for p in parts])
    adv=data['returns']-data['value'];mean=np.average(adv,weights=data['weights']);std=np.sqrt(np.average((adv-mean)**2,weights=data['weights']))
    data['advantage']=(adv-mean)/max(std,1e-6)
    return data

def preference_batch(path,device):
    if path is None or not Path(path).exists():return None
    rows=json.loads(Path(path).read_text());obs=[];better=[];worse=[]
    for row in rows:
        if row['baseline_reward']==row['alternative_reward']:continue
        with np.load(row['capture_path']) as d:obs.append({k:d[k] for k in SHAPES})
        win=row['baseline_reward']>row['alternative_reward']
        better.append(row['baseline_action'] if win else row['alternative_action']);worse.append(row['alternative_action'] if win else row['baseline_action'])
    if not obs:return None
    return tensors({k:np.stack([o[k] for o in obs]) for k in SHAPES},device),torch.tensor(better,device=device),torch.tensor(worse,device=device)

def train(initial,paths,output,cfg,seed,update,device,preference_path=None):
    done=output.with_suffix('.json')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    inputs={'initial_sha256':sha(initial),'data_sha256':[sha(p) for p in paths],
        'preference_sha256':sha(preference_path) if preference_path and Path(preference_path).exists() else None,'config':cfg,'seed':seed,'update':update}
    if done.exists():
        cached=json.loads(done.read_text());assert cached['inputs']==inputs and cached['checkpoint_sha256']==sha(output),'Stale training cache'
        return output,cached
    torch.manual_seed(seed+update);rng=np.random.default_rng(seed+update);m=load(initial,device);frozen=base_hash(m);d=data_from(paths)
    n=len(d['action']);max_lp=max_v=0.
    m.eval()
    with torch.no_grad():
        for start in range(0,n,512):
            ids=np.arange(start,min(start+512,n));lp,v,_=m(tensors({k:d[k][ids] for k in SHAPES},device));a=torch.as_tensor(d['action'][ids],device=device)
            max_lp=max(max_lp,float(np.max(abs(lp.gather(1,a[:,None]).squeeze(1).cpu().numpy()-d['logp'][ids]))))
            max_v=max(max_v,float(np.max(abs(v.cpu().numpy()-d['value'][ids]))))
    if max_lp>3e-4 or max_v>3e-4:raise RuntimeError(f'On-policy parity failed {max_lp} {max_v}')
    # Adapt only from previous training diagnostics, never validation or independent tests.
    previous=Path(initial).with_suffix('.json');lr=cfg['lr']
    if update>=3 and previous.exists():
        old=json.loads(previous.read_text())
        if old.get('greedy_rule_disagreement_rate',1)<.005:lr=min(cfg['max_lr'],lr*2)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=lr,weight_decay=1e-5)
    pref=preference_batch(preference_path,device);metrics=[];stop=False;m.train()
    for epoch in range(cfg['epochs']):
        order=rng.permutation(n)
        for start in range(0,n,cfg['batch_size']):
            ids=order[start:start+cfg['batch_size']];obs=tensors({k:d[k][ids] for k in SHAPES},device)
            get=lambda k:torch.as_tensor(d[k][ids],device=device)
            lp,v,aux=m(obs);a=get('action');w=get('weights');chosen=lp.gather(1,a[:,None]).squeeze(1)
            logratio=chosen-get('logp');ratio=logratio.exp();adv=get('advantage')
            actor=-(torch.minimum(ratio*adv,ratio.clamp(1-cfg['clip'],1+cfg['clip'])*adv)*w).mean()
            value=(F.smooth_l1_loss(v,get('returns'),reduction='none')*w).mean()
            entropy=(-(lp.exp()*lp).sum(-1)*w).mean()
            selected_aux=aux[torch.arange(len(ids),device=device),a]
            targets=get('aux');am=get('aux_mask');survival=F.binary_cross_entropy_with_logits(selected_aux[:,[0,2]],targets[:,[0,2]],reduction='none')
            growth=F.smooth_l1_loss(selected_aux[:,1].sigmoid(),targets[:,1],reduction='none')
            al=torch.stack((survival[:,0],growth,survival[:,1]),-1)
            auxiliary=(al*am*w[:,None]).sum()/am.sum().clamp(min=1)
            imitation=(-lp.gather(1,get('teacher')[:,None]).squeeze(1)*w).mean()
            preference=torch.zeros((),device=device)
            if pref:
                pl,_,_=m(pref[0]);pref_ids=torch.arange(len(pref[1]),device=device)
                preference=F.softplus(-(pl[pref_ids,pref[1]]-pl[pref_ids,pref[2]])).mean()
            loss=actor+cfg['value_coef']*value-cfg['entropy']*entropy+cfg['aux_coef']*auxiliary+cfg['preference_coef']*preference+cfg['imitation']*imitation
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward();gradient=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.)
            if not torch.isfinite(gradient):raise RuntimeError('Nonfinite gradient')
            opt.step();kl=float(((ratio-1)-logratio).detach().mean())
            metrics.append([float(x.detach()) for x in (actor,value,entropy,auxiliary,preference)]+[kl])
            if kl>cfg['kl_stop']:stop=True;break
        if stop:break
    if base_hash(m)!=frozen:raise RuntimeError('Frozen movement backbone changed')
    disagreements=mode_disagreements=0;probe_ids=np.linspace(0,n-1,min(512,n)).astype(int);ablation={};m.eval()
    with torch.no_grad():
        obs=tensors({k:d[k][probe_ids] for k in SHAPES},device);lp,_,_=m(obs)
        greedy=torch.where(lp[:,1:].exp().sum(-1)>lp[:,0].exp(),1+lp[:,1:].argmax(-1),0)
        teacher=torch.as_tensor(d['teacher'][probe_ids],device=device)
        disagreements=int((greedy!=teacher).sum());mode_disagreements=int(((greedy>0)!=(teacher>0)).sum())
        for key in ('tail','context','candidates','base_h'):
            edited=dict(obs);edited[key]=torch.zeros_like(obs[key]);el,_,_=m(edited)
            eg=torch.where(el[:,1:].exp().sum(-1)>el[:,0].exp(),1+el[:,1:].argmax(-1),0)
            ablation[key]={'action_changes':int((eg!=greedy).sum()),'mean_probability_change':float((el.exp()-lp.exp()).abs().mean())}
    m.save(output,stage='ppo',update=update,seed=seed,config=cfg,optimizer=opt.state_dict(),frozen_sha256=frozen)
    result={'decisions':n,'sampled_splits':int((d['action']>0).sum()),'parity':{'logp':max_lp,'value':max_v},'lr':lr,
        'minibatches':len(metrics),'early_stop':stop,'loss_actor_value_entropy_aux_preference_kl':np.mean(metrics,0).tolist(),
        'probe_states':len(probe_ids),'greedy_rule_disagreement_rate':disagreements/len(probe_ids),
        'greedy_mode_disagreement_rate':mode_disagreements/len(probe_ids),'input_probe':ablation,'frozen_sha256':frozen,
        'preference_pairs':len(pref[1]) if pref else 0,'inputs':inputs,'checkpoint_sha256':sha(output)}
    save_json(done,result);print(output.stem,json.dumps(result),flush=True);return output,result
