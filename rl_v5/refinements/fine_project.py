"""Find a less coarse safe parameter step; no reward/test outcomes are consulted."""
from pathlib import Path
import sys,json,hashlib,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'rl_v5'))

def project(metadata_path):
    import numpy as np
    import torch
    from model import load,tensors,setup_torch
    from training import data_from,base_hash
    from features import SHAPES
    setup_torch();device='cuda' if torch.cuda.is_available() else 'cpu'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    metadata_path=Path(metadata_path);meta=json.loads(metadata_path.read_text());initial=Path(meta['initial'])
    raw=metadata_path.with_name(metadata_path.stem+'-raw.pt');assert sha(raw)==meta['raw_training']['checkpoint_sha256']
    paths=[Path(x['file']) for x in meta['sampling']['distribution']]
    assert sha(initial)==meta['initial_sha256'] and [sha(p) for p in paths]==meta['data_sha256']
    data=data_from(paths);n=len(data['action']);weights=[];phases=[]
    for p in paths:
        with np.load(p) as d:phase=d['round']//100
        bins,counts=np.unique(phase,return_counts=True);w=np.empty(len(phase),np.float32)
        for b,c in zip(bins,counts):w[phase==b]=n/(len(paths)*len(bins)*c)
        weights.append(w);phases.append(phase)
    phase=np.concatenate(phases);w=torch.as_tensor(np.concatenate(weights),device=device);obs=tensors({k:data[k] for k in SHAPES},device)
    old=load(initial,device).eval();m=load(raw,device).eval();old_state={k:v.clone() for k,v in old.state_dict().items()};raw_state={k:v.clone() for k,v in m.state_dict().items()}
    def infer(model):
        return torch.cat([model({k:v[start:start+512] for k,v in obs.items()})[0] for start in range(0,n,512)])
    def greedy(lp):return torch.where(lp[:,1:].exp().sum(-1)>lp[:,0].exp(),1+lp[:,1:].argmax(-1),0)
    history=[]
    with torch.no_grad():
        pl=infer(old);g0=greedy(pl)
        def trial(alpha):
            m.load_state_dict({k:old_state[k] if k.startswith('base.') else old_state[k]+alpha*(raw_state[k]-old_state[k]) for k in old_state})
            lp=infer(m);kl=(pl.exp()*(pl-lp)).sum(-1);ph={}
            for b in np.unique(phase):
                ix=torch.as_tensor(phase==b,device=device);ph[int(b)]=float((kl[ix]*w[ix]).sum()/w[ix].sum())
            result=dict(alpha=float(alpha),weighted_kl=float((kl*w).sum()/w.sum()),max_phase_kl=max(ph.values()),
                phase_kl=ph,action_changes=int((greedy(lp)!=g0).sum()),
                rule_disagreements=int((greedy(lp)!=torch.as_tensor(data['teacher'],device=device)).sum()),states=n)
            assert np.isfinite(result['weighted_kl']) and np.isfinite(result['max_phase_kl'])
            history.append(result);return result
        failed=None;passing=None
        for alpha in np.linspace(1,0,21):
            result=trial(float(alpha))
            if result['weighted_kl']<=.02 and result['max_phase_kl']<=.05:passing=float(alpha);break
            failed=float(alpha)
        assert passing is not None
        if failed is not None:
            for _ in range(6):
                mid=(passing+failed)/2;result=trial(mid)
                if result['weighted_kl']<=.02 and result['max_phase_kl']<=.05:passing=mid
                else:failed=mid
        result=trial(passing)
    assert base_hash(old)==base_hash(m)
    output=metadata_path.with_name(metadata_path.stem+'-fine.pt')
    m.save(output,stage='phase-balanced-ppo-fine-projection',config=meta['config'],initial=str(initial),projection_alpha=passing,
        frozen_sha256=base_hash(m),projection_source_sha256=sha(__file__))
    record=dict(source_metadata=str(metadata_path),source_metadata_sha256=sha(metadata_path),projection_source_sha256=sha(__file__),
        checkpoint_sha256=sha(output),initial_sha256=sha(initial),frozen_sha256=base_hash(m),diagnostics=result,search=history)
    output.with_suffix('.json').write_text(json.dumps(record,indent=2));print('FINE_PROJECTION',output.stem,json.dumps(result),flush=True)
    return output,record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('metadata',type=Path);args=p.parse_args();project(args.metadata)
