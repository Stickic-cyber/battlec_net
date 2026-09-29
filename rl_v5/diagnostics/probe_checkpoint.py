"""Bounded observational sensitivity diagnostic; never used as a win estimate."""
from pathlib import Path
import argparse, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'rl_v5'))
import numpy as np
import torch
from model import load,tensors,setup_torch
from features import SHAPES

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('data',type=Path,nargs='+');a=p.parse_args()
    setup_torch();model=load(a.checkpoint).eval();parts=[]
    for path in a.data:
        with np.load(path) as d:
            ids=np.linspace(0,len(d['action'])-1,min(1024,len(d['action']))).astype(int)
            parts.append({k:d[k][ids] for k in SHAPES})
    data={k:np.concatenate([part[k] for part in parts]) for k in SHAPES};n=len(data['mask'])
    def infer(edited):
        with torch.no_grad():
            return torch.cat([model(tensors({k:v[i:i+256] for k,v in edited.items()}))[0] for i in range(0,n,256)])
    def greedy(lp):
        prob=lp.exp().double();prob/=prob.sum(-1,keepdim=True)
        return torch.where(prob[:,1:].sum(-1)>prob[:,0],1+lp[:,1:].argmax(-1),0)
    original=infer(data);original_actions=greedy(original);probes={}
    for key in ('tail','context','candidates','base_h'):
        edited=dict(data);edited[key]=np.roll(data[key],1,axis=0).copy()
        # Preserve the rule priors and action mask while probing learned pathways.
        if key=='context':edited[key][:,[0,20]]=data[key][:,[0,20]]
        if key=='candidates':edited[key][:,:,4]=data[key][:,:,4]
        lp=infer(edited)
        probes[key]=dict(action_changes=int((greedy(lp)!=original_actions).sum()),
            mean_probability_change=float((lp.exp()-original.exp()).abs().mean()))
    result=dict(checkpoint_sha256=sha(a.checkpoint),analysis_sha256=sha(__file__),data_sha256=[sha(p) for p in a.data],
        states=n,greedy_splits=int((original_actions>0).sum()),probes=probes,
        interpretation='Sampled training states. Shuffled features preserve fixed rule-prior fields and masks, but may be jointly inconsistent. This measures sensitivity, not causal feature importance or held-out skill.')
    a.checkpoint.with_suffix('.sensitivity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
