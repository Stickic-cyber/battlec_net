"""Full saved-state comparison; diagnoses action changes without claiming wins."""
from pathlib import Path
import sys,json,argparse,glob,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'rl_v5'))
from model import load,tensors,setup_torch
from training import base_hash
from features import SHAPES
import torch,numpy as np

def compare(old_path,new_path,paths):
    setup_torch();old=load(old_path).eval();new=load(new_path).eval();assert base_hash(old)==base_hash(new)
    counts=dict(states=0,changes=0,rule_disagreements=0,new_split=0,cancel_split=0,size_change=0,rule_new_split=0,rule_cancel_split=0,rule_size_change=0)
    phases={str(i):dict(states=0,changes=0,rule_disagreements=0) for i in range(5)};examples=[]
    def greedy(lp):
        prob=lp.exp().numpy().astype(np.float64);prob/=prob.sum(-1,keepdims=True)
        return np.where(prob[:,1:].sum(-1)>prob[:,0],1+prob[:,1:].argmax(-1),0)
    with torch.no_grad():
        for p in paths:
            with np.load(p) as archive:
                d={k:archive[k] for k in list(SHAPES)+['action','teacher','round']}
                for start in range(0,len(d['action']),512):
                    end=min(start+512,len(d['action']));obs=tensors({k:d[k][start:end] for k in SHAPES})
                    a=greedy(old(obs)[0]);b=greedy(new(obs)[0]);teacher=d['teacher'][start:end];rnd=d['round'][start:end];diff=a!=b;rd=b!=teacher
                    counts['states']+=len(a);counts['changes']+=int(diff.sum());counts['rule_disagreements']+=int(rd.sum())
                    for prefix,x,y in (('',a,b),('rule_',teacher,b)):
                        counts[prefix+'new_split']+=int(((x==0)&(y>0)).sum());counts[prefix+'cancel_split']+=int(((x>0)&(y==0)).sum())
                        counts[prefix+'size_change']+=int(((x>0)&(y>0)&(x!=y)).sum())
                    for phase in range(5):
                        ix=rnd//100==phase;v=phases[str(phase)];v['states']+=int(ix.sum());v['changes']+=int(diff[ix].sum());v['rule_disagreements']+=int(rd[ix].sum())
                    for j in np.flatnonzero(diff)[:max(0,50-len(examples))]:
                        i=start+int(j);c=d['candidates'][i,int(b[j])]
                        examples.append(dict(file=str(p),index=i,round=int(rnd[j]),old=int(a[j]),new=int(b[j]),teacher=int(teacher[j]),
                            body_fraction=float(d['context'][i,7]),child_length=float(c[1]*128),parent_length=float(c[2]*128),birth_exits=float(c[10]*4),birth_space=float(c[11]*128)))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return dict(old=str(old_path),new=str(new_path),old_sha256=sha(old_path),new_sha256=sha(new_path),counts=counts,phase=phases,examples=examples)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);p.add_argument('--pattern',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=compare(a.old,a.new,sorted(glob.glob(a.pattern)));Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r['counts']))
