"""Frozen v4 movement/GRU, trainable split-only birth-map encoder and actor/critic."""
from runtime import ROOT
from base_model import Teacher as BaseTeacher,setup_torch,tensor_obs
from features import SHAPES
import torch
from torch import nn
import numpy as np

VERSION=5

def tensors(data,device='cpu',single=False):
    return {k:torch.as_tensor(v,device=device,dtype=torch.bool if k=='mask' else torch.float32).unsqueeze(0) if single else
        torch.as_tensor(v,device=device,dtype=torch.bool if k=='mask' else torch.float32) for k,v in data.items() if k in SHAPES}

class Teacher(nn.Module):
    def __init__(self,recurrent=False):
        super().__init__();self.recurrent=recurrent;self.base=BaseTeacher(recurrent).requires_grad_(False).eval()
        self.birth=nn.Sequential(nn.Conv2d(29,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),
            nn.GroupNorm(4,32),nn.SiLU(),nn.Flatten(),nn.Linear(32*7*7,64),nn.SiLU())
        self.context=nn.Sequential(nn.Linear(32+256,128),nn.LayerNorm(128),nn.SiLU())
        self.candidate=nn.Sequential(nn.Linear(32+64+128,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.mode=nn.Sequential(nn.Linear(128+64+64,128),nn.SiLU(),nn.Linear(128,1))
        self.size=nn.Linear(64,1)
        self.value=nn.Sequential(nn.Linear(128+64+64,128),nn.SiLU(),nn.Linear(128,1),nn.Tanh())
        self.auxiliary=nn.Linear(64,3)
        nn.init.zeros_(self.mode[-1].weight);nn.init.zeros_(self.mode[-1].bias)
        nn.init.zeros_(self.size.weight);nn.init.zeros_(self.size.bias)
    def train(self,mode=True):
        super().train(mode);self.base.eval();return self
    def forward(self,d):
        b=self.birth(d['tail']);c=self.context(torch.cat((d['context'],d['base_h']),-1))
        z=self.candidate(torch.cat((d['candidates'],b[:,None].expand(-1,6,-1),c[:,None].expand(-1,6,-1)),-1))
        valid=d['mask'].unsqueeze(-1);pool=(z*valid).sum(1)/valid.sum(1).clamp(min=1)
        fused=torch.cat((c,b,pool),-1);ctx=d['context']
        prior=torch.where(ctx[:,20]>.5,1.5,-1.5-4*((ctx[:,0]*500-260)/100).clamp(min=0,max=2.4))
        gate=self.mode(fused).squeeze(-1)+prior;gate=gate.masked_fill(~d['mask'][:,1:].any(-1),-1e9)
        mode=torch.stack((torch.zeros_like(gate),gate),-1).log_softmax(-1)
        sizes=(self.size(z[:,1:]).squeeze(-1)+2*d['candidates'][:,1:,4]).masked_fill(~d['mask'][:,1:],-1e9).log_softmax(-1)
        logp=torch.cat((mode[:,:1],mode[:,1:]+sizes),-1).masked_fill(~d['mask'],-1e9)
        return logp,self.value(fused).squeeze(-1),self.auxiliary(z)
    def save(self,path,**extra):torch.save({'version':VERSION,'recurrent':self.recurrent,'state_dict':self.state_dict(),**extra},path)

def load(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=False);assert ck['version']==5
    m=Teacher(ck['recurrent']).to(device);m.load_state_dict(ck['state_dict']);return m

def initialize(source,output,seed):
    torch.manual_seed(seed);ck=torch.load(source,map_location='cpu',weights_only=False);assert ck['version']==4
    m=Teacher(ck['recurrent']);m.base.load_state_dict(ck['state_dict']);m.save(output,source=str(source),stage='initial',seed=seed)

class Policy:
    def __init__(self,path):setup_torch();self.model=load(path).eval()
    def movement(self,obs,hidden):
        with torch.no_grad():l,v,h=self.model.base(tensor_obs(obs,batch=True),torch.as_tensor(hidden).reshape(1,256))
        return int(l[0,:4].argmax()),h[0].numpy(),l[0,:4].log_softmax(-1).numpy()
    def act(self,data,rng=None):
        with torch.no_grad():lp,v,aux=self.model(tensors(data,single=True));p=lp[0].exp().numpy().astype(np.float64);p/=p.sum()
        # Greedy hierarchical choice uses marginal split probability, then best legal size.
        a=(1+int(lp[0,1:].argmax())) if p[1:].sum()>p[0] else 0
        if rng is not None:a=int(rng.choice(6,p=p))
        return a,float(lp[0,a]),float(v[0]),aux[0].numpy()
