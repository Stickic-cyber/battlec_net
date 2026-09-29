from runtime import ROOT
import numpy as np
import torch
from torch import nn
from base_observation import OBS_SHAPES,ENTITY_DIM,GLOBAL_NAMES,VERSION

def setup_torch():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

class Prior(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(54,64),nn.Tanh())
        self.actor=nn.Sequential(nn.Linear(128,32),nn.Tanh(),nn.Linear(32,1))
        self.critic=nn.Sequential(nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1))
        with np.load(ROOT/'rl_v1/runs/v1-first/deployment-best.npz') as z:
            self.load_state_dict({k:torch.tensor(z[k]) for k in self.state_dict()})
        self.requires_grad_(False)
    def forward(self,x,mask):
        h=self.encoder(x)
        pooled=(h*mask.unsqueeze(-1)).sum(-2)/mask.sum(-1,keepdim=True).clamp(min=1)
        return self.actor(torch.cat((h,pooled.unsqueeze(-2).expand_as(h)),-1)).squeeze(-1)

class Residual(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(8,width),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(8,width))
    def forward(self,x):return torch.nn.functional.silu(x+self.net(x))

class Spatial(nn.Module):
    def __init__(self,channels,width,blocks,out):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(channels,width,3,padding=1),nn.GroupNorm(8,width),nn.SiLU(),
            *[Residual(width) for _ in range(blocks)],nn.AdaptiveAvgPool2d((3,3)),nn.Flatten(),nn.Linear(width*9,out),nn.SiLU())
    def forward(self,x):return self.net(x)

class Teacher(nn.Module):
    def __init__(self,recurrent=False):
        super().__init__();self.recurrent=recurrent
        self.prior=Prior()
        self.memory_map=Spatial(OBS_SHAPES['grid'][0],64,3,128)
        self.local_map=Spatial(OBS_SHAPES['local'][0],32,2,64)
        self.coarse_map=Spatial(OBS_SHAPES['coarse'][0],32,1,64)
        self.entity_in=nn.Linear(ENTITY_DIM,128)
        self.entity_layers=nn.ModuleList([nn.TransformerEncoderLayer(128,4,256,dropout=0,batch_first=True,norm_first=True) for _ in range(2)])
        self.global_net=nn.Sequential(nn.Linear(len(GLOBAL_NAMES),128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.direction=nn.Sequential(nn.Linear(58,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.cross=nn.MultiheadAttention(128,4,dropout=0,batch_first=True)
        self.fusion=nn.Sequential(nn.Linear(512,768),nn.LayerNorm(768),nn.SiLU(),nn.Linear(768,512),nn.LayerNorm(512),nn.SiLU())
        self.temporal=nn.GRUCell(512,256) if recurrent else nn.Sequential(nn.Linear(512,768),nn.Tanh(),nn.Linear(768,256),nn.Tanh())
        self.actor=nn.Sequential(nn.Linear(512,256),nn.SiLU(),nn.Linear(256,64),nn.SiLU(),nn.Linear(64,1))
        self.critic=nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1),nn.Tanh())
        self.split_gate=nn.Sequential(nn.Linear(272,64),nn.SiLU(),nn.Linear(64,1))
        self.split_size=nn.Sequential(nn.Linear(284,64),nn.SiLU(),nn.Linear(64,1))
        for head in (self.actor,self.split_gate,self.split_size):
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)

    def encode(self,obs):
        ent=self.entity_in(obs['entities'])
        for layer in self.entity_layers:ent=layer(ent,src_key_padding_mask=~obs['entity_mask'])
        em=obs['entity_mask'].unsqueeze(-1)
        ep=(ent*em).sum(1)/em.sum(1).clamp(min=1)
        z=self.fusion(torch.cat((self.memory_map(obs['grid']),self.local_map(obs['local']),self.coarse_map(obs['coarse']),ep,self.global_net(obs['global'])),dim=-1))
        eye=torch.eye(4,device=z.device).unsqueeze(0).expand(len(z),-1,-1)
        q=self.direction(torch.cat((obs['legacy'],eye),-1))
        context,_=self.cross(q,ent,ent,key_padding_mask=~obs['entity_mask'],need_weights=False)
        d=torch.cat((q,context),-1)
        prior=self.prior(obs['legacy'],obs['mask'])*.5
        return z,d,prior,obs['split_context'],obs['split_candidates']

    def decoded(self,z,d,prior,split_context,split_candidates,mask,hidden=None):
        if hidden is None:hidden=torch.zeros((len(z),256),device=z.device)
        h=self.temporal(z,hidden) if self.recurrent else self.temporal(z)
        move=self.actor(torch.cat((d,h.unsqueeze(1).expand(-1,4,-1)),-1)).squeeze(-1)+prior
        move_lp=move.masked_fill(~mask[:,:4],-1e9).log_softmax(-1)
        gate=self.split_gate(torch.cat((h,split_context),-1)).squeeze(-1)
        gate=gate+torch.where(split_context[:,1]>0,3.5,-3.5)
        gate=gate.masked_fill(~mask[:,4:].any(-1),-1e9)
        mode=torch.stack((torch.zeros_like(gate),gate),-1)
        mode_lp=mode.log_softmax(-1)
        sizes=self.split_size(torch.cat((h[:,None,:].expand(-1,5,-1),split_context[:,None,:].expand(-1,5,-1),split_candidates),-1)).squeeze(-1)
        sizes=sizes+3*split_candidates[:,:,3]
        size_lp=sizes.masked_fill(~mask[:,4:],-1e9).log_softmax(-1)
        joint=torch.cat((mode_lp[:,:1]+move_lp,mode_lp[:,1:]+size_lp),-1).masked_fill(~mask,-1e9)
        return joint,self.critic(h).squeeze(-1),h

    def forward(self,obs,hidden=None):return self.decoded(*self.encode(obs),obs['action_mask'],hidden)

    def save(self,path,**extra):
        torch.save({'version':VERSION,'recurrent':self.recurrent,'state_dict':self.state_dict(),**extra},path)

def load_model(path,device='cpu'):
    ckpt=torch.load(path,map_location=device,weights_only=False)
    if ckpt['version']!=VERSION:raise ValueError('Observation schema mismatch')
    model=Teacher(ckpt['recurrent']).to(device);model.load_state_dict(ckpt['state_dict'])
    return model

def tensor_obs(obs,device='cpu',batch=False):
    return {k:torch.as_tensor(v,device=device,dtype=torch.bool if k in ('mask','entity_mask','action_mask') else torch.float32).unsqueeze(0) if batch else
        torch.as_tensor(v,device=device,dtype=torch.bool if k in ('mask','entity_mask','action_mask') else torch.float32) for k,v in obs.items()}

class Policy:
    def __init__(self,path):setup_torch();self.model=load_model(path).eval()
    def act(self,obs,hidden,rng=None,rule_split=False,teacher=0):
        with torch.no_grad():
            l,v,h=self.model(tensor_obs(obs,batch=True),torch.as_tensor(hidden).reshape(1,256))
            logp=torch.log_softmax(l[0],-1);p=logp.exp().numpy().astype(np.float64);p/=p.sum()
        if rule_split:
            a=teacher if teacher>=4 else int(l[0,:4].argmax())
        else:a=int(l[0].argmax()) if rng is None else int(rng.choice(9,p=p))
        return a,float(logp[a]),float(v[0]),h[0].numpy()


def initialize_from_v3(source,path,recurrent,seed):
    torch.manual_seed(seed)
    old=torch.load(source,map_location='cpu',weights_only=False)
    assert old['version']==3 and old['recurrent']==recurrent
    model=Teacher(recurrent)
    missing,unexpected=model.load_state_dict(old['state_dict'],strict=False)
    assert not unexpected and all(k.startswith(('split_gate.','split_size.')) for k in missing)
    model.save(path,stage='initial',source=str(source),seed=seed)
    return path
