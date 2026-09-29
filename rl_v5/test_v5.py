from runtime import ROOT
from model import Teacher,initialize,load,setup_torch,tensors
from base_model import load_model,tensor_obs
from features import topology,observe
from training import base_hash,train
from pathlib import Path
from types import SimpleNamespace as NS
import unittest,tempfile,json
import numpy as np
import torch
from helper import Direction

setup_torch()

class Tests(unittest.TestCase):
    def dataset(self):
        with np.load(ROOT/'rl_v5/runs/smoke-a/smoke.npz') as d:return {k:d[k][:12] for k in ('tail','context','candidates','mask','base_h')}
    def test_frozen_transfer_and_movement(self):
        with np.load(ROOT/'rl_v4/runs/teacher-split/episodes/spatial-rollout-1-0.npz') as d:
            from base_observation import OBS_SHAPES
            obs=tensor_obs({k:d[k][:6] for k in OBS_SHAPES});hidden=torch.as_tensor(d['hidden'][:6])
        for name,source in (('spatial','spatial-ppo-4.pt'),('gru','gru-initial.pt')):
            source=ROOT/'rl_v4/runs/teacher-split'/source
            with tempfile.TemporaryDirectory() as td:
                dest=Path(td)/'initial.pt';initialize(source,dest,33);m=load(dest).eval()
            old=load_model(source).eval()
            for k,v in old.state_dict().items():self.assertTrue(torch.equal(v,m.base.state_dict()[k]))
            with torch.no_grad():
                a=old(obs,hidden);b=m.base(obs,hidden)
                for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)
            self.assertTrue(all(not p.requires_grad for p in m.base.parameters()))
    def test_hierarchical_normalization_phase_and_gradients(self):
        m=Teacher();d=self.dataset();d['context'][:,20]=0;d['context'][:,0]=.1
        d['mask'][:]=True;d['mask'][0,1:]=False;d['mask'][1,2:]=False
        obs=tensors(d);lp,v,aux=m(obs);prob=lp.exp()
        torch.testing.assert_close(prob.sum(-1),torch.ones(12));self.assertEqual(float(prob[0,1:].sum()),0)
        torch.testing.assert_close(prob[1,1:].sum(),prob[2,1:].sum())
        late={k:x.clone() for k,x in obs.items()};late['context'][:,0]=.9
        self.assertLess(float(m(late)[0].exp()[2,1:].sum()),float(prob[2,1:].sum())/20)
        before=base_hash(m);(-lp[2:,1].mean()+aux.square().mean()+v.square().mean()).backward()
        self.assertTrue(all(p.grad is None for p in m.base.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        self.assertGreater(float(m.birth[0].weight.grad.abs().sum()),0);self.assertEqual(before,base_hash(m))
        if torch.cuda.is_available():
            with torch.no_grad():
                cpu=m(obs);gpu=m.cuda()(tensors(d,'cuda'))
                for x,y in zip(cpu,gpu):torch.testing.assert_close(x,y.cpu(),atol=3e-5,rtol=3e-5)
    def test_topology_unknown_and_tail_observation_isolation(self):
        dirs=Direction.get_direction_list();world={(x,y):dict(kelp=set(),portals={},has_pearl=False,spawns=False,round=1) for x in range(10) for y in range(10)}
        part=NS(get_team=lambda:'A',get_id=lambda:0,is_head=lambda:False)
        tile=NS(get_position=lambda:NS(x=2,y=3),get_dragon=lambda:part)
        ct=NS(get_position=lambda:NS(x=1,y=1),get_length=lambda:6,get_unit_count=lambda:2,get_tiles=lambda:[tile],get_id=lambda:0,
            get_sonar_echoes=lambda:NS(enemy_head=0),get_dir=lambda:dirs[0])
        r=NS(ct=ct,game=NS(get_round_num=lambda:30,get_unit_limit=lambda:64),map_width=10,map_height=10,
            own_body=[(1,1),(1,2),(1,3)],world_map=world,portal_dest_ready={},_DIRS=dirs,my_team='A',ally_states={},known_king_info=dict(x=-1,y=-1,len=0,round=-99))
        obs=NS(base=np.zeros((27,10,10),np.float32));old=obs.base.copy();choice=NS(role='FARMER',recommended=0,size_mask=[True,True,False,False,False],sizes=[3,2,0,0,0])
        result=observe(r,obs,choice,np.zeros(256,np.float32))
        np.testing.assert_array_equal(obs.base,old)
        self.assertEqual(int(result['tail'][27].sum()),1);self.assertEqual(int(result['tail'][28].sum()),1)
        for d in dirs:world[(1,3)]['kelp'].add(d)
        self.assertEqual(topology(r,(1,3))['exits'],0)
        world[(1,3)]['kelp'].clear();world[(1,3)]['portals'][dirs[0]]=1
        self.assertEqual(topology(r,(1,3),limit=1)['exits'],3)
    def test_real_ppo_freezes_movement(self):
        cfg=json.loads((ROOT/'rl_v5/config.json').read_text());cfg.update(epochs=1,batch_size=128)
        source=ROOT/'rl_v5/runs/smoke-a/spatial-initial.pt'
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'ppo.pt';_,result=train(source,[ROOT/'rl_v5/runs/smoke-a/smoke.npz'],out,cfg,22,1,'cuda' if torch.cuda.is_available() else 'cpu')
            self.assertEqual(base_hash(load(source)),base_hash(load(out)));self.assertGreater(result['minibatches'],0)

if __name__=='__main__':unittest.main(verbosity=2)
