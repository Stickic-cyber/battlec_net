"""Match-only loss weighting control, with the same data, LR and epochs as the pilot."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'rl_v5'))
import numpy as np,torch
from model import setup_torch
from training import train
from fine_project import project
setup_torch();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
run=ROOT/'rl_v5/runs/structured-split';dest=ROOT/'rl_v5/runs/phase-balanced'
initial=run/'spatial-s9201-ppo-1.pt';paths=sorted((run/'episodes').glob('spatial-s9201-rollout-2-*.npz'));assert len(paths)==4
cfg=json.loads((ROOT/'rl_v5/config.json').read_text());cfg.update(lr=.0012,max_lr=.0012,epochs=12,kl_stop=.01,
    refinement='Higher update strength control; original equal-game weights',refinement_source_sha256=sha(__file__))
raw=dest/'spatial-s9201-batch2-highlr-raw.pt'
_,metrics=train(initial,paths,raw,cfg,23201,1,'cuda' if torch.cuda.is_available() else 'cpu')
distribution=[]
for p in paths:
    with np.load(p) as d:b,c=np.unique(d['round']//100,return_counts=True)
    distribution.append(dict(file=str(p),counts={int(x):int(y) for x,y in zip(b,c)}))
meta=dest/'spatial-s9201-batch2-highlr.json';meta.write_text(json.dumps(dict(initial=str(initial),initial_sha256=sha(initial),data_sha256=[sha(p) for p in paths],
    sampling=dict(distribution=distribution),config=cfg,raw_training=metrics),indent=2))
project(meta)
