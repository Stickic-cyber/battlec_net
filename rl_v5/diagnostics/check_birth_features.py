"""Offline confidence audit: compare estimated birth positions with official splits."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
run=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'rl_v5/runs/structured-split'
counts=dict(checked=0,position_matches=0,complete_body_checked=0,complete_body_matches=0,cut_checked=0,cut_matches=0);mismatches=[]
for p in (run/'episodes').glob('*rollout*.npz'):
    rp=p.with_suffix('.replay-metrics.json');tp=p.with_suffix('.telemetry.json')
    if not rp.exists():continue
    r=json.loads(rp.read_text());t=json.loads(tp.read_text());births={b['child']:b for b in r['births']}
    with np.load(p) as d:
        offset={str(i):int(o) for i,o in zip(d['dragon_ids'],d['offsets'])}
        for child,par in t['parents'].items():
            b=births.get(int(child))
            if b is None or 'childHead' not in b:continue
            h=t['histories'][str(par['parent'])][par['parent_history']]
            if h['policy_action']<=0 or h['row']<0:continue
            idx=offset[str(par['parent'])]+h['row'];c=d['candidates'][idx,h['policy_action']];ctx=d['context'][idx]
            w,hgt=int(round(float(ctx[5])*64)),int(round(float(ctx[6])*64))
            point=lambda x,y:dict(x=round(float(x)*w/2+h['x'])%w,y=round(float(y)*hgt/2+h['y'])%hgt)
            match=point(c[8],c[9])==b['childHead'];counts['checked']+=1;counts['position_matches']+=int(match)
            if not match:mismatches.append(dict(episode=p.stem,round=h['round'],parent=par['parent'],child=child,body_fraction=float(c[6]),estimated=point(c[8],c[9]),actual=b['childHead']))
            if c[6]>=.999:counts['complete_body_checked']+=1;counts['complete_body_matches']+=int(match)
            if c[23]>.5:counts['cut_checked']+=1;counts['cut_matches']+=int(point(c[24],c[25])==b['parentTail'])
(run/'birth-feature-audit.json').write_text(json.dumps(dict(counts,mismatches=mismatches),indent=2));print(json.dumps(counts))
