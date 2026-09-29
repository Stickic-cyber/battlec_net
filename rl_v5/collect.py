from runtime import ROOT,hashes
from runner import episode
from pathlib import Path
import json,hashlib

RUN=ROOT/'rl_v5/runs/structured-split'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jobs(maps,seeds,weights,controller='network',collect=False,**extra):
    return [dict(map=m,seed=s,side=side,weights=str(weights),controller=controller,collect=collect,**extra)
        for s in seeds for m in maps for side in ('A','B')]
def collect_jobs(ex,items,label,run=RUN):
    folder=run/'episodes';folder.mkdir(parents=True,exist_ok=True);rows=[]
    for i,j in enumerate(items):
        job=dict(j,output=str(folder/f'{label}-{i}.npz'))
        fingerprint=dict(job,weight_sha256=sha(job['weights']),sources=hashes(),map_sha256=sha(ROOT/'maps'/job['map']))
        meta=folder/f'{label}-{i}.json'
        if meta.exists():
            old=json.loads(meta.read_text());assert old['job']==fingerprint,f'Stale cache {meta}'
            result=old['summary']
            if job['collect']:assert sha(job['output'])==result['data_sha256']
            assert sha(result['telemetry_path'])==result['telemetry_sha256']
            for c in result.get('captures',[]):assert sha(c['path'])==c['sha256']
        else:
            result=ex.submit(episode,job).result();tmp=meta.with_suffix('.tmp');tmp.write_text(json.dumps({'job':fingerprint,'summary':result},indent=2));tmp.replace(meta)
        rows.append(result)
        print(f'{label} {i+1}/{len(items)} {job["map"]} {job["side"]} outcome={result["reward"]} split={result["audit"].get("strategic_splits",0)} sec={result["elapsed_seconds"]}',flush=True)
    (run/f'{label}.json').write_text(json.dumps(rows,indent=2));return rows
