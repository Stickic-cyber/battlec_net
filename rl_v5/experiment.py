"""Reproducible split-only PPO with matched rule and logistics controls."""
from runtime import ROOT,config,hashes
from collect import RUN,jobs,collect_jobs,sha
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse,json,shutil,platform,time

SOURCES={'spatial':ROOT/'rl_v4/runs/teacher-split/spatial-ppo-4.pt',
         'gru':ROOT/'rl_v4/runs/teacher-split/gru-initial.pt'}

def save(p,obj):p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def score(rows):
    r=[x['reward'] for x in rows]
    return dict(wins=r.count(1),draws=r.count(0),losses=r.count(-1),points=sum((x+1)/2 for x in r),games=len(r))
def provenance(run,cfg):
    files=hashes()
    extra=list(SOURCES.values())+list((ROOT/'rl_v4').glob('*.py'))+list((ROOT/'maps').glob('*.map'))
    extra += [ROOT/'方案2-v1/weights.bin',ROOT/'方案2-v2/weights.bin',ROOT/'rl_v1/runs/v1-first/deployment-best.npz']
    for p in extra:files[str(p.relative_to(ROOT))]=sha(p)
    target=run/'manifest.json'
    if target.exists():
        old=json.loads(target.read_text());assert old['files']==files and old['config']==cfg,'Immutable experiment sources changed'
    else:
        from importlib.metadata import version
        save(target,dict(files=files,config=cfg,python=platform.python_version(),torch=version('torch'),unswbc=version('unswbc'),
            protocol='Native CPU teacher against original official SandboxBot. One fresh worker per match. All choices frozen before independent tests.'))
        for rel in files:
            if not rel.endswith(('.py','.json','.map')):continue
            dest=run/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=RUN);args=parser.parse_args()
    run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=True);cfg=config();provenance(run,cfg)
    import torch
    import numpy as np
    from model import initialize,load,setup_torch
    from training import train,base_hash
    setup_torch();device='cuda' if torch.cuda.is_available() else 'cpu';start=time.perf_counter()
    print('DEVICE',device,torch.cuda.get_device_name(0) if device=='cuda' else '',flush=True)
    candidates={};baselines={};branches={};rollouts={};validation={};feeding={};selected={};tests={};training={};test_controls={}
    with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1) as ex:
        for name,source in SOURCES.items():
            initials=[]
            for seed in cfg['training_seeds']:
                p=run/f'{name}-initial-{seed}.pt'
                if not p.exists():initialize(source,p,seed)
                initials.append(p)
            candidates[name]={'rule':dict(weights=str(initials[0]),controller='rule'),
                'aggressive':dict(weights=str(initials[0]),controller='aggressive')}
            for control in ('rule','aggressive'):
                label=f'baseline-{name}-{control}'
                items=jobs(cfg['baseline_maps'],[cfg['baseline_seed']],initials[0],control,capture=control=='rule',save_replay=True)
                baselines[label]=collect_jobs(ex,items,label,run)
            prefs=[]
            for i,row in enumerate(baselines[f'baseline-{name}-rule']):
                if not row['captures']:continue
                c=row['captures'][0]
                with np.load(c['path']) as d:valid=np.flatnonzero(d['mask'][1:])+1
                if not len(valid):continue
                alternative=0 if c['action'] else int(valid[0])
                override=dict(round=c['round'],dragon=c['dragon'],prefix_sha256=c['prefix_sha256'],action=alternative)
                item=dict(map=row['map'],seed=row['seed'],side=row['side'],weights=str(initials[0]),controller='rule',collect=False,override=override,save_replay=True)
                label=f'branch-{name}-{i}';other=collect_jobs(ex,[item],label,run)[0];branches[label]=other
                prefs.append(dict(capture_path=c['path'],baseline_action=c['action'],alternative_action=alternative,
                    baseline_reward=row['reward'],alternative_reward=other['reward'],map=row['map'],side=row['side'],seed=row['seed']))
            preference_path=run/f'preferences-{name}.json';save(preference_path,prefs)
            for si,seed in enumerate(cfg['training_seeds']):
                current=initials[si]
                for update in range(1,cfg['updates']+1):
                    offset=((update-1)%2)*2;label=f'{name}-s{seed}-rollout-{update}'
                    items=jobs(cfg['train_maps'][offset:offset+2],[cfg['rollout_seeds'][si][update-1]],current,collect=True,stochastic=True,sampling_seed=seed,save_replay=True)
                    rows=collect_jobs(ex,items,label,run);rollouts[label]=rows
                    paths=[run/'episodes'/f'{label}-{i}.npz' for i in range(len(rows))]
                    current,metrics=train(current,paths,run/f'{name}-s{seed}-ppo-{update}.pt',cfg,seed,update,device,preference_path)
                    training[current.stem]=metrics
                candidates[name][f'ppo-{seed}']=dict(weights=str(current),controller='network')
        # Validation chooses split policy. No independent test is opened during selection.
        for name,choices in candidates.items():
            validation[name]={}
            for key,choice in choices.items():
                label=f'val-{name}-{key}'
                rows=collect_jobs(ex,jobs(cfg['validation_maps'],cfg['validation_seeds'],**choice,save_replay=True),label,run)
                validation[name][key]=dict(score=score(rows),rows=rows)
                print('VALIDATION',name,key,score(rows),flush=True)
            key=max(choices,key=lambda k:validation[name][k]['score']['points'])
            selected[name]=dict(key=key,**choices[key])
            normal=validation[name][key]['rows'][:4]
            dynamic=collect_jobs(ex,jobs(cfg['validation_maps'],cfg['validation_seeds'][:1],**choices[key],feeding='dynamic',save_replay=True),f'feed-{name}-dynamic',run)
            feeding[name]=dict(original=dict(score=score(normal),rows=normal),dynamic=dict(score=score(dynamic),rows=dynamic))
            selected[name]['feeding']='dynamic' if score(dynamic)['points']>score(normal)['points'] else 'original'
        selection=dict(selected=selected,validation={n:{k:v['score'] for k,v in choices.items()} for n,choices in validation.items()},
            feeding={n:{k:v['score'] for k,v in variants.items()} for n,variants in feeding.items()},
            checkpoint_sha256={n:sha(v['weights']) for n,v in selected.items()},tie_rule='Original rule, aggressive rule, lower training seed; original feeding on ties.')
        selection_path=run/'selection.json'
        if selection_path.exists():assert json.loads(selection_path.read_text())==selection,'Selection changed after independent tests'
        else:save(selection_path,selection)
        print('FROZEN SELECTION',json.dumps(selection),flush=True)
        for name,choice in selected.items():
            options={k:v for k,v in choice.items() if k!='key'}
            rows=collect_jobs(ex,jobs(cfg['test_maps'],cfg['test_seeds'],**options,save_replay=True),f'test-{name}',run)
            tests[name]=dict(score=score(rows),rows=rows);print('TEST',name,score(rows),flush=True)
            if choice['key']=='rule' and choice['feeding']=='original':
                test_controls[name]=dict(score=score(rows),rows=rows,reused=True)
            else:
                baseline=collect_jobs(ex,jobs(cfg['test_maps'],cfg['test_seeds'],**candidates[name]['rule'],save_replay=True),f'test-control-{name}',run)
                test_controls[name]=dict(score=score(baseline),rows=baseline,reused=False)
    counts={};frozen={}
    for name,choices in candidates.items():
        original=base_hash(load(choices['rule']['weights']))
        for key,v in choices.items():
            m=load(v['weights']);assert base_hash(m)==original
            counts[name+'-'+key]=dict(total=sum(p.numel() for p in m.parameters()),trainable=sum(p.numel() for p in m.parameters() if p.requires_grad))
        frozen[name]=original
    result=dict(baselines=baselines,branches=branches,rollouts=rollouts,training=training,validation=validation,feeding=feeding,selection=selection,
        tests=tests,test_controls=test_controls,parameters=counts,frozen_base_sha256=frozen,invocation_seconds=time.perf_counter()-start,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        checkpoint_sha256={f'{n}-{k}':sha(v['weights']) for n,cs in candidates.items() for k,v in cs.items()})
    result['total_games']=sum(len(v) for v in baselines.values())+len(branches)+sum(len(v) for v in rollouts.values())+sum(len(v['rows']) for vs in validation.values() for v in vs.values())+sum(len(v['dynamic']['rows']) for v in feeding.values())+sum(len(v['rows']) for v in tests.values())
    result['total_games']+=sum(len(v['rows']) for v in test_controls.values() if not v['reused'])
    provenance(run,cfg);save(run/'results.json',result);print('COMPLETE',run/'results.json',flush=True)

if __name__=='__main__':main()
