"""Feedback-driven continuation; run only after the immutable reference batch.

Pilot choices use development games only. Refined teachers use new on-policy
rollouts. Fresh test seeds are fixed here before reference tests are observed.
All actual matches remain serial in fresh workers.
"""
from pathlib import Path
import sys,json,hashlib,shutil,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'rl_v5'))
from runtime import hashes,config
from collect import jobs,collect_jobs
from runner import episode
from concurrent.futures import ProcessPoolExecutor

REFERENCE=ROOT/'rl_v5/runs/structured-split'
RUN=ROOT/'rl_v5/runs/refined-teachers'
PROTOCOL=dict(pilot_maps=['arena.map','schooltime.map'],pilot_seed=9151,
    training_maps=[['arena.map','default_small.map'],['autarky.map','schooltime.map']],training_seeds=[9701,9703],
    validation_maps=['dilemma.map','queen_of_spades.map'],validation_seeds=[9411,9417],
    test_maps=['Colosseum.map','devil.map'],test_seeds=[9811,9817],
    execution_comparison='Evaluate refined greedy, refined stochastic, and untrained stochastic-prior split policies on the same 8 validation cases. Stochastic sampling_seed=991, keyed by game seed and dragon ID. This separates learning from random expansion. Select before all new tests.',
    tie_rules='Pilot tie prefers phase weighting. Split ties prefer reference, then refined greedy, untrained stochastic prior, refined stochastic, untrained parent3 prior, refined parent3. Feeding ties prefer original, then dynamic, then dynamic_v2.',
    parent_guard='Added from reference VALIDATION evidence before any independent tests: GRU s9201 made 85/99 splits at length4 into2+2, versus an aggressive rule retaining parent>=3. Compare deterministic refined and initial prior with parent_min=3 on the same eight validation cases. Apply an output probability constraint only; inputs, protected rule splits, and training remain unchanged. If selected, test a matched initial prior with the same constraint and feeding. No other guard thresholds are searched.',
    stochastic_test_control='Whenever the selected policy samples split actions, compare it with the untrained stochastic prior on the same fresh test cases and the SAME selected feeding rule. Reuse rows when the selected policy is that prior. Never use this test to alter selection.',
    preference_supervision='Retain the reference preference coefficient and its verified WLD-changing one-action labels during continuation. No intermediate-growth proxy is converted into a preference label.',
    feeding_confirmation='Screen feeding on 4 paired validation games; confirm the winner on the other 4. Retain original feeding unless the combined 8-game score strictly improves. Choose the preferred architecture on the final combined 8-game validation score before new tests.',
    rule='Never use reference or new independent test outcomes for adjustment or selection. New tests use fresh seeds, not new map geometries.')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def score(rows):
    a=[r['reward'] for r in rows]
    return dict(wins=a.count(1),draws=a.count(0),losses=a.count(-1),points=sum((x+1)/2 for x in a),games=len(a))
def sources():
    result=hashes()
    for p in sorted(Path(__file__).parent.glob('*.py')):result[str(p.relative_to(ROOT))]=sha(p)
    return result

def refined_episode(job):
    if job.get('feeding')=='dynamic_v2':
        import feeding_v2
        sys.modules['feeding']=feeding_v2
    import model
    original_policy=model.Policy
    if job.get('parent_min'):
        assert not job.get('collect') and job.get('controller')=='network','Parent guard is evaluation-only'
        from parent_guard import guarded_policy
        model.Policy=guarded_policy(original_policy,job['parent_min'])
    try:return episode(job)
    finally:model.Policy=original_policy

class SerialExecutor:
    def __init__(self,executor):self.executor=executor
    def submit(self,unused,job):return self.executor.submit(refined_episode,job)

def update(initial,paths,output,cfg,seed,method,preference_path):
    import numpy as np,torch
    import training
    from training import train
    from phase_train import fit
    from fine_project import project
    fingerprint=dict(initial_sha256=sha(initial),data_sha256=[sha(p) for p in paths],preference_sha256=sha(preference_path),cfg=cfg,seed=seed,method=method,sources=sources())
    cache=output.with_suffix('.update.json');final=output.with_name(output.stem+'-fine.pt')
    if cache.exists():
        c=json.loads(cache.read_text());assert c['inputs']==fingerprint and sha(final)==c['checkpoint_sha256'],'Stale refinement update'
        return final,c['projection']
    if method=='phase':
        # Keep the already-audited phase fitter unchanged; supply the original
        # causal preference dataset to its training call within this process.
        original_train=training.train
        def with_preferences(*args,**kwargs):
            kwargs['preference_path']=preference_path
            return original_train(*args,**kwargs)
        training.train=with_preferences
        try:fit(initial,paths,output,cfg,seed)
        finally:training.train=original_train
    else:
        raw=output.with_name(output.stem+'-raw.pt')
        _,metrics=train(initial,paths,raw,cfg,seed,1,'cuda' if torch.cuda.is_available() else 'cpu',preference_path)
        distribution=[]
        for p in paths:
            with np.load(p) as d:b,c=np.unique(d['round']//100,return_counts=True)
            distribution.append(dict(file=str(p),counts={int(x):int(y) for x,y in zip(b,c)}))
        save(output.with_suffix('.json'),dict(initial=str(initial),initial_sha256=sha(initial),data_sha256=[sha(p) for p in paths],
            sampling=dict(distribution=distribution),config=cfg,raw_training=metrics))
    final,projection=project(output.with_suffix('.json'))
    save(cache,dict(inputs=fingerprint,checkpoint_sha256=sha(final),projection=projection));return final,projection

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=RUN);p.add_argument('--pilot-only',action='store_true');args=p.parse_args()
    assert (REFERENCE/'results.json').exists(),'Complete the reference run before starting more matches'
    reference=json.loads((REFERENCE/'results.json').read_text());run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    fixed=sources();manifest=dict(sources=fixed,protocol=PROTOCOL,reference_results_sha256=sha(REFERENCE/'results.json'))
    mp=run/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Immutable refinement sources changed'
    else:
        save(mp,manifest)
        for rel in fixed:
            dest=run/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    def make_jobs(*a,**kw):return jobs(*a,**kw,refinement_sources=fixed,save_replay=True)
    import torch
    from model import setup_torch
    setup_torch();cfg=config();cfg.update(lr=.0012,max_lr=.0012,epochs=12,kl_stop=.01,
        refinement_protocol=PROTOCOL,refinement_driver_sha256=sha(__file__))
    pilots={};rollouts={};updates={};validation={};execution={};feeding={};selected={};tests={};controls={};prior_controls={};confirmation={};final_validation={}
    with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1) as raw_executor:
        ex=SerialExecutor(raw_executor)
        for key,file in [('phase','spatial-s9201-batch2-fine.pt'),('match','spatial-s9201-batch2-highlr-fine.pt')]:
            weights=ROOT/'rl_v5/runs/phase-balanced'/file
            assert sha(weights)==json.loads(weights.with_suffix('.json').read_text())['checkpoint_sha256']
            rows=collect_jobs(ex,make_jobs(PROTOCOL['pilot_maps'],[PROTOCOL['pilot_seed']],weights),f'pilot-{key}',run)
            pilots[key]=dict(score=score(rows),rows=rows,checkpoint_sha256=sha(weights))
        method='match' if pilots['match']['score']['points']>pilots['phase']['score']['points'] else 'phase'
        save(run/'pilot-selection.json',dict(pilots=pilots,method=method,reference_rule=score(reference['baselines']['baseline-spatial-rule']),
            reference_aggressive=score(reference['baselines']['baseline-spatial-aggressive'])))
        print('PILOT METHOD',method,{k:v['score'] for k,v in pilots.items()},flush=True)
        if args.pilot_only:return
        cfg['loss_weighting']=method
        for name in ('spatial','gru'):
            initial=REFERENCE/f'{name}-s9201-ppo-3.pt'
            preferences=REFERENCE/f'preferences-{name}.json'
            assert sha(preferences)==reference['training'][f'{name}-s9201-ppo-3']['inputs']['preference_sha256']
            paths=[REFERENCE/'episodes'/f'{name}-s9201-rollout-4-{i}.npz' for i in range(4)]
            current,diagnostics=update(initial,paths,run/f'{name}-alternate-update4.pt',cfg,24001,method,preferences)
            updates[current.stem]=diagnostics
            for u,(maps,seed) in enumerate(zip(PROTOCOL['training_maps'],PROTOCOL['training_seeds']),1):
                label=f'{name}-refine-rollout-{u}'
                items=make_jobs(maps,[seed],current,collect=True,stochastic=True,sampling_seed=24001)
                rows=collect_jobs(ex,items,label,run);rollouts[label]=rows
                paths=[run/'episodes'/f'{label}-{i}.npz' for i in range(len(items))]
                current,diagnostics=update(current,paths,run/f'{name}-refine-{u}.pt',cfg,24001+u,method,preferences);updates[current.stem]=diagnostics
            rows=collect_jobs(ex,make_jobs(PROTOCOL['validation_maps'],PROTOCOL['validation_seeds'],current),f'val-{name}-refined',run)
            original=reference['selection']['selected'][name]
            baseline=reference['validation'][name][original['key']]
            validation[name]=dict(refined=dict(score=score(rows),rows=rows,weights=str(current)),reference=baseline)
            choices={'reference':{k:v for k,v in original.items() if k!='feeding'},
                'refined':dict(weights=str(current),controller='network',key='refined'),
                'initial-stochastic':dict(weights=str(REFERENCE/f'{name}-initial-9201.pt'),controller='network',key='initial-stochastic',stochastic=True,sampling_seed=991),
                'refined-stochastic':dict(weights=str(current),controller='network',key='refined-stochastic',stochastic=True,sampling_seed=991),
                'initial-parent3':dict(weights=str(REFERENCE/f'{name}-initial-9201.pt'),controller='network',key='initial-parent3',parent_min=3),
                'refined-parent3':dict(weights=str(current),controller='network',key='refined-parent3',parent_min=3)}
            execution[name]={'reference':baseline,'refined':validation[name]['refined']}
            for key in ('refined-stochastic','initial-stochastic','refined-parent3','initial-parent3'):
                extra_rows=collect_jobs(ex,make_jobs(PROTOCOL['validation_maps'],PROTOCOL['validation_seeds'],**{k:v for k,v in choices[key].items() if k!='key'}),f'val-{name}-{key}',run)
                execution[name][key]=dict(score=score(extra_rows),rows=extra_rows)
            chosen_key=max(choices,key=lambda k:execution[name][k]['score']['points'])
            choice=dict(choices[chosen_key]);normal_all=execution[name][chosen_key]['rows'];normal=normal_all[:4]
            options={k:v for k,v in choice.items() if k!='key'}
            if chosen_key=='reference':dynamic=reference['feeding'][name]['dynamic']['rows']
            else:dynamic=collect_jobs(ex,make_jobs(PROTOCOL['validation_maps'],PROTOCOL['validation_seeds'][:1],**options,feeding='dynamic'),f'feed-{name}-dynamic',run)
            revised=collect_jobs(ex,make_jobs(PROTOCOL['validation_maps'],PROTOCOL['validation_seeds'][:1],**options,feeding='dynamic_v2'),f'feed-{name}-dynamic-v2',run)
            feeding[name]={k:dict(score=score(v),rows=v) for k,v in [('original',normal),('dynamic',dynamic),('dynamic_v2',revised)]}
            winner=max(feeding[name],key=lambda k:feeding[name][k]['score']['points']);choice['feeding']=winner;final_rows=normal_all
            if winner!='original':
                extra=collect_jobs(ex,make_jobs(PROTOCOL['validation_maps'],PROTOCOL['validation_seeds'][1:],**options,feeding=winner),f'confirm-feed-{name}-{winner}',run)
                candidate_rows=feeding[name][winner]['rows']+extra
                confirmation[name]=dict(candidate=winner,score=score(candidate_rows),rows=candidate_rows,original_score=score(normal_all))
                if score(candidate_rows)['points']>score(normal_all)['points']:final_rows=candidate_rows
                else:choice['feeding']='original'
            final_validation[name]=dict(score=score(final_rows),rows=final_rows);selected[name]=choice
            print('REFINED VALIDATION',name,validation[name]['refined']['score'],'FEEDING',choice['feeding'],flush=True)
        preferred=max(selected,key=lambda n:final_validation[n]['score']['points'])
        selection=dict(selected=selected,method=method,preferred_architecture=preferred,final_validation_scores={n:v['score'] for n,v in final_validation.items()},
            checkpoint_sha256={k:sha(v['weights']) for k,v in selected.items()},protocol=PROTOCOL)
        sp=run/'selection.json'
        if sp.exists():assert json.loads(sp.read_text())==selection
        else:save(sp,selection)
        print('FROZEN REFINEMENT SELECTION',json.dumps(selection),flush=True)
        for name,choice in selected.items():
            options={k:v for k,v in choice.items() if k!='key'}
            rows=collect_jobs(ex,make_jobs(PROTOCOL['test_maps'],PROTOCOL['test_seeds'],**options),f'test-{name}',run)
            tests[name]=dict(score=score(rows),rows=rows)
            old=reference['selection']['selected'][name]
            signature=lambda x:(sha(x['weights']),x['controller'],x['feeding'],x.get('stochastic',False),x.get('sampling_seed',991),x.get('parent_min'))
            if signature(choice)==signature(old):controls[name]=dict(score=score(rows),rows=rows,reused=True)
            else:
                baseline=collect_jobs(ex,make_jobs(PROTOCOL['test_maps'],PROTOCOL['test_seeds'],**{k:v for k,v in old.items() if k!='key'}),f'test-control-{name}',run)
                controls[name]=dict(score=score(baseline),rows=baseline,reused=False)
            print('REFINED TEST',name,tests[name]['score'],'REFERENCE',controls[name]['score'],flush=True)
            if choice.get('stochastic') or choice.get('parent_min'):
                if choice['key'].startswith('initial-'):prior_controls[name]=dict(score=score(rows),rows=rows,reused=True)
                else:
                    matched={k:choice[k] for k in ('stochastic','sampling_seed','parent_min') if k in choice}
                    prior=collect_jobs(ex,make_jobs(PROTOCOL['test_maps'],PROTOCOL['test_seeds'],REFERENCE/f'{name}-initial-9201.pt',
                        controller='network',feeding=choice['feeding'],**matched),f'test-prior-{name}',run)
                    prior_controls[name]=dict(score=score(prior),rows=prior,reused=False)
                print('MATCHED INITIAL PRIOR TEST',name,prior_controls[name]['score'],flush=True)
    assert sources()==fixed
    result=dict(protocol=PROTOCOL,pilots=pilots,method=method,rollouts=rollouts,updates=updates,validation=validation,execution=execution,feeding=feeding,confirmation=confirmation,final_validation=final_validation,
        selection=selection,tests=tests,test_controls=controls,test_prior_controls=prior_controls)
    # All new games have one collector metadata file, including cached resumed games.
    completed=[]
    for f in (run/'episodes').glob('*.json'):
        if f.name.endswith(('.telemetry.json','.replay-metrics.json')):continue
        obj=json.loads(f.read_text())
        if 'job' in obj:completed.append(f.name)
    result['total_games']=len(completed);save(run/'results.json',result);print('REFINEMENT COMPLETE',run/'results.json',flush=True)

if __name__=='__main__':main()
