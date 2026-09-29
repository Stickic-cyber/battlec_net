"""Serial official engine; frozen movement; split-only records and causal event telemetry."""
from runtime import ROOT,BOT
from base_observation import Observer
from base_actions import prepare,commit
from features import observe,aggressive_choice,SHAPES
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
import importlib.util,sys,io,json,time,hashlib
import numpy as np
from unswbc.engine import EngineModule
from unswbc.bot import ZYGOTE
from unswbc.sandbox import SandboxPool,SandboxBot,warm_interpreter

def module_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

class Dragon:
    def __init__(self,i,init,feed_variant):
        self.helper=module_file(f'helper_v5_{i}',BOT/'helper.py');old=sys.modules.get('helper');sys.modules['helper']=self.helper
        try:self.rules=module_file(f'rules_v5_{i}',BOT/'rule_core.py')
        finally:
            if old is None:sys.modules.pop('helper',None)
            else:sys.modules['helper']=old
        stdin=sys.stdin
        try:sys.stdin=io.StringIO(init.decode());self.rules.ct,self.rules.game=self.helper.init()
        finally:sys.stdin=stdin
        if feed_variant!='original':
            from feeding import install
            install(self.rules,feed_variant)
        self.observer=Observer();self.hidden=np.zeros(256,np.float32);self.records=[];self.history=[];self.audit=Counter()

    def ask(self,i,block,job,policy,rng,prefix):
        stdin=sys.stdin;out=io.StringIO();source='rule';pa=0;features=None;capture=None;choice=None
        try:
            sys.stdin=io.StringIO(block.decode());self.helper.update(self.rules.ct,self.rules.game)
            with redirect_stdout(out):
                r=self.rules;r._dynamic_next_destination=None
                choice=prepare(r);obs=self.observer.observe(r,choice);h_in=self.hidden.copy()
                move,self.hidden,move_lp=policy.movement(obs,h_in)
                active=choice is not None and any(choice.size_mask)
                if active:
                    features=observe(r,self.observer,choice,self.hidden)
                    pa,lp,value,aux=policy.act(features,rng if job.get('stochastic') else None)
                    if job.get('controller')=='rule':pa=choice.teacher-3 if choice.teacher>=4 else 0
                    if job.get('controller')=='aggressive':pa=aggressive_choice(features,choice,r)
                    override=job.get('override')
                    if override and override['round']==r.game.get_round_num() and override['dragon']==i:
                        if prefix!=override['prefix_sha256']:raise RuntimeError('Counterfactual prefix diverged before intervention')
                        pa=override['action'];self.audit['interventions']+=1
                    if not features['mask'][pa]:raise RuntimeError('Illegal split policy action')
                    # Control/override cases are never used as on-policy PPO samples.
                    if job.get('collect'):
                        if job.get('controller')!='network' or override:raise RuntimeError('Off-policy data requested for PPO')
                        self.records.append({'obs':features,'action':pa,'logp':lp,'value':value,'round':r.game.get_round_num(),
                            'history_index':len(self.history),'teacher':choice.teacher-3 if choice.teacher>=4 else 0})
                    if job.get('capture') and 15<=r.game.get_round_num()<=220 and r.ct.get_length()>=6:
                        if features['candidates'][1:,10].max()>=.25:
                            capture={'dragon':i,'round':r.game.get_round_num(),'prefix_sha256':prefix,'action':pa,
                                'sizes':choice.sizes,'features':features}
                    self.audit['opportunities']+=1;self.audit['rule_recommendations']+=int(choice.recommended>0)
                    self.audit[f'opportunity_bucket_{r.game.get_round_num()//100}']+=1
                    self.audit['gate_disagreements']+=int(bool(pa)!=(choice.recommended>0))
                    if pa:
                        self.audit['strategic_splits']+=1;self.audit[f'split_bucket_{r.game.get_round_num()//100}']+=1
                        self.audit['child_segments']+=choice.sizes[pa-1]
                        self.audit['different_sizes']+=int(choice.recommended>0 and choice.sizes[pa-1]!=choice.recommended)
                if choice is not None:
                    actual=pa+3 if active and pa else move
                    source,size=commit(r,choice,actual)
                self.helper.end_turn()
                pos=r.ct.get_position();length=r.ct.get_length();rnd=r.game.get_round_num()
                role=choice.role if choice is not None else ('KING' if obs['global'][36] else 'OTHER')
                row={'round':rnd,'length':length,'x':pos.x,'y':pos.y,'role':role,'source':source,
                    'body_known':len(r.own_body),'policy_action':int(pa),'row':len(self.records)-1 if active and job.get('collect') else -1,
                    'head_king_distance':float(features['context'][12]*64) if features is not None else None}
        finally:sys.stdin=stdin
        data=out.getvalue().encode();self.observer.executed(data)
        row['action']=next((line for line in data.decode().splitlines() if line.startswith(('MOVE ','SPLIT '))),'NONE')
        row['feed']=b'RULE_FEED' in data or b'DYNAMIC_FEED' in data
        row['split_size']=int(row['action'].split()[1]) if row['action'].startswith('SPLIT ') else 0
        if source=='rule' and row['split_size']:self.audit['protected_splits']+=1
        if row['feed']:self.audit['feed_detonations']+=1
        self.history.append(row)
        return data,row,capture

def episode(job):
    warm_interpreter()
    from model import Policy
    policy=Policy(job['weights']);side=job['side'];seed=job['seed'];other='B' if side=='A' else 'A'
    pool=SandboxPool([sys.executable,ZYGOTE,str(ROOT/'方案1/main.py')],cwd=str(ROOT/'方案1'),key=f'v5-{seed}-{other}')
    dragons={};bots={};all_dragons={};rngs={};parents={};deaths={};sources=Counter();errors=[];captures=[]
    prefix=hashlib.sha256();last_reply=None;first_seen=set();birth_errors=[];last_round_notice=-1
    stats={'max_opponent_cpu':0,'max_opponent_memory':0,'opponent_calls':0}
    def spawn(i,init):
        nonlocal last_reply,last_round_notice
        team=next(l.split()[1] for l in init.decode().splitlines() if l.startswith('TEAM '))
        prefix.update(b'SPAWN'+str(i).encode()+init)
        if team==side:
            dragons[i]=Dragon(i,init,job.get('feeding','original'));all_dragons[i]=dragons[i]
            rngs[i]=np.random.default_rng(np.random.SeedSequence([seed,i,job.get('sampling_seed',991)]))
            if last_reply and last_reply['team']==side and last_reply['size'] and not last_reply.get('used'):
                parents[i]={'parent':last_reply['dragon'],'round':last_reply['round'],'size':last_reply['size'],
                    'parent_history':last_reply['history']};last_reply['used']=True
        else:bots[i]=SandboxBot(pool,init=init,name=str(i))
    def reply(i,block):
        nonlocal last_reply,last_round_notice
        if i in bots:
            b=bots[i];data=b.ask(block);stats['opponent_calls']+=1
            if b.error:errors.append({'dragon':i,'error':b.error})
            if b.live:
                stats['max_opponent_cpu']=max(stats['max_opponent_cpu'],b.live[0]);stats['max_opponent_memory']=max(stats['max_opponent_memory'],b.live[1])
            row={'round':int(block.splitlines()[0].split()[1]),'split_size':0};team=other;history=-1
        else:
            data,row,capture=dragons[i].ask(i,block,job,policy,rngs[i],prefix.hexdigest());sources[row['source']]+=1;team=side;history=len(dragons[i].history)-1
            if i not in first_seen:
                first_seen.add(i)
                if i in parents and row['length']!=parents[i]['size']:birth_errors.append({'dragon':i,'expected':parents[i]['size'],'actual':row['length']})
            if capture and len(captures)<job.get('capture_limit',1):captures.append(capture)
        split=next((int(l.split()[1]) for l in data.splitlines() if l.startswith(b'SPLIT ')),0)
        last_reply={'dragon':i,'team':team,'round':row['round'],'size':split,'history':history}
        if job.get('progress',True) and row['round']//100>last_round_notice:
            last_round_notice=row['round']//100
            print(f"ROUND {job['map']} {side} seed={seed} round={row['round']} learner_alive={len(dragons)} sec={time.perf_counter()-start:.1f}",flush=True)
        prefix.update(str(i).encode()+b'\0'+block+b'\0'+data)
        return data
    def death(i,rnd,reason):
        deaths[i]={'round':rnd,'reason':reason,'learner':i in all_dragons};prefix.update(f'DEATH {i} {rnd} {reason}'.encode())
        dragons.pop(i,None);b=bots.pop(i,None)
        if b is not None:b.stop()
    start=time.perf_counter();engine=EngineModule()
    try:result=engine.run((ROOT/'maps'/job['map']).read_bytes(),reply,bot_spawn=spawn,on_death=death,on_notice=lambda _:None,debug=0,seed=seed)
    finally:
        for b in bots.values():b.stop()
        pool.close()
    if errors or birth_errors:raise RuntimeError(f'Execution/lineage errors: {errors} {birth_errors}')
    if job.get('override') and sum(d.audit['interventions'] for d in all_dragons.values())!=1:raise RuntimeError('Intervention did not occur exactly once')
    reward=0 if result.winner is None else (1 if result.winner==side else -1)
    folder=Path(job['output']).parent;folder.mkdir(parents=True,exist_ok=True);stem=Path(job['output']).stem
    if job.get('save_replay'):(folder/f'{stem}.replay').write_bytes(engine.replay('learner' if side=='A' else 'rules','rules' if side=='A' else 'learner'))
    audits=Counter()
    for d in all_dragons.values():
        audits.update(d.audit);audits.update(getattr(d.rules,'_dynamic_audit',{}))
    # Track length growth after accounting for the previous split. This is a net-growth proxy, not food provenance.
    for i,d in all_dragons.items():
        for index,row in enumerate(d.history):
            row['growth']=0 if index==0 else row['length']-d.history[index-1]['length']+d.history[index-1]['split_size']
    telemetry={'parents':parents,'deaths':deaths,'histories':{i:d.history for i,d in all_dragons.items()},'result':asdict(result)}
    tp=folder/f'{stem}.telemetry.json';tp.write_text(json.dumps(telemetry,separators=(',',':')),encoding='utf-8')
    capture_rows=[]
    for j,c in enumerate(captures):
        cp=folder/f'{stem}.capture-{j}.npz';np.savez_compressed(cp,**c.pop('features'));c['path']=str(cp);c['sha256']=hashlib.sha256(cp.read_bytes()).hexdigest();capture_rows.append(c)
    summary={'map':job['map'],'side':side,'seed':seed,'reward':reward,'result':asdict(result),'sources':dict(sources),
        'audit':dict(audits),'errors':errors,'lineage_count':len(parents),'lineage_errors':birth_errors,
        'telemetry_path':str(tp),'telemetry_sha256':hashlib.sha256(tp.read_bytes()).hexdigest(),'captures':capture_rows,
        'prefix_sha256':prefix.hexdigest(),'opponent_budget':stats,'elapsed_seconds':round(time.perf_counter()-start,3)}
    if job.get('collect'):
        flat=[];offsets=[0];ids=[]
        def outcome(actor,at,initial_index=None):
            history=all_dragons[actor].history;death_at=deaths.get(actor,{}).get('round',10000)
            end=at+20;known=end<=result.rounds or death_at<=result.rounds
            growth=sum(max(0,x['growth']) for x in history if at<x['round']<=end)
            return float(death_at>end),min(growth,20)/20,float(known)
        child_at={(p['parent'],p['parent_history']):child for child,p in parents.items()}
        for i,d in all_dragons.items():
            if not d.records:continue
            ids.append(i)
            for rec in d.records:
                at=rec['round'];child=child_at.get((i,rec['history_index'])) if rec['action'] else i
                ps,pg,pk=outcome(i,at)
                if child is not None:cs,cg,ck=outcome(child,at)
                else:cs=cg=ck=0.
                rec.update(aux=np.asarray([cs,cg,ps],np.float32),aux_mask=np.asarray([ck,ck,pk],bool))
                flat.append(rec)
            offsets.append(len(flat))
        if not flat:raise RuntimeError('No strategic split opportunities')
        arrays={k:np.stack([r['obs'][k] for r in flat]) for k in SHAPES}
        for k in ('action','teacher','round'):arrays[k]=np.asarray([r[k] for r in flat],np.int64)
        for k in ('logp','value'):arrays[k]=np.asarray([r[k] for r in flat],np.float32)
        arrays.update(aux=np.stack([r['aux'] for r in flat]),aux_mask=np.stack([r['aux_mask'] for r in flat]),
            offsets=np.asarray(offsets,np.int64),dragon_ids=np.asarray(ids,np.int64),team_reward=np.asarray(reward,np.float32))
        output=Path(job['output']);temp=output.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(output)
        summary.update(decisions=len(flat),sampled_splits=int((arrays['action']>0).sum()),data_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    return summary
