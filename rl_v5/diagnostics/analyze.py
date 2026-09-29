"""Aggregate actual completed matches; can safely run while later games collect."""
from pathlib import Path
from collections import defaultdict,Counter
import json,sys,statistics
ROOT=Path(__file__).resolve().parents[2]

def metrics(row):
    tp=Path(row['telemetry_path']);t=json.loads(tp.read_text());side=row['side'];audit=row['audit'];hist=t['histories']
    out=dict(reward=row['reward'],seconds=row['elapsed_seconds'],rounds=row['result']['rounds'],
        opportunities=audit.get('opportunities',0),strategic_splits=audit.get('strategic_splits',0),
        gate_disagreements=audit.get('gate_disagreements',0),different_sizes=audit.get('different_sizes',0),
        protected_splits=audit.get('protected_splits',0),feed_detonations=audit.get('feed_detonations',0),
        turns=sum(map(len,hist.values())),lineage_errors=len(row['lineage_errors']),errors=len(row['errors']),
        dynamic_home_turns=audit.get('dynamic_home_turns',0),dynamic_king_ack_messages=audit.get('dynamic_king_ack_messages',0),dynamic_ack_detonations=audit.get('dynamic_ack_detonations',0),local_king_reassigned=audit.get('local_king_reassigned',0))
    out['phase']=[dict(opportunities=audit.get(f'opportunity_bucket_{b}',0),splits=audit.get(f'split_bucket_{b}',0)) for b in range(5)]
    surv=[];growth=[]
    for child,p in t['parents'].items():
        parent=hist[str(p['parent'])];idx=p['parent_history']
        if idx<0 or parent[idx]['policy_action']==0:continue
        at=p['round'];end=at+20;death=t['deaths'].get(child,{}).get('round',10000)
        if end<=row['result']['rounds'] or death<=row['result']['rounds']:
            surv.append(float(death>end));growth.append(sum(max(0,h['growth']) for h in hist.get(child,[]) if at<h['round']<=end))
    out.update(child20_observed=len(surv),child20_survived=sum(surv),child20_positive_net_growth=sum(growth))
    rp=tp.with_name(tp.name.replace('.telemetry.json','.replay-metrics.json'))
    if rp.exists():
        r=json.loads(rp.read_text());own=r['final'][side];enemy=r['final']['B' if side=='A' else 'A']
        ds=[d for d in r['deaths'] if d['team']==side and d['feed']]
        out.update(final_longest=own['longestDragon'],final_total=own['totalLength'],final_dragons=own['dragonCount'],enemy_longest=enemy['longestDragon'],
            feed_deposited=sum(d['deposited'] for d in ds),feed_recovered_ally=sum(d['recoveredSameTeam'] for d in ds),
            feed_recovered_enemy=sum(d['recoveredOtherTeam'] for d in ds),feed_direct_final_longest=sum(d['recoveredByFinalLongest'] for d in ds),
            unmatched_pearl_removals=r['consumption']['unmatchedRemovals'])
        out['curve']={}
        for at in (0,100,200,300,400,450,475,499):
            rows=[x for x in r['curves'] if x['round']<=at]
            if rows:out['curve'][str(at)]=rows[-1][side]
    return out

def aggregate(rows):
    totals=Counter();phase=[Counter() for _ in range(5)];curves=defaultdict(list)
    for r in rows:
        totals.update({k:v for k,v in r.items() if isinstance(v,(int,float))})
        for i,p in enumerate(r['phase']):phase[i].update(p)
        for k,v in r.get('curve',{}).items():curves[k].append(v)
    n=len(rows);out=dict(games=n,wins=sum(r['reward']==1 for r in rows),draws=sum(r['reward']==0 for r in rows),losses=sum(r['reward']==-1 for r in rows),totals=dict(totals))
    out['means']={k:v/n for k,v in totals.items()}
    out['child20_survival']=totals['child20_survived']/max(1,totals['child20_observed'])
    out['child20_growth']=totals['child20_positive_net_growth']/max(1,totals['child20_observed'])
    out['feed_recovery']=totals['feed_recovered_ally']/max(1,totals['feed_deposited'])
    out['phase']=[dict(p,split_rate=p['splits']/max(1,p['opportunities'])) for p in phase]
    out['curve_mean']={at:{k:statistics.mean(x[k] for x in vs) for k in vs[0]} for at,vs in curves.items()}
    return out

def main():
    run=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'rl_v5/runs/structured-split'
    groups=defaultdict(list);records={}
    for p in sorted((run/'episodes').glob('*.json')):
        if p.name.endswith(('.telemetry.json','.replay-metrics.json')):continue
        j=json.loads(p.read_text())
        if 'summary' not in j:continue
        row=j['summary'];m=metrics(row);records[p.stem]=m;groups[p.stem.rsplit('-',1)[0]].append(m)
    output=dict(complete=(run/'results.json').exists(),completed_games=len(records),groups={k:aggregate(v) for k,v in groups.items()},records=records)
    (run/'analysis.json').write_text(json.dumps(output,indent=2));print(json.dumps({k:output[k] for k in ('complete','completed_games')}))
if __name__=='__main__':main()
