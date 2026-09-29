"""Compare verified single-action interventions, without turning proxies into wins."""
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[2]

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    run=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'rl_v5/runs/structured-split'
    rows=[]
    for architecture in ('spatial','gru'):
        for index in range(4):
            base=run/'episodes'/f'baseline-{architecture}-rule-{index}.json'
            branch=run/'episodes'/f'branch-{architecture}-{index}-0.json'
            if not base.exists() or not branch.exists():continue
            b=read(base)['summary']; alternative=read(branch); a=alternative['summary']
            capture=b['captures'][0]; intervention=alternative['job']['override']
            assert intervention['prefix_sha256']==capture['prefix_sha256']
            history=read(b['telemetry_path'])['histories'][str(capture['dragon'])]
            observed=next(h for h in history if h['round']==capture['round'])
            target=capture['round']+20;side=b['side']
            def arm(summary):
                telemetry=Path(summary['telemetry_path'])
                replay=telemetry.with_name(telemetry.name.replace('.telemetry.json','.replay-metrics.json'))
                if not replay.exists():return None
                metrics=read(replay)
                curve=next((x[side] for x in metrics['curves'] if x['round']==target),None)
                return dict(reward=summary['reward'],rounds=summary['result']['rounds'],team_after20=curve,final=metrics['final'][side])
            before,after=arm(b),arm(a)
            if before is None or after is None:continue
            rows.append(dict(architecture=architecture,map=b['map'],seed=b['seed'],side=side,
                round=capture['round'],dragon=capture['dragon'],baseline_action=capture['action'],
                alternative_action=intervention['action'],parent_length=observed['length'],role=observed['role'],
                baseline_child_size=capture['sizes'][capture['action']-1] if capture['action'] else 0,
                alternative_child_size=capture['sizes'][intervention['action']-1] if intervention['action'] else 0,
                prefix_verified=True,baseline=before,alternative=after,
                delta_team20=None if before['team_after20'] is None or after['team_after20'] is None else
                    {k:after['team_after20'][k]-before['team_after20'][k] for k in before['team_after20']}))
    comparable=[r for r in rows if r['delta_team20'] is not None]
    summary=dict(interventions=len(rows),terminal_outcome_changed=sum(r['baseline']['reward']!=r['alternative']['reward'] for r in rows),
        complete_20_round_pairs=len(comparable),team_total20_changed=sum(r['delta_team20']['totalLength']!=0 for r in comparable),
        interpretation='Identical prefix, one changed action, then original deterministic policies. Twenty-round team outcomes are intermediate proxies, not terminal preference labels; early-finished arms are censored.')
    (run/'counterfactual-summary.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2),encoding='utf-8')
    print(json.dumps(summary))

if __name__=='__main__':main()
