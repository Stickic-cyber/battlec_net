"""Describe short-horizon split outcomes; these are review flags, not causal errors."""
from pathlib import Path
from collections import Counter, defaultdict
import json, sys

ROOT = Path(__file__).resolve().parents[2]

def summarize(rows):
    out = Counter()
    reasons = Counter()
    for row in rows:
        out['splits'] += 1
        for key in ('child_observed20', 'parent_observed20', 'child_died20', 'parent_died20',
                    'child_nonfeeding_death20', 'parent_nonfeeding_death20', 'both_died20',
                    'child_complete_window_no_positive_growth', 'child_never_acted'):
            out[key] += int(row[key])
        if row['child_died20']:
            reasons[str(row['child_death_reason'])] += 1
    return dict(out, child_death_reasons=dict(reasons))

def main():
    run = Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'rl_v5/runs/structured-split'
    groups = defaultdict(list)
    for meta in sorted((run/'episodes').glob('*.json')):
        if meta.name.endswith(('.telemetry.json', '.replay-metrics.json')):
            continue
        obj = json.loads(meta.read_text())
        if 'summary' not in obj:
            continue
        summary = obj['summary']
        t = json.loads(Path(summary['telemetry_path']).read_text())
        hist, deaths = t['histories'], t['deaths']
        end = summary['result']['rounds']
        label = meta.stem.rsplit('-', 1)[0]
        for child, birth in t['parents'].items():
            parent = str(birth['parent'])
            at = birth['round']
            h = hist[parent][birth['parent_history']]
            if h['policy_action'] <= 0:
                continue
            def actor(dragon):
                history = hist.get(dragon, [])
                d = deaths.get(dragon, {})
                dead = at <= d.get('round', 10000) <= at+20
                feed = dead and any(x['feed'] and x['round']==d['round'] for x in history)
                return dict(observed=(end>=at+20 or dead), dead=dead, nonfeeding=dead and not feed,
                    growth=sum(max(0,x['growth']) for x in history if at<x['round']<=at+20),
                    reason=d.get('reason'), acted=bool(history))
            c, p = actor(child), actor(parent)
            row = dict(episode=meta.stem, map=summary['map'], side=summary['side'], seed=summary['seed'],
                outcome=summary['reward'], parent=int(parent), child=int(child), round=at,
                child_size=birth['size'], parent_length_before=h['length'],
                child_observed20=c['observed'], parent_observed20=p['observed'],
                child_died20=c['dead'], parent_died20=p['dead'], both_died20=c['dead'] and p['dead'],
                child_nonfeeding_death20=c['nonfeeding'], parent_nonfeeding_death20=p['nonfeeding'],
                child_complete_window_no_positive_growth=end>=at+20 and not c['dead'] and c['growth']==0,
                child_never_acted=not c['acted'], child_death_reason=c['reason'])
            groups[label].append(row)
    result = dict(interpretation='Observed review flags only: death may be beneficial or unavoidable; no causal blame is assigned to splitting. Only explicitly marked feeding deaths are excluded from nonfeeding flags. A (no valid action) can also represent intentional sacrifice. Censored windows are not counted as confirmed survival.',
        death_reason_codes={'W':'hit a wall','S':'hit itself','O':'hit another dragon','H':'lost a head-to-head','A':'no valid action'},
        death_reason_source='Installed official unswbc/run.py DEATH_REASONS',
        groups={k:summarize(v) for k,v in groups.items()}, records=dict(groups))
    (run/'split-outcomes.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['groups']))

if __name__ == '__main__':
    main()
