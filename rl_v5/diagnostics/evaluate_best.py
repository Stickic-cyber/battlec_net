"""Evaluate the complete saved teacher configuration, including optional feeding v2."""
from pathlib import Path
import argparse, hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'rl_v5/refinements'))
from followup_experiment import SerialExecutor, sources
from collect import jobs, collect_jobs

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,default=ROOT/'rl_v5/best-policy.json')
    parser.add_argument('--maps',nargs='+',default=['arena.map'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[10001])
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--describe',action='store_true',help='Verify and print configuration without playing matches')
    args=parser.parse_args()
    policy=json.loads(args.policy.read_text(encoding='utf-8'))
    assert sha(policy['weights'])==policy['checkpoint_sha256'],'Checkpoint differs from selection'
    manifest=json.loads(Path(policy['source_manifest']).read_text(encoding='utf-8'))
    assert sources()==manifest['sources'],'Teacher sources differ from the completed experiment'
    options={k:policy[k] for k in ('weights','controller','feeding','stochastic','sampling_seed','parent_min') if k in policy}
    print(json.dumps(dict(architecture=policy['architecture'],options=options,maps=args.maps,seeds=args.seeds,
        scope='Native local teacher versus original sandboxed scheme 1; both sides; serial matches.'),ensure_ascii=False,indent=2),flush=True)
    if args.describe:return
    status=ROOT/'rl_v5/pipeline-status.json'
    if status.exists():
        assert json.loads(status.read_text())['stage']=='complete','Wait for the active training pipeline to finish before starting evaluation'
    run=args.run_dir or ROOT/'rl_v5/runs/best-policy-evaluation'/time.strftime('%Y%m%d-%H%M%S')
    run=run.resolve();run.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1) as raw:
        rows=collect_jobs(SerialExecutor(raw),jobs(args.maps,args.seeds,**options,refinement_sources=manifest['sources'],save_replay=True),'evaluation',run)
    result=dict(policy=policy,maps=args.maps,seeds=args.seeds,rows=rows,
        wins=sum(r['reward']==1 for r in rows),draws=sum(r['reward']==0 for r in rows),losses=sum(r['reward']==-1 for r in rows))
    (run/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RESULTS',run/'results.json',flush=True)

if __name__=='__main__':main()
