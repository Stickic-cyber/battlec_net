"""Exercise both continuation paths with real on-policy GRU data and causal labels.

These checkpoints are diagnostics only and are never candidates for selection.
No extra matches are played by this check.
"""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'rl_v5/refinements'))
from followup_experiment import update,REFERENCE
from runtime import config
import training

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    initial=REFERENCE/'gru-initial-9201.pt'
    data=REFERENCE/'episodes/gru-s9201-rollout-1-0.npz'
    preferences=REFERENCE/'preferences-gru.json'
    job=json.loads(data.with_suffix('.json').read_text())['job']
    assert job['stochastic'] and job['collect'] and job['controller']=='network'
    assert job['weight_sha256']==sha(initial)
    expected=sum(row['baseline_reward']!=row['alternative_reward'] for row in json.loads(preferences.read_text()))
    assert expected>0,'This check requires actual WLD-changing labels'
    original_train,original_data=training.train,training.data_from
    cfg=config();cfg.update(lr=.0003,max_lr=.0003,epochs=1,batch_size=64,kl_stop=.01,
        diagnostic_only='Single actual GRU on-policy match plus verified reference preferences; not a selection candidate.')
    folder=ROOT/'rl_v5/runs/refinement-smoke';folder.mkdir(parents=True,exist_ok=True);results={}
    for method in ('phase','match'):
        output=folder/f'gru-preference-{method}.pt'
        final,projection=update(initial,[data],output,cfg,26001,method,preferences)
        raw=json.loads(output.with_name(output.stem+'-raw.json').read_text())
        assert raw['preference_pairs']==expected
        assert training.train is original_train and training.data_from is original_data
        results[method]=dict(preference_pairs=expected,parity=raw['parity'],projection=projection['diagnostics'],
            checkpoint_sha256=sha(final),global_functions_restored=True)
    (folder/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('PREFERENCE SMOKE PASSED',json.dumps(results),flush=True)

if __name__=='__main__':main()
