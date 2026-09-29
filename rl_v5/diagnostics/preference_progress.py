"""Inspect timing versus exact-size learning on supervised development states."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rl_v5'))
from model import load, tensors, setup_torch
from features import SHAPES


def main():
    setup_torch()
    run = ROOT / 'rl_v5/runs/structured-split'
    refined = ROOT / 'rl_v5/runs/refined-teachers'
    output = []
    for architecture in ('spatial', 'gru'):
        path = run / f'preferences-{architecture}.json'
        if not path.exists():
            continue
        for index, row in enumerate(json.loads(path.read_text())):
            if row['baseline_reward'] == row['alternative_reward']:
                continue
            better = row['baseline_action'] if row['baseline_reward'] > row['alternative_reward'] else row['alternative_action']
            worse = row['alternative_action'] if row['baseline_reward'] > row['alternative_reward'] else row['baseline_action']
            with np.load(row['capture_path']) as archive:
                data = {k: archive[k] for k in SHAPES}
            obs = tensors({k: v[None] for k, v in data.items()})
            paths = [run / f'{architecture}-initial-9201.pt']
            paths += [run / f'{architecture}-s{seed}-ppo-{update}.pt' for seed in (9201, 9203) for update in range(1, 5)]
            paths += [refined / f'{architecture}-refine-2-fine.pt']
            checks = []
            for checkpoint in paths:
                if not checkpoint.exists():
                    continue
                with torch.no_grad():
                    p = load(checkpoint).eval()(obs)[0].exp()[0].numpy()
                split_probability = float(p[1:].sum())
                action = int(1 + p[1:].argmax()) if split_probability > p[0] else 0
                checks.append(dict(checkpoint=checkpoint.stem,
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    split_probability=split_probability, preferred_probability=float(p[better]),
                    worse_probability=float(p[worse]), greedy_action=action,
                    greedy_child_length=int(round(float(data['candidates'][action, 1]) * 128)),
                    timing_matches=bool((action > 0) == (better > 0)), exact_action_matches=bool(action == better)))
            output.append(dict(architecture=architecture, preference_index=index, map=row['map'], side=row['side'],
                seed=row['seed'], preferred_action=better,
                preferred_child_length=int(round(float(data['candidates'][better, 1]) * 128)), checks=checks))
    result = dict(scope='Supervised development states; no held-out or loss-attribution claim.', rows=output)
    (run / 'preference-progression.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
