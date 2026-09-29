"""Verify the exported files and run one synthetic movement forward pass."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'export-manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files_sha256'].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Export file changed: {relative}')
    sys.path.insert(0, str(ROOT / 'rl_v5'))
    import numpy as np
    from model import Policy
    from base_observation import OBS_SHAPES
    from runner import Dragon  # also check the rule/engine imports

    checkpoint = ROOT / 'rl_v5/runs/structured-split/gru-initial-9201.pt'
    policy = Policy(checkpoint)
    obs = {key: np.zeros(shape, dtype=bool if key in ('mask', 'entity_mask', 'action_mask') else np.float32)
           for key, shape in OBS_SHAPES.items()}
    obs['entity_mask'][0] = True
    obs['mask'][:] = True
    obs['action_mask'][:4] = True
    action, hidden, logp = policy.movement(obs, np.zeros(256, np.float32))
    assert hidden.shape == (256,) and logp.shape == (4,)
    assert np.isfinite(hidden).all() and np.isfinite(logp).all()
    assert np.isclose(np.exp(logp).sum(), 1.0) and 0 <= action < 4
    count = sum(p.numel() for p in policy.model.parameters())
    assert count == 2670668
    print(json.dumps({'verified_files': len(manifest['files_sha256']), 'parameters': count,
                      'gru_hidden': len(hidden), 'movement_probabilities': np.exp(logp).tolist(),
                      'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      'synthetic_forward_passed': True}, indent=2))


if __name__ == '__main__':
    main()
