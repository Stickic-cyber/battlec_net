"""Evaluation-only parent-length constraint; no change to actor inputs or training."""
import numpy as np
import torch
from model import tensors


def constrain(probabilities, data, minimum=3):
    assert minimum == 3
    p = np.asarray(probabilities, dtype=np.float64).copy()
    valid = np.asarray(data['mask'], dtype=bool).copy()
    valid[1:] &= np.asarray(data['candidates'])[1:, 2] * 128 >= minimum
    assert valid[0]
    p[~valid] = 0
    total = p.sum()
    if total <= 0:
        p[0] = 1
    else:
        p /= total
    return p


def guarded_policy(base_class, minimum):
    class GuardedPolicy(base_class):
        def act(self, data, rng=None):
            with torch.no_grad():
                lp, v, aux = self.model(tensors(data, single=True))
            p = constrain(lp[0].exp().numpy(), data, minimum)
            action = 1 + int(p[1:].argmax()) if p[1:].sum() > p[0] else 0
            if rng is not None:
                action = int(rng.choice(len(p), p=p))
            return action, float(np.log(max(p[action], np.finfo(float).tiny))), float(v[0]), aux[0].numpy()
    return GuardedPolicy
