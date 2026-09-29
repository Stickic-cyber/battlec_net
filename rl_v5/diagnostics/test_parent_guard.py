"""Check output filtering and worker isolation before the first guarded match."""
from pathlib import Path
import sys, unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'rl_v5/refinements'))
sys.path.insert(0,str(ROOT/'rl_v5'))
import followup_experiment as experiment
import numpy as np
from parent_guard import constrain
import model


class ParentGuardTests(unittest.TestCase):
    def data(self, parents):
        rows=np.zeros((6,32),np.float16)
        rows[:,2]=np.asarray(parents)/128
        return dict(mask=np.ones(6,bool), candidates=rows)

    def test_all_splits_rejected(self):
        data=self.data([4,2,2,2,2,2])
        np.testing.assert_array_equal(constrain([.01,.2,.2,.2,.2,.19],data),[1,0,0,0,0,0])

    def test_only_legal_parents_and_original_mask(self):
        data=self.data([5,3,2,4,2,2]);data['mask'][3]=False
        before=data['mask'].copy();p=constrain([.1,.4,.4,.1,0,0],data)
        np.testing.assert_allclose(p,[.2,.8,0,0,0,0])
        np.testing.assert_array_equal(data['mask'],before)

    def test_worker_restores_policy_and_rejects_training(self):
        original=model.Policy
        with patch.object(experiment,'episode',side_effect=RuntimeError('sentinel')):
            with self.assertRaisesRegex(RuntimeError,'sentinel'):
                experiment.refined_episode(dict(parent_min=3,controller='network'))
        self.assertIs(model.Policy,original)
        with self.assertRaisesRegex(AssertionError,'evaluation-only'):
            experiment.refined_episode(dict(parent_min=3,controller='network',collect=True))
        self.assertIs(model.Policy,original)


if __name__=='__main__':unittest.main()
