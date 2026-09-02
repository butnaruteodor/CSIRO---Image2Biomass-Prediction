"""
Unit tests for the weighted biomass loss and the MLP head.
"""
import unittest
import numpy as np
import torch

from src.training.loss import weighted_biomass_loss
from src.models.heads import BiomassSimpleMLP


class TestWeightedBiomassLoss(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.targets = torch.tensor([[10.0, 5.0, 2.0, 8.0, 20.0],
                                     [0.0, 1.0, 0.0, 3.0, 12.0]])

    def _loss_from_component(self, preds, targets, comp):
        """Loss when only one component deviates (others are exact)."""
        exact = {k: v.clone() for k, v in preds.items()}
        exact[comp] = preds[comp] + 1.0
        return weighted_biomass_loss(
            exact['total'], exact['gdm'], exact['green'],
            exact['clover'], exact['dead'], targets)

    def test_zero_for_perfect_predictions(self):
        p_total = self.targets[:, 4:5].clone()
        p_gdm = self.targets[:, 3:4].clone()
        p_green = self.targets[:, 0:1].clone()
        p_clover = self.targets[:, 2:3].clone()
        p_dead = self.targets[:, 1:2].clone()
        loss = weighted_biomass_loss(p_total, p_gdm, p_green, p_clover, p_dead, self.targets)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_official_weights(self):
        preds = {'total': self.targets[:, 4:5].clone(),
                 'gdm': self.targets[:, 3:4].clone(),
                 'green': self.targets[:, 0:1].clone(),
                 'clover': self.targets[:, 2:3].clone(),
                 'dead': self.targets[:, 1:2].clone()}
        # Deviating each component by 1.0 must contribute its weight
        self.assertAlmostEqual(self._loss_from_component(preds, self.targets, 'total').item(), 0.5, places=5)
        self.assertAlmostEqual(self._loss_from_component(preds, self.targets, 'gdm').item(), 0.2, places=5)
        self.assertAlmostEqual(self._loss_from_component(preds, self.targets, 'green').item(), 0.1, places=5)
        self.assertAlmostEqual(self._loss_from_component(preds, self.targets, 'clover').item(), 0.1, places=5)
        self.assertAlmostEqual(self._loss_from_component(preds, self.targets, 'dead').item(), 0.1, places=5)


class TestBiomassSimpleMLP(unittest.TestCase):

    def test_five_branches_with_softplus(self):
        model = BiomassSimpleMLP(2048)
        # Five independent branches exist
        for name in ['head_total', 'head_gdm', 'head_green', 'head_clover', 'head_dead']:
            self.assertTrue(hasattr(model, name), name)
        feats = torch.randn(4, 2048)
        model.eval()
        with torch.no_grad():
            out = model(feats)
        self.assertEqual(len(out), 5)
        for p in out:
            self.assertEqual(p.shape, (4, 1))
            # Softplus guarantees non-negative predictions on every branch
            self.assertTrue(bool((p >= 0).all()))

    def test_hidden_dimensions(self):
        model = BiomassSimpleMLP(2048)
        head = model.head_total
        linears = [m for m in head if isinstance(m, torch.nn.Linear)]
        self.assertEqual([m.in_features for m in linears], [2048, 1024, 512])
        self.assertEqual([m.out_features for m in linears], [1024, 512, 1])


if __name__ == "__main__":
    unittest.main()