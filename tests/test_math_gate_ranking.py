import unittest

import torch

from src.math_gate_ranking import combined_loss, pairwise_ranking_loss, weights_digest


class RankingTests(unittest.TestCase):
    def test_correct_order_has_lower_loss(self):
        labels = torch.tensor([1, 0])
        good = torch.tensor([[0., 2.], [0., -2.]])
        bad = good.flip(0)
        self.assertLess(pairwise_ranking_loss(good, labels), pairwise_ranking_loss(bad, labels))

    def test_gradients_increase_positive_score_and_decrease_negative(self):
        logits = torch.zeros((2, 2), requires_grad=True)
        pairwise_ranking_loss(logits, torch.tensor([1, 0])).backward()
        self.assertLess(logits.grad[0, 1], 0)
        self.assertGreater(logits.grad[1, 1], 0)

    def test_single_class_batch_has_finite_zero_gradient(self):
        for label in (0, 1):
            logits = torch.ones((3, 2), requires_grad=True)
            loss = pairwise_ranking_loss(logits, torch.full((3,), label))
            loss.backward()
            self.assertEqual(loss.item(), 0.)
            self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_zero_coefficient_exactly_matches_ce_gradient(self):
        labels = torch.tensor([0, 0, 1])
        weights = torch.tensor([1., 5.])
        a = torch.tensor([[.2, .1], [.3, .2], [.4, .1]], requires_grad=True)
        b = a.detach().clone().requires_grad_()
        total, _, _ = combined_loss(a, labels, weights, 0.)
        ce = torch.nn.functional.cross_entropy(b, labels, weight=weights)
        total.backward()
        ce.backward()
        self.assertEqual(total.item(), ce.item())
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_stable_for_extreme_logits(self):
        logits = torch.tensor([[1000., -1000.], [-1000., 1000.]], requires_grad=True)
        loss = pairwise_ranking_loss(logits, torch.tensor([1, 0]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_pair_mean_and_order_invariance(self):
        logits = torch.tensor([[0., 2.], [0., 1.], [0., -2.], [0., -1.]])
        labels = torch.tensor([1, 1, 0, 0])
        expected = torch.stack([torch.nn.functional.softplus(torch.tensor(n - p))
                                for p in (2., 1.) for n in (-2., -1.)]).mean()
        torch.testing.assert_close(pairwise_ranking_loss(logits, labels), expected)
        torch.testing.assert_close(pairwise_ranking_loss(logits.flip(0), labels.flip(0)), expected)

    def test_bad_labels_and_shapes_rejected(self):
        for logits, labels in [(torch.zeros(3), torch.zeros(3)),
                               (torch.zeros(3, 2), torch.tensor([0, 1, 2]))]:
            with self.assertRaises(ValueError):
                pairwise_ranking_loss(logits, labels)

    def test_digest_detects_weight_changes(self):
        model = torch.nn.Linear(3, 2)
        initial = weights_digest(model)
        with torch.no_grad():
            model.bias.add_(1.)
        self.assertNotEqual(initial, weights_digest(model))


if __name__ == '__main__':
    unittest.main()
