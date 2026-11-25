import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.transformer_policy import TorchPolicyMLP, predict_transformer_actions


class IdentityScaler:
    def transform(self, values):
        return values


class TransformerBatchPredictionTests(unittest.TestCase):
    def test_batch_prediction_uses_direct_threshold(self):
        model = TorchPolicyMLP(2, (), 2)
        with torch.no_grad():
            model.network[0].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            model.network[0].bias.zero_()
        bundle = {
            "model": model,
            "encoder_name": "fake",
            "encoder_batch_size": 2,
            "max_length": 16,
            "max_text_chars": 100,
            "pooling": "mean",
            "layers": [-1],
            "pure_text": True,
            "device": "cpu",
            "scaler": IdentityScaler(),
            "action_labels": ["direct", "reflect_1"],
            "direct_threshold": 0.6,
        }
        features = np.asarray([[3.0, 0.0], [0.0, 3.0]], dtype="float32")
        with patch("src.transformer_policy._encode_hf_hidden", return_value=(features, {})):
            actions = predict_transformer_actions(bundle, [{"question": "a"}, {"question": "b"}])
        self.assertEqual(actions, ["direct", "reflect_1"])


if __name__ == "__main__":
    unittest.main()
