"""
Unit tests for the BiLSTM neural network model.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import build_bilstm_model
import config

class TestModel(unittest.TestCase):
    def setUp(self):
        self.num_classes = 18
        self.max_len = 30
        self.vocab_size = 1000
        self.model = build_bilstm_model(
            vocab_size=self.vocab_size,
            max_length=self.max_len,
            num_classes=self.num_classes
        )

    def test_model_structure(self):
        self.assertIsNotNone(self.model)
        # Check layer types exist in model
        layer_names = [layer.name for layer in self.model.layers]
        self.assertIn("embedding", layer_names)
        self.assertIn("bidirectional_lstm", layer_names)
        self.assertIn("intent_probabilities", layer_names)

    def test_forward_pass_shape_and_probabilities(self):
        batch_size = 4
        # Create dummy batch of random token IDs
        dummy_input = np.random.randint(1, self.vocab_size, size=(batch_size, self.max_len))
        predictions = self.model.predict(dummy_input, verbose=0)

        # Output shape must be (batch_size, num_classes)
        self.assertEqual(predictions.shape, (batch_size, self.num_classes))

        # Softmax probabilities for each sample must sum to approximately 1.0
        prob_sums = np.sum(predictions, axis=1)
        np.testing.assert_allclose(prob_sums, np.ones(batch_size), atol=1e-5)

if __name__ == "__main__":
    unittest.main()
