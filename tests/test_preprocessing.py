"""
Unit tests for NLP preprocessing and tokenization pipeline.
"""
import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing import clean_text, TextPreprocessor
import config

class TestPreprocessing(unittest.TestCase):
    def test_clean_text(self):
        sample = "  Hello, CAN you tell me what's the weather like?!  "
        cleaned = clean_text(sample)
        self.assertEqual(cleaned, "hello can you tell me what's the weather like")

    def test_preprocessor_pipeline(self):
        texts = [
            "hello good morning",
            "what is the weather today",
            "set an alarm for 7 am",
            "goodbye see you later"
        ]
        labels = ["greeting", "weather", "alarm", "goodbye"]

        preprocessor = TextPreprocessor(max_len=10)
        preprocessor.fit(texts, labels)

        self.assertTrue(preprocessor.is_fitted)
        self.assertEqual(len(preprocessor.get_classes()), 4)

        # Test sequence transformation
        padded = preprocessor.transform_texts(["hello weather"])
        self.assertEqual(padded.shape, (1, 10))
        self.assertIsInstance(padded, np.ndarray)

        # Test label encoding and inverse
        encoded = preprocessor.transform_labels(["weather", "greeting"])
        self.assertEqual(len(encoded), 2)
        decoded = [preprocessor.inverse_transform_label(c) for c in encoded]
        self.assertEqual(decoded, ["weather", "greeting"])

    def test_oov_handling(self):
        texts = ["tell me a joke", "check account balance"]
        labels = ["tell_joke", "balance"]
        preprocessor = TextPreprocessor(max_len=10)
        preprocessor.fit(texts, labels)

        # Unseen words should be mapped to the OOV token index
        padded = preprocessor.transform_texts(["supercalifragilistic"])
        self.assertEqual(padded.shape, (1, 10))
        # OOV index should be 1 by default in Keras Tokenizer
        oov_idx = preprocessor.tokenizer.word_index[config.OOV_TOKEN]
        self.assertEqual(padded[0][0], oov_idx)

if __name__ == "__main__":
    unittest.main()
