"""
Unit tests for the end-to-end inference service and response mapping.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference import IntentClassifierService
import config

class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = IntentClassifierService.get_instance()

    def test_greeting_intent(self):
        res = self.service.predict("hello there good morning")
        self.assertEqual(res["raw_intent"], "greeting")
        self.assertGreater(res["confidence"], 0.70)
        self.assertFalse(res["is_low_confidence"])
        from src.responses import INTENT_RESPONSES
        self.assertIn(res["response"], INTENT_RESPONSES["greeting"])

    def test_weather_intent(self):
        res = self.service.predict("will it rain this afternoon")
        self.assertEqual(res["raw_intent"], "weather")
        self.assertGreater(res["confidence"], 0.70)
        self.assertFalse(res["is_low_confidence"])

    def test_empty_input(self):
        res = self.service.predict("   ")
        self.assertEqual(res["intent"], "none")
        self.assertEqual(res["confidence"], 0.0)
        self.assertTrue(res["is_low_confidence"])
        self.assertIn("didn't hear anything", res["response"])

    def test_low_confidence_threshold_override(self):
        # Gibberish below threshold triggers capability fallback response
        res = self.service.predict("flim flam blorp quux zorp xyz", confidence_threshold=0.99999)
        self.assertTrue(res["is_low_confidence"])
        self.assertEqual(res["response"], config.FALLBACK_RESPONSE)

        # Question below threshold provides question-relevant intent response
        res_q = self.service.predict("what is the weather today", confidence_threshold=0.99999)
        self.assertTrue(res_q["is_low_confidence"])
        self.assertIn("weather", res_q["response"])

    def test_location_intent_and_intended_response(self):
        res = self.service.predict("where is vit vellore located")
        self.assertEqual(res["raw_intent"], "directions")
        self.assertGreater(res["confidence"], 0.85)
        self.assertFalse(res["is_low_confidence"])
        self.assertEqual(res["response"], "It is located in Andhra Pradesh.")

    def test_other_location_queries(self):
        res = self.service.predict("where is the taj mahal located")
        self.assertEqual(res["raw_intent"], "directions")
        self.assertIn("Agra", res["response"])

if __name__ == "__main__":
    unittest.main()
