"""
Inference Service for Voice-Enabled Chatbot.
Provides real-time intent classification, confidence scoring,
threshold gating, and response retrieval.
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.preprocessing import TextPreprocessor, clean_text
from src.model import load_trained_model
from src.responses import get_response

class IntentClassifierService:
    """
    Singleton service managing the trained BiLSTM model and preprocessor
    for low-latency inference in the Streamlit application.
    """
    _instance: Optional["IntentClassifierService"] = None

    def __init__(self,
                 model_path: Optional[Path] = None,
                 preprocessor: Optional[TextPreprocessor] = None):
        self.model_path = model_path or config.MODEL_PATH
        self.preprocessor = preprocessor or TextPreprocessor.load()
        self.model = load_trained_model(self.model_path)
        self.classes = self.preprocessor.get_classes()
        print(f"IntentClassifierService loaded with {len(self.classes)} intents.")

    @classmethod
    def get_instance(cls) -> "IntentClassifierService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(self,
                raw_text: str,
                confidence_threshold: float = config.CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
        """
        End-to-end inference for a single input text utterance.

        Returns a dictionary containing:
        - raw_text: Original input
        - cleaned_text: Preprocessed text
        - intent: Predicted intent label
        - confidence: Probability score (float 0.0 to 1.0)
        - confidence_pct: Formatted percentage string
        - is_low_confidence: Boolean indicating if score was below threshold
        - response: Chatbot response string
        - all_scores: Dictionary of intent -> probability for all classes
        """
        cleaned = clean_text(raw_text)

        # Handle empty or blank input
        if not cleaned:
            return {
                "raw_text": raw_text,
                "cleaned_text": "",
                "intent": "none",
                "raw_intent": "none",
                "confidence": 0.0,
                "confidence_pct": "0.0%",
                "is_low_confidence": True,
                "response": "I didn't hear anything. Please speak into the microphone or type your message.",
                "all_scores": {c: 0.0 for c in self.classes}
            }

        # Tokenize and pad sequence
        padded_seq = self.preprocessor.transform_texts([cleaned])

        # Model forward pass
        probabilities = self.model.predict(padded_seq, verbose=0)[0]

        top_idx = int(np.argmax(probabilities))
        top_intent = self.classes[top_idx]
        confidence = float(probabilities[top_idx])

        # Confidence threshold check
        is_low_confidence = confidence < confidence_threshold
        response = get_response(top_intent, confidence, threshold=confidence_threshold, raw_text=raw_text)

        # Top 5 distribution for UI inspection
        top_scores = {
            self.classes[i]: float(probabilities[i])
            for i in np.argsort(probabilities)[::-1]
        }

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "intent": top_intent if not is_low_confidence else f"{top_intent} (Uncertain)",
            "raw_intent": top_intent,
            "confidence": confidence,
            "confidence_pct": f"{confidence * 100:.1f}%",
            "is_low_confidence": is_low_confidence,
            "response": response,
            "all_scores": top_scores
        }

if __name__ == "__main__":
    service = IntentClassifierService.get_instance()
    test_queries = [
        "What is the weather like in New York today?",
        "Can you transfer 50 dollars to my savings account?",
        "Please wake me up at 6 am tomorrow",
        "Tell me a funny joke to brighten my day",
        "xyzabc 12345 gibberish words"
    ]
    for q in test_queries:
        res = service.predict(q)
        print(f"\nUser:       {q}")
        print(f"Intent:     {res['intent']} (Conf: {res['confidence_pct']})")
        print(f"Response:   {res['response']}")
