"""
Unit tests for Speech Recognition module.
"""
import sys
import unittest
import wave
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.speech import SpeechRecognizerHandler, transcribe_audio

class TestSpeechRecognition(unittest.TestCase):
    def setUp(self):
        self.handler = SpeechRecognizerHandler()

    def test_empty_audio_bytes(self):
        result = self.handler.transcribe_audio_bytes(b"")
        self.assertFalse(result["success"])
        self.assertEqual(result["text"], "")
        self.assertIn("Empty audio data", result["error"])

    def test_corrupt_audio_bytes(self):
        corrupt_data = b"this is completely invalid audio binary data"
        result = self.handler.transcribe_audio_bytes(corrupt_data)
        self.assertFalse(result["success"])
        self.assertEqual(result["text"], "")
        self.assertIsNotNone(result["error"])

    def test_valid_silent_wav_audio(self):
        # Create a valid 1-second silent WAV file in-memory
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)      # Mono
            wav_file.setsampwidth(2)     # 16-bit
            wav_file.setframerate(16000) # 16kHz
            # 1 second of silence (zero bytes)
            wav_file.writeframes(b"\x00" * 32000)

        wav_bytes = buf.getvalue()
        result = self.handler.transcribe_audio_bytes(wav_bytes)
        # Silence should not crash; it should either return UnknownValueError gracefully
        self.assertFalse(result["success"])
        self.assertIn("not recognized", result["error"])

if __name__ == "__main__":
    unittest.main()
