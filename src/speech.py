"""
Speech Recognition Module for Voice-Enabled Chatbot.
Provides robust conversion of audio input into text using SpeechRecognition.
Architecturally decoupled from the downstream BiLSTM intent classification model.
"""
import io
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import speech_recognition as sr

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class SpeechRecognizerHandler:
    """
    Handles speech-to-text conversion from audio bytes or microphone sources.
    Uses Google Web Speech API via SpeechRecognition with zero external C-dependencies.
    """
    def __init__(self, energy_threshold: int = 300, pause_threshold: float = 0.8):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = True

    def transcribe_audio_bytes(self,
                               audio_bytes: bytes,
                               sample_rate: Optional[int] = None,
                               sample_width: Optional[int] = None) -> Dict[str, Any]:
        """
        Transcribe raw audio bytes (WAV format from browser mic or audio file).

        Returns:
            dict containing:
            - success: bool
            - text: str (recognized text if successful)
            - error: Optional[str] (error description if failed)
        """
        if not audio_bytes or len(audio_bytes) == 0:
            return {
                "success": False,
                "text": "",
                "error": "Empty audio data received. Please record again."
            }

        try:
            audio_data = None

            # First attempt: Read standard WAV/AIFF/FLAC container
            try:
                audio_file = io.BytesIO(audio_bytes)
                with sr.AudioFile(audio_file) as source:
                    audio_data = self.recognizer.record(source)
            except Exception as file_err:
                # Second attempt: Raw PCM AudioData using provided sample_rate and sample_width
                if sample_rate and sample_width:
                    audio_data = sr.AudioData(audio_bytes, int(sample_rate), int(sample_width))
                elif audio_bytes.startswith(b"\x1aE\xdf\xa3"):
                    raise ValueError("Audio was received in WebM container. Please record using format='wav'.")
                else:
                    raise file_err

            # Transcribe via Google Web Speech API
            recognized_text = self.recognizer.recognize_google(audio_data, language="en-US")
            recognized_text = recognized_text.strip()

            return {
                "success": True,
                "text": recognized_text,
                "error": None
            }

        except sr.UnknownValueError:
            return {
                "success": False,
                "text": "",
                "error": "Speech was not recognized. Please speak clearly and try again."
            }
        except sr.RequestError as e:
            return {
                "success": False,
                "text": "",
                "error": f"Speech recognition service request error: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "error": f"Audio processing error: {str(e)}"
            }

    def transcribe_local_microphone(self) -> Dict[str, Any]:
        """
        Transcribe directly from local system microphone if available.
        """
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                recognized_text = self.recognizer.recognize_google(audio_data, language="en-US")
                return {
                    "success": True,
                    "text": recognized_text.strip(),
                    "error": None
                }
        except sr.WaitTimeoutError:
            return {
                "success": False,
                "text": "",
                "error": "Listening timed out. No speech detected."
            }
        except sr.UnknownValueError:
            return {
                "success": False,
                "text": "",
                "error": "Speech was not recognized. Please speak clearly and try again."
            }
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "error": f"Microphone access error: {str(e)}"
            }

# Module-level convenience function
_handler_instance = None

def get_speech_handler() -> SpeechRecognizerHandler:
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = SpeechRecognizerHandler()
    return _handler_instance

def transcribe_audio(audio_bytes: bytes,
                     sample_rate: Optional[int] = None,
                     sample_width: Optional[int] = None) -> Dict[str, Any]:
    return get_speech_handler().transcribe_audio_bytes(audio_bytes, sample_rate, sample_width)
