"""
Auto-submitting Voice Recorder Streamlit Component.
Records audio in browser with Voice Activity Detection (VAD)
and automatically submits when user stops speaking (YouTube style).
"""
import base64
from pathlib import Path
from typing import Optional, Dict, Any
import streamlit.components.v1 as components

COMPONENT_DIR = Path(__file__).resolve().parent

_component_func = components.declare_component(
    "auto_voice_recorder",
    path=str(COMPONENT_DIR)
)

def auto_voice_recorder(key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Renders an auto-submitting microphone button that detects speech
    and automatically stops and returns WAV audio bytes when the user
    pauses / stops speaking for ~1.3 seconds (like YouTube).

    Returns:
        dict with keys:
            - id: int
            - bytes: bytes (PCM 16-bit Mono WAV format)
            - sample_rate: int
            - sample_width: int (2)
            - format: str ('wav')
        or None if no recording yet.
    """
    component_value = _component_func(key=key, default=None)
    if component_value and "audio_base64" in component_value:
        raw_b64 = component_value["audio_base64"]
        audio_bytes = base64.b64decode(raw_b64)
        return {
            "id": component_value.get("id"),
            "bytes": audio_bytes,
            "sample_rate": component_value.get("sample_rate", 16000),
            "sample_width": component_value.get("sample_width", 2),
            "format": component_value.get("format", "wav")
        }
    return None
