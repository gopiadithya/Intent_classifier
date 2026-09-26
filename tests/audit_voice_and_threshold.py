"""
Audit verification script for Voice Pipeline and Out-of-Scope Threshold Handling.
"""
import io
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.speech import transcribe_audio
from src.inference import IntentClassifierService

def audit_voice_and_threshold():
    print("=" * 70)
    print("AUDITING VOICE PIPELINE & OUT-OF-SCOPE THRESHOLD GATING")
    print("=" * 70)

    # 1. Voice pipeline check (empty and synthetic silent WAV)
    print("\n--- 1. Testing Voice Pipeline Audio Handlers ---")
    res_empty = transcribe_audio(b"")
    print(f"Empty Audio Input Handling: Success={res_empty['success']}, Error='{res_empty['error']}'")
    assert not res_empty["success"], "Empty audio must not succeed"

    # Synthetic silent audio
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00" * 16000)
    res_silence = transcribe_audio(buf.getvalue())
    print(f"Silent Audio Input Handling: Success={res_silence['success']}, Error='{res_silence['error']}'")
    assert not res_silence["success"], "Silence must return clean recognition error, not crash"

    # 2. Text Fallback Verification
    print("\n--- 2. Testing Text Fallback Queries ---")
    service = IntentClassifierService.get_instance()
    sample_tests = [
        ("what is the weather today", "weather"),
        ("how much balance is in checking", "balance"),
        ("can you set an alarm for seven am", "alarm"),
        ("transfer money to savings", "transfer")
    ]
    for text, expected in sample_tests:
        pred = service.predict(text)
        print(f"Text Input: '{text}' -> Intent: {pred['intent']} (Conf: {pred['confidence_pct']})")
        print(f"Chatbot Response: '{pred['response']}'")
        assert pred["raw_intent"] == expected, f"Expected {expected}, got {pred['raw_intent']}"
        assert not pred["is_low_confidence"], "High-confidence standard query failed threshold"

    # 3. Out-of-Scope / Gibberish Threshold Gating Verification
    print("\n--- 3. Testing Out-of-Scope & Gibberish Inputs ---")
    gibberish_inputs = [
        "flim flam blorp quux zorp xyz",
        "asdkjfhqwuer kjasdhf",
        "supercalifragilisticexpialidocious quantum blorp",
        "123987 456098 777777"
    ]
    for g in gibberish_inputs:
        pred = service.predict(g, confidence_threshold=config.CONFIDENCE_THRESHOLD)
        print(f"\nGibberish Input: '{g}'")
        print(f"Predicted Raw Intent: {pred['raw_intent']} | Conf: {pred['confidence_pct']} | Flagged Uncertain: {pred['is_low_confidence']}")
        print(f"Returned Response: '{pred['response']}'")
        assert pred["is_low_confidence"], f"Gibberish '{g}' was accepted with excessive confidence!"
        assert pred["response"] == config.FALLBACK_RESPONSE, "Fallback response was not returned!"

    print("\n" + "=" * 70)
    print("ALL AUDIT CHECKS PASSED: Voice Pipeline & Confidence Gating are 100% Robust!")
    print("=" * 70)

if __name__ == "__main__":
    audit_voice_and_threshold()
