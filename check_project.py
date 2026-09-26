"""
check_project.py — Comprehensive Automated Project Audit Script
Voice-Enabled Chatbot | Deep Learning Lab Assessment

Verifies:
1. File structure completeness
2. Dataset integrity (counts, balance, leakage)
3. Model artifact integrity (architecture, shapes, classes)
4. Preprocessing pipeline consistency
5. Inference pipeline end-to-end correctness
6. Response honesty (no false action claims)
7. Path portability (no hardcoded absolute paths)
8. Requirements file completeness
9. Documentation existence
"""
import sys
import os
import json
import re
import pickle
import io
from pathlib import Path

# Fix Windows console encoding for emoji
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"
total_checks = 0
passed_checks = 0
failed_checks = 0
warned_checks = 0

def check(name, condition, detail=""):
    global total_checks, passed_checks, failed_checks
    total_checks += 1
    if condition:
        passed_checks += 1
        print(f"  {PASS}  {name}")
    else:
        failed_checks += 1
        print(f"  {FAIL}  {name}")
        if detail:
            print(f"         → {detail}")

def warn(name, detail=""):
    global total_checks, warned_checks
    total_checks += 1
    warned_checks += 1
    print(f"  {WARN}  {name}")
    if detail:
        print(f"         → {detail}")

# ============================================================
print("=" * 70)
print("COMPREHENSIVE PROJECT AUDIT")
print("Voice-Enabled Chatbot | BiLSTM Intent Classification")
print("=" * 70)

# ============================================================
# 1. FILE STRUCTURE
# ============================================================
print("\n[1/9] FILE STRUCTURE VERIFICATION")
required_files = [
    "config.py", "app.py", "train.py", "evaluate.py",
    "requirements.txt", "README.md", ".gitignore",
    "src/__init__.py", "src/preprocessing.py", "src/model.py",
    "src/inference.py", "src/responses.py", "src/speech.py",
    "src/download_data.py", "src/analyze_dataset.py",
    "tests/__init__.py", "tests/test_preprocessing.py",
    "tests/test_model.py", "tests/test_inference.py",
    "tests/test_speech.py", "tests/run_all_tests.py",
    "tests/benchmark_test.py",
    "data/raw/clinc150_full.json",
    "data/processed/train.csv", "data/processed/val.csv", "data/processed/test.csv",
    "models/chatbot_model.keras", "models/tokenizer.pkl",
    "models/label_encoder.pkl", "models/metadata.pkl",
    "reports/project_report.md", "reports/demo_examples.md",
    "reports/evaluation_results.json", "reports/training_metrics.json",
    "reports/figures/confusion_matrix.png",
    "reports/figures/training_history.png",
    "reports/figures/intent_distribution.png",
    "reports/figures/sequence_length_distribution.png",
    "reports/figures/split_distribution.png",
]

for f in required_files:
    check(f"File exists: {f}", (BASE_DIR / f).exists())

# ============================================================
# 2. DATASET INTEGRITY
# ============================================================
print("\n[2/9] DATASET INTEGRITY")
import pandas as pd

TARGET_INTENTS = [
    "greeting", "goodbye", "thank_you", "what_can_i_ask_you", "tell_joke",
    "weather", "directions", "restaurant_suggestion", "book_hotel", "travel_suggestion",
    "calendar", "reminder", "alarm",
    "balance", "spending_history", "pay_bill", "transfer", "freeze_account"
]

try:
    with open(BASE_DIR / "data/raw/clinc150_full.json", "r") as f:
        raw_data = json.load(f)
    check("Raw JSON loads successfully", True)
    
    for split, expected_per_intent in [("train", 100), ("val", 20), ("test", 30)]:
        filtered = [d for d in raw_data[split] if d[1] in TARGET_INTENTS]
        expected_total = expected_per_intent * 18
        check(f"Raw {split} filtered count = {expected_total}", len(filtered) == expected_total,
              f"Got {len(filtered)}")
except Exception as e:
    check("Raw dataset loads", False, str(e))

try:
    train_df = pd.read_csv(BASE_DIR / "data/processed/train.csv")
    val_df = pd.read_csv(BASE_DIR / "data/processed/val.csv")
    test_df = pd.read_csv(BASE_DIR / "data/processed/test.csv")
    
    check("train.csv has 1800 rows", len(train_df) == 1800, f"Got {len(train_df)}")
    check("val.csv has 360 rows", len(val_df) == 360, f"Got {len(val_df)}")
    check("test.csv has 540 rows", len(test_df) == 540, f"Got {len(test_df)}")
    
    check("train.csv has 18 unique intents", train_df["intent"].nunique() == 18,
          f"Got {train_df['intent'].nunique()}")
    check("All 18 target intents in train", set(train_df["intent"].unique()) == set(TARGET_INTENTS))
    
    # Data leakage
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])
    train_test_overlap = train_texts & test_texts
    check("Zero train-test text overlap", len(train_test_overlap) == 0,
          f"Found {len(train_test_overlap)} overlapping texts")
    
    train_val_overlap = train_texts & val_texts
    if len(train_val_overlap) <= 1:
        warn(f"Train-val overlap: {len(train_val_overlap)} text(s)",
             "Minor: from original CLINC150 dataset, same intent")
    else:
        check("Minimal train-val overlap", len(train_val_overlap) <= 1,
              f"Found {len(train_val_overlap)} overlapping texts")
except Exception as e:
    check("Processed CSVs load", False, str(e))

# ============================================================
# 3. MODEL ARTIFACT INTEGRITY
# ============================================================
print("\n[3/9] MODEL ARTIFACT INTEGRITY")
try:
    from tensorflow.keras.models import load_model
    model = load_model(BASE_DIR / "models/chatbot_model.keras")
    check("Model loads successfully", True)
    
    check("Model input shape is (None, 30)", model.input_shape == (None, 30),
          f"Got {model.input_shape}")
    check("Model output shape is (None, 18)", model.output_shape == (None, 18),
          f"Got {model.output_shape}")
    
    layer_names = [l.name for l in model.layers]
    check("Has embedding layer", "embedding" in layer_names)
    check("Has bidirectional LSTM", "bidirectional_lstm" in layer_names or
          any("bidirectional" in n for n in layer_names))
    check("Has softmax output", any("intent" in n or "dense" in n for n in layer_names))
    
    # Check model file size < 50 MB (Git limit)
    model_size_mb = (BASE_DIR / "models/chatbot_model.keras").stat().st_size / (1024 * 1024)
    check(f"Model size < 50 MB (actual: {model_size_mb:.2f} MB)", model_size_mb < 50)
    
except Exception as e:
    check("Model loads", False, str(e))

try:
    with open(BASE_DIR / "models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    check("Tokenizer loads", True)
    check(f"Tokenizer vocab > 0 (actual: {len(tokenizer.word_index)})",
          len(tokenizer.word_index) > 0)
    
    with open(BASE_DIR / "models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    check("Label encoder loads", True)
    check("Label encoder has 18 classes", len(label_encoder.classes_) == 18,
          f"Got {len(label_encoder.classes_)}")
    check("Label encoder classes match target intents",
          set(label_encoder.classes_) == set(TARGET_INTENTS))
    
    with open(BASE_DIR / "models/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    check("Metadata loads", True)
    check("Metadata max_len = 30", metadata.get("max_len") == 30,
          f"Got {metadata.get('max_len')}")
    check("Metadata num_classes = 18", metadata.get("num_classes") == 18,
          f"Got {metadata.get('num_classes')}")
except Exception as e:
    check("Artifacts load", False, str(e))

# ============================================================
# 4. INFERENCE PIPELINE
# ============================================================
print("\n[4/9] INFERENCE PIPELINE")
try:
    from src.inference import IntentClassifierService
    service = IntentClassifierService()
    
    test_cases = [
        ("hello there", "greeting"),
        ("what is the weather today", "weather"),
        ("set an alarm for 7 am", "alarm"),
        ("transfer money to savings", "transfer"),
        ("tell me a joke", "tell_joke"),
    ]
    
    for text, expected in test_cases:
        result = service.predict(text)
        check(f"Predict '{text}' → {expected}",
              result["raw_intent"] == expected,
              f"Got {result['raw_intent']} (conf={result['confidence']:.4f})")
    
    # Empty input
    result = service.predict("")
    check("Empty input returns 'none' intent", result["intent"] == "none")
    check("Empty input has 0.0 confidence", result["confidence"] == 0.0)
    
    # Low confidence threshold override
    result = service.predict("hello", confidence_threshold=0.999)
    check("Threshold override marks low confidence",
          result["is_low_confidence"] == True or result["confidence"] >= 0.999)
    
except Exception as e:
    check("Inference pipeline", False, str(e))

# ============================================================
# 5. RESPONSE HONESTY
# ============================================================
print("\n[5/9] RESPONSE HONESTY AUDIT")
try:
    from src.responses import INTENT_RESPONSES
    
    dishonest_phrases = [
        "money transfer initiated", "transfer initiated", "transfer completed",
        "alarm successfully set", "alarm set successfully", "alarm has been set",
        "account has been frozen", "account frozen successfully", "card has been blocked",
        "appointment has been booked", "booking confirmed", "reservation confirmed",
        "payment has been completed", "payment completed successfully", "bill paid successfully",
        "reminder has been set", "reminder set successfully",
        "balance is $", "you have $",
        "here are your transactions",
    ]
    
    all_honest = True
    for intent, responses in INTENT_RESPONSES.items():
        for response in responses:
            for phrase in dishonest_phrases:
                if phrase.lower() in response.lower():
                    check(f"No dishonest claim in '{intent}'", False,
                          f"Found '{phrase}' in: {response[:60]}...")
                    all_honest = False
    
    if all_honest:
        check("All 18 intent responses are academically honest", True)
    
    check("All 18 intents have responses", len(INTENT_RESPONSES) == 18,
          f"Got {len(INTENT_RESPONSES)} intents")
    check("All intents match target list",
          set(INTENT_RESPONSES.keys()) == set(TARGET_INTENTS))
    
except Exception as e:
    check("Response system", False, str(e))

# ============================================================
# 6. PATH PORTABILITY
# ============================================================
print("\n[6/9] PATH PORTABILITY (No Hardcoded Absolute Paths)")
source_files = list((BASE_DIR / "src").glob("*.py")) + [
    BASE_DIR / "app.py", BASE_DIR / "train.py", BASE_DIR / "evaluate.py",
    BASE_DIR / "config.py"
]
hardcoded_path_pattern = re.compile(r'["\'](?:[A-Z]:\\|/home/|/Users/)')
path_issues = []
for src_file in source_files:
    content = src_file.read_text(encoding="utf-8", errors="ignore")
    matches = hardcoded_path_pattern.findall(content)
    if matches:
        path_issues.append((src_file.name, matches))

check("Zero hardcoded absolute paths in source", len(path_issues) == 0,
      f"Found in: {path_issues}")

# ============================================================
# 7. REQUIREMENTS FILE
# ============================================================
print("\n[7/9] REQUIREMENTS FILE")
req_content = (BASE_DIR / "requirements.txt").read_text()
required_packages = ["tensorflow", "streamlit", "SpeechRecognition",
                     "scikit-learn", "numpy", "pandas", "matplotlib", "seaborn",
                     "streamlit-mic-recorder"]
for pkg in required_packages:
    check(f"requirements.txt has {pkg}",
          pkg.lower() in req_content.lower(),
          f"'{pkg}' not found")

# ============================================================
# 8. EVALUATION RESULTS CONSISTENCY
# ============================================================
print("\n[8/9] EVALUATION RESULTS")
try:
    with open(BASE_DIR / "reports/evaluation_results.json") as f:
        eval_results = json.load(f)
    
    check("eval results has accuracy", "accuracy" in eval_results)
    check("Test accuracy ≈ 93.70%",
          abs(eval_results.get("accuracy", 0) - 0.9370) < 0.01,
          f"Got {eval_results.get('accuracy')}")
    check("Test samples = 540", eval_results.get("test_samples") == 540)
    check("Correct count = 506", eval_results.get("correct_count") == 506)
    check("Misclassified count = 34", eval_results.get("misclassified_count") == 34)
except Exception as e:
    check("Evaluation results", False, str(e))

try:
    with open(BASE_DIR / "reports/training_metrics.json") as f:
        train_metrics = json.load(f)
    check("Training metrics has best_epoch", "best_epoch" in train_metrics)
    check("Best epoch = 11", train_metrics.get("best_epoch") == 11,
          f"Got {train_metrics.get('best_epoch')}")
except Exception as e:
    check("Training metrics", False, str(e))

# ============================================================
# 9. DOCUMENTATION
# ============================================================
print("\n[9/9] DOCUMENTATION")
check("README.md exists and > 5KB",
      (BASE_DIR / "README.md").stat().st_size > 5000)
check("project_report.md exists and > 10KB",
      (BASE_DIR / "reports/project_report.md").stat().st_size > 10000)
check("demo_examples.md exists and > 2KB",
      (BASE_DIR / "reports/demo_examples.md").stat().st_size > 2000)

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)
print(f"  Total Checks:   {total_checks}")
print(f"  Passed:          {passed_checks}  {PASS}")
print(f"  Failed:          {failed_checks}  {'(NONE)' if failed_checks == 0 else FAIL}")
print(f"  Warnings:        {warned_checks}  {'(NONE)' if warned_checks == 0 else WARN}")
print(f"\n  Overall Status:  {'✅ PROJECT AUDIT PASSED' if failed_checks == 0 else '❌ PROJECT AUDIT FAILED'}")
print("=" * 70)

sys.exit(0 if failed_checks == 0 else 1)
