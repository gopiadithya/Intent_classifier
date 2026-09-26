"""
Deployment Readiness Pre-Flight Checklist Script.
Verifies all artifacts, dependencies, path portability, and model weights
prior to public deployment on Streamlit Community Cloud.
"""
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def audit_deployment():
    print("=" * 60)
    print("AUDITING DEPLOYMENT READINESS")
    print("=" * 60)

    checklist = []

    # 1. Essential root files
    essential_files = ["app.py", "config.py", "requirements.txt", ".gitignore"]
    for fname in essential_files:
        p = BASE_DIR / fname
        exists = p.exists()
        checklist.append((f"Root file: {fname}", exists, str(p)))
        print(f"[{'PASS' if exists else 'FAIL'}] Essential file: {fname}")

    # 2. Required model artifacts
    model_files = [
        "chatbot_model.keras",
        "tokenizer.pkl",
        "label_encoder.pkl",
        "metadata.pkl"
    ]
    total_model_bytes = 0
    for mname in model_files:
        p = BASE_DIR / "models" / mname
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        total_model_bytes += size
        checklist.append((f"Model artifact: {mname}", exists, f"{size / 1024:.1f} KB"))
        print(f"[{'PASS' if exists else 'FAIL'}] Model artifact: {mname} ({size / 1024:.1f} KB)")

    total_mb = total_model_bytes / (1024 * 1024)
    is_git_safe = total_mb < 50.0  # Well below GitHub's 100MB limit
    checklist.append(("Model size < 50MB (GitHub Safe)", is_git_safe, f"{total_mb:.2f} MB"))
    print(f"[{'PASS' if is_git_safe else 'FAIL'}] Total Model Size: {total_mb:.2f} MB (Git-safe, no LFS or S3 required)")

    # 3. Path portability check (no absolute local paths hardcoded)
    hardcoded_issues = []
    current_script = Path(__file__).resolve()
    for py_file in BASE_DIR.rglob("*.py"):
        if py_file.resolve() == current_script:
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            if "C:\\Users\\" in content or "/home/" in content:
                hardcoded_issues.append(str(py_file.relative_to(BASE_DIR)))
        except Exception:
            pass

    has_no_hardcoded = len(hardcoded_issues) == 0
    checklist.append(("Zero hardcoded absolute paths", has_no_hardcoded, hardcoded_issues))
    print(f"[{'PASS' if has_no_hardcoded else 'FAIL'}] Path portability check: {len(hardcoded_issues)} violations found")

    # 4. Dry-run inference test
    try:
        sys.path.insert(0, str(BASE_DIR))
        from src.inference import IntentClassifierService
        svc = IntentClassifierService.get_instance()
        res = svc.predict("What is the forecast today?")
        inference_works = res["raw_intent"] == "weather"
        checklist.append(("In-process inference dry-run", inference_works, f"Predicted: {res['intent']}"))
        print(f"[{'PASS' if inference_works else 'FAIL'}] Dry-run inference: {res['intent']} ({res['confidence_pct']})")
    except Exception as e:
        checklist.append(("In-process inference dry-run", False, str(e)))
        print(f"[FAIL] Inference test error: {e}")

    # Summary
    all_passed = all(item[1] for item in checklist)
    print("=" * 60)
    print(f"DEPLOYMENT AUDIT RESULT: {'ALL CHECKS PASSED (100% READY)' if all_passed else 'SOME CHECKS FAILED'}")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    ready = audit_deployment()
    sys.exit(0 if ready else 1)
