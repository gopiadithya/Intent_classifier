"""
Comprehensive test runner for the entire Voice-Enabled Chatbot project.
Runs all unit and integration tests across preprocessing, model, inference, and speech.
"""
import sys
import unittest
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def run_suite():
    loader = unittest.TestLoader()
    start_dir = Path(__file__).resolve().parent
    suite = loader.discover(start_dir=str(start_dir), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*50)
    print(f"Total Tests Run:    {result.testsRun}")
    print(f"Total Failures:     {len(result.failures)}")
    print(f"Total Errors:       {len(result.errors)}")
    print(f"Suite Status:       {'PASSED' if result.wasSuccessful() else 'FAILED'}")
    print("="*50)

    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_suite()
    sys.exit(0 if success else 1)
