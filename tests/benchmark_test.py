"""
Benchmark testing script running 20 representative queries through the complete
NLP + BiLSTM + Confidence + Response pipeline.
"""
import sys
from pathlib import Path
import pandas as pd

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference import IntentClassifierService

def run_benchmark():
    service = IntentClassifierService.get_instance()

    test_queries = [
        # Conversational
        ("hello good morning assistant", "greeting"),
        ("see you later goodbye", "goodbye"),
        ("thanks a lot for your help", "thank_you"),
        ("what kind of things can i ask you", "what_can_i_ask_you"),
        ("can you tell me a good joke", "tell_joke"),

        # Daily Utility & Productivity
        ("what is the weather like outside right now", "weather"),
        ("give me driving directions to downtown", "directions"),
        ("can you suggest a good restaurant nearby", "restaurant_suggestion"),
        ("i need to book a hotel room for next weekend", "book_hotel"),
        ("where should i travel for my next vacation", "travel_suggestion"),
        ("what meetings do i have on my calendar today", "calendar"),
        ("remind me to call the doctor tomorrow morning", "reminder"),
        ("set an alarm for 6:30 am", "alarm"),

        # Banking & Account Management
        ("how much money is in my bank balance", "balance"),
        ("show me my spending history for this month", "spending_history"),
        ("i need to pay my electricity bill", "pay_bill"),
        ("transfer fifty dollars from savings to checking", "transfer"),
        ("please freeze my bank card immediately", "freeze_account"),

        # Edge cases & Confidence Gating
        ("", "none"),
        ("flim flam blorp quux zorp xyz", "out_of_scope_or_low_conf")
    ]

    print("\n" + "="*80)
    print("RUNNING 20 REPRESENTATIVE BENCHMARK QUERIES THROUGH END-TO-END PIPELINE")
    print("="*80)

    results = []
    correct_matches = 0

    for query, expected in test_queries:
        pred = service.predict(query)
        is_match = (pred["raw_intent"] == expected) if expected != "out_of_scope_or_low_conf" else pred["is_low_confidence"]
        if is_match or (expected == "none" and pred["intent"] == "none"):
            correct_matches += 1

        results.append({
            "Input Query": query if query else "<EMPTY INPUT>",
            "Expected": expected,
            "Predicted Intent": pred["intent"],
            "Confidence": pred["confidence_pct"],
            "Low Conf Flag": pred["is_low_confidence"],
            "Response Preview": pred["response"][:60] + "..." if len(pred["response"]) > 60 else pred["response"]
        })

        print(f"Query:      '{query}'")
        print(f"Predicted:  {pred['intent']} | Confidence: {pred['confidence_pct']} | LowConf: {pred['is_low_confidence']}")
        print(f"Response:   {pred['response']}")
        print("-" * 80)

    df = pd.DataFrame(results)
    print("\nSUMMARY TABLE:")
    print(df[["Input Query", "Expected", "Predicted Intent", "Confidence", "Low Conf Flag"]].to_string())

    print("\n" + "="*80)
    print(f"Benchmark Match Rate: {correct_matches} / {len(test_queries)} ({correct_matches/len(test_queries)*100:.1f}%)")
    print("="*80)

    return df

if __name__ == "__main__":
    run_benchmark()
