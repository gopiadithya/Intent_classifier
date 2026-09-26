"""
Model Evaluation Script for Voice-Enabled Chatbot.
Evaluates the trained BiLSTM model on the untouched test set (540 samples across 18 intents),
generates accuracy, precision, recall, f1-score, confusion matrix, and error analysis.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.preprocessing import TextPreprocessor
from src.model import load_trained_model

def evaluate():
    print("=" * 60)
    print("STARTING TEST SET EVALUATION")
    print("=" * 60)

    # 1. Load Untouched Test Set
    test_df = pd.read_csv(config.PROCESSED_TEST_FILE)
    print(f"Loaded {len(test_df)} test samples from: {config.PROCESSED_TEST_FILE}")

    X_test_raw = test_df["text"].values
    y_test_raw = test_df["intent"].values

    # 2. Load Preprocessor and Trained Model
    print("Loading preprocessor and trained BiLSTM model...")
    preprocessor = TextPreprocessor.load()
    model = load_trained_model()

    # 3. Transform Test Data
    X_test = preprocessor.transform_texts(X_test_raw)
    y_true_indices = preprocessor.transform_labels(y_test_raw)
    classes = preprocessor.get_classes()

    # 4. Predict Probabilities and Classes
    print("Generating predictions on test set...")
    y_probs = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=0)
    y_pred_indices = np.argmax(y_probs, axis=1)
    confidence_scores = np.max(y_probs, axis=1)

    y_pred_labels = [classes[idx] for idx in y_pred_indices]

    # 5. Compute Quantitative Metrics
    acc = accuracy_score(y_true_indices, y_pred_indices)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_indices, y_pred_indices, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_indices, y_pred_indices, average="weighted", zero_division=0
    )

    report_str = classification_report(
        y_true_indices, y_pred_indices, target_names=classes, digits=4
    )
    report_dict = classification_report(
        y_true_indices, y_pred_indices, target_names=classes, output_dict=True
    )

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Test Accuracy:    {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Macro Precision:          {macro_p:.4f}")
    print(f"Macro Recall:             {macro_r:.4f}")
    print(f"Macro F1-Score:           {macro_f1:.4f}")
    print(f"Weighted F1-Score:        {weighted_f1:.4f}")
    print(f"Average Confidence:       {np.mean(confidence_scores):.4f} ({np.mean(confidence_scores)*100:.2f}%)")
    print("-" * 60)
    print("\nDetailed Classification Report:")
    print(report_str)

    # 6. Confusion Matrix Visualization
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Normalized Ratio"}
    )
    plt.title(f"Normalized Confusion Matrix (Test Accuracy: {acc*100:.2f}%)", fontsize=15, weight="bold")
    plt.xlabel("Predicted Intent Label", fontsize=12)
    plt.ylabel("True Intent Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    cm_path = config.FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_path}")

    # 7. Qualitative Error Analysis: Correct & Misclassified Examples
    correct_examples = []
    misclassified_examples = []

    for i in range(len(test_df)):
        item = {
            "text": X_test_raw[i],
            "true_intent": y_test_raw[i],
            "predicted_intent": y_pred_labels[i],
            "confidence": float(confidence_scores[i])
        }
        if y_test_raw[i] == y_pred_labels[i]:
            correct_examples.append(item)
        else:
            misclassified_examples.append(item)

    print("\n" + "=" * 60)
    print("SAMPLE CORRECT PREDICTIONS (First 5):")
    print("=" * 60)
    for sample in correct_examples[:5]:
        print(f"Text:       '{sample['text']}'")
        print(f"Intent:     {sample['predicted_intent']} (Conf: {sample['confidence']*100:.2f}%)")
        print("-" * 40)

    print("\n" + "=" * 60)
    print(f"SAMPLE MISCLASSIFIED PREDICTIONS (Total {len(misclassified_examples)} out of {len(test_df)}):")
    print("=" * 60)
    for sample in misclassified_examples[:10]:
        print(f"Text:       '{sample['text']}'")
        print(f"True:       {sample['true_intent']}")
        print(f"Predicted:  {sample['predicted_intent']} (Conf: {sample['confidence']*100:.2f}%)")
        print("-" * 40)

    # 8. Save Metrics to JSON
    results = {
        "test_samples": len(test_df),
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mean_confidence": float(np.mean(confidence_scores)),
        "correct_count": len(correct_examples),
        "misclassified_count": len(misclassified_examples),
        "sample_misclassified": misclassified_examples[:10],
        "sample_correct": correct_examples[:10],
        "classification_report": report_dict
    }

    eval_json_path = config.REPORTS_DIR / "evaluation_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nEvaluation metrics successfully saved to: {eval_json_path}")

    return results

if __name__ == "__main__":
    evaluate()
