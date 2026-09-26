"""
Dataset Analysis and Pre-Extraction Script for Voice Chatbot.
Filters CLINC150 dataset to the 18 target intents, generates analytical visualizations,
and exports clean train/val/test CSV splits.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Set plot styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
sns.set_theme(style="whitegrid")

def extract_and_analyze():
    print(f"Loading raw CLINC150 from: {config.RAW_DATA_FILE}")
    with open(config.RAW_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_set = set(config.TARGET_INTENTS)
    splits = ["train", "val", "test"]
    split_dfs = {}

    for split in splits:
        records = []
        for text, intent in data[split]:
            if intent in target_set:
                records.append({"text": text.strip(), "intent": intent})
        df = pd.DataFrame(records)
        df["word_count"] = df["text"].apply(lambda s: len(s.split()))
        split_dfs[split] = df
        
        # Save processed CSVs
        out_csv = getattr(config, f"PROCESSED_{split.upper()}_FILE")
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(df)} samples for '{split}' to: {out_csv}")

    # Dataset Summary Statistics
    train_df = split_dfs["train"]
    val_df = split_dfs["val"]
    test_df = split_dfs["test"]
    combined_df = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test")
    ], ignore_index=True)

    print("\n" + "="*50)
    print("DATASET ANALYSIS REPORT (Actual Measured Statistics)")
    print("="*50)
    print(f"Number of Selected Intents: {len(config.TARGET_INTENTS)}")
    print(f"Intents: {', '.join(config.TARGET_INTENTS)}")
    print(f"Training Samples:   {len(train_df)} ({len(train_df)/len(config.TARGET_INTENTS):.0f} per intent)")
    print(f"Validation Samples: {len(val_df)} ({len(val_df)/len(config.TARGET_INTENTS):.0f} per intent)")
    print(f"Test Samples:       {len(test_df)} ({len(test_df)/len(config.TARGET_INTENTS):.0f} per intent)")
    print(f"Total Selected:     {len(combined_df)}")
    print("="*50)

    # Class balance check
    train_counts = train_df["intent"].value_counts()
    val_counts = val_df["intent"].value_counts()
    test_counts = test_df["intent"].value_counts()
    print("\nSamples per Intent across Splits:")
    summary_table = pd.DataFrame({
        "Train": train_counts,
        "Validation": val_counts,
        "Test": test_counts,
        "Total": train_counts + val_counts + test_counts
    })
    print(summary_table)

    # Sentence Length Analysis
    all_lengths = combined_df["word_count"]
    print("\nSentence Length Statistics (Words):")
    print(f"Minimum words:     {all_lengths.min()}")
    print(f"Maximum words:     {all_lengths.max()}")
    print(f"Mean words:        {all_lengths.mean():.2f}")
    print(f"Median words:      {all_lengths.median():.2f}")
    print(f"95th Percentile:   {np.percentile(all_lengths, 95):.2f}")
    print(f"99th Percentile:   {np.percentile(all_lengths, 99):.2f}")
    print("="*50)

    # Figure 1: Intent Class Distribution
    plt.figure(figsize=(12, 6))
    order = sorted(config.TARGET_INTENTS)
    sns.countplot(data=combined_df, y="intent", hue="split", order=order, palette="viridis")
    plt.title("Sample Distribution per Intent across Train, Val, and Test Splits", fontsize=14, weight="bold")
    plt.xlabel("Sample Count", fontsize=12)
    plt.ylabel("Intent Name", fontsize=12)
    plt.legend(title="Split")
    plt.tight_layout()
    fig1_path = config.FIGURES_DIR / "intent_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig1_path}")

    # Figure 2: Sequence Length Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(combined_df["word_count"], bins=20, kde=True, color="#2b5c8f")
    plt.axvline(x=config.MAX_SEQUENCE_LENGTH, color="red", linestyle="--", linewidth=2, label=f"Max Pad Length ({config.MAX_SEQUENCE_LENGTH})")
    plt.axvline(x=np.percentile(all_lengths, 95), color="green", linestyle=":", linewidth=2, label=f"95th Percentile ({np.percentile(all_lengths, 95):.1f})")
    plt.title("Utterance Length Distribution (Word Count)", fontsize=14, weight="bold")
    plt.xlabel("Number of Words per Utterance", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.tight_layout()
    fig2_path = config.FIGURES_DIR / "sequence_length_distribution.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig2_path}")

    # Figure 3: Split Proportion
    plt.figure(figsize=(6, 6))
    split_counts = [len(train_df), len(val_df), len(test_df)]
    plt.pie(split_counts, labels=["Train (66.7%)", "Val (13.3%)", "Test (20.0%)"],
            autopct="%1.1f%%", colors=["#3498db", "#f39c12", "#2ecc71"], startangle=140,
            textprops={"fontsize": 11})
    plt.title("Dataset Split Proportion (CLINC150 Subset)", fontsize=14, weight="bold")
    plt.tight_layout()
    fig3_path = config.FIGURES_DIR / "split_distribution.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig3_path}")

    return {
        "num_intents": len(config.TARGET_INTENTS),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "max_words": int(all_lengths.max()),
        "p95_words": float(np.percentile(all_lengths, 95)),
        "p99_words": float(np.percentile(all_lengths, 99))
    }

if __name__ == "__main__":
    extract_and_analyze()
