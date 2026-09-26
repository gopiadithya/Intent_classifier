"""
Data Acquisition Script for CLINC150 Dataset.
Downloads the official CLINC150 data_full.json and inspects its schema.
"""
import os
import sys
import json
import urllib.request
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def download_clinc150():
    """Download CLINC150 dataset if not already present."""
    if config.RAW_DATA_FILE.exists():
        print(f"Dataset already exists at: {config.RAW_DATA_FILE}")
    else:
        print(f"Downloading CLINC150 from: {config.CLINC150_URL} ...")
        urllib.request.urlretrieve(config.CLINC150_URL, config.RAW_DATA_FILE)
        print(f"Downloaded successfully to: {config.RAW_DATA_FILE}")

def inspect_raw_dataset():
    """Inspect splits, counts, and available intent names in CLINC150."""
    with open(config.RAW_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Keys in dataset:", list(data.keys()))
    
    # Standard splits in CLINC150 data_full.json: 'train', 'val', 'test', 'oos_train', 'oos_val', 'oos_test'
    for split in ["train", "val", "test"]:
        if split in data:
            print(f"Total samples in '{split}' split: {len(data[split])}")

    # Gather all unique in-scope intents from train split
    train_intents = set([intent for _, intent in data["train"] if intent != "oos"])
    val_intents = set([intent for _, intent in data["val"] if intent != "oos"])
    test_intents = set([intent for _, intent in data["test"] if intent != "oos"])

    print(f"Total unique in-scope intents: {len(train_intents)}")
    print(f"Train/Val intent consistency: {train_intents == val_intents}")
    print(f"Train/Test intent consistency: {train_intents == test_intents}")

    all_sorted_intents = sorted(list(train_intents))
    print("\nSample of available intents (first 30):")
    print(", ".join(all_sorted_intents[:30]))

    return all_sorted_intents

if __name__ == "__main__":
    download_clinc150()
    intents = inspect_raw_dataset()
