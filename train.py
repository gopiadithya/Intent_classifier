"""
Training Script for Voice-Enabled Chatbot Intent Classification.
Loads processed data, fits preprocessor, trains BiLSTM neural network,
saves best model checkpoints and artifacts, and generates training plots.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.preprocessing import TextPreprocessor
from src.model import build_bilstm_model

def train():
    print("=" * 60)
    print("STARTING MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Processed Datasets
    print(f"Loading training data from:   {config.PROCESSED_TRAIN_FILE}")
    print(f"Loading validation data from: {config.PROCESSED_VAL_FILE}")
    train_df = pd.read_csv(config.PROCESSED_TRAIN_FILE)
    val_df = pd.read_csv(config.PROCESSED_VAL_FILE)

    X_train_raw = train_df["text"].values
    y_train_raw = train_df["intent"].values

    X_val_raw = val_df["text"].values
    y_val_raw = val_df["intent"].values

    print(f"Loaded {len(train_df)} training samples across {train_df['intent'].nunique()} intents.")
    print(f"Loaded {len(val_df)} validation samples.")

    # 2. Build and Fit Text Preprocessor (Train Split Only)
    preprocessor = TextPreprocessor(
        vocab_size=config.VOCAB_SIZE,
        max_len=config.MAX_SEQUENCE_LENGTH,
        oov_token=config.OOV_TOKEN
    )
    preprocessor.fit(X_train_raw, y_train_raw)

    # 3. Transform Text and Labels
    X_train = preprocessor.transform_texts(X_train_raw)
    y_train = preprocessor.transform_labels(y_train_raw)

    X_val = preprocessor.transform_texts(X_val_raw)
    y_val = preprocessor.transform_labels(y_val_raw)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")

    # 4. Save Preprocessor Artifacts
    preprocessor.save(
        tokenizer_path=config.TOKENIZER_PATH,
        label_encoder_path=config.LABEL_ENCODER_PATH,
        metadata_path=config.METADATA_PATH
    )

    # 5. Build BiLSTM Model
    num_classes = len(preprocessor.get_classes())
    vocab_size = min(len(preprocessor.tokenizer.word_index) + 1, config.VOCAB_SIZE)

    model = build_bilstm_model(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        max_length=config.MAX_SEQUENCE_LENGTH,
        lstm_units=config.LSTM_UNITS,
        dense_units=config.DENSE_UNITS,
        dropout_rate=config.DROPOUT_RATE,
        num_classes=num_classes,
        learning_rate=config.LEARNING_RATE
    )
    print("\nModel Architecture Summary:")
    model.summary()

    # 6. Configure Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 7. Train Model
    print("\nTraining BiLSTM model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # 8. Save Training History Plot
    epochs_range = range(1, len(history.history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs_range, history.history["accuracy"], "o-", label="Training Accuracy", color="#2980b9")
    ax1.plot(epochs_range, history.history["val_accuracy"], "s--", label="Validation Accuracy", color="#27ae60")
    ax1.set_title("Training and Validation Accuracy", fontsize=13, weight="bold")
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss Plot
    ax2.plot(epochs_range, history.history["loss"], "o-", label="Training Loss", color="#c0392b")
    ax2.plot(epochs_range, history.history["val_loss"], "s--", label="Validation Loss", color="#e67e22")
    ax2.set_title("Training and Validation Loss", fontsize=13, weight="bold")
    ax2.set_xlabel("Epochs", fontsize=11)
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    history_fig_path = config.FIGURES_DIR / "training_history.png"
    plt.savefig(history_fig_path, dpi=300)
    plt.close()
    print(f"\nTraining curves saved to: {history_fig_path}")

    # Report best metrics
    best_epoch = int(np.argmin(history.history["val_loss"]))
    best_val_loss = history.history["val_loss"][best_epoch]
    best_val_acc = history.history["val_accuracy"][best_epoch]
    final_train_acc = history.history["accuracy"][best_epoch]
    final_train_loss = history.history["loss"][best_epoch]

    print("\n" + "="*50)
    print("TRAINING RUN RESULTS")
    print("="*50)
    print(f"Total Epochs Run:        {len(epochs_range)}")
    print(f"Best Validation Epoch:   {best_epoch + 1}")
    print(f"Training Loss:           {final_train_loss:.4f}")
    print(f"Training Accuracy:       {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
    print(f"Validation Loss:         {best_val_loss:.4f}")
    print(f"Validation Accuracy:     {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"Model saved to:          {config.MODEL_PATH}")
    print("="*50)

    # Save training metrics to JSON for evaluation and reporting
    training_metrics = {
        "epochs_run": len(epochs_range),
        "best_epoch": best_epoch + 1,
        "train_loss": float(final_train_loss),
        "train_accuracy": float(final_train_acc),
        "val_loss": float(best_val_loss),
        "val_accuracy": float(best_val_acc)
    }
    with open(config.REPORTS_DIR / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(training_metrics, f, indent=4)

    return model, history

if __name__ == "__main__":
    train()
