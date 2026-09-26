"""
Bidirectional LSTM Model Architecture for Intent Classification.
Implements the genuine Deep Learning architecture with Embedding, BiLSTM,
Dropout, Dense, and Softmax classification layers.
"""
import sys
from pathlib import Path
from typing import Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Input
)
from tensorflow.keras.optimizers import Adam

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def build_bilstm_model(
    vocab_size: int = config.VOCAB_SIZE,
    embedding_dim: int = config.EMBEDDING_DIM,
    max_length: int = config.MAX_SEQUENCE_LENGTH,
    lstm_units: int = config.LSTM_UNITS,
    dense_units: int = config.DENSE_UNITS,
    dropout_rate: float = config.DROPOUT_RATE,
    num_classes: int = len(config.TARGET_INTENTS),
    learning_rate: float = config.LEARNING_RATE
) -> Model:
    """
    Constructs and compiles the BiLSTM Intent Classification model.

    Architecture:
    1. Input layer: (batch_size, max_length)
    2. Embedding: projects token IDs into dense semantic space
    3. Bidirectional LSTM: extracts contextual representations in both directions
    4. Dropout: prevents co-adaptation of features (regularization)
    5. Dense (ReLU): intermediate non-linear feature projection
    6. Dropout: secondary regularization layer
    7. Dense (Softmax): class probability distribution across all intents
    """
    model = Sequential([
        Input(shape=(max_length,), dtype=tf.int32, name="input_sequence"),
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name="embedding"
        ),
        Bidirectional(
            LSTM(lstm_units, return_sequences=False),
            name="bidirectional_lstm"
        ),
        Dropout(dropout_rate, name="dropout_1"),
        Dense(dense_units, activation="relu", name="dense_features"),
        Dropout(dropout_rate, name="dropout_2"),
        Dense(num_classes, activation="softmax", name="intent_probabilities")
    ], name="BiLSTM_Intent_Classifier")

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def load_trained_model(model_path: Optional[Path] = None) -> Model:
    """Load a pre-trained Keras model from disk."""
    path = model_path or config.MODEL_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    return tf.keras.models.load_model(str(path))

if __name__ == "__main__":
    print("Building model architecture with default configuration:")
    clf = build_bilstm_model()
    clf.summary()
