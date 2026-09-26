"""
NLP Preprocessing Pipeline for Voice-Enabled Chatbot.
Handles text cleaning, tokenization, sequence conversion, padding,
and label encoding/decoding.
"""
import re
import sys
import pickle
from pathlib import Path
from typing import Tuple, List, Union
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def clean_text(text: str) -> str:
    """
    Standardize and clean raw text utterance.
    - Converts to lowercase
    - Normalizes contractions / special characters
    - Strips excess whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Normalize common apostrophes
    text = text.replace("’", "'").replace("`", "'")
    # Remove unwanted punctuation but keep alphanumeric and whitespace
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

class TextPreprocessor:
    """
    Manages Tokenizer and LabelEncoder lifecycles for reproducible
    training and real-time inference.
    """
    def __init__(self,
                 vocab_size: int = config.VOCAB_SIZE,
                 max_len: int = config.MAX_SEQUENCE_LENGTH,
                 oov_token: str = config.OOV_TOKEN,
                 padding: str = config.PADDING_TYPE,
                 truncating: str = config.TRUNCATING_TYPE):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.oov_token = oov_token
        self.padding = padding
        self.truncating = truncating

        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token=self.oov_token,
            lower=True
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        """Fit Tokenizer on training texts and LabelEncoder on labels."""
        cleaned_texts = [clean_text(t) for t in texts]
        self.tokenizer.fit_on_texts(cleaned_texts)
        self.label_encoder.fit(labels)
        self.is_fitted = True

        actual_vocab_size = len(self.tokenizer.word_index) + 1
        num_classes = len(self.label_encoder.classes_)
        print(f"Tokenizer fitted: {actual_vocab_size} distinct words in vocabulary.")
        print(f"LabelEncoder fitted: {num_classes} distinct intent classes.")

    def transform_texts(self, texts: Union[List[str], np.ndarray, pd.Series]) -> np.ndarray:
        """Convert raw texts to padded numerical sequence arrays."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        cleaned = [clean_text(t) for t in texts]
        seqs = self.tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(
            seqs,
            maxlen=self.max_len,
            padding=self.padding,
            truncating=self.truncating
        )
        return padded

    def transform_labels(self, labels: Union[List[str], np.ndarray, pd.Series]) -> np.ndarray:
        """Encode textual intent names to integer class IDs."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        return self.label_encoder.transform(labels)

    def inverse_transform_label(self, class_id: int) -> str:
        """Decode integer class ID back to original intent label string."""
        return self.label_encoder.inverse_transform([class_id])[0]

    def get_classes(self) -> List[str]:
        """Return list of encoded intent class names."""
        return list(self.label_encoder.classes_)

    def save(self,
             tokenizer_path: Union[str, Path] = config.TOKENIZER_PATH,
             label_encoder_path: Union[str, Path] = config.LABEL_ENCODER_PATH,
             metadata_path: Union[str, Path] = config.METADATA_PATH):
        """Serialize tokenizer, label encoder, and config metadata to disk."""
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)
        with open(label_encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

        metadata = {
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "oov_token": self.oov_token,
            "padding": self.padding,
            "truncating": self.truncating,
            "classes": list(self.label_encoder.classes_),
            "actual_vocab_size": len(self.tokenizer.word_index) + 1,
            "num_classes": len(self.label_encoder.classes_)
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        print(f"Serialized preprocessor artifacts to {config.MODELS_DIR}")

    @classmethod
    def load(cls,
             tokenizer_path: Union[str, Path] = config.TOKENIZER_PATH,
             label_encoder_path: Union[str, Path] = config.LABEL_ENCODER_PATH,
             metadata_path: Union[str, Path] = config.METADATA_PATH) -> "TextPreprocessor":
        """Load serialized tokenizer, label encoder, and metadata."""
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        with open(metadata_path, "rb") as f:
            meta = pickle.load(f)

        instance = cls(
            vocab_size=meta.get("vocab_size", config.VOCAB_SIZE),
            max_len=meta.get("max_len", config.MAX_SEQUENCE_LENGTH),
            oov_token=meta.get("oov_token", config.OOV_TOKEN),
            padding=meta.get("padding", config.PADDING_TYPE),
            truncating=meta.get("truncating", config.TRUNCATING_TYPE)
        )
        instance.tokenizer = tokenizer
        instance.label_encoder = label_encoder
        instance.is_fitted = True
        return instance
