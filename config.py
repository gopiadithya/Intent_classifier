import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset URLs & Filenames
CLINC150_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
RAW_DATA_FILE = RAW_DATA_DIR / "clinc150_full.json"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_VAL_FILE = PROCESSED_DATA_DIR / "val.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

# Model Artifact Paths
MODEL_PATH = MODELS_DIR / "chatbot_model.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
METADATA_PATH = MODELS_DIR / "metadata.pkl"

# Selected Intents for the Chatbot (18 balanced conversational and utility intents)
TARGET_INTENTS = [
    "greeting",
    "goodbye",
    "thank_you",
    "what_can_i_ask_you",
    "tell_joke",
    "weather",
    "directions",
    "restaurant_suggestion",
    "book_hotel",
    "travel_suggestion",
    "calendar",
    "reminder",
    "alarm",
    "balance",
    "spending_history",
    "pay_bill",
    "transfer",
    "freeze_account"
]

# NLP & Model Hyperparameters
VOCAB_SIZE = 5000          # Maximum vocabulary size
OOV_TOKEN = "<OOV>"        # Out of vocabulary token
MAX_SEQUENCE_LENGTH = 30   # Sequence padding length (will be tuned based on data analysis)
PADDING_TYPE = "post"
TRUNCATING_TYPE = "post"

EMBEDDING_DIM = 128
LSTM_UNITS = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

# Confidence Threshold for Inference
CONFIDENCE_THRESHOLD = 0.50
FALLBACK_RESPONSE = "I couldn't confidently identify the type of request. I can currently help with topics such as weather, travel, banking, reminders, alarms, directions, and general conversation."
