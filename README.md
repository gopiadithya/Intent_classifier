# 🎙️ Voice-Enabled Chatbot Using Speech Recognition and Deep Learning-Based Intent Classification

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![UI](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-93.70%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An academic Deep Learning and Natural Language Processing project that converts spoken audio into text, processes the utterance using a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network to classify the user's intent, and delivers an appropriate conversational response.

---

## 📌 Project Overview

Modern voice conversational systems bridge spoken language interaction and digital automated services. This project implements an explainable, end-to-end intent-classification pipeline:

1. **Voice Capture**: Ingests user speech in real-time through the web browser microphone with silence auto-submit.
2. **Speech Recognition**: Transcribes spoken audio waveforms into normalized text using Google Speech Recognition.
3. **NLP Preprocessing**: Tokenizes utterances and applies sequence padding.
4. **Deep Learning Classification**: Employs an **Embedding + Bidirectional LSTM** neural network to classify the recognized text into one of 18 intent categories.
5. **Confidence & Semantic Evaluation**: Evaluates softmax class probabilities and semantic keyword evidence to ensure reliable classification.
6. **Intent-Relevant Response Module**: Rather than returning generic misunderstandings or ungrounded generative LLM hallucinations, the response layer selects a direct, semantically intended answer corresponding to the predicted intent category (including documented demonstration knowledge mappings for questions like VIT Vellore, Python, and Machine Learning).
7. **Streamlit Interactive UI**: An academic interface displaying the voice input, recognized speech, detected intent, model confidence score, and conversational response.

---

## 🏗️ System Architecture

```
                 🎙️ USER SPOKEN AUDIO
                           │
                           ▼
          [ 1. Speech Recognition Engine ]
            (AudioFile / Ambient Noise Filter)
                           │
                           ▼
             📝 RECOGNIZED TEXT UTTERANCE
                           │
                           ▼
             [ 2. NLP Preprocessing ]
       - Case Folding & Character Normalization
       - Word-to-Index Tokenizer (<OOV> Mapping)
       - Padded Integer Sequences (Length = 30)
                           │
                           ▼
        [ 3. Deep Learning Classifier (BiLSTM) ]
       ┌──────────────────────────────────────┐
       │  Embedding Layer (Vocab=1308, Dim=128)│
       │  Bidirectional LSTM (64 Hidden Units) │
       │  Dropout Regularization (Rate = 0.3)  │
       │  Dense Representation (64 Units, ReLU)│
       │  Dropout Regularization (Rate = 0.3)  │
       │  Softmax Output (18 Intent Classes)  │
       └──────────────────────────────────────┘
                           │
                           ▼
            🎯 PREDICTED INTENT & CONFIDENCE
                           │
                  Semantic / Topic Match?
                     /           \
                 YES              NO (Unintelligible Gibberish)
                 /                  \
                ▼                    ▼
     [ 4. Response Mapping ]     [ Fallback Handler ]
   (Direct & Intended Answers)  (Domain Capability Guide)
                \                    /
                 ▼                  ▼
             💬 CHATBOT CONVERSATIONAL RESPONSE
                           │
                           ▼
              🖥️ STREAMLIT USER INTERFACE
```

---

## 📊 Dataset: CLINC150

The model is trained on a curated subset of the publicly available, benchmark [CLINC150 (OOS-Eval) Dataset](https://github.com/clinc/oos-eval). Rather than using all 150 intents, we curated **18 balanced, practical intents** covering four conversational domains:

* **Conversational Courtesy**: `greeting`, `goodbye`, `thank_you`, `what_can_i_ask_you`, `tell_joke`
* **Personal Productivity & Daily Utility**: `weather`, `calendar`, `reminder`, `alarm`
* **Travel & Lifestyle**: `directions`, `restaurant_suggestion`, `travel_suggestion`, `book_hotel`
* **Banking & Financial Services**: `balance`, `spending_history`, `pay_bill`, `transfer`, `freeze_account`

### Dataset Statistics (Actual Measured Counts)

| Split | Number of Samples | Samples per Intent | Class Balance |
| :--- | :---: | :---: | :---: |
| **Train Split** | 1,800 | 100 per intent | 100% Balanced |
| **Validation Split** | 360 | 20 per intent | 100% Balanced |
| **Test Split (Untouched)** | 540 | 30 per intent | 100% Balanced |
| **Total Selected Dataset** | **2,700** | **150 per intent** | **18 Classes** |

* **Sentence Length (Words)**: Min = 1, Max = 24, Mean = 8.01 words, 95th Percentile = 14.0 words.
* **Sequence Padding**: Maximum sequence length is set to $30$ with post-padding, guaranteeing 100% coverage without utterance truncation.

---

## 🧠 Model Architecture & Hyperparameters

```python
Model: "BiLSTM_Intent_Classifier"
+--------------------------------------------------------------------------+
| Layer (type)                    | Output Shape           |       Param # |
|---------------------------------+------------------------+---------------|
| embedding (Embedding)           | (None, 30, 128)        |       167,424 |
| bidirectional_lstm (Bidirect.)  | (None, 128)            |        98,816 |
| dropout_1 (Dropout)             | (None, 128)            |             0 |
| dense_features (Dense)          | (None, 64)             |         8,256 |
| dropout_2 (Dropout)             | (None, 64)             |             0 |
| intent_probabilities (Dense)    | (None, 18)             |         1,170 |
+--------------------------------------------------------------------------+
Total params: 827,000 (3.15 MB)
Trainable params: 275,666 (1.05 MB)
Non-trainable params: 0 (0.00 B)
Optimizer params: 551,334 (2.10 MB)
```

* **Optimizer**: Adam (Initial Learning Rate = $0.001$, dynamically scaled via `ReduceLROnPlateau`)
* **Loss Function**: Sparse Categorical Cross-Entropy
* **Regularization**: Dropout ($0.30$ after BiLSTM, $0.30$ after Dense feature layer)
* **Early Stopping**: Monitored `val_loss` with patience of 5 epochs (best weights automatically restored)

---

## 📈 Experimental Results

### Training Convergence
* **Training Epochs**: 16 epochs (EarlyStopping selected optimal checkpoint at **Epoch 11**)
* **Training Loss**: `0.0315`
* **Training Accuracy**: `99.39%`
* **Validation Loss**: `0.2073`
* **Validation Accuracy**: `94.17%`

### Untouched Test Set Evaluation (540 Samples)
* **Overall Test Accuracy**: **93.70%** (506 out of 540 correctly classified)
* **Macro Precision**: **0.9393**
* **Macro Recall**: **0.9370**
* **Macro F1-Score**: **0.9371**
* **Weighted F1-Score**: **0.9371**
* **Mean Prediction Confidence**: **96.41%**

### Per-Intent Performance Summary

| Intent | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `alarm` | 1.0000 | 0.9667 | **0.9831** | 30 |
| `balance` | 0.9000 | 0.9000 | **0.9000** | 30 |
| `book_hotel` | 0.9667 | 0.9667 | **0.9667** | 30 |
| `calendar` | 0.8750 | 0.9333 | **0.9032** | 30 |
| `directions` | 1.0000 | 0.9333 | **0.9655** | 30 |
| `freeze_account` | 0.9375 | 1.0000 | **0.9677** | 30 |
| `goodbye` | 0.9630 | 0.8667 | **0.9123** | 30 |
| `greeting` | 0.9062 | 0.9667 | **0.9355** | 30 |
| `pay_bill` | 1.0000 | 0.9667 | **0.9831** | 30 |
| `reminder` | 0.8333 | 0.8333 | **0.8333** | 30 |
| `restaurant_suggestion` | 0.9310 | 0.9000 | **0.9153** | 30 |
| `spending_history` | 0.9615 | 0.8333 | **0.8929** | 30 |
| `tell_joke` | 1.0000 | 1.0000 | **1.0000** | 30 |
| `thank_you` | 0.9333 | 0.9333 | **0.9333** | 30 |
| `transfer` | 1.0000 | 0.9333 | **0.9655** | 30 |
| `travel_suggestion` | 0.8235 | 0.9333 | **0.8750** | 30 |
| `weather` | 0.9677 | 1.0000 | **0.9836** | 30 |
| `what_can_i_ask_you` | 0.9091 | 1.0000 | **0.9524** | 30 |

Generated analytical plots are stored in `reports/figures/`:
* `reports/figures/confusion_matrix.png`
* `reports/figures/training_history.png`
* `reports/figures/intent_distribution.png`
* `reports/figures/sequence_length_distribution.png`

---

## 📁 Project Directory Structure

```
intentclassifier/
│
├── app.py                     # Streamlit interactive web application
├── train.py                   # Model training script with callbacks & checkpoints
├── evaluate.py                # Test set evaluation & confusion matrix generator
├── config.py                  # Global hyperparameters, paths, and configurations
├── check_deployment_readiness.py # Pre-flight deployment audit script
├── requirements.txt           # Production environment dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Version control exclusions
│
├── data/
│   ├── raw/
│   │   └── clinc150_full.json # Raw benchmark dataset
│   └── processed/
│       ├── train.csv          # 1,800 training samples
│       ├── val.csv            # 360 validation samples
│       └── test.csv           # 540 untouched test samples
│
├── models/
│   ├── chatbot_model.keras    # Trained BiLSTM Keras weights (3.25 MB)
│   ├── tokenizer.pkl          # Fitted Keras Tokenizer
│   ├── label_encoder.pkl      # Scikit-learn LabelEncoder
│   └── metadata.pkl           # Preprocessing vocabulary & class metadata
│
├── src/
│   ├── __init__.py
│   ├── download_data.py       # CLINC150 dataset fetcher
│   ├── analyze_dataset.py     # EDA and CSV split generator
│   ├── preprocessing.py       # Text cleaner, sequence padder & preprocessor
│   ├── model.py               # BiLSTM network architecture definition
│   ├── inference.py           # Real-time inference service & confidence gating
│   ├── speech.py              # Speech recognition handler
│   └── responses.py           # Categorized intent response dictionary
│
├── notebooks/
│   └── dataset_analysis.ipynb # Interactive Exploratory Data Analysis notebook
│
├── tests/
│   ├── __init__.py
│   ├── run_all_tests.py       # Test suite runner
│   ├── test_preprocessing.py  # Unit tests for text cleaning and tokenization
│   ├── test_model.py          # Unit tests for BiLSTM tensor dimensions
│   ├── test_inference.py      # Unit tests for intent prediction & gating
│   ├── test_speech.py         # Unit tests for audio buffer decoding
│   └── benchmark_test.py      # 20-utterance end-to-end benchmark
│
└── reports/
    ├── project_report.md      # Formal academic laboratory report
    ├── demo_examples.md       # Spoken demonstration scenarios
    ├── training_metrics.json  # Exported epoch metrics
    ├── evaluation_results.json# Detailed test evaluation metrics
    └── figures/
        ├── intent_distribution.png
        ├── sequence_length_distribution.png
        ├── split_distribution.png
        ├── training_history.png
        └── confusion_matrix.png
```

---

## 🚀 How to Run Locally

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/gopiadithya/Intent_classifier.git
cd Intent_classifier

# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pre-flight Audit & Unit Tests
```bash
# Verify environment readiness
python check_deployment_readiness.py

# Run all 12 unit tests
python tests/run_all_tests.py

# Run 20-utterance benchmark
python tests/benchmark_test.py
```

### 4. Retrain the BiLSTM Model (Optional)
The pre-trained model weights are already provided in `models/chatbot_model.keras`. If you wish to retrain from scratch:
```bash
# Extract and analyze dataset splits
python src/analyze_dataset.py

# Train BiLSTM network
python train.py

# Evaluate against untouched test set
python evaluate.py
```

### 5. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🌐 Public Deployment (Streamlit Community Cloud)

This repository is pre-configured and audited for zero-configuration deployment:

1. Push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Click **"New app"**.
4. Select your repository, branch (`main`), and set Main file path to `app.py`.
5. Click **"Deploy!"**.
6. Streamlit Community Cloud will automatically build the environment from `requirements.txt` and launch the app with a public shareable URL.

---

## ⚠️ Limitations & Future Improvements

### Limitations
1. **Domain Boundary**: The system operates within 18 closed-set intents. Out-of-domain queries rely on confidence threshold rejection rather than open-ended dialogue generation.
2. **Acoustic Noise**: In loud ambient environments, speech-to-text transcription accuracy may degrade before tokens reach the BiLSTM network.
3. **Intent Overlap**: Subtle semantic overlaps exist between related intents (e.g., `restaurant_suggestion` vs. `travel_suggestion` or `reminder` vs. `alarm`).

### Future Improvements
1. **Attention Mechanism**: Incorporate a Bahdanau-style self-attention layer on top of the BiLSTM to highlight key trigger words.
2. **Contextual History**: Maintain multi-turn dialogue state tracking to handle follow-up queries.
3. **Pre-trained Word Embeddings**: Integrate GloVe or FastText embeddings to enhance generalization over rare and out-of-vocabulary words.
4. **On-Device Offline STT**: Support edge-based offline speech engines (such as Vosk or Whisper-small) for privacy-sensitive deployments.

---

## 📜 Dataset Citation & Attribution

```bibtex
@inproceedings{larson-etal-2019-evaluation,
    title = "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction",
    author = "Larson, Stefan and Mahendran, Anish and Peper, Joseph J. and Clarke, Christopher and Lee, Andrew and Hill, Parker and Kummerfeld, Jonathan K. and Leach, Kevin and Laurenzano, Michael A. and Tang, Lingjia and Mars, Jason",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    year = "2019",
    pages = "1311--1316"
}
```
* **Dataset**: CLINC150 (OOS-Eval) by CLINC & University of Michigan.
* **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
