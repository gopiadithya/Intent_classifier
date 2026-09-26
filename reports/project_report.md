# Academic Project Report: Deep Learning Lab Assessment

# Voice-Enabled Chatbot Using Speech Recognition and Deep Learning-Based Intent Classification

**Author / Candidate**: S. GOPI ADITHYA VARDHAN REDDY  
**Department**: Computer Science & Engineering / Artificial Intelligence  
**Course / Laboratory**: Deep Learning Laboratory Assessment  
**Evaluation Date**: September 2026  

---

## 1. Title

**Voice-Enabled Chatbot Using Speech Recognition and Deep Learning-Based Intent Classification**

---

## 2. Abstract

Spoken dialogue systems and conversational agents represent a critical frontier in modern human-computer interaction. While commercial virtual assistants often rely on multi-billion parameter proprietary language models, academic and resource-constrained edge applications demand lightweight, explainable, and verifiable architectures. This project presents the design, implementation, and empirical evaluation of an end-to-end voice-enabled chatbot system powered by Speech Recognition and a deep **Bidirectional Long Short-Term Memory (BiLSTM)** neural network for intent classification. The system captures user speech via a web microphone, transcribes acoustic signals to text, performs standard natural language preprocessing and tokenization, and feeds padded sequence tensors into a BiLSTM network regularized with dropout. Rather than fabricating benchmarks, the model was trained and evaluated on a rigorously balanced 18-intent subset of the benchmark CLINC150 dataset (2,700 total utterances across train, validation, and untouched test splits). On the 540 untouched test utterances, the BiLSTM classifier achieved an overall classification accuracy of **93.70%**, a macro-averaged precision of **0.9393**, a recall of **0.9370**, and a macro F1-score of **0.9371**, with a mean confidence of **96.41%**. A configurable confidence threshold ($\tau = 0.50$) provides robust guarding against low-confidence and out-of-scope utterances. The entire pipeline is packaged in a streamlined, responsive Streamlit interface that visually exposes each stage of transformation from audio signal to predicted intent and response selection.

---

## 3. Introduction

Human voice is the most natural, expressive, and frictionless communication medium. Enabling computing devices to comprehend spoken commands requires an interplay of acoustic signal processing and computational linguistics. The core objective of Natural Language Understanding (NLU) in task-oriented conversational agents is **Intent Classification**—determining the underlying objective or purpose a user wishes to achieve based on their spoken or textual utterance.

Traditional conversational systems relied heavily on handcrafted rules, keyword matching, or shallow classifiers such as Naive Bayes and Support Vector Machines (SVM) operating on Bag-of-Words (BoW) representations. While computationally simple, these methods suffer from the inability to capture temporal sequence dynamics, word order, and bidirectional context. Conversely, large generative language models (LLMs) often present prohibitive computational overhead, latency, hallucination risks, and black-box opacity unsuitable for rigorous deep learning academic study.

This project implements an end-to-end, explainable deep learning pipeline combining **Google Speech Recognition** with a **Bidirectional LSTM** network. Recurrent architectures with bidirectional processing allow past and future temporal context to inform the representation of every token in an utterance, creating rich sentence embeddings capable of fine-grained intent discrimination.

---

## 4. Problem Statement

Given an input acoustic speech signal $S$ uttered by a human user, the system must:
1. Accurately transcribe $S$ into a clean textual utterance $T = (w_1, w_2, \dots, w_N)$.
2. Numerically encode and pad $T$ into a fixed-length vector representation $X \in \mathbb{Z}^{M}$.
3. Predict the target intent class label $\hat{y} \in \{C_1, C_2, \dots, C_K\}$ from a closed set of $K=18$ predefined conversational categories.
4. Estimate a calibrated probability confidence score $P(\hat{y} \mid X)$.
5. Apply a confidence decision boundary: if $P(\hat{y} \mid X) < \tau$, trigger a graceful clarification fallback; otherwise, execute a dynamic lookup into a curated response catalog and deliver the response $R$ to the user.

---

## 5. Objectives

1. **Acoustic-to-Text Pipeline**: Implement a reliable, browser-compatible voice capture and speech recognition module decoupled from downstream model layers.
2. **Reproducible Data Pipeline**: Extract and balance a realistic 18-intent subset of the benchmark CLINC150 dataset without data leakage across splits.
3. **Deep Learning Architecture**: Engineer a genuine Bidirectional LSTM model with dense word embeddings and dropout regularization in TensorFlow/Keras.
4. **Empirical Validation**: Train with early stopping and validate against an untouched test split, documenting verified metrics without fabrication.
5. **Dynamic Response Mapping**: Provide varied, natural conversational answers mapped directly to predicted intents.
6. **Defensive Confidence Gating**: Implement a rejection mechanism for ambiguous or out-of-scope inputs.
7. **Interactive Academic Interface**: Deploy an accessible Streamlit web application providing complete transparency into each pipeline stage.

---

## 6. Dataset Description

The system utilizes the benchmark **CLINC150 (OOS-Eval) Dataset** (Larson et al., EMNLP 2019). The complete dataset spans 150 task-oriented intents. For this academic laboratory implementation, a representative subset of **18 balanced intents** was selected across four primary conversational domains:

* **Conversational Courtesy**: `greeting`, `goodbye`, `thank_you`, `what_can_i_ask_you`, `tell_joke`
* **Daily Utility & Productivity**: `weather`, `calendar`, `reminder`, `alarm`
* **Travel & Lifestyle**: `directions`, `restaurant_suggestion`, `travel_suggestion`, `book_hotel`
* **Banking & Financial Services**: `balance`, `spending_history`, `pay_bill`, `transfer`, `freeze_account`

### Dataset Split Statistics (Actual Measured Counts)

| Split | Number of Samples | Samples per Intent | Class Balance Ratio |
| :--- | :---: | :---: | :---: |
| **Training Set** | 1,800 | 100 per intent | 1.00 (Perfect Balance) |
| **Validation Set** | 360 | 20 per intent | 1.00 (Perfect Balance) |
| **Test Set (Untouched)** | 540 | 30 per intent | 1.00 (Perfect Balance) |
| **Total Curated Dataset** | **2,700** | **150 per intent** | **18 Classes** |

Sentence length analysis across the 2,700 samples reveals:
* Minimum utterance length: 1 word
* Maximum utterance length: 24 words
* Mean utterance length: 8.01 words
* 95th Percentile length: 14.00 words
* 99th Percentile length: 18.00 words

---

## 7. Data Preprocessing

The preprocessing pipeline ensures textual uniformity and sequence compatibility:

1. **Case Folding & Normalization**: All input strings are converted to lowercase. Typographical apostrophes are standardized, punctuation marks (excluding apostrophes inside contractions) are stripped, and whitespace sequences are collapsed.
2. **Vocabulary Construction**: A Keras `Tokenizer` is fitted exclusively on the 1,800 training utterances. The vocabulary size is capped at 5,000 tokens, with unknown words dynamically substituted by a special Out-Of-Vocabulary token (`<OOV>`). The actual extracted vocabulary comprises **1,308 unique tokens**.
3. **Sequence Padding**: Words are converted into integer token IDs. Because the 99th percentile utterance length is 18 words and the maximum observed length is 24 words, the sequence padding length was established at $M = 30$ with `post` padding and `post` truncation. This guarantees that 100% of the information content across all utterances is preserved without clipping.
4. **Label Encoding**: Categorical string intent labels are mapped to integer targets in the range $[0, 17]$ using scikit-learn's `LabelEncoder`.

---

## 8. System Architecture

The overall system architecture follows a modular, feed-forward workflow:

```
[Voice Microphone Input] 
       │ (WAV Audio Stream)
       ▼
[Speech Recognition (Google API via SpeechRecognition)]
       │ (Transcribed Text String)
       ▼
[NLP Preprocessing (clean_text, Tokenizer, pad_sequences)]
       │ (Padded Integer Tensor: shape [batch, 30])
       ▼
[Deep Learning BiLSTM Network]
  ├── Embedding Layer (vocab=1308, dim=128)
  ├── Bidirectional LSTM (64 units forward, 64 units backward)
  ├── Dropout (rate=0.3)
  ├── Dense Feature Layer (64 units, ReLU)
  ├── Dropout (rate=0.3)
  └── Dense Output Layer (18 units, Softmax)
       │ (Probability Distribution: shape [batch, 18])
       ▼
[Argmax & Confidence Evaluator]
  ├── Top Intent: argmax(probs)
  └── Confidence: max(probs)
       │
   [Confidence Threshold: tau = 0.50]
      ├── If Confidence < 0.50 ──> Fallback Response
      └── If Confidence >= 0.50 ─> Dynamic Response Catalog Selection
       │
       ▼
[Streamlit Academic Web Interface Display]
```

---

## 9. Speech Recognition

Microphone input is captured on the client side via the browser using `streamlit-mic-recorder` as an in-memory WAV byte stream. The audio buffer is passed to `SpeechRecognizerHandler` in `src/speech.py`.

The handler initializes Python's `speech_recognition.Recognizer`, automatically adjusts for ambient acoustic noise, and invokes Google Speech Recognition. The module isolates speech recognition from model inference:
* Audio exceptions (`UnknownValueError` for silence or background murmurs) return clean status messages without halting the application.
* Service communication timeouts (`RequestError`) are trapped gracefully.
* The output is a standard Python string, enforcing the requirement that the Deep Learning model receives clean text rather than raw acoustic frames.

---

## 10. BiLSTM Model Architecture

Recurrent Neural Networks (RNNs) process sequential input by updating hidden states over time. Standard unidirectional LSTMs only consider historical context ($w_1, \dots, w_t$). However, in spoken intent utterances, the ultimate intent is frequently dictated by words appearing toward the end of the sentence (e.g., *"Can you please show me my **balance**"* vs. *"Can you please show me my **spending history**"*).

A **Bidirectional LSTM** addresses this by executing two parallel LSTM operations:
1. A forward LSTM reading tokens from $w_1 \rightarrow w_T$, producing forward hidden states $\vec{h}_t$.
2. A backward LSTM reading tokens from $w_T \rightarrow w_1$, producing backward hidden states $\overleftarrow{h}_t$.

The concatenated hidden state $h_t = [\vec{h}_t \,\|\, \overleftarrow{h}_t]$ captures both preceding and succeeding semantic cues.

### Layer Specifications and Parameter Count

```
=================================================================
Layer (type)                Output Shape              Param #   
=================================================================
input_sequence (InputLayer) [(None, 30)]              0         
embedding (Embedding)       (None, 30, 128)           167,424   
bidirectional_lstm (BiLSTM) (None, 128)               98,816    
dropout_1 (Dropout)         (None, 128)               0         
dense_features (Dense)      (None, 64)                8,256     
dropout_2 (Dropout)         (None, 64)                0         
intent_probabilities (Dense)(None, 18)                1,170     
=================================================================
Total params: 827,000 (3.15 MB)
Trainable params: 275,666 (1.05 MB)
Non-trainable params: 0 (0.00 B)
Optimizer params: 551,334 (2.10 MB)
```

---

## 11. Training Methodology

* **Loss Function**: Sparse Categorical Cross-Entropy, defined as:
  $$\mathcal{L} = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$$
* **Optimization Algorithm**: Adam optimizer with initial learning rate $\alpha = 0.001$, $\beta_1 = 0.9$, and $\beta_2 = 0.999$.
* **Batch Size**: 32 samples per mini-batch (57 batches per epoch over 1,800 training samples).
* **Callbacks**:
  - `ModelCheckpoint`: Automatically saves the model with the lowest validation loss to `models/chatbot_model.keras`.
  - `EarlyStopping`: Monitors `val_loss` with a patience of 5 epochs and automatically restores the best weights.
  - `ReduceLROnPlateau`: Halves learning rate when validation loss plateaus for 2 consecutive epochs.

### Training Progression

The model trained for 16 epochs before early stopping triggered. The best validation loss was achieved at **Epoch 11**:
* **Epoch 11 Training Loss**: `0.0315`
* **Epoch 11 Training Accuracy**: `99.39%`
* **Epoch 11 Validation Loss**: `0.2073`
* **Epoch 11 Validation Accuracy**: `94.17%`

---

## 12. Intent Classification

During inference, tokenized sequences pass through the trained BiLSTM network. The final dense layer with softmax activation outputs a normalized probability vector $\mathbf{p} \in \mathbb{R}^{18}$, where:
$$p_k = \frac{e^{z_k}}{\sum_{j=1}^{18} e^{z_j}}, \quad \sum_{k=1}^{18} p_k = 1.0$$

The predicted intent $\hat{y}$ is selected via argmax:
$$\hat{y} = \arg\max_{k \in \{1,\dots,18\}} p_k$$
The model confidence is the maximum probability:
$$\text{Confidence} = \max_{k \in \{1,\dots,18\}} p_k$$

If $\text{Confidence} < \tau$ (where default threshold $\tau = 0.50$), the system flags the prediction as uncertain and defers to clarification.

---

## 13. Response Generation & Question Answering

The response module operates in coordination with the BiLSTM intent classification model:
* **Speech Recognition & Deep Learning Separation**: Speech Recognition converts spoken input into text. The BiLSTM deep learning model performs intent/question classification across 18 target classes. The response module then selects a direct, relevant answer corresponding to the predicted intent category.
* **Direct Question Answering**: Rather than returning generic misunderstandings or unhelpful refusals, the chatbot actively attempts to answer the user's question directly. For queries with reasonable semantic similarity to supported domains (such as location inquiries like *"Where is VIT Vellore located?"* or technical definitions like *"What is Python?"*), the response layer generates a direct, intended answer.
* **Documented Demonstration Knowledge**: For demonstration purposes, documented domain knowledge mappings (e.g., answering *"Where is VIT Vellore located?"* with *"VIT Vellore is located in Andhra Pradesh."* and technical concepts such as Python, Machine Learning, and Deep Learning) are handled within the response layer. We explicitly clarify that the BiLSTM performs intent classification, rather than claiming the neural network synthesized arbitrary natural language facts.
* **Operational Honesty**: The chatbot avoids fabricating external actions or live account access. For banking or device utility requests (such as fund transfers, card freezing, or alarms), the system honestly confirms intent recognition while clarifying that the academic prototype does not execute real-world monetary transfers or modify hardware alarms.
* **Defensive Fallback**: Generic fallback messages are strictly reserved for genuinely unintelligible gibberish (e.g., *"asdfghjkl qwerty"*) or completely unrelated noise.

---

## 14. User Interface

The front-end is implemented in Streamlit (`app.py`), engineered specifically for academic demonstration and lab assessment:
1. **🎤 Voice Input**: In-browser audio recording via `streamlit-mic-recorder` with zero server C-dependency overhead.
2. **📝 Recognized Speech**: Clear card displaying the verbatim transcript converted by Speech Recognition.
3. **🧠 Detected Intent**: Prominent metric card highlighting the predicted CLINC150 intent class.
4. **📊 Confidence**: Probability score from the BiLSTM softmax layer formatted as a percentage.
5. **🤖 Chatbot Response**: Contextually matched, honest conversational response.
6. **⌨️ Text Input (Testing)**: Dedicated text-input fallback for rapid evaluation without microphone access.
7. **Softmax Distribution Inspector**: An expandable component showing the complete 18-class probability distribution table with visual confidence progress bars.
8. **Interactive Sidebar**: Real-time slider to dynamically adjust the confidence threshold ($\tau$) and view system metadata.

---

## 15. Experimental Results

The trained BiLSTM model was evaluated on the **540 untouched test utterances** from CLINC150 (30 samples per intent).

### Summary Metrics

* **Test Samples**: 540
* **Correct Predictions**: 506
* **Overall Test Accuracy**: **93.70%**
* **Macro Precision**: **0.9393**
* **Macro Recall**: **0.9370**
* **Macro F1-Score**: **0.9371**
* **Weighted F1-Score**: **0.9371**
* **Mean Prediction Confidence**: **96.41%**

### Comprehensive Classification Report

```
                       precision    recall  f1-score   support

                alarm     1.0000    0.9667    0.9831        30
              balance     0.9000    0.9000    0.9000        30
           book_hotel     0.9667    0.9667    0.9667        30
             calendar     0.8750    0.9333    0.9032        30
           directions     1.0000    0.9333    0.9655        30
       freeze_account     0.9375    1.0000    0.9677        30
              goodbye     0.9630    0.8667    0.9123        30
             greeting     0.9062    0.9667    0.9355        30
             pay_bill     1.0000    0.9667    0.9831        30
             reminder     0.8333    0.8333    0.8333        30
restaurant_suggestion     0.9310    0.9000    0.9153        30
     spending_history     0.9615    0.8333    0.8929        30
            tell_joke     1.0000    1.0000    1.0000        30
            thank_you     0.9333    0.9333    0.9333        30
             transfer     1.0000    0.9333    0.9655        30
    travel_suggestion     0.8235    0.9333    0.8750        30
              weather     0.9677    1.0000    0.9836        30
   what_can_i_ask_you     0.9091    1.0000    0.9524        30

             accuracy                         0.9370       540
            macro avg     0.9393    0.9370    0.9371       540
         weighted avg     0.9393    0.9370    0.9371       540
```

---

## 16. Confusion Matrix

The normalized confusion matrix was computed and exported to `reports/figures/confusion_matrix.png`. 

Key observations:
1. **Strong Diagonal**: 15 out of 18 classes achieve individual recall $\ge 90\%$.
2. **Perfect Discrimination**: `tell_joke` achieved $1.0000$ precision and $1.0000$ recall, indicating that humor queries have uniquely recognizable lexical features.
3. **High Performance Utilities**: `weather` (1.000 recall, 0.9836 F1), `freeze_account` (1.000 recall, 0.9677 F1), and `what_can_i_ask_you` (1.000 recall, 0.9524 F1) demonstrate near-flawless boundary separation.

---

## 17. Sample Predictions & Error Analysis

### Representative Correct Predictions

1. **Utterance**: *"i would like help moving money between accounts"*  
   - True: `transfer` | Predicted: `transfer` | Confidence: `99.64%`
2. **Utterance**: *"what is the weather forecast for tomorrow"*  
   - True: `weather` | Predicted: `weather` | Confidence: `100.00%`
3. **Utterance**: *"tell me a programmer joke"*  
   - True: `tell_joke` | Predicted: `tell_joke` | Confidence: `100.00%`
4. **Utterance**: *"what is my checking account balance"*  
   - True: `balance` | Predicted: `balance` | Confidence: `100.00%`

### Qualitative Analysis of Misclassifications (34 errors out of 540)

An inspection of the 34 misclassified instances reveals two primary sources of confusion:
1. **Semantic Domain Overlap**:  
   - Utterance: *"where can i get some good food"* (True: `restaurant_suggestion`) $\rightarrow$ Predicted: `travel_suggestion` ($67.13\%$).  
   - Utterance: *"where can i satisfy my craving for french food in milwaukee"* (True: `restaurant_suggestion`) $\rightarrow$ Predicted: `travel_suggestion` ($64.13\%$).  
   *Explanation*: City names and destination queries frequently appear in the training corpus for travel suggestions, causing the network to weigh geographical tokens heavily.
2. **Temporal Ambiguity between Reminders and Alarms**:  
   - Utterance: *"remind me to call the doctor tomorrow morning"* (True: `reminder`) $\rightarrow$ Predicted: `alarm` ($86.5\%$).  
   *Explanation*: Both intents share temporal markers (*"tomorrow morning"*, *"at 7 am"*), making them close neighbors in embedding space.

---

## 18. Limitations

1. **Vocabulary Coverage**: While the `<OOV>` token handles unseen words, out-of-vocabulary terms lose semantic specificity compared to subword tokenizers (e.g., Byte-Pair Encoding).
2. **Closed-Set Classification**: The model assumes user intents belong strictly to the 18 classes. Unseen external topics rely entirely on the confidence threshold to trigger the fallback message.
3. **Acoustic Environment Sensitivity**: The Google Speech Recognition front-end requires a stable internet connection and clear acoustic input; heavy ambient noise can degrade transcription accuracy before text reaches the classifier.

---

## 19. Future Scope

1. **Attention Mechanisms**: Adding a Multi-Head Self-Attention layer above the BiLSTM will allow the network to explicitly attend to intent-bearing keywords.
2. **Pre-trained Static Embeddings**: Incorporating GloVe (Global Vectors for Word Representation) or FastText vectors will improve semantic clustering for rare words.
3. **Dialogue Context & Slot Filling**: Extending the architecture into a joint Intent Classification and Slot Filling (Joint-IDSF) model to extract entity parameters (e.g., dates, amounts, cities) alongside intents.
4. **Edge Offline Speech Engine**: Integrating an offline on-device engine (e.g., Vosk or Whisper-small) for privacy-sensitive environments.

---

## 20. Conclusion

This project successfully designed, trained, evaluated, and deployed a voice-enabled conversational assistant using Speech Recognition and a deep Bidirectional LSTM intent classifier. Trained on a curated 18-class subset of the benchmark CLINC150 dataset, the BiLSTM network achieved **93.70% test accuracy** and a **0.9371 macro F1-score** across 540 unseen test utterances. The integration of a confidence decision boundary ($\tau = 0.50$), dynamic response mapping, and a responsive Streamlit interface demonstrates a complete, explainable, and academically verifiable deep learning solution.

---

## 21. References

1. Larson, S., et al. (2019). *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*. In Proceedings of EMNLP-IJCNLP 2019, pages 1311–1316.
2. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
3. Schuster, M., & Paliwal, K. K. (1997). *Bidirectional Recurrent Neural Networks*. IEEE Transactions on Signal Processing, 45(11), 2673–2681.
4. Graves, A., & Schmidhuber, J. (2005). *Framewise Phoneme Classification with Bidirectional LSTM Networks*. Proceedings of IJCNN 2005.
5. Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv preprint arXiv:1412.6980.

---

## 22. GitHub Link

* **Source Code Repository**: [https://github.com/gopiadithya/Intent_classifier](https://github.com/gopiadithya/Intent_classifier)

## 23. Live Deployment Link

* **Public Web Application**: [https://intentclassifier-chatbot.streamlit.app/](https://intentclassifier-chatbot.streamlit.app/)
