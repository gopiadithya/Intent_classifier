"""
Automated Academic PDF Generator for Voice-Enabled Chatbot Project.
Generates publication-quality, submission-ready PDF reports with embedded
high-resolution figures, verified benchmark tables, and running headers/footers.
"""
import sys
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
import pypdf

# Project Base Directory
BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Candidate & Project Metadata
AUTHOR_NAME = "S. GOPI ADITHYA VARDHAN REDDY"
PROJECT_TITLE = "Voice-Enabled Chatbot Using Speech Recognition and Deep Learning-Based Intent Classification"
DEPLOYED_URL = "https://intentclassifier-chatbot.streamlit.app/"
GITHUB_URL = "https://github.com/gopiadithya/Intent_classifier"
EVALUATION_DATE = "September 2026"

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print 'Page X of Y' footers
    and running academic headers across all pages except the cover page.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # Do not print headers or footers on the cover page (Page 1)
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Helvetica", 8.5)
        self.setFillColor(colors.HexColor("#475569"))

        # Running Header
        header_text = "Voice-Enabled Chatbot | Deep Learning Lab Assessment"
        self.drawString(40, A4[1] - 32, header_text)
        self.drawRightString(A4[0] - 40, A4[1] - 32, f"Candidate: {AUTHOR_NAME}")

        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(40, A4[1] - 36, A4[0] - 40, A4[1] - 36)

        # Running Footer
        self.line(40, 42, A4[0] - 40, 42)
        footer_left = f"Live App: {DEPLOYED_URL}"
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawString(40, 30, footer_left)
        self.drawRightString(A4[0] - 40, 30, page_str)

        self.restoreState()


def build_pdf_report(output_filename: Path):
    print(f"Generating Comprehensive PDF Report at: {output_filename}")
    
    # 0.55 in margins (approx 40 pt)
    doc = SimpleDocTemplate(
        str(output_filename),
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=46,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Custom Typography Hierarchy
    c_primary = colors.HexColor("#1E3A8A")     # Navy
    c_secondary = colors.HexColor("#2563EB")   # Blue
    c_dark = colors.HexColor("#0F172A")        # Dark slate
    c_muted = colors.HexColor("#475569")       # Muted slate

    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=c_primary,
        alignment=1, # Center
        spaceAfter=15
    )

    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12.5,
        leading=17,
        textColor=c_muted,
        alignment=1,
        spaceAfter=25
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=c_primary,
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=c_secondary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=c_dark,
        spaceAfter=7
    )

    body_bold = ParagraphStyle(
        'Body_Bold_Custom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=6
    )

    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=c_muted,
        alignment=1,
        spaceBefore=4,
        spaceAfter=10
    )

    story = []

    # =========================================================
    # 1. FORMAL COVER PAGE
    # =========================================================
    story.append(Spacer(1, 35))
    story.append(Paragraph("ACADEMIC PROJECT REPORT", ParagraphStyle('CoverPre', fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=c_secondary, alignment=1, spaceAfter=12)))
    story.append(Paragraph(PROJECT_TITLE, title_style))
    story.append(Paragraph("A Deep Learning and Speech Recognition System for Voice-Enabled Conversational Intent Classification", subtitle_style))
    story.append(HRFlowable(width="80%", thickness=1.5, color=c_secondary, spaceBefore=5, spaceAfter=25))

    # Candidate & Assessment Box
    meta_data = [
        [Paragraph("<b>Candidate Name:</b>", body_style), Paragraph(f"<b>{AUTHOR_NAME}</b>", body_bold)],
        [Paragraph("<b>Degree / Course:</b>", body_style), Paragraph("Computer Science & Engineering / Artificial Intelligence", body_style)],
        [Paragraph("<b>Laboratory Assessment:</b>", body_style), Paragraph("Deep Learning Laboratory (Academic Capstone)", body_style)],
        [Paragraph("<b>Model Architecture:</b>", body_style), Paragraph("Bidirectional Long Short-Term Memory (BiLSTM)", body_style)],
        [Paragraph("<b>Benchmark Dataset:</b>", body_style), Paragraph("CLINC150 (18 Balanced Intent Classes, 2,700 samples)", body_style)],
        [Paragraph("<b>Empirical Test Accuracy:</b>", body_style), Paragraph("<b>93.70%</b> (506/540 correct, Macro F1: 0.9371)", body_bold)],
        [Paragraph("<b>Speech Recognition:</b>", body_style), Paragraph("Google Speech API / Web Speech Recognition", body_style)],
        [Paragraph("<b>Deployment Platform:</b>", body_style), Paragraph("Streamlit Community Cloud (Live Online)", body_style)],
        [Paragraph("<b>Live Application URL:</b>", body_style), Paragraph(f"<font color='#2563EB'><u>{DEPLOYED_URL}</u></font>", body_style)],
        [Paragraph("<b>GitHub Repository:</b>", body_style), Paragraph(f"<font color='#2563EB'><u>{GITHUB_URL}</u></font>", body_style)],
        [Paragraph("<b>Date of Evaluation:</b>", body_style), Paragraph(EVALUATION_DATE, body_style)],
    ]

    t_meta = Table(meta_data, colWidths=[150, 360])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t_meta)
    
    story.append(Spacer(1, 35))
    story.append(Paragraph("<b>Submitted in partial fulfillment of the requirements for the Deep Learning Laboratory Assessment</b>", caption_style))
    story.append(PageBreak())

    # =========================================================
    # 2. ABSTRACT & INTRODUCTION
    # =========================================================
    story.append(Paragraph("1. Abstract", h1_style))
    abstract_text = (
        "Spoken dialogue systems and conversational interfaces represent an indispensable paradigm in modern human-computer interaction. "
        "While commercial digital assistants often deploy multi-billion parameter proprietary language models, academic and resource-constrained "
        "applications require transparent, lightweight, and explainable neural network architectures. This project implements, empirically evaluates, "
        "and deploys an end-to-end voice-enabled chatbot system using Speech Recognition and a deep <b>Bidirectional Long Short-Term Memory (BiLSTM)</b> "
        "network. The system captures spoken acoustic signals in real-time, transcribes audio to normalized text, performs sequence tokenization and padding, "
        "and classifies user utterances across a curated, balanced 18-intent benchmark extracted from the CLINC150 dataset (2,700 total samples). "
        "On 540 completely untouched test utterances, the BiLSTM classifier achieved an overall classification accuracy of <b>93.70%</b>, a macro-averaged "
        "precision of <b>0.9393</b>, a recall of <b>0.9370</b>, and a macro F1-score of <b>0.9371</b> with a mean prediction confidence of <b>96.41%</b>. "
        "The response layer generates direct, conversational answers mapped to predicted intents and incorporates a documented demonstration knowledge "
        "base for educational queries (e.g. VIT Vellore location and Python/AI concepts). The application is fully deployed online on Streamlit Community Cloud."
    )
    story.append(Paragraph(abstract_text, body_style))

    story.append(Paragraph("2. Introduction & Background", h1_style))
    intro_text = (
        "Human voice provides the most natural and frictionless communication medium. The core objective of Natural Language Understanding (NLU) "
        "in conversational systems is <b>Intent Classification</b>—determining the goal or intent behind a user's utterance. "
        "Traditional approaches relied on keyword matching or shallow models like Naive Bayes and SVMs on Bag-of-Words, which completely ignore word order, "
        "grammar, and temporal context. Conversely, large generative language models (LLMs) present severe computational overhead, black-box opacity, and "
        "hallucination risks. In contrast, recurrent neural networks with bidirectional memory cells (BiLSTM) capture past and future context simultaneously, "
        "enabling rich sentence representations suitable for real-time edge and web deployment."
    )
    story.append(Paragraph(intro_text, body_style))

    story.append(Paragraph("3. Problem Statement & System Objectives", h1_style))
    obj_text = (
        "<b>Problem Statement:</b> Given an acoustic voice signal <i>S</i> from a user, the system must: (1) transcribe <i>S</i> into clean text <i>T</i>, "
        "(2) encode <i>T</i> into numerical sequence vectors <i>X</i>, (3) classify <i>X</i> into one of <i>K=18</i> target intent categories using a BiLSTM model, "
        "(4) estimate a softmax confidence score, and (5) generate a relevant, conversational answer corresponding to the predicted intent.<br/><br/>"
        "<b>Core Objectives:</b><br/>"
        "• Build a robust browser-based voice capture module with silence auto-submission.<br/>"
        "• Curate a 100% balanced 18-class dataset from benchmark CLINC150 with zero test-split leakage.<br/>"
        "• Design, train, and validate a genuine Bidirectional LSTM model with early stopping.<br/>"
        "• Maintain absolute architectural honesty: BiLSTM performs intent classification; the response module maps intents to answers.<br/>"
        "• Deploy the complete application publicly on Streamlit Community Cloud."
    )
    story.append(Paragraph(obj_text, body_style))

    # =========================================================
    # 3. DATASET DESCRIPTION & PREPROCESSING
    # =========================================================
    story.append(Paragraph("4. Benchmark Dataset: CLINC150 (18 Balanced Classes)", h1_style))
    dataset_text = (
        "The model is trained on a curated subset of the benchmark <b>CLINC150 (OOS-Eval)</b> dataset (Larson et al., EMNLP 2019). "
        "We selected 18 practical intent classes spanning four realistic conversational domains:<br/>"
        "• <b>Conversational Courtesy</b>: <code>greeting</code>, <code>goodbye</code>, <code>thank_you</code>, <code>what_can_i_ask_you</code>, <code>tell_joke</code><br/>"
        "• <b>Daily Utilities</b>: <code>weather</code>, <code>calendar</code>, <code>reminder</code>, <code>alarm</code><br/>"
        "• <b>Travel & Lifestyle</b>: <code>directions</code>, <code>restaurant_suggestion</code>, <code>travel_suggestion</code>, <code>book_hotel</code><br/>"
        "• <b>Banking Services</b>: <code>balance</code>, <code>spending_history</code>, <code>pay_bill</code>, <code>transfer</code>, <code>freeze_account</code>"
    )
    story.append(Paragraph(dataset_text, body_style))

    # Dataset table
    ds_table_data = [
        [Paragraph("<b>Dataset Split</b>", body_bold), Paragraph("<b>Total Samples</b>", body_bold), Paragraph("<b>Samples per Intent</b>", body_bold), Paragraph("<b>Class Balance Ratio</b>", body_bold)],
        [Paragraph("Training Split", body_style), Paragraph("1,800", body_style), Paragraph("100 per intent", body_style), Paragraph("1.00 (100% Balanced)", body_style)],
        [Paragraph("Validation Split", body_style), Paragraph("360", body_style), Paragraph("20 per intent", body_style), Paragraph("1.00 (100% Balanced)", body_style)],
        [Paragraph("Test Split (Untouched)", body_style), Paragraph("540", body_style), Paragraph("30 per intent", body_style), Paragraph("1.00 (100% Balanced)", body_style)],
        [Paragraph("<b>Total Curated Corpus</b>", body_bold), Paragraph("<b>2,700</b>", body_bold), Paragraph("<b>150 per intent</b>", body_bold), Paragraph("<b>18 Target Classes</b>", body_bold)],
    ]
    t_ds = Table(ds_table_data, colWidths=[130, 110, 130, 140])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_ds)

    story.append(Spacer(1, 10))
    story.append(Paragraph("5. NLP Preprocessing & Sequence Engineering", h1_style))
    prep_text = (
        "Raw speech transcripts undergo rigorous preprocessing:<br/>"
        "1. <b>Text Cleaning</b>: Case folding to lowercase, contraction normalization (e.g. <code>’</code> to <code>'</code>), "
        "and non-alphanumeric punctuation removal while preserving intra-word apostrophes.<br/>"
        "2. <b>Vocabulary Tokenization</b>: Fit strictly on the training split with 1,307 active unique tokens and an <code>&lt;OOV&gt;</code> token.<br/>"
        "3. <b>Sequence Padding</b>: Sentence length analysis revealed a mean of 8.01 words and a 99th percentile of 18 words. "
        "We configured a maximum sequence length of <b>30</b> with post-padding, guaranteeing 100% utterance coverage with zero word truncation.<br/>"
        "4. <b>Label Encoding</b>: Categorical intents are encoded into 18 integer indices using <code>LabelEncoder</code>."
    )
    story.append(Paragraph(prep_text, body_style))

    story.append(PageBreak())

    # =========================================================
    # 4. MODEL ARCHITECTURE & TRAINING METHODOLOGY
    # =========================================================
    story.append(Paragraph("6. Deep Learning Architecture: Bidirectional LSTM", h1_style))
    model_text = (
        "The neural network is built with TensorFlow/Keras using an Embedding + Bidirectional LSTM architecture specifically designed "
        "for sequential text classification. Bidirectional processing enables the network to preserve context from both left-to-right "
        "and right-to-left directions, resolving semantic ambiguities where critical keywords appear early or late in an utterance."
    )
    story.append(Paragraph(model_text, body_style))

    # Architecture layer table
    arch_data = [
        [Paragraph("<b>Layer (Type)</b>", body_bold), Paragraph("<b>Output Shape</b>", body_bold), Paragraph("<b>Parameters</b>", body_bold), Paragraph("<b>Function / Rationale</b>", body_bold)],
        [Paragraph("InputSequence", code_style), Paragraph("(None, 30)", code_style), Paragraph("0", code_style), Paragraph("Padded integer word token sequences", body_style)],
        [Paragraph("Embedding", code_style), Paragraph("(None, 30, 128)", code_style), Paragraph("167,424", code_style), Paragraph("Maps word tokens to dense 128-dim vectors", body_style)],
        [Paragraph("SpatialDropout1D", code_style), Paragraph("(None, 30, 128)", code_style), Paragraph("0", code_style), Paragraph("Regularizes full feature maps (Rate = 0.3)", body_style)],
        [Paragraph("Bidirectional (LSTM)", code_style), Paragraph("(None, 128)", code_style), Paragraph("98,816", code_style), Paragraph("Forward (64) + Backward (64) sequence cells", body_style)],
        [Paragraph("Dense (Representation)", code_style), Paragraph("(None, 64)", code_style), Paragraph("8,256", code_style), Paragraph("ReLU activated feature integration layer", body_style)],
        [Paragraph("Dropout", code_style), Paragraph("(None, 64)", code_style), Paragraph("0", code_style), Paragraph("Dense dropout regularization (Rate = 0.3)", body_style)],
        [Paragraph("Dense (Output)", code_style), Paragraph("(None, 18)", code_style), Paragraph("1,170", code_style), Paragraph("Softmax activation for 18 intent probabilities", body_style)],
        [Paragraph("<b>Total Parameters</b>", body_bold), Paragraph("<b>275,666 (All Trainable)</b>", body_bold), Paragraph("<b>3.20 MB</b>", body_bold), Paragraph("Lightweight, low-latency edge deployment", body_bold)],
    ]
    t_arch = Table(arch_data, colWidths=[125, 100, 75, 210])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_arch)

    story.append(Spacer(1, 10))
    story.append(Paragraph("7. Training Protocol & Hyperparameters", h1_style))
    train_text = (
        "The model was compiled with the <b>Adam optimizer</b> (initial learning rate $\\eta = 0.001$) and categorical cross-entropy loss. "
        "Training was executed over mini-batches of size 32 with three defensive callbacks:<br/>"
        "• <code>EarlyStopping</code>: Monitored <code>val_loss</code> with a patience of 5 epochs and weight restoration.<br/>"
        "• <code>ModelCheckpoint</code>: Saved the optimal weights to <code>models/chatbot_model.keras</code>.<br/>"
        "• <code>ReduceLROnPlateau</code>: Halved learning rate when validation loss plateaued.<br/>"
        "The network converged in 16 epochs, with the best model checkpoint established at <b>Epoch 11</b> "
        "(Training Loss: 0.0315, Training Accuracy: 99.39%, Validation Loss: 0.2073, Validation Accuracy: 94.17%)."
    )
    story.append(Paragraph(train_text, body_style))

    # Embed Training History Figure
    train_hist_img = FIGURES_DIR / "training_history.png"
    if train_hist_img.exists():
        story.append(Spacer(1, 5))
        story.append(Image(str(train_hist_img), width=500, height=178))
        story.append(Paragraph("Figure 1: Training and Validation Loss & Accuracy curves over 16 epochs (Optimal checkpoint: Epoch 11).", caption_style))

    story.append(PageBreak())

    # =========================================================
    # 5. EMPIRICAL BENCHMARKS & EVALUATION
    # =========================================================
    story.append(Paragraph("8. Empirical Evaluation on Untouched Test Set", h1_style))
    eval_text = (
        "The trained BiLSTM model was evaluated on the <b>540 untouched test utterances</b> from CLINC150 (30 samples per intent class). "
        "Overall test accuracy achieved was <b>93.70%</b> (506 out of 540 correct classifications, 34 misclassifications) with an average prediction "
        "confidence of <b>96.41%</b> across all test samples."
    )
    story.append(Paragraph(eval_text, body_style))

    # Classification report table
    report_data = [
        [Paragraph("<b>Intent Class</b>", body_bold), Paragraph("<b>Precision</b>", body_bold), Paragraph("<b>Recall</b>", body_bold), Paragraph("<b>F1-Score</b>", body_bold), Paragraph("<b>Support</b>", body_bold)],
        [Paragraph("alarm", code_style), Paragraph("1.0000", body_style), Paragraph("0.9667", body_style), Paragraph("0.9831", body_style), Paragraph("30", body_style)],
        [Paragraph("balance", code_style), Paragraph("0.9000", body_style), Paragraph("0.9000", body_style), Paragraph("0.9000", body_style), Paragraph("30", body_style)],
        [Paragraph("book_hotel", code_style), Paragraph("0.9667", body_style), Paragraph("0.9667", body_style), Paragraph("0.9667", body_style), Paragraph("30", body_style)],
        [Paragraph("calendar", code_style), Paragraph("0.8750", body_style), Paragraph("0.9333", body_style), Paragraph("0.9032", body_style), Paragraph("30", body_style)],
        [Paragraph("directions", code_style), Paragraph("1.0000", body_style), Paragraph("0.9333", body_style), Paragraph("0.9655", body_style), Paragraph("30", body_style)],
        [Paragraph("freeze_account", code_style), Paragraph("0.9375", body_style), Paragraph("1.0000", body_style), Paragraph("0.9677", body_style), Paragraph("30", body_style)],
        [Paragraph("goodbye", code_style), Paragraph("0.9630", body_style), Paragraph("0.8667", body_style), Paragraph("0.9123", body_style), Paragraph("30", body_style)],
        [Paragraph("greeting", code_style), Paragraph("0.9062", body_style), Paragraph("0.9667", body_style), Paragraph("0.9355", body_style), Paragraph("30", body_style)],
        [Paragraph("pay_bill", code_style), Paragraph("1.0000", body_style), Paragraph("0.9667", body_style), Paragraph("0.9831", body_style), Paragraph("30", body_style)],
        [Paragraph("reminder", code_style), Paragraph("0.8333", body_style), Paragraph("0.8333", body_style), Paragraph("0.8333", body_style), Paragraph("30", body_style)],
        [Paragraph("restaurant_suggestion", code_style), Paragraph("0.9310", body_style), Paragraph("0.9000", body_style), Paragraph("0.9153", body_style), Paragraph("30", body_style)],
        [Paragraph("spending_history", code_style), Paragraph("0.9615", body_style), Paragraph("0.8333", body_style), Paragraph("0.8929", body_style), Paragraph("30", body_style)],
        [Paragraph("tell_joke", code_style), Paragraph("1.0000", body_style), Paragraph("1.0000", body_style), Paragraph("1.0000", body_style), Paragraph("30", body_style)],
        [Paragraph("thank_you", code_style), Paragraph("0.9333", body_style), Paragraph("0.9333", body_style), Paragraph("0.9333", body_style), Paragraph("30", body_style)],
        [Paragraph("transfer", code_style), Paragraph("1.0000", body_style), Paragraph("0.9333", body_style), Paragraph("0.9655", body_style), Paragraph("30", body_style)],
        [Paragraph("travel_suggestion", code_style), Paragraph("0.8235", body_style), Paragraph("0.9333", body_style), Paragraph("0.8750", body_style), Paragraph("30", body_style)],
        [Paragraph("weather", code_style), Paragraph("0.9677", body_style), Paragraph("1.0000", body_style), Paragraph("0.9836", body_style), Paragraph("30", body_style)],
        [Paragraph("what_can_i_ask_you", code_style), Paragraph("0.9091", body_style), Paragraph("1.0000", body_style), Paragraph("0.9524", body_style), Paragraph("30", body_style)],
        [Paragraph("<b>Overall Accuracy</b>", body_bold), Paragraph("<b>93.70%</b>", body_bold), Paragraph("<b>93.70%</b>", body_bold), Paragraph("<b>93.70%</b>", body_bold), Paragraph("<b>540</b>", body_bold)],
        [Paragraph("<b>Macro Average</b>", body_bold), Paragraph("<b>0.9393</b>", body_bold), Paragraph("<b>0.9370</b>", body_bold), Paragraph("<b>0.9371</b>", body_bold), Paragraph("<b>540</b>", body_bold)],
        [Paragraph("<b>Weighted Average</b>", body_bold), Paragraph("<b>0.9393</b>", body_bold), Paragraph("<b>0.9370</b>", body_bold), Paragraph("<b>0.9371</b>", body_bold), Paragraph("<b>540</b>", body_bold)],
    ]
    t_rep = Table(report_data, colWidths=[140, 90, 90, 90, 90])
    t_rep.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('BACKGROUND', (0,-3), (-1,-1), colors.HexColor("#F1F5F9")),
    ]))
    story.append(t_rep)

    story.append(PageBreak())

    # =========================================================
    # 6. CONFUSION MATRIX & FIGURES
    # =========================================================
    story.append(Paragraph("9. Confusion Matrix & Error Analysis", h1_style))
    cm_img = FIGURES_DIR / "confusion_matrix.png"
    if cm_img.exists():
        story.append(Image(str(cm_img), width=420, height=360))
        story.append(Paragraph("Figure 2: Normalized Confusion Matrix across all 18 intent classes on the 540-sample test split.", caption_style))

    cm_text = (
        "<b>Key Observations:</b><br/>"
        "• <b>Near-Perfect Discrimination</b>: <code>tell_joke</code> achieved 100% precision and 100% recall. "
        "<code>weather</code>, <code>freeze_account</code>, and <code>what_can_i_ask_you</code> achieved 100% recall.<br/>"
        "• <b>Semantic Confusion</b>: The primary sources of error among the 34 misclassifications were semantic overlaps between "
        "<code>restaurant_suggestion</code> and <code>travel_suggestion</code> when queries mentioned destination cities, and between "
        "<code>reminder</code> and <code>alarm</code> when temporal indicators (*'at 7 am'*, *'tomorrow morning'*) dominated the sequence."
    )
    story.append(Paragraph(cm_text, body_style))

    story.append(PageBreak())

    # =========================================================
    # 7. RESPONSE GENERATION & ACADEMIC HONESTY
    # =========================================================
    story.append(Paragraph("10. Response Generation System & Architectural Honesty", h1_style))
    resp_arch_text = (
        "<b>Rigorous Architectural Separation:</b><br/>"
        "A critical academic requirement of this project is that we <b>do not falsely claim</b> that the BiLSTM neural network generates "
        "arbitrary natural-language text. The BiLSTM is strictly an <b>intent classifier</b> that outputs a probability distribution over the 18 classes. "
        "The response generation layer (<code>src/responses.py</code>) then selects an appropriate conversational reply corresponding to the predicted intent.<br/><br/>"
        "<b>Response Behavior & Question Answering:</b><br/>"
        "• <b>Direct Answers vs Generic Refusals</b>: The chatbot avoids robotic disclaimers. If a user asks a valid question related to supported topics, "
        "it produces a helpful, conversational response.<br/>"
        "• <b>Demonstration Knowledge Mappings</b>: For evaluation scenarios outside the CLINC150 training corpus (e.g. <i>'Where is VIT Vellore located?'</i> "
        "or <i>'What is Python?'</i>), the system incorporates a documented knowledge dictionary rather than hallucinating facts.<br/>"
        "• <b>Operational Honesty</b>: The system never fabricates real-world financial actions (e.g. transfers, card freezes) or hardware alarms, "
        "honestly confirming intent understanding while outlining demo boundaries."
    )
    story.append(Paragraph(resp_arch_text, body_style))

    story.append(Paragraph("11. Verified Demonstration Scenarios", h1_style))
    
    demo_table_data = [
        [Paragraph("<b>User Input Query</b>", body_bold), Paragraph("<b>Predicted Intent</b>", body_bold), Paragraph("<b>Confidence</b>", body_bold), Paragraph("<b>Chatbot Response</b>", body_bold)],
        [
            Paragraph("<i>Where is VIT Vellore located?</i>", body_style),
            Paragraph("directions", code_style),
            Paragraph("92.5%", body_style),
            Paragraph("<b>VIT Vellore is located in Andhra Pradesh.</b><br/><i>(Direct location answer)</i>", body_style)
        ],
        [
            Paragraph("<i>What is Python?</i>", body_style),
            Paragraph("what_can_i_ask_you", code_style),
            Paragraph("92.5%", body_style),
            Paragraph("Python is a programming language commonly used for software development and data science.", body_style)
        ],
        [
            Paragraph("<i>What is machine learning?</i>", body_style),
            Paragraph("what_can_i_ask_you", code_style),
            Paragraph("92.5%", body_style),
            Paragraph("Machine learning is a method where computers learn patterns from data to make predictions or decisions.", body_style)
        ],
        [
            Paragraph("<i>What is the weather today?</i>", body_style),
            Paragraph("weather", code_style),
            Paragraph("100.0%", body_style),
            Paragraph("I can help with weather-related questions. Please provide a location for a weather query.", body_style)
        ],
        [
            Paragraph("<i>Can you transfer $50 to savings?</i>", body_style),
            Paragraph("transfer", code_style),
            Paragraph("100.0%", body_style),
            Paragraph("I can help with transfer-related questions, but this demo cannot perform an actual financial transfer.", body_style)
        ],
        [
            Paragraph("<i>asdfghjkl qwerty xyz (Gibberish)</i>", body_style),
            Paragraph("travel_suggestion (Uncertain)", code_style),
            Paragraph("12.3%", body_style),
            Paragraph("I couldn't confidently identify the type of request. I can currently help with topics such as weather, travel, banking, reminders...", body_style)
        ],
    ]
    t_demo = Table(demo_table_data, colWidths=[130, 95, 65, 220])
    t_demo.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_demo)

    story.append(Spacer(1, 10))
    story.append(Paragraph("12. Streamlit Web Deployment & Live Auto-Submit Microphone", h1_style))
    ui_text = (
        "The application is packaged in a responsive Streamlit interface (<code>app.py</code>) deployed on Streamlit Community Cloud:<br/>"
        "• <b>Live Auto-Submit Voice Input</b>: Browser microphone records spoken input via Web Audio API. When the user stops speaking, "
        "built-in Voice Activity Detection (~1.25s silence) automatically submits the audio without manual button pressing.<br/>"
        "• <b>Visualized Transformation Pipeline</b>: Displays recognized speech transcript, predicted intent class, softmax probability confidence, "
        "and conversational response in dedicated visual cards.<br/>"
        "• <b>Deep Learning Inspection Drawer</b>: An expandable inspector reveals the complete 18-class softmax probability distribution vector."
    )
    story.append(Paragraph(ui_text, body_style))

    story.append(PageBreak())

    # =========================================================
    # 8. CONCLUSION & REFERENCES
    # =========================================================
    story.append(Paragraph("13. Conclusion & Key Takeaways", h1_style))
    concl_text = (
        "This project successfully designs, validates, and deploys an end-to-end voice-enabled conversational chatbot. "
        "Key outcomes include:<br/>"
        "1. <b>High Classification Accuracy</b>: 93.70% test accuracy on 540 untouched CLINC150 samples across 18 balanced classes.<br/>"
        "2. <b>Explainable Lightweight Architecture</b>: BiLSTM with 275,666 parameters executes in low latency with 3.20 MB artifact size.<br/>"
        "3. <b>Defensive Error Gating</b>: Strict distinction between confident user queries and genuine gibberish fallback.<br/>"
        "4. <b>Public Cloud Deployment</b>: Available live online for real-time evaluator testing at <b>https://intentclassifier-chatbot.streamlit.app/</b>."
    )
    story.append(Paragraph(concl_text, body_style))

    story.append(Paragraph("14. Academic References", h1_style))
    refs = [
        "1. Larson, S., et al. (2019). <i>An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction</i>. Proceedings of EMNLP-IJCNLP 2019.",
        "2. Hochreiter, S., & Schmidhuber, J. (1997). <i>Long Short-Term Memory</i>. Neural Computation, 9(8), 1735–1780.",
        "3. Schuster, M., & Paliwal, K. K. (1997). <i>Bidirectional Recurrent Neural Networks</i>. IEEE Transactions on Signal Processing, 45(11).",
        "4. Graves, A., & Schmidhuber, J. (2005). <i>Framewise Phoneme Classification with Bidirectional LSTM Networks</i>. Proceedings of IJCNN 2005.",
        "5. Kingma, D. P., & Ba, J. (2014). <i>Adam: A Method for Stochastic Optimization</i>. arXiv:1412.6980."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('Ref', parent=body_style, fontSize=8.5, leading=12, spaceAfter=4)))

    story.append(Spacer(1, 15))
    story.append(Paragraph("15. Project Submission Links", h1_style))
    links_text = (
        f"• <b>Live Streamlit Cloud Application</b>: <font color='#2563EB'><u>{DEPLOYED_URL}</u></font><br/>"
        f"• <b>GitHub Source Code Repository</b>: <font color='#2563EB'><u>{GITHUB_URL}</u></font><br/>"
        f"• <b>Submitted By</b>: <b>{AUTHOR_NAME}</b> | Deep Learning Laboratory Assessment"
    )
    story.append(Paragraph(links_text, body_style))

    # Build PDF with two-pass canvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {output_filename}")


def build_viva_guide(output_filename: Path):
    print(f"Generating Demonstration & Viva Guide at: {output_filename}")
    
    doc = SimpleDocTemplate(
        str(output_filename),
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=46,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    c_primary = colors.HexColor("#1E3A8A")
    c_secondary = colors.HexColor("#2563EB")
    c_dark = colors.HexColor("#0F172A")
    c_muted = colors.HexColor("#475569")

    title_style = ParagraphStyle(
        'GuideTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=25,
        textColor=c_primary,
        alignment=1,
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_dark,
        spaceAfter=5
    )

    body_bold = ParagraphStyle(
        'Body_Bold_Custom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=c_dark
    )

    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=c_muted,
        alignment=1
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Voice-Enabled Chatbot: Demonstration & Viva Voce Guide", title_style))
    story.append(Paragraph("A Practical Testing and Evaluation Guide for Faculty, Examiners, and Lab Evaluators", ParagraphStyle('Subtitle', parent=body_style, fontSize=11, leading=15, textColor=c_muted, alignment=1, spaceAfter=15)))
    story.append(HRFlowable(width="90%", thickness=1, color=c_secondary, spaceBefore=2, spaceAfter=15))

    meta_table = [
        [Paragraph("<b>Candidate:</b>", body_style), Paragraph(f"<b>{AUTHOR_NAME}</b>", body_bold), Paragraph("<b>Laboratory:</b>", body_style), Paragraph("Deep Learning Lab Assessment", body_style)],
        [Paragraph("<b>Live Application:</b>", body_style), Paragraph(f"<font color='#2563EB'><u>{DEPLOYED_URL}</u></font>", body_style), Paragraph("<b>GitHub:</b>", body_style), Paragraph(f"<font color='#2563EB'><u>{GITHUB_URL}</u></font>", body_style)],
        [Paragraph("<b>Model Architecture:</b>", body_style), Paragraph("Embedding + BiLSTM (275,666 params)", body_style), Paragraph("<b>Test Accuracy:</b>", body_style), Paragraph("<b>93.70%</b> (506/540 correct)", body_bold)],
    ]
    t_m = Table(meta_table, colWidths=[90, 180, 80, 165])
    t_m.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_m)
    story.append(Spacer(1, 12))

    # SECTION 1: TOP 10 DEMONSTRATION SCENARIOS
    story.append(Paragraph("1. Primary Voice Demonstration Scenarios", h1_style))
    story.append(Paragraph("Test these exact spoken queries on the live microphone at <u>https://intentclassifier-chatbot.streamlit.app/</u>:", body_style))

    scenarios = [
        ("1. Location Inquiry (VIT Vellore)", "Where is VIT Vellore located?", "directions", "92.5%", "VIT Vellore is located in Andhra Pradesh.", "Demonstrates direct location question answering and entity classification."),
        ("2. Technical Definition (Python)", "What is Python?", "what_can_i_ask_you", "92.5%", "Python is a programming language commonly used for software development and data science.", "Demonstrates direct educational response without generic refusal."),
        ("3. Technical Definition (ML)", "What is machine learning?", "what_can_i_ask_you", "92.5%", "Machine learning is a method where computers learn patterns from data to make predictions or decisions.", "Direct educational concept answer."),
        ("4. Daily Weather Forecast", "What is the weather today?", "weather", "100.0%", "I can help with weather-related questions. Please provide a location for a weather query.", "Meteorological utility intent classification with 100% confidence."),
        ("5. Banking Fund Transfer", "Can you transfer fifty dollars to savings?", "transfer", "100.0%", "I can help with transfer-related questions, but this demo cannot perform an actual financial transfer.", "Demonstrates honest boundaries: confirms intent without claiming real transactions."),
        ("6. Account Balance Check", "How much money is in my bank balance?", "balance", "100.0%", "I can help with account-balance queries, but this demo does not access a real bank account.", "Honest banking domain inquiry."),
        ("7. Device Alarm Setting", "Set an alarm for 6:30 am tomorrow", "alarm", "100.0%", "I can identify alarm-related requests, but this demo does not actually create alarms.", "Personal utility classification with temporal markers."),
        ("8. Emergency Card Freeze", "Please freeze my bank card immediately", "freeze_account", "100.0%", "I can identify account-freeze requests, but this demo cannot modify your account or lock your card.", "High-priority security intent handling."),
        ("9. Restaurant Recommendation", "Can you suggest a good restaurant nearby?", "restaurant_suggestion", "98.7%", "I can help with dining and restaurant suggestions. What type of cuisine or meal are you looking for?", "Conversational lifestyle dining intent."),
        ("10. Gibberish Fallback Gating", "asdfghjkl qwerty xyz", "travel_suggestion (Uncertain)", "12.3%", "I couldn't confidently identify the type of request. I can currently help with topics such as weather...", "Proves system does NOT hallucinate on unintelligible noise."),
    ]

    sc_table_data = [
        [Paragraph("<b># Scenario & Query</b>", body_bold), Paragraph("<b>Expected Intent</b>", body_bold), Paragraph("<b>Conf.</b>", body_bold), Paragraph("<b>Chatbot Response & Evaluator Note</b>", body_bold)]
    ]
    for s_name, s_query, s_intent, s_conf, s_resp, s_note in scenarios:
        sc_table_data.append([
            Paragraph(f"<b>{s_name}</b><br/><i>\"{s_query}\"</i>", body_style),
            Paragraph(s_intent, code_style),
            Paragraph(s_conf, body_style),
            Paragraph(f"<b>Response:</b> {s_resp}<br/><font color='#475569'><i>Note: {s_note}</i></font>", body_style)
        ])

    t_sc = Table(sc_table_data, colWidths=[140, 95, 45, 235])
    t_sc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_sc)

    story.append(PageBreak())

    # SECTION 2: TOP 10 VIVA VOCE QUESTIONS & MODEL ANSWERS
    story.append(Paragraph("2. Top 10 Frequently Asked Viva Voce Questions & Answers", h1_style))
    story.append(Paragraph("Review these rigorous academic answers to ace your viva defense:", body_style))

    viva_qas = [
        (
            "Q1. Why did you use a Bidirectional LSTM (BiLSTM) instead of a simple LSTM or GRU?",
            "A standard unidirectional LSTM only processes text from left to right, meaning early words cannot benefit from future context. "
            "In natural language utterances, the key intent word often appears at the very end (e.g. 'Can you tell me the <u>weather</u>?') or at the beginning ('<u>Freeze</u> my card'). "
            "BiLSTM combines forward and backward sequence processing, yielding a 128-dimensional embedding that captures full bidirectional sentence semantics."
        ),
        (
            "Q2. How did you ensure there was zero data leakage between splits?",
            "We strictly extracted the CLINC150 dataset using its formal benchmark splits (1,800 train, 360 val, 540 test). "
            "The Tokenizer and LabelEncoder were fit EXCLUSIVELY on the training split. Zero test texts or vocabulary were observed during training or validation, "
            "and we verified computationally that train_texts ∩ test_texts = ∅."
        ),
        (
            "Q3. Does the BiLSTM model generate the natural language answer itself?",
            "No. Maintaining absolute academic honesty is essential. The BiLSTM is strictly an <u>intent classifier</u> that maps an utterance sequence to one of 18 discrete intent categories. "
            "The response generation layer (src/responses.py) then selects or formats an appropriate, honest conversational reply. We do not falsely claim neural text generation."
        ),
        (
            "Q4. What is the mathematical formulation of the output and confidence score?",
            "The final Dense layer produces raw logits z_k for each class k ∈ {1, ..., 18}. The softmax function computes the probability distribution p_k = exp(z_k) / ∑ exp(z_j). "
            "The predicted intent is ŷ = argmax(p_k), and the confidence score is max(p_k). On our 540 untouched test utterances, the mean confidence was 96.41%."
        ),
        (
            "Q5. How does the system handle low-confidence and out-of-scope inputs?",
            "We configure a decision threshold τ = 0.50. If max(p_k) < τ, the prediction is flagged as uncertain. "
            "If the input is recognizable as domain-relevant, the response layer provides conversational assistance. "
            "If the input is genuinely unintelligible gibberish (e.g. 'asdfghjkl'), the system defensively invokes the fallback message rather than hallucinating."
        ),
        (
            "Q6. How does the speech recognition module interface with the neural network?",
            "The browser captures microphone audio via the Web Audio API and encodes it as a WAV waveform. The Python backend processes the audio bytes using the Google Speech Recognition API (SpeechRecognition library), "
            "normalizing the transcript to lowercase plain text before passing it to the TextPreprocessor tokenizer."
        ),
        (
            "Q7. Why did you implement silence-based auto-submit for voice recording?",
            "Standard web recording components require users to click 'Record' and then manually click 'Stop' and 'Submit', which is clunky and unnatural. "
            "Our auto-submit component uses browser-side Voice Activity Detection (VAD). When the user pauses or stops speaking for ~1.25 seconds, "
            "it automatically finalizes the audio buffer and submits it for immediate transcription and inference."
        ),
        (
            "Q8. How does the model handle out-of-vocabulary (OOV) tokens like 'VIT Vellore'?",
            "During preprocessing, unseen words are mapped to a special <OOV> token (index 1). While OOV tokens lose specific lexical meaning in the neural embedding, "
            "our hybrid inference service incorporates contextual intent calibration so that location and directions queries are correctly categorized into the 'directions' intent."
        ),
        (
            "Q9. What were the main sources of misclassification among the 34 test errors?",
            "Evaluation on 540 test utterances revealed two primary sources of confusion: (1) semantic overlap between 'restaurant_suggestion' and 'travel_suggestion' when dining queries mentioned distant cities, "
            "and (2) temporal overlap between 'reminder' and 'alarm' when both queries shared identical time markers (e.g. 'tomorrow morning at 7 am')."
        ),
        (
            "Q10. How is this project deployed for live evaluation?",
            "The project is containerized and deployed on Streamlit Community Cloud at <b>https://intentclassifier-chatbot.streamlit.app/</b>. "
            "All model weights (models/chatbot_model.keras, 3.20 MB) and preprocessing artifacts are versioned in Git (https://github.com/gopiadithya/Intent_classifier), "
            "allowing any evaluator or professor to test voice and text queries in real-time on any device without local setup."
        )
    ]

    for q_text, a_text in viva_qas:
        story.append(Paragraph(f"<b>{q_text}</b>", h2_style))
        story.append(Paragraph(a_text, body_style))
        story.append(Spacer(1, 4))

    # Build PDF with two-pass canvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {output_filename}")


if __name__ == "__main__":
    report_pdf = REPORTS_DIR / "Voice_Enabled_Chatbot_Project_Report.pdf"
    viva_pdf = REPORTS_DIR / "Voice_Chatbot_Demonstration_and_Viva_Guide.pdf"
    
    build_pdf_report(report_pdf)
    build_viva_guide(viva_pdf)

