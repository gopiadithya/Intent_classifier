"""
Streamlit Web Application: Voice-Enabled Chatbot
Using Speech Recognition and Deep Learning-Based Intent Classification (BiLSTM).
Academic Project for Deep Learning Lab Assessment.
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.speech import transcribe_audio
from src.inference import IntentClassifierService
from src.components.voice_recorder import auto_voice_recorder

# ---------------------------------------------------------
# Page Configuration & Rich CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Voice-Enabled Chatbot | BiLSTM Intent Classification",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 1.4rem;
    }
    .pipeline-badge {
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 10px 18px;
        font-size: 0.92rem;
        font-weight: 500;
        margin-bottom: 1.6rem;
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        backdrop-filter: blur(8px);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        backdrop-filter: blur(8px);
    }
    .speech-box {
        background: rgba(148, 163, 184, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 1.15rem;
        font-weight: 500;
    }
    .response-card {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 5px solid #10b981;
        border-radius: 12px;
        padding: 18px 22px;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-top: 8px;
    }
    .warning-card {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 5px solid #f59e0b;
        border-radius: 12px;
        padding: 18px 22px;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Academic Information & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/microphone.png", width=110)
    st.title("Project Overview")
    st.markdown("**Voice-Enabled Chatbot**")
    st.caption("Deep Learning Lab Assessment")

    st.divider()
    st.markdown("### 📊 System Metadata")
    st.markdown("- **Architecture**: Bidirectional LSTM (BiLSTM)")
    st.markdown("- **Dataset**: CLINC150 (18 Selected Classes)")
    st.markdown("- **Test Accuracy**: **93.70%** (506/540 correct)")
    st.markdown("- **Speech Recognition**: Web Speech API / Google STT")
    st.markdown("- **Voice Input**: Auto-Submit Silence Detection")

    st.divider()
    st.markdown("### ⚙️ Inference Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=config.CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Predictions with confidence below this value are flagged as uncertain."
    )

    st.divider()
    with st.expander("📚 View 18 Supported Intents"):
        for it in sorted(config.TARGET_INTENTS):
            st.markdown(f"- `{it}`")

    st.divider()
    st.caption("Deep Learning Lab Assessment • Academic Project")

# ---------------------------------------------------------
# Load Inference Service (Cached)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained BiLSTM model & tokenizer...")
def get_service():
    if not config.MODEL_PATH.exists() or not config.TOKENIZER_PATH.exists():
        return None
    try:
        return IntentClassifierService.get_instance()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

classifier_service = get_service()

if classifier_service is None:
    st.error("⚠️ Model artifacts missing! Please run `python train.py` first to generate model weights.")
    st.stop()

# Initialize session state for speech / text query
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "query_source" not in st.session_state:
    st.session_state.query_source = "None"
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None
if "recorded_audio_bytes" not in st.session_state:
    st.session_state.recorded_audio_bytes = None

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown('<div class="main-header">🎙️ Voice-Enabled Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Speech Recognition + Deep Learning-Based Intent Classification (BiLSTM)</div>',
    unsafe_allow_html=True
)

# Pipeline Visualizer
st.markdown("""
<div class="pipeline-badge">
    <span>🎤 <b>1. Voice Input</b></span>
    <span>➔</span>
    <span>📝 <b>2. Recognized Speech</b></span>
    <span>➔</span>
    <span>🧠 <b>3. Detected Intent</b></span>
    <span>➔</span>
    <span>📊 <b>4. Confidence</b></span>
    <span>➔</span>
    <span>🤖 <b>5. Chatbot Response</b></span>
</div>
""", unsafe_allow_html=True)

# Input Section: Voice (Auto-Submit) & Fallback Text Input
col_voice, col_text = st.columns([1.1, 1], gap="large")

with col_voice:
    st.markdown("### 🎙️ Voice Input")
    st.caption("Click to speak — automatically submits when you stop speaking:")
    
    # Pure Auto-Submit Microphone Component
    audio = auto_voice_recorder(key="auto_voice_recorder")

    if audio and audio.get("id") != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio.get("id")
        audio_bytes = audio.get("bytes")
        sample_rate = audio.get("sample_rate")
        sample_width = audio.get("sample_width")
        st.session_state.recorded_audio_bytes = audio_bytes

        with st.spinner("Transcribing speech..."):
            transcription = transcribe_audio(
                audio_bytes,
                sample_rate=sample_rate,
                sample_width=sample_width
            )

        if transcription["success"]:
            st.session_state.user_query = transcription["text"]
            st.session_state.query_source = "Voice Microphone"
            st.rerun()
        else:
            st.error(f"⚠️ {transcription['error']}")

with col_text:
    st.markdown("### ⌨️ Text Input")
    st.caption("Or type a query for instant intent classification:")
    with st.form(key="text_input_form", clear_on_submit=False):
        text_val = st.text_input(
            "Enter query:",
            placeholder="e.g., What is the weather like in New York today?",
            label_visibility="collapsed"
        )
        submit_btn = st.form_submit_button("Classify Query", use_container_width=True)

    if submit_btn and text_val.strip():
        st.session_state.user_query = text_val.strip()
        st.session_state.query_source = "Text Input"
        st.session_state.recorded_audio_bytes = None
        st.rerun()

# Quick Benchmark Test Chips for Evaluators
st.write("")
st.caption("💡 Quick test samples:")
sample_chips = [
    "What is the weather today?",
    "Can you help me transfer money?",
    "Set an alarm for 7 am",
    "Where can I get good food?",
    "What is my account balance?",
    "Tell me a funny joke",
    "I want to freeze my card"
]
chip_cols = st.columns(len(sample_chips))
for i, chip in enumerate(sample_chips):
    if chip_cols[i].button(chip, key=f"chip_{i}", use_container_width=True):
        st.session_state.user_query = chip
        st.session_state.query_source = "Sample Benchmark Query"
        st.session_state.recorded_audio_bytes = None
        st.rerun()

st.divider()

# ---------------------------------------------------------
# Processing & Displaying Pipeline Results
# ---------------------------------------------------------
if st.session_state.user_query:
    query = st.session_state.user_query
    source = st.session_state.query_source

    # Run Deep Learning Intent Classification
    prediction = classifier_service.predict(query, confidence_threshold=conf_threshold)

    # 1. Recognized Speech / Query
    st.markdown("### 📝 Recognized Speech")
    st.markdown(
        f'<div class="speech-box">"{prediction["raw_text"]}" &nbsp;&nbsp;<span style="opacity:0.65; font-size:0.85rem; font-weight:normal;">[{source}]</span></div>',
        unsafe_allow_html=True
    )

    if st.session_state.recorded_audio_bytes:
        st.audio(st.session_state.recorded_audio_bytes, format="audio/wav")

    st.write("")

    # 2. Detected Intent & Confidence Cards
    col_intent, col_conf = st.columns(2)
    with col_intent:
        st.markdown("### 🧠 Detected Intent")
        if prediction["is_low_confidence"]:
            st.markdown(
                f'<div class="metric-card"><div style="font-size:1.35rem; font-weight:700; color:#f59e0b;">⚠️ {prediction["intent"]}</div><div style="opacity:0.75; font-size:0.85rem; margin-top:6px;">Below Threshold ({conf_threshold*100:.0f}%) — Contextually Handled</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="metric-card"><div style="font-size:1.35rem; font-weight:700; color:#3b82f6;">🏷️ {prediction["intent"]}</div><div style="opacity:0.75; font-size:0.85rem; margin-top:6px;">CLINC150 Target Intent Class</div></div>',
                unsafe_allow_html=True
            )

    with col_conf:
        st.markdown("### 📊 Confidence")
        st.markdown(
            f'<div class="metric-card"><div style="font-size:1.35rem; font-weight:700; color:#10b981;">📈 {prediction["confidence_pct"]}</div><div style="opacity:0.75; font-size:0.85rem; margin-top:6px;">Softmax Probability Score</div></div>',
            unsafe_allow_html=True
        )

    st.write("")

    # 3. Chatbot Response
    st.markdown("### 🤖 Chatbot Response")
    card_class = "warning-card" if prediction["is_low_confidence"] else "response-card"
    st.markdown(
        f'<div class="{card_class}"><b>🤖 Assistant:</b><br>{prediction["response"]}</div>',
        unsafe_allow_html=True
    )

    # 4. Softmax Class Distribution Breakdown (For Professor/Evaluator Inspection)
    st.write("")
    with st.expander("🔍 Deep Learning Layer Inspection (Softmax Probability Distribution across all 18 Intents)"):
        st.write("Full output vector from `Dense(18, activation='softmax')`:")
        scores_df = pd.DataFrame([
            {"Intent": k, "Probability": f"{v * 100:.2f}%", "Raw Score": v}
            for k, v in list(prediction["all_scores"].items())
        ])
        st.dataframe(
            scores_df,
            column_config={
                "Raw Score": st.column_config.ProgressColumn(
                    "Confidence Bar",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),
            },
            hide_index=True,
            use_container_width=True
        )

else:
    st.info("👆 Use the **Microphone** to speak, or enter a sentence into the **Text Input** box to run the classification pipeline.")

# Academic Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    Voice-Enabled Chatbot Academic Project • Speech Recognition + BiLSTM Intent Classification • Deep Learning Lab Assessment
</div>
""", unsafe_allow_html=True)
