"""
app.py
Streamlit web application for Amazon Review Sentiment Analysis.

Usage:
    streamlit run app.py
"""

import sys
import os
import streamlit as st

# Add project root to path so src/ imports work
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .result-positive {
        background: #d4edda;
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .result-negative {
        background: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .result-confidence {
        font-size: 1.1rem;
        color: #555;
    }
    .char-counter {
        font-size: 0.8rem;
        color: #888;
        text-align: right;
    }
    .footer-text {
        font-size: 0.85rem;
        color: #999;
        text-align: center;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("💬 Amazon Review Sentiment Analyzer")
st.markdown(
    "Analyzes Amazon product reviews using a fine-tuned **BERT** transformer "
    "model to predict whether a review is **Positive** or **Negative**."
)
st.divider()


# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource(show_spinner="Loading BERT model…")
def load_predictor():
    from src.predict import predict_sentiment, predict_batch
    return predict_sentiment, predict_batch


model_available = os.path.exists("models/bert_sentiment_model")

if model_available:
    predict_sentiment, predict_batch = load_predictor()
else:
    st.warning(
        "⚠️ Trained model not found. Run `python src/train.py` first to train "
        "the model, then restart the app.",
        icon="⚠️",
    )
    predict_sentiment = None


# ── Input section ─────────────────────────────────────────────────────────────
st.subheader("Enter a Review")

review_input = st.text_area(
    label="Review text",
    placeholder="Enter Amazon product review here…",
    height=160,
    label_visibility="collapsed",
)

char_count = len(review_input)
st.markdown(
    f'<p class="char-counter">{char_count} characters</p>',
    unsafe_allow_html=True,
)

analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")


# ── Analysis result ───────────────────────────────────────────────────────────
if analyze_btn:
    if not review_input.strip():
        st.error("Please enter a review before clicking Analyze.")
    elif predict_sentiment is None:
        st.error("Model is not loaded. Train the model first.")
    else:
        with st.spinner("Analyzing…"):
            result = predict_sentiment(review_input)

        label      = result["sentiment_label"]
        confidence = result["confidence_score"]
        is_pos     = label == "positive"

        st.markdown("### 📊 Analysis Result")

        css_class  = "result-positive" if is_pos else "result-negative"
        icon       = "✅" if is_pos else "❌"
        label_text = "POSITIVE" if is_pos else "NEGATIVE"

        st.markdown(f"""
        <div class="{css_class}">
            <div class="result-label">{icon} {label_text}</div>
            <div class="result-confidence">{confidence:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")  # spacing

        # Sentiment bar
        pos_score = confidence if is_pos else 100 - confidence
        neg_score = 100 - pos_score

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Positive Score", f"{pos_score:.1f}%")
        with col2:
            st.metric("Negative Score", f"{neg_score:.1f}%")

        st.progress(int(pos_score), text=f"Positive ← {'█' * int(pos_score // 5)}{'░' * (20 - int(pos_score // 5))} → Negative")

        st.divider()


# ── Example reviews ───────────────────────────────────────────────────────────
st.subheader("📝 Try an Example Review")

examples = {
    "⭐⭐⭐⭐⭐ Great product": (
        "Absolutely love this product! It arrived quickly, packaging was perfect, "
        "and it works exactly as described. Very high quality materials and the "
        "design is beautiful. Would definitely buy again and highly recommend!"
    ),
    "⭐ Terrible purchase": (
        "Complete waste of money. The item broke within two days of light use. "
        "The quality is extremely poor — nothing like the photos. Customer support "
        "was unhelpful and refused to issue a refund. Stay away from this seller."
    ),
    "⭐⭐⭐⭐ Solid buy, minor issues": (
        "Overall a good product for the price. Setup was straightforward and it "
        "works well. Docking a star because the instructions were unclear and one "
        "small piece felt flimsy. But for the price point, I am happy with it."
    ),
}

selected = st.selectbox("Choose an example:", list(examples.keys()))

if st.button("📋 Load Example", use_container_width=False):
    st.session_state["example_text"] = examples[selected]
    st.rerun()

if "example_text" in st.session_state:
    st.text_area(
        "Example review (copy and paste into the box above):",
        value=st.session_state["example_text"],
        height=100,
        disabled=True,
    )

st.divider()


# ── How to use ────────────────────────────────────────────────────────────────
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    1. **Paste or type** an Amazon product review into the text box above.
    2. Click **🔍 Analyze Sentiment** to run the BERT model.
    3. View the **sentiment result** (Positive / Negative) and **confidence score**.
    4. Use the **Sentiment Bar** to see the positive vs. negative split.
    5. Try one of the **example reviews** to see how the model works.
    """)

with st.expander("🤖 How the Model Works"):
    st.markdown("""
    - The model is **bert-base-uncased** fine-tuned on 50,000+ Amazon reviews.
    - Each review is tokenized into sub-word tokens using the BERT tokenizer (max 128 tokens).
    - The `[CLS]` token embedding is passed through a classification head to output
      **Positive (1)** or **Negative (0)**.
    - Confidence is the **softmax probability** of the predicted class (0–100%).
    """)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    💬 Amazon Review Sentiment Analysis &nbsp;|&nbsp;
    Built with BERT + HuggingFace Transformers + Streamlit &nbsp;|&nbsp;
    <a href="https://github.com/YOUR_USERNAME/amazon-review-sentiment" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)
