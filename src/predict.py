"""
src/predict.py
Load the fine-tuned BERT model and run inference on single or batch reviews.

Usage (standalone test):
    python src/predict.py
"""

import os
import sys
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH     = "models/bert_sentiment_model"
TOKENIZER_PATH = "models/bert_sentiment_tokenizer"
MAX_LENGTH     = 128

# ── Load model once at module level ──────────────────────────────────────────

_tokenizer = None
_model     = None
_device    = None


def _load_model():
    """Load tokenizer + model from disk (lazy, called once)."""
    global _tokenizer, _model, _device

    if _model is not None:
        return  # already loaded

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Trained model not found at '{MODEL_PATH}'.")
        print("   Run  python src/train.py  first.")
        sys.exit(1)

    print("Loading BERT model…")
    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    _model     = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    _model.to(_device)
    _model.eval()
    print(f"Model loaded on {_device}.")


# ── Single prediction ─────────────────────────────────────────────────────────

def predict_sentiment(review_text: str) -> dict:
    """
    Predict sentiment for a single review string.

    Parameters
    ----------
    review_text : str
        Raw review text (no preprocessing needed).

    Returns
    -------
    dict with keys:
        sentiment_label  : "positive" or "negative"
        confidence_score : float  (0–100 %)
        raw_label        : int    (1 = positive, 0 = negative)
    """
    _load_model()

    encoding = _tokenizer(
        review_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    input_ids      = encoding["input_ids"].to(_device)
    attention_mask = encoding["attention_mask"].to(_device)

    with torch.no_grad():
        outputs = _model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits

    probabilities   = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    raw_label       = int(np.argmax(probabilities))
    confidence      = float(probabilities[raw_label]) * 100.0
    sentiment_label = "positive" if raw_label == 1 else "negative"

    return {
        "sentiment_label":  sentiment_label,
        "confidence_score": round(confidence, 2),
        "raw_label":        raw_label,
    }


# ── Batch prediction ──────────────────────────────────────────────────────────

def predict_batch(reviews: list[str], batch_size: int = 32) -> list[dict]:
    """
    Predict sentiment for a list of review strings.

    Parameters
    ----------
    reviews    : list of str
    batch_size : int  (number of reviews processed at once)

    Returns
    -------
    list of dicts (same schema as predict_sentiment)
    """
    _load_model()
    results = []

    for start in range(0, len(reviews), batch_size):
        batch = reviews[start : start + batch_size]

        encoding = _tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(_device)
        attention_mask = encoding["attention_mask"].to(_device)

        with torch.no_grad():
            outputs = _model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

        probs  = torch.softmax(logits, dim=-1).cpu().numpy()

        for prob_row in probs:
            raw_label  = int(np.argmax(prob_row))
            confidence = float(prob_row[raw_label]) * 100.0
            results.append({
                "sentiment_label":  "positive" if raw_label == 1 else "negative",
                "confidence_score": round(confidence, 2),
                "raw_label":        raw_label,
            })

        print(f"  Processed {min(start + batch_size, len(reviews))}/{len(reviews)} reviews…")

    return results


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_reviews = [
        "This product is absolutely amazing! Best purchase I have ever made.",
        "Complete waste of money. Broke after two days. Totally disappointed.",
        "It is okay, nothing special but it gets the job done.",
        "Fast shipping, great quality, exactly as described. Very happy!",
        "Do NOT buy this. The seller is a scam and the product is fake.",
    ]

    print("\n" + "="*60)
    print("  SENTIMENT PREDICTION TEST")
    print("="*60)

    for review in test_reviews:
        result = predict_sentiment(review)
        label  = result["sentiment_label"].upper()
        conf   = result["confidence_score"]
        icon   = "✅" if label == "POSITIVE" else "❌"
        print(f"\nReview  : {review[:65]}…" if len(review) > 65 else f"\nReview  : {review}")
        print(f"Result  : {icon} {label}  ({conf:.1f}% confidence)")

    print("\n" + "="*60)
    print("Batch prediction test:")
    batch_results = predict_batch(test_reviews)
    for i, r in enumerate(batch_results):
        print(f"  [{i+1}] {r['sentiment_label']:8s}  {r['confidence_score']:.1f}%")
    print("="*60)
