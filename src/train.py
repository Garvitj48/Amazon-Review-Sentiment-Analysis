"""
src/train.py
Fine-tunes bert-base-uncased for binary sentiment classification
on the Amazon review dataset.

Usage:
    python src/train.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH       = "data/amazon_reviews.csv"
MODEL_SAVE_PATH = "models/bert_sentiment_model"
TOKENIZER_SAVE  = "models/bert_sentiment_tokenizer"
os.makedirs("models", exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
MODEL_NAME   = "bert-base-uncased"
MAX_LENGTH   = 128
BATCH_SIZE   = 16
EPOCHS       = 3
LEARNING_RATE = 2e-5
RANDOM_STATE = 42


# ── Dataset class ─────────────────────────────────────────────────────────────

class ReviewDataset(Dataset):
    """PyTorch Dataset that tokenizes reviews on the fly."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Metrics function ──────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy":  round(acc, 4),
        "f1":        round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
    }


# ── Main training pipeline ────────────────────────────────────────────────────

def main():
    # 1. Load data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at '{DATA_PATH}'.")
        print("   Run  python download_data.py  first.")
        sys.exit(1)

    print("Loading dataset…")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["review_text", "sentiment"])
    texts  = df["review_text"].tolist()
    labels = df["sentiment"].astype(int).tolist()
    print(f"   {len(texts):,} reviews loaded.")

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 3. Tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}…")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = ReviewDataset(X_train, y_train, tokenizer)
    test_dataset  = ReviewDataset(X_test,  y_test,  tokenizer)

    # 4. Model
    print(f"Loading model: {MODEL_NAME}…")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir               = "./results/checkpoints",
        num_train_epochs         = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate            = LEARNING_RATE,
        weight_decay             = 0.01,
        eval_strategy            = "epoch",
        save_strategy            = "epoch",
        load_best_model_at_end   = True,
        metric_for_best_model    = "f1",
        logging_dir              = "./results/logs",
        logging_steps            = 100,
        save_total_limit         = 2,
        fp16                     = torch.cuda.is_available(),  # mixed precision on GPU
        report_to                = "none",
    )

    # 6. Trainer
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = test_dataset,
        compute_metrics = compute_metrics,
    )

    # 7. Train
    print("\n" + "="*50)
    print("Starting training…")
    print("="*50)
    trainer.train()

    # 8. Final evaluation
    print("\nRunning final evaluation on test set…")
    metrics = trainer.evaluate(test_dataset)

    print("\n" + "="*50)
    print("  MODEL PERFORMANCE")
    print("="*50)
    print(f"  Accuracy  : {metrics.get('eval_accuracy',  'N/A')}")
    print(f"  F1 Score  : {metrics.get('eval_f1',        'N/A')}")
    print(f"  Precision : {metrics.get('eval_precision', 'N/A')}")
    print(f"  Recall    : {metrics.get('eval_recall',    'N/A')}")
    print("="*50)

    # 9. Save model & tokenizer
    print(f"\nSaving model to '{MODEL_SAVE_PATH}'…")
    model.save_pretrained(MODEL_SAVE_PATH)
    print(f"Saving tokenizer to '{TOKENIZER_SAVE}'…")
    tokenizer.save_pretrained(TOKENIZER_SAVE)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
