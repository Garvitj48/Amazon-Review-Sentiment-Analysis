"""
analyze_reviews.py
Batch-analyze the full Amazon review dataset and save summary statistics.

Usage:
    python analyze_reviews.py
"""

import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from src.predict import predict_batch

DATA_PATH   = "data/amazon_reviews.csv"
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "sentiment_analysis_summary.csv")
BATCH_SIZE  = 64

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    # 1. Load dataset
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at '{DATA_PATH}'. Run download_data.py first.")
        sys.exit(1)

    print("Loading dataset…")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["review_text"])
    reviews = df["review_text"].tolist()
    print(f"   {len(reviews):,} reviews loaded.")

    # 2. Batch predict
    print("\nRunning batch sentiment analysis…")
    results = []
    for start in tqdm(range(0, len(reviews), BATCH_SIZE), desc="Batches", unit="batch"):
        batch  = reviews[start : start + BATCH_SIZE]
        preds  = predict_batch(batch, batch_size=BATCH_SIZE)
        results.extend(preds)

    # 3. Merge predictions back into DataFrame
    pred_df = pd.DataFrame(results)
    df = df.reset_index(drop=True)
    df["predicted_sentiment"] = pred_df["sentiment_label"]
    df["confidence_score"]    = pred_df["confidence_score"]

    # 4. Summary statistics
    total     = len(df)
    pos_count = (df["predicted_sentiment"] == "positive").sum()
    neg_count = (df["predicted_sentiment"] == "negative").sum()
    avg_conf  = df["confidence_score"].mean()

    print("\n" + "="*50)
    print("  BATCH ANALYSIS SUMMARY")
    print("="*50)
    print(f"  Total reviews analyzed : {total:,}")
    print(f"  Positive reviews       : {pos_count:,}  ({pos_count/total*100:.1f}%)")
    print(f"  Negative reviews       : {neg_count:,}  ({neg_count/total*100:.1f}%)")
    print(f"  Average confidence     : {avg_conf:.2f}%")
    print("="*50)

    # 5. Save detailed results CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Full results saved to '{OUTPUT_FILE}'")

    # 6. Save compact summary CSV
    summary = pd.DataFrame([{
        "total_reviews":       total,
        "positive_count":      int(pos_count),
        "positive_pct":        round(pos_count / total * 100, 2),
        "negative_count":      int(neg_count),
        "negative_pct":        round(neg_count / total * 100, 2),
        "avg_confidence_pct":  round(avg_conf, 2),
    }])
    summary_path = os.path.join(RESULTS_DIR, "summary_stats.csv")
    summary.to_csv(summary_path, index=False)
    print(f"   Summary stats saved to '{summary_path}'")


if __name__ == "__main__":
    main()
