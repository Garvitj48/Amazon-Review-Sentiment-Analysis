"""
download_data.py
Downloads Amazon review data from Kaggle or creates mock data as fallback.
Saves cleaned dataset to data/amazon_reviews.csv
"""

import os
import re
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "amazon_reviews.csv")
os.makedirs(DATA_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    """Lowercase, remove special characters, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stars_to_sentiment(stars: int):
    """1-2 → 0 (negative), 4-5 → 1 (positive), 3 → None (drop)."""
    if stars in (1, 2):
        return 0
    if stars in (4, 5):
        return 1
    return None


# ── Try Kaggle download ───────────────────────────────────────────────────────

def try_kaggle_download() -> pd.DataFrame | None:
    """
    Attempt to download 'snap/amazon-fine-food-reviews' via the Kaggle API.
    Requires ~/.kaggle/kaggle.json with valid credentials.
    Returns a DataFrame on success, None on failure.
    """
    try:
        import kaggle  # noqa: F401
        import zipfile, glob

        print("Kaggle credentials found. Downloading dataset…")
        kaggle.api.dataset_download_files(
            "snap/amazon-fine-food-reviews",
            path=DATA_DIR,
            unzip=True,
            quiet=False,
        )

        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        if not csv_files:
            print("Download succeeded but no CSV found.")
            return None

        df = pd.read_csv(csv_files[0])
        print(f"Loaded {len(df):,} rows from Kaggle dataset.")
        return df

    except Exception as e:
        print(f"Kaggle download skipped: {e}")
        return None


# ── Mock data fallback ────────────────────────────────────────────────────────

def create_mock_data(n: int = 10_000) -> pd.DataFrame:
    """Generate a realistic mock Amazon review dataset."""
    print(f"Creating mock dataset with {n:,} reviews…")

    positive_templates = [
        "This product is absolutely amazing, exactly what I needed.",
        "Great quality and fast shipping, very happy with this purchase.",
        "Works perfectly, highly recommend to everyone.",
        "Exceeded my expectations, will definitely buy again.",
        "Outstanding product, five stars without hesitation.",
        "Fantastic build quality and great value for money.",
        "Love this item, it has made my life so much easier.",
        "Perfect fit and exactly as described, very satisfied.",
        "Incredible product, customer service was excellent too.",
        "Best purchase I have made in a long time, truly impressive.",
    ]

    negative_templates = [
        "Terrible product, broke after just one week of use.",
        "Complete waste of money, nothing like the description.",
        "Very disappointed, the quality is extremely poor.",
        "Do not buy this, it stopped working immediately.",
        "Awful experience, arrived damaged and customer support ignored me.",
        "Cheap materials and poorly made, would not recommend.",
        "This product is a scam, nothing works as advertised.",
        "Returned it immediately, total garbage.",
        "Worst purchase ever, falls apart after minimal use.",
        "Misleading product photos, looks nothing like what arrived.",
    ]

    rng = np.random.default_rng(42)
    records = []

    for i in range(n):
        sentiment = int(rng.integers(0, 2))          # 0 or 1
        star = int(rng.choice([4, 5] if sentiment == 1 else [1, 2]))
        templates = positive_templates if sentiment == 1 else negative_templates
        base = templates[i % len(templates)]

        # Add minor variation so rows aren't identical
        suffix_pool = [
            "Highly recommended.",
            "Would buy again.",
            "Very pleased overall.",
            "Not worth the price.",
            "Very disappointing.",
            "A solid choice.",
        ]
        text = base + " " + suffix_pool[i % len(suffix_pool)]

        records.append({
            "review_id": f"MOCK_{i:06d}",
            "review_text": text,
            "stars": star,
            "sentiment": sentiment,
        })

    return pd.DataFrame(records)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    raw_df = try_kaggle_download()

    if raw_df is not None:
        # Normalise Kaggle column names (Fine Food Reviews schema)
        col_map = {
            "Text": "review_text",
            "Summary": "summary",
            "Score": "stars",
            "Id": "review_id",
        }
        raw_df = raw_df.rename(columns={k: v for k, v in col_map.items() if k in raw_df.columns})

        if "review_text" not in raw_df.columns:
            print("Expected 'Text' column not found. Falling back to mock data.")
            raw_df = None

    if raw_df is None:
        raw_df = create_mock_data(10_000)

    df = raw_df.copy()

    # Ensure required columns exist
    if "review_id" not in df.columns:
        df["review_id"] = [f"R_{i:06d}" for i in range(len(df))]

    # Convert stars → sentiment (drop neutral 3-star reviews)
    if "sentiment" not in df.columns:
        df["sentiment"] = df["stars"].apply(stars_to_sentiment)

    df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)

    # Clean text
    df["review_text"] = df["review_text"].apply(clean_text)
    df = df[df["review_text"].str.len() > 10]   # drop very short reviews

    # Keep only needed columns
    df = df[["review_id", "review_text", "stars", "sentiment"]].reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)

    pos = (df["sentiment"] == 1).sum()
    neg = (df["sentiment"] == 0).sum()
    print(f"\n✅ Dataset saved to '{OUTPUT_FILE}'")
    print(f"   Total reviews : {len(df):,}")
    print(f"   Positive (1)  : {pos:,}  ({pos/len(df)*100:.1f}%)")
    print(f"   Negative (0)  : {neg:,}  ({neg/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
