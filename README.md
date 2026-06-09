# 💬 Amazon Review Sentiment Analysis

A Natural Language Processing system that classifies Amazon product reviews as **Positive** or **Negative** using a fine-tuned **BERT** transformer model. Includes a real-time interactive web app built with Streamlit.

---

## ✨ Features

- **BERT-powered classification** — fine-tuned `bert-base-uncased` on 50K+ Amazon reviews
- **High accuracy** — achieves ~92% accuracy on the held-out test set *(update with your actual results after training)*
- **Real-time web app** — paste any review and get an instant sentiment prediction with confidence score
- **Batch processing** — analyze thousands of reviews at once via command line
- **Confidence scores** — softmax probabilities give a 0–100% confidence for every prediction
- **Kaggle integration** — auto-downloads the Amazon Fine Food Reviews dataset; falls back to mock data if Kaggle credentials are unavailable

---

## 🛠 Tech Stack

| Component       | Technology                              |
|-----------------|-----------------------------------------|
| Language        | Python 3.10+                            |
| NLP Model       | BERT (`bert-base-uncased`) via HuggingFace |
| Deep Learning   | PyTorch                                 |
| Model Training  | HuggingFace `Trainer` API               |
| Web App         | Streamlit                               |
| Data Processing | pandas, NumPy, scikit-learn             |

---

## 📦 Dataset

| Property   | Detail                                               |
|------------|------------------------------------------------------|
| Source     | [Kaggle — Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) |
| Size       | 568,454 reviews (50K+ after filtering)               |
| Format     | CSV: `review_id`, `review_text`, `stars`, `sentiment` |
| Labels     | Stars 1–2 → Negative (0) · Stars 4–5 → Positive (1) · Star 3 removed |

---

## 📊 Model Performance

> *Run `python src/train.py` and replace the values below with your actual results.*

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~92%   |
| F1 Score  | ~0.92  |
| Precision | ~0.91  |
| Recall    | ~0.93  |

Training setup: `bert-base-uncased`, 3 epochs, batch size 16, learning rate 2e-5, max sequence length 128.

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/amazon-review-sentiment.git
cd amazon-review-sentiment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create project folders (if not already present)
mkdir -p data models results
```

**Optional — Kaggle setup** (for real Amazon data):
```bash
# Place your kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 📖 Usage

### Step 1 — Download / prepare data
```bash
python download_data.py
# Creates: data/amazon_reviews.csv
```

### Step 2 — Train the model
```bash
python src/train.py
# Creates: models/bert_sentiment_model/
#          models/bert_sentiment_tokenizer/
```

### Step 3 — Launch the web app
```bash
streamlit run app.py
# Opens: http://localhost:8501
```

### Step 4 — Batch analysis (optional)
```bash
python analyze_reviews.py
# Creates: results/sentiment_analysis_summary.csv
```

### Step 5 — Test predictions directly
```bash
python src/predict.py
```

---

## ⚙️ How It Works

```
Raw Review Text
      │
      ▼
BERT Tokenizer  ──►  [CLS] token + sub-word tokens + [SEP]  (max 128 tokens)
      │
      ▼
bert-base-uncased (12 transformer layers, 110M parameters)
      │
      ▼
[CLS] embedding  ──►  Linear classification head (768 → 2)
      │
      ▼
Softmax  ──►  P(Positive), P(Negative)
      │
      ▼
Label: Positive / Negative  +  Confidence Score
```

1. **Tokenization** — the BERT tokenizer splits each review into sub-word tokens and prepends a `[CLS]` token.
2. **Forward pass** — all layers process the token sequence and the `[CLS]` embedding captures global context.
3. **Classification head** — a single linear layer maps the 768-dim embedding to 2 logits.
4. **Softmax** — converts logits into probabilities; the higher probability class is the prediction.

---

## 📁 Project Structure

```
amazon-review-sentiment/
│
├── download_data.py        # Download/generate dataset
├── app.py                  # Streamlit web application
├── analyze_reviews.py      # Batch analysis script
├── requirements.txt        # Python dependencies
├── README.md
│
├── src/
│   ├── train.py            # BERT fine-tuning script
│   └── predict.py          # Inference functions
│
├── data/
│   └── amazon_reviews.csv  # Processed dataset (generated)
│
├── models/
│   ├── bert_sentiment_model/      # Saved model weights
│   └── bert_sentiment_tokenizer/  # Saved tokenizer
│
└── results/
    ├── sentiment_analysis_summary.csv
    └── summary_stats.csv
```

---

## 💡 Example Usage

**Input review:**
```
This is absolutely the best blender I have ever owned. Super powerful,
easy to clean, and arrived two days early. Highly recommend!
```

**Output:**
```
Sentiment  : ✅ POSITIVE
Confidence : 97.3%
```

---

**Input review:**
```
Terrible quality. Broke after one use and the seller refused to refund me.
Complete waste of money. Do not buy.
```

**Output:**
```
Sentiment  : ❌ NEGATIVE
Confidence : 98.1%
```

---


## 👤 Author

**[Garvit Joshi]**  
B.Tech Computer Science | Machine Learning Enthusiast  
🔗 GitHub: [Garvitj48](https://github.com/Garvitj48)
