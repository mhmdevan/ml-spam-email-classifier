# 📧 Spam Email Classifier (TF‑IDF + Linear SVM + LogReg + RF + NB)

End‑to‑end **spam vs ham** email/text classifier using **scikit‑learn** and **TF‑IDF**.  
The code is written as a small, production‑like Python package (no Jupyter), ready for GitHub and interview walkthroughs.

This project is not a toy:

- real‑world Kaggle dataset with ~83k labeled emails
- multiple models: **Logistic Regression, Random Forest, Linear SVM, Naive Bayes**
- **Precision‑Recall curves** and **PR‑AUC** for class‑imbalance‑aware evaluation
- **threshold tuning** on the spam score (maximize F1 on spam)
- **CLI predictor**, **FastAPI HTTP API**
- **active learning** helper for manual labeling
- simple **monitoring/drift report**
- **MLflow** experiment tracking (local)
- and a basic **pytest** test suite

---

## 1. Problem & Goal

- **Problem:** Binary classification – decide if a given email/message is **spam** (1) or **ham** (0).
- **Input:** Raw email/SMS text.
- **Output:** A label (`SPAM` / `HAM`), plus a spam score / probability when the model supports it.
- **Business view:**
  - False negatives (spam → ham) = security/phishing risk.
  - False positives (ham → spam) = user frustration, missed important messages.

What this repo demonstrates:

1. Clean **text → TF‑IDF → ML model** pipeline.
2. Comparison of several models on high‑dimensional sparse features.
3. Proper **classification metrics**: Accuracy, Precision, Recall, F1, ROC‑AUC, PR‑AUC, Confusion Matrix.
4. **Threshold tuning** focused on the spam class.
5. Serving the model via **CLI** and **HTTP API**.
6. First steps to **active learning** and **monitoring**.

---

## 2. Dataset (Real Emails, Not Synthetic)

The project uses a **real spam email dataset** from Kaggle:

> **Spam Email Classification Dataset** – P. Singhvi  
> Kaggle: `email-spam-classification-dataset`  

This dataset is referenced in recent research as a real email corpus for spam detection,  
and contains tens of thousands of labeled emails. It is suitable for serious ML experiments, not just a toy sample.

In this project:

- the CSV is stored as:

  ```text
  data/spam_emails.csv
  ```

- typical schema (after normalization):

  - `text` – raw email or message text
  - `label` – `"ham"` or `"spam"`

The original CSV is **not committed** to the repo (Kaggle terms).  
Download it manually and place it in `data/spam_emails.csv`.

Optionally, you can add:

- `data/unlabeled_emails.csv` – raw emails without labels, used by the **active learning** and **monitoring** scripts.

---

## 3. Project Structure

```text
1.2-spam-classifier/
├─ data/
│   ├─ spam_emails.csv                      # labeled Kaggle dataset (not committed)
│   └─ unlabeled_emails.csv                 # optional unlabeled data for active learning / monitoring
├─ models/
│   ├─ spam_classifier.joblib               # best trained model (e.g. calibrated Linear SVM)
│   ├─ tfidf_vectorizer.joblib              # fitted TfidfVectorizer
│   └─ model_metadata.json                  # model name, paths, labels, tuned threshold, n_samples
├─ output/
│   ├─ plots/
│   │   ├─ label_distribution.png           # ham vs spam counts
│   │   ├─ message_length_hist.png          # histogram of email length
│   │   └─ precision_recall_curves.png      # PR curves for all models
│   ├─ metrics/
│   │   └─ classification_metrics.json      # all metrics (default + tuned) for all models
│   ├─ active_learning/
│   │   └─ uncertain_emails_top_10.csv      # most uncertain unlabeled emails to review
│   └─ monitoring/
│       └─ daily_stats.csv                  # simple drift/volume report
├─ src/
│   ├─ __init__.py
│   ├─ text_preprocessing.py                # cleaning: lowercasing, URL/EMAIL/NUM masking, etc.
│   ├─ train_spam_classifier.py             # training + evaluation + PR curves + MLflow logging
│   ├─ predict_spam.py                      # CLI / library prediction (loads model + vectorizer + metadata)
│   ├─ api.py                               # FastAPI app exposing /health and /predict
│   ├─ active_learning.py                   # select most uncertain unlabeled emails for human labeling
│   └─ monitoring_report.py                 # simple monitoring over unlabeled batch
├─ tests/
│   ├─ test_api.py                          # FastAPI TestClient tests (health + spam/ham predictions)
│   ├─ test_predict.py                      # smoke tests for predict_single()
│   ├─ test_predict_cli.py                  # tests CLI output via subprocess
│   └─ test_training.py                     # checks that training produces all expected artifacts
├─ requirements.txt
└─ README.md
```

---

## 4. Installation

### 4.1. Create virtualenv

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### 4.2. Install requirements

```bash
pip install -r requirements.txt
```

If you want to use the **MLflow UI**:

```bash
pip install mlflow
```

---

## 5. Training Pipeline (End‑to‑End)

Run:

```bash
python -m src.train_spam_classifier
```

You should see logs similar to:

```text
[INFO] Loaded data from data/spam_emails.csv
[INFO] Raw shape: 83448 rows, 2 columns
[INFO] Detected schema: text/label
[CLEAN] Dropped 0 rows with missing text/label
[INFO] Final cleaned shape (before split): (83448, 3)
...
[SPLIT] Train size: 66758, Test size: 16690
[SPLIT] Train spam ratio: 0.526, Test spam ratio: 0.526
...
[TRAIN] Fitted LogisticRegression
[TRAIN] Fitted RandomForestClassifier
[TRAIN] Fitted LinearSVM (margin-based)
[TRAIN] Fitted LinearSVM_calibrated with Platt scaling (sigmoid)
[TRAIN] Fitted NaiveBayes (MultinomialNB)
...
[SELECT] Best model by tuned F1: LinearSVM_calibrated
...
[DONE] Training + evaluation (with PR curves & tuned thresholds) completed.
```

### 5.1. Loading & schema normalization

`train_spam_classifier.py`:

- loads `data/spam_emails.csv`
- detects common schemas:
  - `["text", "label"]`
  - `["text", "target"]`
  - `["v1", "v2"]` (SMS spam style)
- normalizes to:

  ```text
  text      -> raw message text
  label     -> "ham" / "spam"
  text_clean -> cleaned text used for modeling
  ```

- drops rows with missing `text`/`label`
- lowercases and normalizes:
  - URLs → `URL`
  - email addresses → `EMAIL`
  - digits → `NUM`
  - strips extra whitespace
- drops rows where `text_clean` becomes empty

### 5.2. Quick EDA plots

Before training, the script generates:

- `output/plots/label_distribution.png` – spam vs ham counts
- `output/plots/message_length_hist.png` – histogram of character lengths

These are meant to be embedded in the README / GitHub:

```markdown
![Label distribution](output/plots/label_distribution.png)
![Message length histogram](output/message_length_hist.png)
```

### 5.3. Train/Test split

- stratified split with `test_size=0.2`
- preserves spam ratio between train and test
- logs sizes and ratios

### 5.4. Vectorization (TF‑IDF)

Uses `TfidfVectorizer` with:

- `ngram_range=(1, 2)` (unigrams + bigrams)
- `max_features=20000`
- `min_df=5`, `max_df=0.95`
- `stop_words="english"`

Crucial design point:

- **fit** on training data only,
- **transform** both train and test with the same fitted vectorizer.

The fitted vectorizer is later saved to:

```text
models/tfidf_vectorizer.joblib
```

### 5.5. Models

The following models are trained on the TF‑IDF features:

1. **LogisticRegression**
   - `class_weight="balanced"`
   - `solver="liblinear"`, `max_iter=1000`
2. **RandomForestClassifier**
   - `n_estimators=300`
   - `class_weight="balanced"`
   - `n_jobs=-1`
3. **LinearSVM** (`LinearSVC`)
   - `class_weight="balanced"`
   - used with `decision_function` scores
4. **LinearSVM_calibrated**
   - `LinearSVC` wrapped in `CalibratedClassifierCV` (Platt scaling / sigmoid)
   - provides calibrated probabilities for spam
5. **NaiveBayes** (`MultinomialNB`)
   - classic probabilistic text classifier

Each model is evaluated on the **same** test split to make metrics comparable.

### 5.6. Metrics, PR curves & threshold tuning

For each model, the script computes:

- **default metrics** at the standard threshold:
  - 0.5 for probability‑based models
  - 0.0 for margin‑based models (decision function)
- **classification metrics** (spam as positive class):
  - Accuracy
  - Precision, Recall, F1
  - ROC‑AUC
  - PR‑AUC (area under Precision‑Recall curve)
  - Confusion matrix
  - Full `classification_report`
- **Precision‑Recall curve** on the test set
- **tuned F1**:
  - scan over thresholds from the PR curve
  - pick the threshold that maximizes **F1** on spam

The curves for all models are combined into:

```text
output/plots/precision_recall_curves.png
```

Embeddable in README:

```markdown
![Precision-Recall curves](output/plots/precision_recall_curves.png)
```

Example log lines:

```text
[EVAL] LogisticRegression      | Default: Acc=0.9853, ... F1=0.9861 | Tuned F1=0.9869 @ thr=0.4311 | ROC-AUC=0.9982 | PR-AUC=0.9981
[EVAL] RandomForest           | Default: Acc=0.9874, ... F1=0.9880 | Tuned F1=0.9882 @ thr=0.4634 | ROC-AUC=0.9984 | PR-AUC=0.9981
[EVAL] LinearSVM              | Default: Acc=0.9898, ... F1=0.9903 | Tuned F1=0.9907 @ thr=0.0125 | ROC-AUC=0.9986 | PR-AUC=0.9983
[EVAL] LinearSVM_calibrated   | Default: Acc=0.9902, ... F1=0.9907 | Tuned F1=0.9908 @ thr=0.4894 | ROC-AUC=0.9987 | PR-AUC=0.9984
[EVAL] NaiveBayes             | Default: Acc=0.9739, ... F1=0.9750 | Tuned F1=0.9809 @ thr=0.3221 | ROC-AUC=0.9968 | PR-AUC=0.9965
```

### 5.7. Model selection & artifacts

- The **selection criterion** is the **best tuned F1** on the spam class.
- In the shown run, the best model is:

  - `LinearSVM_calibrated` (LinearSVC with Platt scaling)

The training script saves:

- `models/tfidf_vectorizer.joblib`
- `models/spam_classifier.joblib` (best model)
- `models/model_metadata.json`:

  ```jsonc
  {
    "best_model_name": "LinearSVM_calibrated",
    "vectorizer_path": "models/tfidf_vectorizer.joblib",
    "model_path": "models/spam_classifier.joblib",
    "labels": {"ham": 0, "spam": 1},
    "n_samples": 83448,
    "best_threshold": 0.4894
  }
  ```

- `output/metrics/classification_metrics.json`:

  ```jsonc
  {
    "best_model_name": "LinearSVM_calibrated",
    "metrics": {
      "LogisticRegression": {
        "default": { ... },
        "tuned":   { ... },
        "roc_auc": 0.9982,
        "pr_auc":  0.9981
      },
      "RandomForest": {
        "...": "..."
      },
      "LinearSVM": {
        "...": "..."
      },
      "LinearSVM_calibrated": {
        "...": "..."
      },
      "NaiveBayes": {
        "...": "..."
      }
    }
  }
  ```

---

## 6. Inference – CLI

Once training is done, you can classify arbitrary text from the command line.

Example:

```bash
python -m src.predict_spam --text "Congratulations, you have won a free iPhone. Click here now!"
```

Example output:

```text
[LOAD] Loaded vectorizer from models/tfidf_vectorizer.joblib
[LOAD] Loaded classifier from models/spam_classifier.joblib

[RESULT]
Predicted label: SPAM (int=1)
Estimated spam probability: 0.987
Threshold used: 0.4894

[DEBUG] Cleaned text preview:
congratulations you have won a free iphone click here now
```

For a ham‑like message:

```bash
python -m src.predict_spam --text "Hi John, here are the slides for tomorrow's meeting."
```

Output:

```text
Predicted label: HAM (int=0)
Estimated spam probability: 0.013
Threshold used: 0.4894
```

You can also use it programmatically:

```python
from src.predict_spam import predict_single

result = predict_single("You have been selected for a FREE cash prize!")
print(result)
# {
#   "label_int": 1,
#   "label_str": "spam",
#   "spam_probability": 0.992,
#   "score": 3.42,               # e.g. decision_function score
#   "threshold_used": 0.4894,
#   "model_name": "LinearSVM_calibrated",
#   "cleaned_text": "you have been selected for a free cash prize"
# }
```

---

## 7. HTTP API (FastAPI)

### 7.1. Start the API

First, train the model:

```bash
python -m src.train_spam_classifier
```

Then run:

```bash
uvicorn src.api:app --reload
```

- API base: <http://127.0.0.1:8000>
- Swagger UI: <http://127.0.0.1:8000/docs>

### 7.2. Endpoints

#### `GET /health`

Health check:

```json
{
  "status": "ok",
  "model_name": "LinearSVM_calibrated",
  "has_vectorizer": true
}
```

(if loading was successful)

#### `POST /predict`

Request:

```json
{
  "text": "Congratulations! You have been selected for a FREE prize. Click here now!"
}
```

Response (example):

```json
{
  "label_int": 1,
  "label_str": "SPAM",
  "spam_probability": 0.987,
  "threshold_used": 0.4894,
  "model_name": "LinearSVM_calibrated",
  "cleaned_text": "congratulations you have been selected for a free prize click here now"
}
```

You can call it with `curl`:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hi, here is the agenda for tomorrow\'s standup."}'
```

---

## 8. Active Learning Helper

The project includes a simple **active learning** helper that picks the most uncertain emails from an unlabeled pool.

Run:

```bash
python -m src.active_learning
```

It will:

1. Load the trained model + vectorizer.
2. Load `data/unlabeled_emails.csv` (you provide this file).
3. Compute an **uncertainty score** for each message (e.g. probability close to the tuned threshold).
4. Select the **top‑K most uncertain** examples (default: 10).
5. Save them to:

   ```text
   output/active_learning/uncertain_emails_top_10.csv
   ```

Log example:

```text
[AL] Saved top-10 most uncertain emails to output/active_learning/uncertain_emails_top_10.csv
```

You can then open that CSV and manually label these borderline cases to improve the dataset in the next training run.

---

## 9. Monitoring / Drift Report

`monitoring_report.py` gives a tiny example of **offline monitoring** on a batch of (unlabeled) emails.

Run:

```bash
python -m src.monitoring_report
```

It will:

1. Load the model/vectorizer.
2. Load `data/unlabeled_emails.csv`.
3. Classify each message as spam/ham.
4. Aggregate **daily statistics**, including things like:
   - date
   - number of messages
   - spam rate
   - average text length
   - changes in average length vs the first day (simple drift indicator)
5. Save to:

   ```text
   output/monitoring/daily_stats.csv
   ```

Example log:

```text
[MON] Saved daily stats to output/monitoring/daily_stats.csv

[MON] Daily stats preview:
         date  ...  avg_length_delta_vs_first
0  2025-11-29  ...                        0.0
```

This is deliberately simple, but it shows how to:

- plug the model into a daily batch job,
- start tracking basic **data drift** signals.

---

## 10. Experiment Tracking with MLflow (Optional)

If `mlflow` is installed, `train_spam_classifier` can log:

- parameters (vectorizer config, model types)
- metrics (acc/prec/recall/F1, ROC‑AUC, PR‑AUC)
- artifacts (plots, metrics JSON)

to a local MLflow tracking directory (e.g. `./mlruns`).

To launch the UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open the address shown in the console (typically <http://127.0.0.1:5000>) and compare runs.

If you don’t care about experiment tracking, you can ignore MLflow, the core pipeline still works.

---

## 11. Testing

Run all tests with:

```bash
python -m pytest
```

You should see something like:

```text
============================ test session starts ============================
platform darwin -- Python 3.11, pytest-9.x
collected 7 items

tests/test_api.py ...
tests/test_predict.py .
tests/test_predict_cli.py ..
tests/test_training.py .

================== 7 passed, 1 warning in XX.XXs ==================
```

High‑level coverage:

- **`test_training.py`**
  - calls the training function on a small sample
  - checks that:
    - vectorizer/model/metadata/metrics files exist
    - metrics JSON has entries for all models
    - tuned threshold is present

- **`test_predict.py`**
  - calls `predict_single()` on spam‑like and ham‑like texts
  - asserts labels and probabilities/scores are in valid ranges

- **`test_predict_cli.py`**
  - runs `python -m src.predict_spam` via `subprocess`
  - checks that CLI prints `Predicted label: SPAM` / `HAM` as expected

- **`test_api.py`**
  - uses `TestClient` from FastAPI
  - hits `/health` and `/predict`, verifies 200 responses and sensible outputs

This puts the project clearly in the **“production‑minded”** camp, not just notebook hacking.

---

## 12. How to Pitch This Project in an Interview

A compact way to explain this repo:

> “I built an end‑to‑end spam email classifier on a real Kaggle dataset (~83k emails).  
> I normalize and clean the text, use TF‑IDF with unigrams/bigrams, and train four models  
> (Logistic Regression, Random Forest, Linear SVM, Naive Bayes). I evaluate all of them  
> with ROC‑AUC and PR‑AUC, then tune the spam threshold using Precision‑Recall curves  
> to maximize F1 on the spam class.  
> The best model (calibrated Linear SVM) is saved along with the TF‑IDF vectorizer and  
> metadata in joblib/JSON, and I expose it via a CLI script and a FastAPI endpoint.  
> On top of that, I added a small active‑learning loop to surface the most uncertain emails  
> for manual labeling, a monitoring script that aggregates daily spam/ham stats and simple  
> drift signals, and a pytest suite to keep the training and prediction pipelines stable.”

---

## 14. License

```text
MIT License

Copyright (c) 2025 Mohammad Eslamnia
```
