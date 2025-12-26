# 📧 Spam Email Classifier → NLP / Transformers / LLM Lab (v1 + B1–B4)

End‑to‑end **spam vs ham** email/text classification project that starts with a brutally strong classical baseline (**TF‑IDF + Linear SVM**) and evolves into a modern NLP lab:
**spaCy preprocessing**, **Sentence Transformers embeddings**, **DistilBERT fine‑tuning**, **PEFT/LoRA**, **ONNX export + FastAPI serving**, and **LLM orchestration with LangGraph + n8n**.

This repo is intentionally written like a small production package (no notebooks) and is meant to be walked through in an interview:

- **What problem?** spam detection as a risk trade‑off.
- **What did you try?** classical vs embeddings vs fine‑tuned transformers vs PEFT.
- **Why those choices?** accuracy vs cost vs latency vs deployment.
- **How did you evaluate?** ROC‑AUC, PR‑AUC, F1, threshold tuning, confusion matrix.
- **What broke?** missing deps, slow spaCy, ONNX export API mismatch/opset, Apple MPS quirks.
- **How did you fix it?** caching, multiprocessing, truncation, stable ONNX export path, backend switching.

---

## 0) What’s inside (high level)

### ✅ v1 (classic, production‑minded)

- Kaggle spam dataset (~83k)
- TF‑IDF vectorizer + multiple models:
  - Logistic Regression, Random Forest, Linear SVM, Naive Bayes
- Metrics: Accuracy, Precision, Recall, F1, ROC‑AUC, **PR‑AUC**
- **threshold tuning** to maximize F1 on spam
- CLI predictor
- FastAPI HTTP API
- Active learning helper (uncertainty sampling)
- Monitoring / drift report (simple)
- MLflow local tracking (optional)
- pytest test suite

### ✅ B1 (modernize NLP + compare model families)

- spaCy integration (tokenization/lemmatization/stopwords optional, caching, multiprocess)
- Sentence Transformers → embeddings → LogReg (fast semantic baseline)
- DistilBERT fine‑tuning for spam/ham
- Compare vs TF‑IDF+SVM baseline under same split & metrics

### ✅ B2 (PEFT / LoRA)

- Linear probe variant (cheap adaptation)
- LoRA variant (parameter‑efficient fine‑tune)
- Saved model artifacts

### ✅ B3 (Serving + ONNX + latency benchmark)

- Export transformer models to ONNX (Optimum)
- Export sklearn baseline as joblib
- FastAPI serving that can switch between **onnx** and **sklearn**
- Smoke tests + latency benchmark

### ✅ B4 (LLM orchestration + automation)

- Orchestrator service: spam detection → if spam → redact/truncate → LLM categorization/summary
- LangGraph flow + n8n workflow integration blueprint

---

## 1) Problem & Goal

- **Problem:** Binary classification – decide if a given email/message is **spam** (1) or **ham** (0).
- **Input:** Raw email/SMS text.
- **Output:** A label (`spam` / `ham`) + spam score/probability + decision threshold.

### Business view

- **False negatives** (spam → ham) = phishing & security risk.
- **False positives** (ham → spam) = user frustration, missed important messages.

So I explicitly track:

- **PR‑AUC** (more meaningful under class imbalance)
- **threshold tuning** (maximize F1 on the spam class or enforce a recall target)

---

## 2) Dataset (Real Emails, Not Synthetic)

Uses a real spam email dataset (~83k labeled samples). Put it here:

```text
data/spam_emails.csv
```

Typical normalized schema (after normalization step in code):

- `text` – raw email/message body
- `label` – `"ham"` or `"spam"`

The raw CSV is not committed (Kaggle terms). Download manually and place in `data/spam_emails.csv`.

Optional:

- `data/unlabeled_emails.csv` — unlabeled pool for active learning / monitoring.

---

## 3) Project Structure (FULL: v1 + B1–B4)

> You complained I “forgot the old version” in the map. Here it is — **everything**: old v1 scripts + new B‑phases.

```text
spam-classifier/
├─ data/
│  ├─ spam_emails.csv                          # Kaggle dataset (not committed)
│  └─ unlabeled_emails.csv                     # optional (active learning / monitoring)
│
├─ models/
│  # ===== v1 artifacts (classic pipeline) =====
│  ├─ spam_classifier.joblib                   # best classical model
│  ├─ tfidf_vectorizer.joblib                  # fitted vectorizer
│  └─ model_metadata.json                      # best model name, threshold, paths, etc.
│
│  # ===== B-phase artifacts =====
│  ├─ b1/
│  │  └─ distilbert_spam/                      # fine-tuned DistilBERT (HF format)
│  ├─ b2/
│  │  ├─ linear_probe_distilbert-base-uncased/  # saved model (full weights)
│  │  └─ lora_distilbert-base-uncased/          # PEFT adapter (and meta)
│  └─ b3/
│     ├─ sklearn_tfidf_svm/
│     │  ├─ model.joblib
│     │  └─ meta.json
│     ├─ onnx_distilbert_ft_opset18/
│     │  ├─ model.onnx
│     │  ├─ tokenizer.json / vocab files
│     │  └─ meta.json
│     └─ onnx_distilbert_lora_merged_opset18/
│        ├─ model.onnx
│        ├─ tokenizer.json / vocab files
│        └─ meta.json
│
├─ output/
│  # ===== v1 outputs =====
│  ├─ plots/
│  │  ├─ label_distribution.png
│  │  ├─ message_length_hist.png
│  │  └─ precision_recall_curves.png
│  ├─ metrics/
│  │  └─ classification_metrics.json
│  ├─ active_learning/
│  │  └─ uncertain_emails_top_10.csv
│  └─ monitoring/
│     └─ daily_stats.csv
│
│  # ===== B-phase outputs =====
│  ├─ b1/
│  │  ├─ metrics_b1.json
│  │  └─ compare_b1.csv
│  ├─ b2/
│  │  ├─ metrics_b2.json
│  │  └─ compare_b2.csv
│  └─ b3/
│     └─ latency.csv
│
├─ src/
│  # ===== v1 classic pipeline =====
│  ├─ __init__.py
│  ├─ text_preprocessing.py                    # URL/EMAIL/NUM masking, cleanup
│  ├─ train_spam_classifier.py                 # training + evaluation + PR curves + MLflow
│  ├─ predict_spam.py                          # CLI / library prediction
│  ├─ api.py                                   # FastAPI app (v1) exposing /health + /predict
│  ├─ active_learning.py                       # uncertainty sampling for unlabeled pool
│  └─ monitoring_report.py                     # drift/volume stats over unlabeled batch
│
│  # ===== B1/B2 NLP lab =====
│  ├─ nlp_lab/
│  │  ├─ run_b1.py                             # orchestrates B1 experiments
│  │  ├─ run_b2.py                             # orchestrates B2 experiments
│  │  ├─ spacy_pipeline.py                     # spaCy preprocessor + caching
│  │  └─ models/
│  │     ├─ tfidf_svm.py                        # baseline wrapper (B1)
│  │     ├─ st_logreg.py                        # sentence-transformer embeddings + LogReg
│  │     ├─ bert_finetune.py                    # DistilBERT fine-tune
│  │     └─ peft_lora.py                        # PEFT/LoRA training (B2)
│
│  # ===== B3 serving (multi-backend) =====
│  └─ serving/
│     ├─ app.py                                # FastAPI app (B3) switchable backend
│     ├─ backends.py                           # ONNX + sklearn runtime
│     ├─ export_sklearn.py                      # export joblib bundle for serving
│     ├─ export_to_onnx.py                      # export to ONNX via optimum-cli
│     ├─ smoke_onnx.py                          # validates ONNX runtime
│     └─ benchmark_latency.py                   # throughput/latency benchmark
│
│  # ===== B4 orchestrator (LLM + LangGraph + n8n) =====
│  └─ orchestrator/
│     ├─ app.py                                # FastAPI triage service
│     ├─ llm_graph.py                           # LangGraph definition
│     ├─ redact.py                              # PII redact + truncation
│     ├─ schemas.py                             # Pydantic request/response models
│     └─ db.py                                  # optional persistence layer
│
├─ tests/
│  ├─ test_api.py
│  ├─ test_predict.py
│  ├─ test_predict_cli.py
│  └─ test_training.py
│
├─ requirements.txt
└─ README.md
```

---

## 4) Installation

### 4.1 Create virtualenv

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install -U pip
```

### 4.2 Install v1 requirements (classic)

```bash
pip install -r requirements.txt
```

If your `requirements.txt` only had these older packages:

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
pytest
mlflow
pika
httpx
```

### 4.3 Install B1/B2/B3 extras (modern NLP / Transformers / ONNX)

```bash
# spaCy + language detect
pip install spacy langdetect tqdm
python -m spacy download en_core_web_sm

# transformers stack
pip install "transformers>=4.40" "datasets>=2.18" "accelerate>=0.28" "torch"
pip install sentence-transformers

# PEFT / LoRA
pip install peft

# ONNX export/runtime via Optimum
pip install "optimum[onnxruntime]" onnxruntime
```

If you want cleaner ONNX tooling:

```bash
pip install onnx onnxruntime-tools
```

---

## 5) v1 Training Pipeline (End‑to‑End, TF‑IDF + Models)

Run:

```bash
python -m src.train_spam_classifier
```

You should see logs similar to:

```text
[INFO] Loaded data from data/spam_emails.csv
[INFO] Raw shape: 83448 rows, 2 columns
[INFO] Detected schema: text/label
...
[SPLIT] Train size: 66758, Test size: 16690
...
[TRAIN] Fitted LogisticRegression
[TRAIN] Fitted RandomForestClassifier
[TRAIN] Fitted LinearSVM (margin-based)
[TRAIN] Fitted LinearSVM_calibrated (Platt scaling)
[TRAIN] Fitted NaiveBayes (MultinomialNB)
...
[SELECT] Best model by tuned F1: LinearSVM_calibrated
[DONE] Saved models + plots + metrics JSON
```

### 5.1 Loading & schema normalization

`src/train_spam_classifier.py` handles common Kaggle/SMS schemas:

- `["text", "label"]`
- `["text", "target"]`
- `["v1", "v2"]` (SMS spam datasets)

Normalizes into:

- `text` (raw)
- `label` (`ham`/`spam`)
- `text_clean` (cleaned)

### 5.2 Preprocessing (v1)

`src/text_preprocessing.py` typically includes:

- lowercase
- normalize URLs → `URL`
- normalize emails → `EMAIL`
- digits → `NUM`
- strip extra whitespace
- remove empty messages

### 5.3 TF‑IDF vectorization

Key idea: fit on train only, transform train/test with same vectorizer.
Typical config:

- ngrams (1,2)
- `max_features`
- `min_df`, `max_df`
- stopwords

### 5.4 Metrics & threshold tuning

Compute:

- Accuracy, Precision, Recall, F1
- ROC‑AUC, **PR‑AUC**
- confusion matrix
- PR curve
- threshold scan to maximize F1 on spam (or set recall constraint)

Artifacts:

- `output/metrics/classification_metrics.json`
- `output/plots/precision_recall_curves.png`

### 5.5 Saved artifacts (v1)

- `models/tfidf_vectorizer.joblib`
- `models/spam_classifier.joblib`
- `models/model_metadata.json` (includes best threshold)

---

## 6) v1 Inference – CLI

Example:

```bash
python -m src.predict_spam --text "Congratulations, you have won a free iPhone. Click here now!"
```

Programmatic usage:

```python
from src.predict_spam import predict_single

res = predict_single("You have been selected for a FREE cash prize!")
print(res["label_str"], res["spam_probability"], res["threshold_used"])
```

---

## 7) v1 HTTP API (FastAPI)

Start v1 API:

```bash
uvicorn src.api:app --reload
```

- Swagger: <http://127.0.0.1:8000/docs>
- `GET /health`
- `POST /predict`

---

## 8) v1 Active Learning Helper

Run:

```bash
python -m src.active_learning
```

Uses `data/unlabeled_emails.csv`, picks most uncertain messages, writes:

```text
output/active_learning/uncertain_emails_top_10.csv
```

---

## 9) v1 Monitoring / Drift Report

Run:

```bash
python -m src.monitoring_report
```

Writes:

```text
output/monitoring/daily_stats.csv
```

---

## 10) MLflow tracking (Optional, v1)

If enabled, training logs params/metrics/artifacts to `./mlruns`.
Launch UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

---

# ===========================

# B‑PHASES (B1 → B4)

# ===========================

## 11) B1 — Modern NLP Experiments (spaCy + Sentence Transformers + DistilBERT)

B1 is a comparison lab using the same dataset but different model families.

### 11.1 B1 Baseline (TF‑IDF + Calibrated LinearSVC)

```bash
B1_MODELS=tfidf_svm B1_SPACY_LEMMA=0 python -m src.nlp_lab.run_b1
```

Your observed run example:

- `f1(tuned)=0.9916`

### 11.2 SentenceTransformer embeddings + Logistic Regression

```bash
B1_MODELS=tfidf_svm,st_logreg B1_SPACY_LEMMA=0 python -m src.nlp_lab.run_b1
```

Your observed run example:

- `st_logreg f1(tuned)=0.9550` (baseline still wins on this dataset)

### 11.3 spaCy lemmatization (when you want it, and why it can be slow)

The spaCy path is heavier. You solved “it looks stuck” by:

- enabling multiprocessing
- batching
- truncation
- caching lemma arrays

Run:

```bash
B1_MODELS=tfidf_svm \
B1_SPACY_LEMMA=1 \
B1_SPACY_NPROC=4 \
B1_SPACY_MAXCHARS=3000 \
python -m src.nlp_lab.run_b1
```

You also saw:
`[W108] rule-based lemmatizer did not find POS annotation ...`
Meaning: lemmatizer expects POS tags. If lemma doesn’t help your metric, keep it OFF (your baseline didn’t need it).

### 11.4 DistilBERT fine‑tuning

You already run this with sampling to keep it practical:

```bash
B1_MODELS=bert \
B1_BERT_TRAIN_MAX=20000 \
B1_BERT_EVAL_MAX=5000 \
python -m src.nlp_lab.run_b1
```

Artifacts:

- `output/b1/metrics_b1.json`
- `output/b1/compare_b1.csv`
- `models/b1/distilbert_spam/`

---

## 12) B2 — PEFT / LoRA (parameter‑efficient adaptation)

### 12.1 Linear probe

```bash
B2_VARIANT=linear_probe \
B2_TRAIN_MAX=5000 \
B2_EVAL_MAX=2000 \
python -m src.nlp_lab.run_b2
```

Sample result you produced:

- trainable params ≈ 592k / 66.9M
- `f1_tuned ≈ 0.9445`

### 12.2 LoRA

```bash
B2_VARIANT=lora \
B2_TRAIN_MAX=5000 \
B2_EVAL_MAX=2000 \
B2_EPOCHS=2 \
B2_LR=2e-4 \
B2_LORA_R=16 \
B2_LORA_ALPHA=32 \
B2_LORA_DROPOUT=0.05 \
python -m src.nlp_lab.run_b2
```

Sample result you produced:

- trainable params ≈ 1.18M / 68.1M
- `f1_tuned ≈ 0.9728`

Artifacts:

- `output/b2/metrics_b2.json`
- `output/b2/compare_b2.csv`
- `models/b2/...`

---

## 13) B3 — Serving + ONNX + Multi‑backend FastAPI + Benchmark

### 13.1 Export sklearn baseline

```bash
python -m src.serving.export_sklearn
```

### 13.2 Export transformer to ONNX

You hit an Optimum API mismatch (`opset` argument) and solved it using `optimum-cli` path inside the exporter.

Export fine‑tuned DistilBERT:

```bash
python -m src.serving.export_to_onnx \
  --source models/b1/distilbert_spam \
  --out models/b3/onnx_distilbert_ft_opset18 \
  --threshold 0.5 \
  --opset 18 \
  --task text-classification \
  --device cpu
```

Export LoRA model (merged for export):

```bash
python -m src.serving.export_to_onnx \
  --source models/b2/lora_distilbert-base-uncased \
  --out models/b3/onnx_distilbert_lora_merged_opset18 \
  --threshold 0.5 \
  --opset 18 \
  --task text-classification \
  --device cpu
```

### 13.3 Smoke test ONNX

```bash
python -m src.serving.smoke_onnx --model-dir models/b3/onnx_distilbert_lora_merged_opset18
```

### 13.4 Serve (B3 API) — choose backend

ONNX backend:

```bash
B3_BACKEND=onnx B3_MODEL_DIR=models/b3/onnx_distilbert_ft_opset18 \
uvicorn src.serving.app:app --host 127.0.0.1 --port 8000 --reload
```

sklearn backend:

```bash
B3_BACKEND=sklearn B3_MODEL_DIR=models/b3/sklearn_tfidf_svm \
uvicorn src.serving.app:app --host 127.0.0.1 --port 8000 --reload
```

Test:

```bash
curl -s http://127.0.0.1:8000/health ; echo
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts":["free money click now!!!","hey can we reschedule the meeting?"]}' ; echo
```

### 13.5 Benchmark latency

```bash
python -m src.serving.benchmark_latency
```

Writes:

```text
output/b3/latency.csv
```

---

## 14) B4 — LLM Orchestration + n8n Automation (LangGraph)

For now you said: “skip eval because RAM is huge”. B4 does not require big eval; it’s system integration.

### Goal

Turn “spam probability” into a workflow:

- If spam → decide subtype (phishing/scam/promo/malware)
- Produce short explanation + recommended action
- Save to DB / send notification

### Core design

1. **Classifier service** (B3) predicts `spam_prob` and `is_spam`
2. If spam, call **Triage Orchestrator** (B4):
   - redact PII (emails/phones/IDs/card-like patterns)
   - truncate long bodies
   - call LLM via LangGraph with a structured output schema
3. Persist results or push to downstream workflow

### Run orchestrator (example)

```bash
uvicorn src.orchestrator.app:app --host 127.0.0.1 --port 8010 --reload
```

### Example triage request

```bash
curl -s -X POST http://127.0.0.1:8010/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "source":"manual",
    "message_id":"demo-1",
    "from_addr":"promo@example.com",
    "subject":"WIN a gift card NOW",
    "received_at":"2025-12-24T00:00:00Z",
    "text":"Congratulations! You have won. Click this link to claim.",
    "spam_prob":0.99,
    "is_spam":true,
    "spam_model":"onnx_distilbert_ft_opset18"
  }' ; echo
```

### n8n workflow blueprint (practical)

Nodes (typical):

1. **IMAP Email Trigger** (or Gmail node)
2. **HTTP Request** → call `POST http://classifier/predict`
3. **IF** node: `is_spam == true`
4. If true: **HTTP Request** → call `POST http://triage/v1/triage`
5. **Postgres** node (save triage result) OR **Telegram/Email** node (notify)

This is the “system” story that turns a model into automation.

---

## 15) Testing

Run:

```bash
python -m pytest
```

---

## 16) How to pitch this in an interview (script you can actually say)

> “I built an end‑to‑end spam classifier on a real Kaggle corpus (~83k emails).  
> I started with TF‑IDF + linear models because they’re fast and surprisingly strong for spam. I evaluated with PR‑AUC, ROC‑AUC, F1, and tuned thresholds to maximize spam F1.  
> Then I built an NLP lab comparing sentence‑transformer embeddings + logreg and DistilBERT fine‑tuning, and finally added PEFT/LoRA to show parameter‑efficient adaptation.  
> For deployment I created a FastAPI service with swappable backends: sklearn for ultra‑fast throughput and ONNX for transformer portability, plus a benchmark script.  
> Finally I added a LangGraph + n8n orchestration layer that turns spam predictions into automated triage and notifications.”

---

## 17) License

MIT License

Copyright (c) 2025 Mohammad Eslamnia
