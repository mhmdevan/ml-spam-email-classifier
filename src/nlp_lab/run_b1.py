from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

from .dataset import load_spam_csv, stratified_split
from .spacy_pipeline import SpacyPreprocessor, SpacyConfig
from .metrics import evaluate_binary_classifier

from .models.tfidf_svm import train_tfidf_svm, predict_proba as tfidf_predict_proba
from .models.st_logreg import train_st_logreg, predict_proba as st_predict_proba
from .models.bert_finetune import fine_tune_bert, BertConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../1.2-spam-classifier
OUT_DIR = PROJECT_ROOT / "output" / "b1"
MODELS_DIR = PROJECT_ROOT / "models" / "b1"


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_models_env(default: List[str]) -> List[str]:
    models_env = os.getenv("B1_MODELS")
    if not models_env:
        return default
    items = [m.strip() for m in models_env.split(",") if m.strip()]
    return items if items else default


def run(
    csv_path: str = str(PROJECT_ROOT / "data" / "spam_emails.csv"),
    spacy_model: str = "en_core_web_sm",
    remove_stopwords: bool = False,
    enable_ner: bool = False,
    models: Optional[List[str]] = None,
    # BERT sampling limits (keep training reasonable on laptop)
    max_train_samples_bert: Optional[int] = 20000,
    max_eval_samples_bert: Optional[int] = 5000,
) -> Path:
    """
    B1 benchmark runner.

    Controls (environment variables):
      - B1_MODELS: comma-separated list from {tfidf_svm, st_logreg, bert}
          Example: B1_MODELS=tfidf_svm,st_logreg

      - B1_SPACY_LEMMA: 1/0 (default 0)
          If 1: TF-IDF baseline uses spaCy lemmatized text (cached).
          If 0: TF-IDF uses raw normalized text (fast).

      - B1_SPACY_NPROC: int (default 1)
      - B1_SPACY_MAXCHARS: int (default 4000)
      - B1_SPACY_BATCH: int (default 64)

      - B1_BERT_TRAIN_MAX: int (override max_train_samples_bert)
      - B1_BERT_EVAL_MAX: int (override max_eval_samples_bert)
    """
    _ensure_dirs()

    if models is None:
        models = ["tfidf_svm", "st_logreg", "bert"]
    models = _parse_models_env(models)

    # ---- Controls via env vars (no CLI yet) ----
    use_spacy_lemma = os.getenv("B1_SPACY_LEMMA", "0") == "1"
    spacy_nproc = int(os.getenv("B1_SPACY_NPROC", "1"))
    spacy_maxchars = int(os.getenv("B1_SPACY_MAXCHARS", "4000"))
    spacy_batch = int(os.getenv("B1_SPACY_BATCH", "64"))

    bert_train_max_env = os.getenv("B1_BERT_TRAIN_MAX")
    bert_eval_max_env = os.getenv("B1_BERT_EVAL_MAX")
    if bert_train_max_env:
        max_train_samples_bert = int(bert_train_max_env)
    if bert_eval_max_env:
        max_eval_samples_bert = int(bert_eval_max_env)

    print("[B1] Loading dataset:", csv_path)
    df = load_spam_csv(Path(csv_path))
    split = stratified_split(df)

    print(f"[B1] Dataset size: {len(df)} | train={len(split.x_train_raw)} test={len(split.x_test_raw)}")
    print(f"[B1] Models: {models}")
    print(f"[B1] spaCy lemma: {'ON' if use_spacy_lemma else 'OFF'} | nproc={spacy_nproc} batch={spacy_batch} maxchars={spacy_maxchars}")

    results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    # ---- Optional spaCy instance for lemma/NER ----
    sp: Optional[SpacyPreprocessor] = None

    # If we need spaCy for lemma OR NER, initialize once.
    if use_spacy_lemma or enable_ner:
        sp = SpacyPreprocessor(
            SpacyConfig(
                model_name=spacy_model,
                remove_stopwords=remove_stopwords,
                enable_ner=enable_ner,
                n_process=spacy_nproc,
                max_chars=spacy_maxchars,
                batch_size=spacy_batch,
                disable_attribute_ruler=True,
            )
        )

    # ---- Optional NER dump (sample) ----
    if enable_ner:
        assert sp is not None
        sample_n = min(200, len(split.x_test_raw))
        print(f"[B1] Extracting NER sample: n={sample_n}")
        ents = sp.extract_ner(split.x_test_raw[:sample_n])
        ner_path = OUT_DIR / "ner_sample.jsonl"
        with ner_path.open("w", encoding="utf-8") as f:
            for i in range(sample_n):
                f.write(json.dumps({"text": split.x_test_raw[i], "entities": ents[i]}, ensure_ascii=False) + "\n")
        print(f"[B1] NER sample saved: {ner_path}")

    # ---- spaCy lemma cache (only if enabled) ----
    if use_spacy_lemma:
        assert sp is not None
        cache_key = (
            f"{csv_path}|{spacy_model}|stop={remove_stopwords}|ner={enable_ner}"
            f"|nproc={spacy_nproc}|maxchars={spacy_maxchars}|batch={spacy_batch}"
        )
        cache_id = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:12]

        cache_train = OUT_DIR / f"lemma_train_{cache_id}.npy"
        cache_test = OUT_DIR / f"lemma_test_{cache_id}.npy"

        if cache_train.exists() and cache_test.exists():
            print(f"[B1] Loading spaCy lemma cache: {cache_id}")
            x_train_feat = np.load(cache_train, allow_pickle=True)
            x_test_feat = np.load(cache_test, allow_pickle=True)
        else:
            print(f"[B1] Building spaCy lemma cache: {cache_id}")
            x_train_feat = np.array(sp.lemmatize_texts(split.x_train_raw), dtype=object)
            x_test_feat = np.array(sp.lemmatize_texts(split.x_test_raw), dtype=object)
            np.save(cache_train, x_train_feat, allow_pickle=True)
            np.save(cache_test, x_test_feat, allow_pickle=True)
            print(f"[B1] Saved spaCy lemma cache: {cache_train.name}, {cache_test.name}")
    else:
        # Fast path: use raw normalized texts from dataset split
        x_train_feat = split.x_train_raw
        x_test_feat = split.x_test_raw

    # ---- TF-IDF + SVM baseline (uses x_*_feat) ----
    if "tfidf_svm" in models:
        print("[B1] Training TFIDF+CalibratedLinearSVC ...")
        bundle = train_tfidf_svm(x_train_feat, split.y_train)
        scores = tfidf_predict_proba(bundle, x_test_feat)

        mb = evaluate_binary_classifier(split.y_test, scores, default_threshold=0.5)
        results["tfidf_svm"] = {
            "input": "spacy_lemma" if use_spacy_lemma else "raw_norm",
            "roc_auc": mb.roc_auc,
            "pr_auc": mb.pr_auc,
            "default": mb.default,
            "tuned": mb.tuned,
        }
        rows.append(
            {
                "model": f"TFIDF+CalibratedLinearSVC ({'spaCy lemma' if use_spacy_lemma else 'raw_norm'})",
                "roc_auc": mb.roc_auc,
                "pr_auc": mb.pr_auc,
                "f1_default": mb.default["f1"],
                "f1_tuned": mb.tuned["f1"],
            }
        )
        print(f"[B1] TFIDF+SVM done | f1(tuned)={mb.tuned['f1']:.4f}")

    # ---- SentenceTransformer embeddings + LogReg ----
    if "st_logreg" in models:
        print("[B1] Training SentenceTransformer + LogReg ...")
        bundle = train_st_logreg(split.x_train_raw, split.y_train)
        scores = st_predict_proba(bundle, split.x_test_raw)

        mb = evaluate_binary_classifier(split.y_test, scores, default_threshold=0.5)
        results["st_logreg"] = {
            "st_model_name": bundle.st_model_name,
            "roc_auc": mb.roc_auc,
            "pr_auc": mb.pr_auc,
            "default": mb.default,
            "tuned": mb.tuned,
        }
        rows.append(
            {
                "model": f"SentenceTransformer({bundle.st_model_name}) + LogReg",
                "roc_auc": mb.roc_auc,
                "pr_auc": mb.pr_auc,
                "f1_default": mb.default["f1"],
                "f1_tuned": mb.tuned["f1"],
            }
        )
        print(f"[B1] ST+LogReg done | f1(tuned)={mb.tuned['f1']:.4f}")

    # ---- Fine-tune DistilBERT ----
    if "bert" in models:
        model_dir = str(MODELS_DIR / "distilbert_spam")
        print("[B1] Fine-tuning DistilBERT ...")
        print(f"[B1] BERT sampling | train_max={max_train_samples_bert} eval_max={max_eval_samples_bert}")

        bert_report = fine_tune_bert(
            split.x_train_raw,
            split.y_train,
            split.x_test_raw,
            split.y_test,
            out_dir=model_dir,
            cfg=BertConfig(),
            max_train_samples=max_train_samples_bert,
            max_eval_samples=max_eval_samples_bert,
        )
        results["bert"] = bert_report
        rows.append(
            {
                "model": "DistilBERT fine-tuned",
                "roc_auc": bert_report["roc_auc"],
                "pr_auc": bert_report["pr_auc"],
                "f1_default": bert_report["default"]["f1"],
                "f1_tuned": bert_report["tuned"]["f1"],
            }
        )
        print(f"[B1] DistilBERT done | f1(tuned)={bert_report['tuned']['f1']:.4f}")

    # ---- Persist results ----
    metrics_path = OUT_DIR / "metrics_b1.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    compare_df = pd.DataFrame(rows)
    if not compare_df.empty:
        compare_df = compare_df.sort_values("f1_tuned", ascending=False)

    compare_path = OUT_DIR / "compare_b1.csv"
    compare_df.to_csv(compare_path, index=False)

    print(f"[B1] Saved metrics: {metrics_path}")
    print(f"[B1] Saved comparison: {compare_path}")

    return compare_path


if __name__ == "__main__":
    out = run()
    print(f"[B1] Done. Compare file: {out}")
