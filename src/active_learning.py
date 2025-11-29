# src/active_learning.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from joblib import load

from .text_preprocessing import basic_clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "active_learning"

UNLABELED_PATH = DATA_DIR / "unlabeled_emails.csv"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "spam_classifier.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def load_model_bundle() -> Dict[str, Any]:
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Vectorizer/model not found. Run training first "
            "(python -m src.train_spam_classifier)."
        )
    vectorizer = load(VECTORIZER_PATH)
    model = load(MODEL_PATH)

    best_threshold = None
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        best_threshold = metadata.get("best_threshold", None)

    return {"vectorizer": vectorizer, "model": model, "best_threshold": best_threshold}


def compute_uncertainty(scores: np.ndarray, score_type: str) -> np.ndarray:
    """
    Convert continuous scores to an "uncertainty" measure.

    If we have probabilities p(spam):
      - uncertainty = |p - 0.5|  (smaller = more uncertain)
    If we have raw decision_function scores:
      - we map them through a heuristic: uncertainty = |score| (closer to 0 = more uncertain)
    """
    if score_type == "proba":
        # scores are probabilities in [0,1], shape (n_samples,)
        return np.abs(scores - 0.5)
    # decision_function or any unbounded scores
    return np.abs(scores)


def get_scores(model, X_vec) -> tuple[np.ndarray, str]:
    """
    Get scores and score_type for an arbitrary classifier, similar to get_scores_and_type.
    """
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_vec)[:, 1]
        s_type = "proba"
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_vec)
        s_type = "decision_function"
    else:
        raise ValueError(
            f"Model {type(model).__name__} does not support predict_proba "
            "or decision_function; unsuitable for active learning scoring."
        )
    return scores, s_type


def run_active_learning(top_k: int = 100) -> Path:
    """
    Active learning helper:
      - load unlabeled emails from data/unlabeled_emails.csv (column: text)
      - compute spam score and uncertainty for each
      - select top_k most uncertain samples
      - save them to output/active_learning/uncertain_emails.csv

    These are the rows you should manually label and add back into training data.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not UNLABELED_PATH.exists():
        raise FileNotFoundError(
            f"Unlabeled emails file not found: {UNLABELED_PATH}\n"
            "Create it with at least a 'text' column."
        )

    df = pd.read_csv(UNLABELED_PATH)
    if "text" not in df.columns:
        raise KeyError("Expected column 'text' in unlabeled_emails.csv")

    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["text_clean"] = df["text"].map(basic_clean_text)

    bundle = load_model_bundle()
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    X_vec = vectorizer.transform(df["text_clean"].values)

    scores, score_type = get_scores(model, X_vec)
    uncertainty = compute_uncertainty(scores, score_type)

    # Lower uncertainty => more ambiguous, so sort ascending
    df["score"] = scores
    df["uncertainty"] = uncertainty

    df_sorted = df.sort_values("uncertainty", ascending=True).reset_index(drop=True)
    top_k = min(top_k, df_sorted.shape[0])
    df_top = df_sorted.head(top_k)

    out_path = OUTPUT_DIR / f"uncertain_emails_top_{top_k}.csv"
    df_top.to_csv(out_path, index=False)
    print(f"[AL] Saved top-{top_k} most uncertain emails to {out_path}")

    return out_path


if __name__ == "__main__":
    run_active_learning(top_k=100)
