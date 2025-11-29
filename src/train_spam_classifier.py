# src/train_spam_classifier.py

from __future__ import annotations

import json
import mlflow
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from .text_preprocessing import basic_clean_text

# -----------------------------
# 1. Global paths / constants
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "spam_emails.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"


def ensure_dirs() -> None:
    """Create models / output directories if they do not exist."""
    for path in (MODELS_DIR, PLOTS_DIR, METRICS_DIR):
        path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 2. Data loading / preparation
# -----------------------------

def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """
    Load raw spam dataset from CSV.

    Supported schemas:
      - v1 (label), v2 (text)                 # e.g. SMS Spam Collection
      - text, target                          # e.g. Spam Assassin dataset
      - text, label
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded data from {csv_path}")
    print(f"[INFO] Raw shape: {df.shape[0]} rows, {df.shape[1]} columns")

    cols = set(df.columns)

    if {"v1", "v2"}.issubset(cols):
        # Kaggle SMS dataset style
        df = df.rename(columns={"v1": "label", "v2": "text"})
        print("[INFO] Detected schema: v1/v2 → renamed to label/text")
    elif {"text", "target"}.issubset(cols):
        df = df.rename(columns={"target": "label"})
        print("[INFO] Detected schema: text/target → renamed to label")
    elif {"text", "label"}.issubset(cols):
        print("[INFO] Detected schema: text/label")
    else:
        raise ValueError(
            "Unsupported CSV schema. Expected one of: "
            "[v1,v2], [text,target], [text,label]. Columns found: "
            f"{list(df.columns)}"
        )

    df = df[["text", "label"]].copy()

    # Drop rows with missing text or label
    before = df.shape[0]
    df = df.dropna(subset=["text", "label"])
    after = df.shape[0]
    print(f"[CLEAN] Dropped {before - after} rows with missing text/label")

    # Normalize labels to 0 (ham) / 1 (spam)
    if df["label"].dtype == object:
        df["label"] = (
            df["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda x: 1 if x == "spam" else 0)
        )
    else:
        df["label"] = df["label"].astype(int)

    # Filter any weird labels outside {0,1}
    valid_mask = df["label"].isin([0, 1])
    removed = (~valid_mask).sum()
    if removed > 0:
        print(f"[CLEAN] Removed {removed} rows with non-binary labels")
    df = df.loc[valid_mask].reset_index(drop=True)

    # Basic text cleaning
    df["text"] = df["text"].astype(str)
    df["text_clean"] = df["text"].map(basic_clean_text)

    # Remove empty-cleaned texts
    empty_mask = df["text_clean"].str.strip() == ""
    empty_count = empty_mask.sum()
    if empty_count > 0:
        df = df.loc[~empty_mask].reset_index(drop=True)
        print(f"[CLEAN] Removed {empty_count} rows with empty cleaned text")

    print(f"[INFO] Final cleaned shape (before split): {df.shape}")
    return df


def train_test_split_text(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split cleaned dataframe into train/test on text_clean and label (stratified).
    """
    X = df["text_clean"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"[SPLIT] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(
        f"[SPLIT] Train spam ratio: {y_train.mean():.3f}, "
        f"Test spam ratio: {y_test.mean():.3f}"
    )
    return X_train, X_test, y_train, y_test


# -----------------------------
# 3. TF-IDF vectorization
# -----------------------------

def fit_vectorizer(X_train: np.ndarray) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on the training texts only.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=5,
        max_df=0.95,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    print(
        "[VEC] TF-IDF fitted on training data. "
        f"Shape: {X_train_vec.shape[0]} samples × {X_train_vec.shape[1]} features"
    )
    return vectorizer


# -----------------------------
# 4. Models
# -----------------------------

def train_models(
    X_train_vec,
    y_train,
) -> Dict[str, Any]:
    """
    Train multiple models on the same vectorized training data:
      - Logistic Regression
      - Random Forest
      - Linear SVM (margin-based)
      - Linear SVM (probability-calibrated via Platt scaling)
      - Naive Bayes (MultinomialNB)
    Returns a dict with fitted models.
    """
    models: Dict[str, Any] = {}

    # 1) Logistic Regression
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )
    log_reg.fit(X_train_vec, y_train)
    models["LogisticRegression"] = log_reg
    print("[TRAIN] Fitted LogisticRegression")

    # 2) Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf_clf.fit(X_train_vec, y_train)
    models["RandomForest"] = rf_clf
    print("[TRAIN] Fitted RandomForestClassifier")

    # 3) Linear SVM (margin only)
    svm_clf = LinearSVC(
        class_weight="balanced",
        random_state=42,
    )
    svm_clf.fit(X_train_vec, y_train)
    models["LinearSVM"] = svm_clf
    print("[TRAIN] Fitted LinearSVM (margin-based)")

     # 4) Linear SVM + probability calibration (Platt / sigmoid)
    svm_base = LinearSVC(
        class_weight="balanced",
        random_state=42,
    )

    # NOTE:
    # In newer scikit-learn versions the argument is called `estimator`
    # (old name was `base_estimator` and now raises a TypeError).
    svm_calibrated = CalibratedClassifierCV(
        estimator=svm_base,
        method="sigmoid",   # Platt scaling; you could also try "isotonic"
        cv=3,
    )
    svm_calibrated.fit(X_train_vec, y_train)
    models["LinearSVM_calibrated"] = svm_calibrated
    print("[TRAIN] Fitted LinearSVM_calibrated with Platt scaling (sigmoid)")
    
    # 5) Naive Bayes
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train_vec, y_train)
    models["NaiveBayes"] = nb_clf
    print("[TRAIN] Fitted NaiveBayes (MultinomialNB)")

    return models


# -----------------------------
# 5. Helpers for scores & metrics
# -----------------------------

def get_scores_and_type(model, X) -> Tuple[np.ndarray, str]:
    """
    Get continuous scores for each sample and identify score type.

    If the model supports predict_proba, we use probabilities for class 1 (spam).
    Else if it supports decision_function, we use that.
    """
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X)[:, 1]
        score_type = "proba"
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X)
        score_type = "decision_function"
    else:
        raise ValueError(
            f"Model {type(model).__name__} does not support predict_proba "
            "or decision_function; cannot compute PR/ROC curves."
        )
    return y_scores, score_type


def compute_default_threshold(score_type: str) -> float:
    """
    Default decision threshold depending on score type.
      - probabilities: 0.5
      - decision_function: 0.0
    """
    if score_type == "proba":
        return 0.5
    return 0.0


def predict_with_threshold(y_scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert continuous scores to binary predictions using a threshold.
    """
    return (y_scores >= threshold).astype(int)


def tune_threshold_via_pr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, Any]:
    """
    Search for the best decision threshold using the precision-recall curve,
    optimizing F1 score on the spam class (label=1).

    We use sklearn.precision_recall_curve to obtain:
      precisions, recalls, thresholds

    thresholds.shape = (n_points - 1,)
    precisions/recalls.shape = (n_points,)

    For each threshold i:
      - threshold_i = thresholds[i]
      - precision_i = precisions[i+1]
      - recall_i = recalls[i+1]

    We pick the threshold that maximizes F1.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    best_idx: Optional[int] = None
    best_f1: float = -1.0
    best_threshold: float = 0.5
    best_precision: float = 0.0
    best_recall: float = 0.0
    best_accuracy: float = 0.0
    best_cm: Optional[np.ndarray] = None

    if thresholds.size == 0:
        # Degenerate case: all scores the same; fall back to default 0.5
        default_pred = predict_with_threshold(y_scores, 0.5)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            default_pred,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
        acc = accuracy_score(y_true, default_pred)
        cm = confusion_matrix(y_true, default_pred, labels=[0, 1])
        return {
            "threshold": 0.5,
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
        }

    # thresholds[i] corresponds to precisions[i+1], recalls[i+1]
    for i, thr in enumerate(thresholds):
        p = precisions[i + 1]
        r = recalls[i + 1]
        if (p + r) == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_idx = i
            best_threshold = float(thr)
            best_precision = float(p)
            best_recall = float(r)

    # Compute accuracy and confusion at best_threshold
    y_pred_best = predict_with_threshold(y_scores, best_threshold)
    best_accuracy = float(accuracy_score(y_true, y_pred_best))
    best_cm = confusion_matrix(y_true, y_pred_best, labels=[0, 1])

    return {
        "threshold": best_threshold,
        "accuracy": best_accuracy,
        "precision": best_precision,
        "recall": best_recall,
        "f1": float(best_f1),
        "confusion_matrix": best_cm.tolist(),
    }


def evaluate_model(
    name: str,
    model,
    y_test: np.ndarray,
    y_scores: np.ndarray,
    score_type: str,
) -> Dict[str, Any]:
    """
    Compute metrics for a single classifier on the test set.

    We evaluate two operating points:
      - "default": fixed threshold (0.5 for probabilities, 0.0 for scores)
      - "tuned": best threshold from precision-recall curve (maximizing F1)

    Metrics:
      - accuracy
      - precision / recall / f1 (binary, spam = 1)
      - roc_auc (if scores monotonic wrt spam)
      - pr_auc (Average Precision)
      - confusion_matrix for default and tuned
      - classification_report for default point
    """
    default_threshold = compute_default_threshold(score_type)
    y_pred_default = predict_with_threshold(y_scores, default_threshold)

    # Default metrics
    acc = accuracy_score(y_test, y_pred_default)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred_default,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    cm_default = confusion_matrix(y_test, y_pred_default, labels=[0, 1])

    cls_report = classification_report(
        y_test,
        y_pred_default,
        labels=[0, 1],
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    )

    # ROC-AUC and PR-AUC use continuous scores
    try:
        roc_auc = roc_auc_score(y_test, y_scores)
    except ValueError:
        roc_auc = None

    try:
        pr_auc = average_precision_score(y_test, y_scores)
    except ValueError:
        pr_auc = None

    # Threshold tuning via PR curve
    tuned = tune_threshold_via_pr(y_test, y_scores)

    metrics: Dict[str, Any] = {
        "model_name": name,
        "score_type": score_type,
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
        "default": {
            "threshold": float(default_threshold),
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm_default.tolist(),
            "classification_report": cls_report,
        },
        "tuned": tuned,
    }

    print(
        f"[EVAL] {name} | "
        f"Default: Acc={acc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f} | "
        f"Tuned F1={tuned['f1']:.4f} @ thr={tuned['threshold']:.4f} | "
        f"ROC-AUC={metrics['roc_auc']} | PR-AUC={metrics['pr_auc']}"
    )

    return metrics


def pick_best_model(
    models: Dict[str, Any],
    y_test: np.ndarray,
    scores_by_model: Dict[str, np.ndarray],
    score_types: Dict[str, str],
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Evaluate all models and return:
      - best model name (by tuned F1 on spam class)
      - full metrics bundle for all models
    """
    all_metrics: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        y_scores = scores_by_model[name]
        score_type = score_types[name]
        metrics = evaluate_model(name, model, y_test, y_scores, score_type)
        all_metrics[name] = metrics

    def f1_tuned(m_name: str) -> float:
        tuned = all_metrics[m_name].get("tuned")
        if tuned is None:
            return all_metrics[m_name]["default"]["f1"]
        return float(tuned["f1"])

    best_name = max(all_metrics.keys(), key=f1_tuned)
    print(f"[SELECT] Best model by tuned F1: {best_name}")

    return best_name, all_metrics


# -----------------------------
# 6. Plots (EDA + PR curves)
# -----------------------------

def plot_label_distribution(labels: np.ndarray) -> None:
    """
    Bar plot: ham vs spam counts.
    """
    ham_count = int((labels == 0).sum())
    spam_count = int((labels == 1).sum())

    plt.figure(figsize=(5, 4))
    plt.bar(["ham", "spam"], [ham_count, spam_count])
    plt.title("Label distribution (ham vs spam)")
    plt.ylabel("Count")
    plt.tight_layout()

    out_path = PLOTS_DIR / "label_distribution.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"[PLOT] Saved label distribution to {out_path}")


def plot_message_length_hist(lengths: np.ndarray) -> None:
    """
    Histogram of message lengths (characters).
    """
    plt.figure(figsize=(7, 4))
    plt.hist(lengths, bins=50, edgecolor="black")
    plt.title("Message length distribution (characters)")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.tight_layout()

    out_path = PLOTS_DIR / "message_length_hist.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"[PLOT] Saved message length histogram to {out_path}")


def plot_pr_curves(
    y_test: np.ndarray,
    scores_by_model: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """
    Plot Precision-Recall curves for all models on the same figure.

    This is particularly useful when classes are imbalanced:
    PR curve is more informative than ROC in skewed datasets.
    """
    plt.figure(figsize=(8, 6))

    for name, scores in scores_by_model.items():
        precisions, recalls, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        plt.plot(
            recalls,
            precisions,
            label=f"{name} (AP={ap:.3f})",
        )

    # Baseline: positive class ratio
    pos_ratio = y_test.mean()
    plt.hlines(
        pos_ratio,
        xmin=0.0,
        xmax=1.0,
        colors="gray",
        linestyles="dashed",
        label=f"Baseline (pos ratio={pos_ratio:.3f})",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves (spam = positive class)")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved PR curves for all models to {out_path}")


# -----------------------------
# 7. Main training pipeline
# -----------------------------

def log_experiments_mlflow(all_metrics: Dict[str, Dict[str, Any]]) -> None:
    """
    Log experiments for each model to MLflow.
    Each model gets its own run, with:
      - params: model_name, score_type
      - metrics: default_* and tuned_* (accuracy, precision, recall, f1)
      - metrics: roc_auc, pr_auc
    """
    # Local file-based tracking inside the repo
    tracking_dir = PROJECT_ROOT / "mlruns"
    mlflow.set_tracking_uri(str(tracking_dir))
    mlflow.set_experiment("spam_classifier")

    for model_name, m in all_metrics.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("score_type", m.get("score_type", "unknown"))

            # Core metrics (default threshold)
            default = m.get("default", {})
            for key in ("accuracy", "precision", "recall", "f1"):
                if key in default:
                    mlflow.log_metric(f"default_{key}", float(default[key]))

            # Tuned metrics (threshold from PR curve)
            tuned = m.get("tuned", {})
            for key in ("accuracy", "precision", "recall", "f1"):
                if key in tuned:
                    mlflow.log_metric(f"tuned_{key}", float(tuned[key]))
            if "threshold" in tuned and tuned["threshold"] is not None:
                mlflow.log_metric("tuned_threshold", float(tuned["threshold"]))

            # Global metrics based on scores
            if m.get("roc_auc") is not None:
                mlflow.log_metric("roc_auc", float(m["roc_auc"]))
            if m.get("pr_auc") is not None:
                mlflow.log_metric("pr_auc", float(m["pr_auc"]))

            # You can also log artifacts later (e.g. PR curve image, metrics json)


def run_training(sample_size: int | None = None) -> Dict[str, Any]:
    """
    End-to-end training run:
      - load data
      - clean / normalize
      - train/test split
      - TF-IDF vectorization
      - train multiple models
      - evaluate (ROC, PR, default vs tuned threshold)
      - choose best model by tuned F1
      - save best model + vectorizer + metadata + metrics
      - plot label distribution, message length, PR curves

    Returns a dict with:
      {
        "best_model_name": str,
        "metrics": {...},   # metrics for all models
      }

    `sample_size` can be used in tests to train on a smaller subset.
    """
    ensure_dirs()

    df = load_raw_data(DATA_PATH)

    if sample_size is not None and sample_size < df.shape[0]:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"[DEBUG] Using a sample of {sample_size} rows for training")

    # Class-imbalance quick analysis: label distribution
    plot_label_distribution(df["label"].values)
    msg_lengths = df["text"].astype(str).str.len().values
    plot_message_length_hist(msg_lengths)

    # Split and vectorize
    X_train, X_test, y_train, y_test = train_test_split_text(df)
    vectorizer = fit_vectorizer(X_train)

    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train models
    models = train_models(X_train_vec, y_train)

    # Compute scores on test set for each model
    scores_by_model: Dict[str, np.ndarray] = {}
    score_types: Dict[str, str] = {}

    for name, model in models.items():
        scores, s_type = get_scores_and_type(model, X_test_vec)
        scores_by_model[name] = scores
        score_types[name] = s_type

    # Plot PR curves using all models
    pr_plot_path = PLOTS_DIR / "precision_recall_curves.png"
    plot_pr_curves(y_test, scores_by_model, pr_plot_path)

    # Evaluate and pick best by tuned F1
    best_name, all_metrics = pick_best_model(
        models=models,
        y_test=y_test,
        scores_by_model=scores_by_model,
        score_types=score_types,
    )
    
    # Log experiments to MLflow for all models
    log_experiments_mlflow(all_metrics)

    # Persist vectorizer and best model
    best_model = models[best_name]

    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    model_path = MODELS_DIR / "spam_classifier.joblib"
    metadata_path = MODELS_DIR / "model_metadata.json"
    metrics_path = METRICS_DIR / "classification_metrics.json"

    dump(vectorizer, vectorizer_path)
    dump(best_model, model_path)
    print(f"[SAVE] Saved TF-IDF vectorizer to {vectorizer_path}")
    print(f"[SAVE] Saved best classifier ({best_name}) to {model_path}")

    # Save metrics and metadata
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Extract tuned threshold for the best model
    tuned_info = all_metrics[best_name].get("tuned", {})
    best_threshold = tuned_info.get("threshold", None)

    metadata = {
        "best_model_name": best_name,
        "vectorizer_path": str(vectorizer_path),
        "model_path": str(model_path),
        "n_samples": int(df.shape[0]),
        "labels": {"ham": 0, "spam": 1},
        "best_threshold": best_threshold,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SAVE] Saved metrics to {metrics_path}")
    print(f"[SAVE] Saved model metadata to {metadata_path}")
    print("[DONE] Training + evaluation (with PR curves & tuned thresholds) completed.")

    return {
        "best_model_name": best_name,
        "metrics": all_metrics,
    }


if __name__ == "__main__":
    # Full training run on the entire dataset
    run_training(sample_size=None)
