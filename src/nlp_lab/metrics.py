from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)


@dataclass
class MetricBundle:
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    default: Dict[str, Any]
    tuned: Dict[str, Any]


def _predict_with_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def tune_threshold_for_best_f1(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    if thresholds.size == 0:
        thr = 0.5
        y_pred = _predict_with_threshold(y_scores, thr)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
        return {
            "threshold": float(thr),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        }

    best_thr = 0.5
    best_f1 = -1.0
    best_p = 0.0
    best_r = 0.0

    for i, thr in enumerate(thresholds):
        p = precisions[i + 1]
        r = recalls[i + 1]
        if (p + r) == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_p = float(p)
            best_r = float(r)

    y_pred = _predict_with_threshold(y_scores, best_thr)
    return {
        "threshold": float(best_thr),
        "precision": float(best_p),
        "recall": float(best_r),
        "f1": float(best_f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    default_threshold: float = 0.5,
) -> MetricBundle:
    y_pred = _predict_with_threshold(y_scores, default_threshold)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)

    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        roc_auc = None

    try:
        pr_auc = float(average_precision_score(y_true, y_scores))
    except ValueError:
        pr_auc = None

    tuned = tune_threshold_for_best_f1(y_true, y_scores)

    return MetricBundle(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        default={
            "threshold": float(default_threshold),
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        },
        tuned=tuned,
    )
