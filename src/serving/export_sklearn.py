from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import joblib

from ..nlp_lab.dataset import load_spam_csv, stratified_split
from ..nlp_lab.metrics import evaluate_binary_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "models" / "b3" / "sklearn_tfidf_svm"


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT))
        return out.decode("utf-8").strip()
    except Exception:
        return None


def export_sklearn_baseline(
    csv_path: Path,
    out_dir: Path,
    seed: int = 42,
    max_train: Optional[int] = None,
) -> Path:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_spam_csv(csv_path)
    split = stratified_split(df)

    x_train = split.x_train_raw
    y_train = split.y_train
    x_test = split.x_test_raw
    y_test = split.y_test

    # optional downsample for speed while debugging
    if max_train is not None and max_train < len(x_train):
        idx = np.random.RandomState(seed).choice(len(x_train), size=max_train, replace=False)
        x_train = x_train[idx]
        y_train = y_train[idx]

    # validation split for threshold tuning (avoid tuning on test!)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.15,
        random_state=seed,
        stratify=y_train,
    )

    # Baseline pipeline
    # - TfidfVectorizer for sparse lexical features
    # - LinearSVC is strong for text
    # - CalibratedClassifierCV gives predict_proba
    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                strip_accents="unicode",
            )),
            ("clf", CalibratedClassifierCV(
                estimator=LinearSVC(C=1.0),
                method="sigmoid",
                cv=3,
            )),
        ]
    )

    pipe.fit(x_tr, y_tr)

    # tune threshold on val
    val_probs = pipe.predict_proba(x_val)[:, 1]
    mb_val = evaluate_binary_classifier(y_val, val_probs, default_threshold=0.5)
    tuned_thr = float(mb_val.tuned["threshold"])

    # evaluate on test using tuned threshold
    test_probs = pipe.predict_proba(x_test)[:, 1]
    mb_test = evaluate_binary_classifier(y_test, test_probs, default_threshold=tuned_thr)

    # save
    joblib.dump(pipe, out_dir / "model.joblib")

    meta: Dict[str, Any] = {
        "model": "tfidf(1-2gram) + LinearSVC calibrated(sigmoid)",
        "threshold": tuned_thr,
        "seed": seed,
        "max_train": max_train,
        "git_sha": _git_sha(),
        "val": {
            "default_0.5": mb_val.default,
            "tuned": mb_val.tuned,
        },
        "test_at_tuned_threshold": {
            "accuracy": mb_test.default["accuracy"],
            "precision": mb_test.default["precision"],
            "recall": mb_test.default["recall"],
            "f1": mb_test.default["f1"],
            "confusion_matrix": mb_test.default["confusion_matrix"],
            "roc_auc": mb_test.roc_auc,
            "pr_auc": mb_test.pr_auc,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[B3] Saved sklearn model: {out_dir / 'model.joblib'}")
    print(f"[B3] Saved sklearn meta : {out_dir / 'meta.json'}")
    return out_dir


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "spam_emails.csv"))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=0, help="0 means use full train")
    args = p.parse_args()

    max_train = None if args.max_train == 0 else int(args.max_train)
    export_sklearn_baseline(Path(args.csv), Path(args.out), seed=args.seed, max_train=max_train)


if __name__ == "__main__":
    main()
