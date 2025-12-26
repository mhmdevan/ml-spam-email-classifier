from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from ..nlp_lab.dataset import load_spam_csv, stratified_split
from ..nlp_lab.metrics import evaluate_binary_classifier

from .backends import OnnxTextClassifier, TorchTextClassifier, SklearnTextClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "b3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _sample(x: np.ndarray, y: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0 or n >= len(x):
        return x, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx], y[idx]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["onnx", "torch", "sklearn"], required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "spam_emails.csv"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-len", type=int, default=256)

    # optional sampling (keeps it practical)
    p.add_argument("--train-max", type=int, default=0, help="0=full")
    p.add_argument("--val-max", type=int, default=0, help="0=full")
    p.add_argument("--test-max", type=int, default=0, help="0=full")

    args = p.parse_args()

    df = load_spam_csv(Path(args.csv))
    split = stratified_split(df)

    x_train = split.x_train_raw
    y_train = split.y_train
    x_test = split.x_test_raw
    y_test = split.y_test

    # optional sampling
    x_train, y_train = _sample(x_train, y_train, args.train_max, args.seed)
    x_test, y_test = _sample(x_test, y_test, args.test_max, args.seed + 1)

    # Split train -> train/val for threshold tuning (NO peeking at test)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.15,
        random_state=args.seed,
        stratify=y_train,
    )
    x_val, y_val = _sample(x_val, y_val, args.val_max, args.seed + 2)

    model_dir = Path(args.model_dir).resolve()

    if args.backend == "onnx":
        model = OnnxTextClassifier(model_dir, max_length=args.max_len)
    elif args.backend == "torch":
        model = TorchTextClassifier(model_dir, max_length=args.max_len)
    else:
        model = SklearnTextClassifier(model_dir)

    # Tune threshold on VAL
    val_probs = model.predict_proba(list(x_val))
    val_report = evaluate_binary_classifier(y_val, val_probs, default_threshold=0.5)
    tuned_thr = float(val_report.tuned["threshold"])

    # Evaluate on TEST using tuned threshold
    test_probs = model.predict_proba(list(x_test))
    test_report = evaluate_binary_classifier(y_test, test_probs, default_threshold=tuned_thr)

    out: Dict[str, Any] = {
        "backend": args.backend,
        "model_dir": str(model_dir),
        "seed": args.seed,
        "max_len": args.max_len,
        "splits": {
            "train": int(len(x_tr)),
            "val": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "threshold": {
            "tuned_on_val": tuned_thr,
        },
        "val": {
            "roc_auc": float(val_report.roc_auc),
            "pr_auc": float(val_report.pr_auc),
            "default": val_report.default,
            "tuned": val_report.tuned,
        },
        "test": {
            "roc_auc": float(test_report.roc_auc),
            "pr_auc": float(test_report.pr_auc),
            "default_at_tuned_threshold": test_report.default,
            "tuned": test_report.tuned,
        },
    }

    tag = model_dir.name.replace("/", "_")
    out_path = OUT_DIR / f"eval_{args.backend}_{tag}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[B3] Saved: {out_path}")


if __name__ == "__main__":
    main()
