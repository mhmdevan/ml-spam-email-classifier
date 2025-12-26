from __future__ import annotations

import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .backends import OnnxTextClassifier, SklearnTextClassifier, TorchTextClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "b3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _p(x: List[float], q: float) -> float:
    return float(np.quantile(np.array(x, dtype=float), q))


def bench_one(
    name: str,
    model,
    texts: List[str],
    batch_size: int,
    runs: int = 50,
    warmup: int = 10,
) -> Dict[str, Any]:
    # warmup
    for _ in range(warmup):
        model.predict_proba(texts[:batch_size])

    times: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.predict_proba(texts[:batch_size])
        t1 = time.perf_counter()
        times.append(t1 - t0)

    p50 = _p(times, 0.50)
    p95 = _p(times, 0.95)
    avg = float(np.mean(times))
    throughput = batch_size / avg if avg > 0 else 0.0

    return {
        "backend": name,
        "batch_size": batch_size,
        "runs": runs,
        "warmup": warmup,
        "p50_ms": p50 * 1000.0,
        "p95_ms": p95 * 1000.0,
        "avg_ms": avg * 1000.0,
        "throughput_items_per_sec": throughput,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", default=str(PROJECT_ROOT / "models" / "b3" / "onnx_distilbert_ft_opset18"))
    parser.add_argument("--skl-dir", default=str(PROJECT_ROOT / "models" / "b3" / "sklearn_tfidf_svm"))
    parser.add_argument("--torch-dir", default=str(PROJECT_ROOT / "models" / "b1" / "distilbert_spam"))
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batches", default="1,4,8,16,32,64")
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    skl_dir = Path(args.skl_dir)
    torch_dir = Path(args.torch_dir)

    batches = [int(x.strip()) for x in args.batches.split(",") if x.strip()]

    # Instantiate models
    onnx = OnnxTextClassifier(onnx_dir, max_length=args.max_len)
    skl = SklearnTextClassifier(skl_dir)
    torch_backend = TorchTextClassifier(torch_dir, max_length=args.max_len)

    # Use realistic-ish texts (repeat to have enough for slicing)
    texts = [
        "free money click now!!!",
        "hey can we reschedule the meeting?",
        "limited time offer!!! claim your prize now",
        "invoice attached please review and confirm payment",
        "Congratulations! You've won a gift card. Click to redeem.",
        "Team update: standup moved to 11:30 tomorrow.",
        "URGENT: verify your account immediately to avoid suspension",
        "Lunch tomorrow? Let me know what time works.",
    ] * 300

    rows: List[Dict[str, Any]] = []
    for bs in batches:
        rows.append(bench_one("sklearn", skl, texts, batch_size=bs, runs=args.runs, warmup=args.warmup))
        rows.append(bench_one("torch", torch_backend, texts, batch_size=bs, runs=args.runs, warmup=args.warmup))
        rows.append(bench_one("onnx", onnx, texts, batch_size=bs, runs=args.runs, warmup=args.warmup))

    df = pd.DataFrame(rows)
    out = OUT_DIR / "latency.csv"
    df.to_csv(out, index=False)

    # Print a compact view sorted by batch/backend
    print(df.sort_values(["batch_size", "backend"]).reset_index(drop=True))
    print(f"[B3] Saved: {out}")


if __name__ == "__main__":
    main()
