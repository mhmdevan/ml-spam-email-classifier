from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np

from .dataset import load_spam_csv, stratified_split
from .models.peft_seqcls import train_b2, B2Config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "b2"
MODELS_DIR = PROJECT_ROOT / "models" / "b2"


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None or v == "" else int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None or v == "" else float(v)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else str(v)


def run(
    csv_path: str = str(PROJECT_ROOT / "data" / "spam_emails.csv"),
) -> Path:
    _ensure_dirs()

    # Controls
    variant = _env_str("B2_VARIANT", "lora")  # linear_probe | lora | qlora
    base_model = _env_str("B2_BASE_MODEL", "distilbert-base-uncased")

    train_max = os.getenv("B2_TRAIN_MAX")
    eval_max = os.getenv("B2_EVAL_MAX")
    train_max = None if not train_max else int(train_max)
    eval_max = None if not eval_max else int(eval_max)

    cfg = B2Config(
        base_model_name=base_model,
        variant=variant,
        max_length=_env_int("B2_MAX_LENGTH", 256),
        num_train_epochs=_env_float("B2_EPOCHS", 2.0),
        per_device_train_batch_size=_env_int("B2_TRAIN_BS", 16),
        per_device_eval_batch_size=_env_int("B2_EVAL_BS", 32),
        learning_rate=_env_float("B2_LR", 2e-4),
        weight_decay=_env_float("B2_WD", 0.01),
        seed=_env_int("B2_SEED", 42),
        lora_r=_env_int("B2_LORA_R", 16),
        lora_alpha=_env_int("B2_LORA_ALPHA", 32),
        lora_dropout=_env_float("B2_LORA_DROPOUT", 0.05),
        dataloader_num_workers=_env_int("B2_WORKERS", 0),
        dataloader_pin_memory=False,
    )

    print("[B2] Loading dataset:", csv_path)
    df = load_spam_csv(Path(csv_path))
    split = stratified_split(df)

    print(f"[B2] Dataset size: {len(df)} | train={len(split.x_train_raw)} test={len(split.x_test_raw)}")
    print(f"[B2] Variant: {variant} | base_model: {base_model}")
    print(f"[B2] Sampling: train_max={train_max} eval_max={eval_max}")
    print(f"[B2] HP: epochs={cfg.num_train_epochs} lr={cfg.learning_rate} bs={cfg.per_device_train_batch_size} max_len={cfg.max_length}")

    out_dir = str(MODELS_DIR / f"{variant}_{base_model.replace('/', '_')}")
    report = train_b2(
        split.x_train_raw,
        split.y_train,
        split.x_test_raw,
        split.y_test,
        out_dir=out_dir,
        cfg=cfg,
        max_train_samples=train_max,
        max_eval_samples=eval_max,
    )

    metrics_path = OUT_DIR / "metrics_b2.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    rows: List[Dict[str, Any]] = [{
        "variant": variant,
        "base_model": base_model,
        "trainable_params": report["meta"]["params"]["trainable"],
        "total_params": report["meta"]["params"]["total"],
        "roc_auc": report["roc_auc"],
        "pr_auc": report["pr_auc"],
        "f1_default": report["default"]["f1"],
        "f1_tuned": report["tuned"]["f1"],
        "saved": report["meta"]["saved"],
        "out_dir": report["out_dir"],
        "device": report["meta"]["device"],
        "quantization": report["meta"]["quantization"],
    }]

    compare_df = pd.DataFrame(rows)
    compare_path = OUT_DIR / "compare_b2.csv"
    compare_df.to_csv(compare_path, index=False)

    print(f"[B2] Saved metrics: {metrics_path}")
    print(f"[B2] Saved comparison: {compare_path}")
    print(f"[B2] Model artifacts: {out_dir}")

    return compare_path


if __name__ == "__main__":
    out = run()
    print(f"[B2] Done. Compare file: {out}")
